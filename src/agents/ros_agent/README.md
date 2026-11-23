# ros_agent: Daksha agent (Q&A + real-robot VLM pick-and-place)

One ROS 2 agent, one web page, with an optional Placo debug viewer:

| Tab | What | Moves the robot? |
|---|---|---|
| **Ask** | read-only Q&A: URDF/SRDF, live state, any topic, cameras, LLM (Ollama) | never |
| **Robot tasks** | the `local_daksha_rax` VLM pipeline driving the **real** Daksha | only with **Execute** on |
| **IK Debug** | live Placo viewer from `aruco_hand_eye` | no; viewer node is forced to dry run |

```bash
cd ~/robot && source install/setup.bash
ros2 launch ros_agent agent.launch.py                 # http://<robot>:8150/
ros2 launch ros_agent agent.launch.py ik_debug:=true  # also embed Placo IK debug view
ros2 launch ros_agent agent_with_aruco_reach.launch.py  # agent + /aruco_reach/go_pose
#   reach_execute:=true      arm /joint_cmd execution (default: dry run)
#   calib_ui:=true           also start the hand-eye calibration UI
#   pipeline:=false          Ask tab only
#   pipeline_execute:=true   start armed (default: dry run)
#   pipeline_model:=9b       pipeline VLM (default qwen3.5:35b, config/pipeline.yaml)
#   llm_model:=qwen3.5:35b   Ask-tab model
```

Needs Ollama on :11434 (`ollama list`), the cameras (`camera_launch`), the arm
bringup (`/joint_states`, `joint_command_limiter`), a world-camera calibration from
`aruco_hand_eye`, and **for motion, `ros2 launch aruco_hand_eye aruco_reach.launch.py`**.
The pipeline sends target positions to aruco_reach's actions
(`aruco_hand_eye` actions). aruco_reach is the only `/joint_cmd` sender and owns
the execute switch, so the page's **Enable execute** toggles aruco_reach's.

## Layout

```
ros_agent/
├── ros_agent/
│   ├── agent_node.py      main: Ask agent + PipelineNode on one executor
│   ├── web.py             the page's server (Ask API + /api/pipeline/*)
│   ├── pipeline_node.py   ROS 2 node around local_pipeline's LocalAgent
│   ├── real_backend.py    the daksha_* tools on the real robot
│   └── ...                Ask agent (tools, topics, cameras, tracker, LLM client)
├── web/chat.html          the page (tabs: Ask | Robot tasks)
├── config/
│   ├── agent.yaml, robots.yaml   Ask agent
│   └── pipeline.yaml             real-robot pipeline settings
├── launch/agent.launch.py
├── local_daksha_rax/      the RAX/MuJoCo project, moved here as-is (not installed;
│                          has COLCON_IGNORE and its own README / LOCAL_PIPELINE.md)
└── test/
```

`local_daksha_rax` keeps working on its own (sim pipeline, RAX harness, scripts);
`source local_daksha_rax/scripts/activate_local.sh` as before. The ROS node imports
its `local_pipeline` from the source tree at runtime.

## How the pipeline runs on the real robot

The planner, skills and VLM client are unchanged `local_pipeline` code. Only the
robot behind them is swapped: `LocalAgent(..., backend=RosRobotBackend)`, which
implements the same five tools the sim provides:

| Tool | Real-robot implementation |
|---|---|
| `daksha_observe` | world camera frame + its calibrated pose (`aruco_hand_eye` result, `world_calibration`) |
| `daksha_pixel_to_position` | aligned depth if `camera.depth_topic` is live; otherwise the camera ray hitting a plane `plane_offset_m` above the table |
| `daksha_move_eef` | table / workspace checks here, then **`/aruco_reach/move_to_pose`** (IK, straight-line or joint trajectory, self-collision check, `min_tcp_z` = table floor along the whole path) |
| `daksha_move_joints` | **`/aruco_reach/move_joints`** |
| `daksha_gripper` | **`/aruco_reach/gripper`** (width + stall on an object) |
| `daksha_home` | **`/aruco_reach/move_joints` named `home`** (gen2 `HomeMoveService.DEFAULT_HOME_POSITION`) |

Robot frame = URDF `world` (x forward, y left, z up). `right_joint_6` is sent
negated to cancel `joint_command_limiter`'s inversion (measured; see
`aruco_hand_eye/README.md` → Joint signs).

## Using Robot tasks

1. **Calibration:** the world camera must be calibrated in `aruco_hand_eye`
   (calibration within 10 mm / 2°, or **Use anyway** for this session).
2. **Table height:** lay the 80 mm ArUco (ID 0) flat on the table and press
   **Measure table** (or set `scene.table_z`). The live camera can identify
   arbitrary objects, but actions that need 3-D positions require table height.
3. Talk to it: *look* · *what is on the table?* · *pick up the box* ·
   *move the bottle to the left* · *home* · *open*. Objects named in commands
   are matched against the live camera; the `objects` setting is not an allowlist.
4. It starts in **dry run**: it perceives and plans, then says *"Dry run, not
   moving: planned …"*. Press **Enable execute** to move the robot.
   **STOP / Esc** interrupts at any time.

## ROS interface (pipeline)

| Name | Type | |
|---|---|---|
| `/pipeline/command` | `std_msgs/String` (sub) | instruction or question |
| `/pipeline/stop` | `std_srvs/Trigger` | interrupt now, drop the queue |
| `/pipeline/set_execute` | `std_srvs/SetBool` | arm / disarm real motion |
| `/pipeline/measure_table` | `std_srvs/Trigger` | table height from the ArUco on the table |
| `/pipeline/status` | `std_msgs/String` (pub, 2 Hz) | JSON state |

```bash
ros2 topic pub --once /pipeline/command std_msgs/String "{data: 'look'}"
ros2 service call /pipeline/set_execute std_srvs/srv/SetBool "{data: true}"
ros2 service call /pipeline/stop std_srvs/srv/Trigger
```

The Ask tab shows the left wrist, right wrist, and world camera feeds together; click
a feed to select it in the main view. It reads the same `camera_topics.yaml` as the
robot camera panel. The Ask agent's interface (`~/request`, `~/response`,
`~/describe_robot`, ...) is unchanged; see `agent_node.py`.

Launch with `ik_debug:=true` to add the **IK Debug** tab. It embeds the existing
Placo/Viser viewer from `aruco_hand_eye` at port 7000, showing live joints,
interactive TCP goals, and reach-node IK plans. The debug reach node is forced
to dry run; use the Robot tasks tab for pipeline motion.

`agent_with_aruco_reach.launch.py` starts both the agent and `aruco_reach`. Use
`/aruco_reach/go_pose` to submit a detected marker pose from another detector;
the reach node runs IK and collision-checked planning, then publishes to
`/joint_cmd` only after execution is enabled. Its default is dry run. For the
service type and request format, see `aruco_hand_eye/README.md`.

## Safety

- **Dry run by default:** nothing is published on `/joint_cmd` until Execute is on.
- **Refused targets:** anything outside the `workspace` box or below `table_z + min_clearance_m`.
- **Refused paths:** any path that self-collides or takes the TCP below the table.
- **Stops while moving:** operator STOP (every control tick), stale `/joint_states`,
  a joint lagging > 0.35 rad for 0.5 s, or a stall.
- **Not modelled:** objects on the table and people; there's no environment
  collision checking. Keep a hand near STOP.
- Don't run it together with VR teleop or `aruco_reach` executing (all use `/joint_cmd`).

## Tests

```bash
cd ~/robot && source install/setup.bash
cd src/agents/ros_agent && PYTHONPATH=.:$PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```

`test_real_backend.py` drives `RosRobotBackend` with a fake node (no robot): it
checks pixel → table position, the refusals and dry run (nothing published), an
executed move reaching its target, STOP mid-move, home to the gen2 pose, and
gripper stall detection.
