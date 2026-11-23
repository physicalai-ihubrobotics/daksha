# local_daksha_rax: Daksha + local Qwen VLM (Ollama: qwen3.5:35b / qwen3.5:9b / qwen3-vl:8b)

This folder is a separate copy of `daksha_rax`. Nothing in `../daksha_rax` or `../rax` is changed.
It adds two things on top of the copy:

1. **`local_pipeline/`**: an interactive, interruptible pipeline. You talk to Daksha in the fruit
   scene, the selected local Ollama vision model (default `qwen3.5:35b`) sees the head camera, plans, and finds the objects, and the robot
   carries out the plan. The browser shows the live sim and qwen3-vl's detections drawn on the camera image.
2. **RAX integration**: the robot backend `daksha_sim_ollama` adds a `vlm_query` MCP tool backed by the
   same local model. That is the `<vlm>` tool that RAX's `vlm-perception` skill expects (copied into
   `agent_skills/vlm-perception/`). A Claude Code RAX session can then use local VLM perception.


> **Real robot (ROS 2):** this folder now lives inside the `ros_agent` package. The same
> pipeline (planner, skills, VLM) runs on the real Daksha through `ros_agent`'s
> `RosRobotBackend`, in the **Robot tasks** tab of `ros2 launch ros_agent agent.launch.py`
> (http://<robot>:8150). See `../README.md`. Everything below (MuJoCo sim, RAX) works as before.

## Requirements

```bash
cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
source scripts/activate_local.sh
scripts/setup_local.sh       # installs MuJoCo into daksha_venv and ensures the selected Ollama model exists
# Ollama should be running on :11434; start it with `ollama serve` if needed
```

## 1. Interactive pipeline (interact + interrupt)

```bash
cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
MUJOCO_GL=egl PYTHONPATH=$PWD /home/ihub/daksha_venv/bin/python -m local_pipeline.run
# open http://localhost:8092
```

Choose the model with `--model` (default `35b`):

| `--model` | Ollama model | detection on this Thor |
|---|---|---|
| `35b` (default) | `qwen3.5:35b` (MoE, vision) | ~3–5 s, best plans |
| `9b` | `qwen3.5:9b` | ~6–7 s |
| `8b` | `qwen3-vl:8b` | ~5–8 s |

Any other Ollama model name with vision also works. For the RAX MCP tool, set `LOCAL_VLM_MODEL=35b|9b|8b` before `launch`.

Other options: `--seed 1003`, `--task fruit_toy_sorting`, `--realtime 2` (sim twice as fast as wall clock), `--host 0.0.0.0`,
`--port`, `--model`, `--ollama http://host:11434`, `--no-repl`, `--do look` (run a command at start).

Type at the `you>` prompt or in the browser:

| You say | What happens |
|---|---|
| `look` | arms move aside, qwen3-vl boxes every fruit and the bowl, depth turns the boxes into 3-D positions, overlay shown |
| `put the apple in the bowl` | perceive → pick → place → home (retries once) |
| `put all the fruit in the bowl` / free text | qwen3-vl plans the actions from the instruction + the current camera image |
| `what is on the table?` (any question) | answered from the live frame **without stopping the robot** |
| `stop` / STOP button / **Esc** / **Ctrl-C** | **interrupts at the next 100 Hz control tick**; the arm holds where it stopped, queued commands are dropped |
| a new motion command while one runs | **barge-in**: the running command is interrupted, then the new one runs |
| `home`, `open`, `reset`, `reset 1005` | home the arms, open the grippers, new layout |

Run files go to `runtime/local_pipeline/<name>/` (`stream/latest.jpg` live, `stream/vlm.jpg` overlay,
`workspace/frames/` every observation).

### Say it your own way

Every message goes to the local VLM with the live head-camera image, a scene memory (where each object was last seen,
what each hand holds, what is running) and the last few chat turns. There is no fixed phrase list. Examples that work:
"grab the red one and drop it in the white dish", "pick the apple and hold it" (picks and **keeps holding**), "now put it
in the bowl", "put the banana next to the orange", "move the orange to the left side", "hand me the orange", "lift your
right hand a bit", "let go", "tidy up the table", "then go home" (queued after the current task), "apple ko bowl mein daal do".
Questions and chat are answered without stopping the robot. Unknown objects get a clarifying question.

The robot does only what you asked: it looks from where its arms are (it moves them out of the camera's view only if a
needed object is hidden), goes straight to the object without homing first, and puts things down on free spots that keep
clear of the bowl and the other fruit. Keyword rules are only a fallback for when the model gives no answer.

### How it works

```
operator text ─┬─ question ──► qwen3-vl (live head frame) ──► answer          (robot keeps moving)
               ├─ stop ──────► interrupt Event ──► _monitor() → MotionStopped (arm holds)
               └─ command ───► interrupt current ─► queue ─► worker thread (only one touching the sim)
                                   rule or qwen3-vl plan ─► pick_place(obj, bowl):
                                   clear view → daksha_observe → qwen3-vl detect (bbox + point)
                                   → daksha_pixel_to_position on the box → 3-D centre (banana: yaw)
                                   → daksha_move_eef / gripper → place → home
```

- `local_pipeline/ollama_vlm.py`: stdlib Ollama client. Detection uses a JSON schema
  (qwen3-vl otherwise sometimes writes a `<think>` block that runs out of tokens). It converts
  Qwen's 0..1000 coordinates to pixels.
- `local_pipeline/robot.py`: `InterruptibleBackend` (`DakshaSimBackend` + interrupt + real-time pacing) and
  the skills. If the VLM gives no answer, colour masks are the fallback (the overlay marks it `[fallback]`).
- `local_pipeline/agent.py`: worker, planner, interrupt rules, overlay. `local_pipeline/web.py`: the UI.
- The pill at the top of the UI shows the simulator's trusted task check (`inside`, `COMPLETE`), which is separate
  from what the VLM believes.

Speed on this Jetson Thor: a question or plan takes 2–5 s, a 4-object detection takes 5–8 s (the reply is prefilled as
`{"objects": [`, with schema-constrained decoding as the retry), and a cold model load takes about 75 s (the pipeline warms it
at start). STOP halts the arm in about 0.5 s, even while a VLM call is still running. After placing, the pipeline
re-checks with qwen3-vl and retries anything that is not well inside the bowl.

## 2. RAX agent session with local VLM perception

```bash
/home/ihub/daksha_venv/bin/python -m harness.launcher prepare --name q1 --robot daksha_sim_ollama
/home/ihub/daksha_venv/bin/python scripts/viewer.py --name q1 &
/home/ihub/daksha_venv/bin/python -m harness.launcher launch --name q1
```

The agent then has `vlm_query(frame_id, camera, detect=[...] | prompt=..., json=...)` next to the usual
`daksha_*` tools, and controllers can call it with `ctx.call('vlm_query', ...)`.

## 2b. RAX agent on a local Ollama model, and on the real robot

The RAX agent is Claude Code; with `--llm ollama` it talks to Ollama's Anthropic-compatible
`/v1/messages` endpoint instead of Claude, and the harness, sandbox, MCP tools and skills
are unchanged. A model copy with a large context (`rax-<model>:64k`, same weights) is made
automatically, because Ollama's default context window truncates the agent's instructions.

```bash
source scripts/activate_local.sh && export PYTHONPATH=$PWD MUJOCO_GL=egl
$DAKSHA_PYTHON -m harness.launcher prepare --name s1 --robot daksha_sim_ollama
$DAKSHA_PYTHON -m harness.launcher launch  --name s1 --llm ollama --allow-unsandboxed
#   --ollama-model 9b   --ollama-ctx 32768
```

`--allow-unsandboxed` is needed on this machine because bubblewrap is unavailable (true for
Claude too).

**Real robot:** `--robot daksha_ros` (in a shell with the ROS 2 workspace sourced, and
`ros2 launch aruco_hand_eye aruco_reach.launch.py` running). Same tool names; perception
from ros_agent; motion through aruco_reach's actions, in dry run until its execute switch is
on. On the real robot `start_episode` does not rearrange the table (you do), and `evaluate`
cannot verify the task. Explore and write the controller in sim first.

## 3. Human-posture IK (`kinematics/` package)

`local_pipeline/human_ik.py` is a separate script and module. It solves IK with your `kinematics/kinematics/arm_ik_solver.py`
(`RobotKinematics`: placo, URDF joint limits, posture task, and the self-collision constraint from
`kinematics/config/collision_pairs.json`) on `assets/daksha/robot_collision.urdf`, and keeps a **human arm posture**:

- **TCP position**: a *hard* task (exact), softened only if infeasible;
- **approach / finger axis**: TCP x axis = approach (e.g. straight down), optional finger axis or a full quaternion;
- **elbow down**: the elbow (`link_5`) is pulled softly to about 17 cm below the shoulder (`link_3`), a little forward and
  tucked in, with the upper arm hanging and no "chicken wing";
- **posture task**: redundant joints relax toward a human reference posture, which is solved from the arm hanging straight down
  (all-zero joints) with the elbow bent;
- **multi-seed**: current, human, hanging and sim-home seeds; it picks the solution that reaches the target with the lowest elbow cost.

**Robot model: `assets/daksha/daksha_raw.xml`** (MuJoCo MJCF, copied from `daksha_rax/scratch/`). placo reads only URDF, so
`local_pipeline/mjcf_to_urdf.py` converts it to `runtime/ik/daksha_raw.urdf`: body tree, joint axes, limits and inertia from the XML,
meshes from `assets/daksha/meshes`, and self-collision shapes taken from `robot_collision.urdf`'s boxes and cylinders by link name
(`--collision mesh` uses the XML's own meshes instead, which is exact but much slower). Placo FK on the converted file equals MuJoCo on
`daksha_raw.xml` (0.0000 mm over 200 random poses). `--robot-model urdf` switches back to `robot_collision.urdf`.

```bash
PYTHONPATH=$PWD /home/ihub/daksha_venv/bin/python -m local_pipeline.mjcf_to_urdf          # regenerate the URDF
```

The placo world frame equals the sim's robot frame (FK matches to 0.1 mm). placo 0.11.0 and the other Python
dependencies come from `/home/ihub/daksha_venv`; MuJoCo is installed there by `scripts/setup_local.sh`. The project
sets `PYTHONPATH` to this folder so its `services/` and `local_pipeline/` modules resolve locally:

```bash
cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
# solve one target (prints joints, errors, elbow vs shoulder)
PYTHONPATH=$PWD /home/ihub/daksha_venv/bin/python -m local_pipeline.human_ik --arm left --pos 0.30 0.20 0.90 --approach 0 0 -1
# sim IK vs human IK on sample targets -> runtime/ik/demo/sim_vs_human_ik.png
PYTHONPATH=$PWD MUJOCO_GL=egl /home/ihub/daksha_venv/bin/python -m local_pipeline.human_ik --demo
# the interactive pipeline + web UI with human-posture IK and qwen3.5:35b
PYTHONPATH=$PWD MUJOCO_GL=egl /home/ihub/daksha_venv/bin/python -m local_pipeline.run --model 35b --ik human --host 0.0.0.0
```

With `--ik human`, every reach, grasp descent and lift (a straight line, solved every 1 cm and warm-started so the posture stays
continuous), every place, and home (the human ready posture) are solved by this IK. They are then executed through the sim's
own joint-limit check, collision sweep, contact monitor and STOP. Two exceptions:

- if the human IK finds no solution (it happened for some banana grasps, where the finger axis plus the narrow wrist ask too much),
  that move falls back to the sim's IK, and the log says so;
- the short "clear the camera view" gesture before each detection uses the sim's sideways pose, because Daksha's reach has no
  human-like pose that clears the head camera.

Tested: the full fruit task completed on layouts 1000, 1003 (robot_collision.urdf) and 1005 (daksha_raw.xml) with qwen3.5:35b and `--ik human`.

Python use:

```python
from local_pipeline.human_ik import HumanPostureIK
ik = HumanPostureIK('left')
r = ik.solve([0.30, 0.20, 0.90], approach=(0, 0, -1), state=current_joint_dict, tolerance_deg=30)
r['q'], r['ok'], r['position_error_m'], r['human_cost']
qs = ik.solve_line(q_now, [0.30, 0.20, 0.87])   # straight line, posture kept continuous
```
