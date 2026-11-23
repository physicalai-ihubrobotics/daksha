# Daksha RAX — MuJoCo

A RAX-style robot-agent harness for **Daksha** in MuJoCo, built like RAX's `real/franka`
and `real/rainbow` branches. Claude Code drives the robot only through MCP tools. It learns
the task from its own episodes, then writes `code/<project>/controller.py:main(ctx)`, which is
evaluated alone on held-out layouts.

Default task: **pick the orange, the apple and the banana and place them in the bowl.**
The `fruit_toy_sorting` task adds three small toys and a separate bowl for them.

Nothing in `~/uni/daksha` is modified: the URDF, meshes and joint limits are copies in `assets/daksha/`.

## Layout

| Path | What |
|---|---|
| `assets/daksha/` | copied URDF, meshes, collision meshes, `arm_limit.yaml` |
| `services/daksha_sim/scene.py` | MuJoCo scene: Daksha + table + bowl + fruit, actuators, cameras |
| `services/daksha_sim/kinematics.py` | IK/FK on the TCP (approach-cone targets for the narrow wrist) |
| `services/daksha_sim/backend.py` | the robot backend: `daksha_*` tools, validation, contact monitor, cameras |
| `services/daksha_sim/tasks.py` | seeded layouts (checked reachable) and the trusted success check |
| `services/daksha_sim/sorting.py` | seeded fruit/toy layouts and the two-bowl completion check |
| `services/controller/frontend.py` | the session's MCP server: `status`, `start_episode`, `evaluate`, `finish` + robot tools |
| `harness/launcher.py` | prepare a session, start the robot service, run Claude Code |
| `harness/mcp_bridge.py` | stdio MCP bridge Claude Code spawns |
| `auto_research_agent/` | `AGENTS.md`, prompts, `api/runtime.py` (Python `Context`) |
| `agent_skills/` | skills copied into each workspace (`robot-interface`, `grasping`, ...) |
| `scripts/evaluate_controller.py` | formal evaluation on held-out seeds 0..N-1 |
| `scripts/viewer.py` | live browser view of a session |
| `scripts/characterize.py` | measured motion/perception numbers → `docs/characterization.json` |
| `services/calibration.py` | camera extrinsics loader (RAX `real/franka` format): `fixed` head camera, `tcp` wrist cameras |
| `services/calibration_check.py` | FK-projection check: renders the gripper from the calibrated camera, compares with the image and depth |
| `scripts/export_sim_calibration.py` | writes the sim cameras' extrinsics as calibration files (template; `--perturb-*` for a wrong one) |
| `scripts/check_calibration.py` | moves the arms, checks every camera's calibration, saves overlays + `report.json` |
| `examples/baseline_controller/` | hand-written perception-only controller (operator reference, hidden from the agent) |
| `local_tests/oracle_pick_place.py` | feasibility check with privileged poses |

## Robot tools (what the agent gets)

`daksha_status`, `daksha_observe` (world/left/right RGB + depth + calibration),
`daksha_pixel_to_position`, `daksha_check` (dry-run IK/FK), `daksha_move_eef`,
`daksha_move_joints`, `daksha_gripper`, `daksha_home`, plus `status`, `start_episode`,
`evaluate` and `finish`. These are the same tool names a real-Daksha ROS 2 backend
would register (`services/backends.py`), so a controller written in sim can be pointed at hardware.

## Run

```bash
cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
source scripts/activate_local.sh
scripts/setup_local.sh
export PYTHONPATH=$PWD MUJOCO_GL=egl

# 1. a session (workspace + Claude Code settings)
/home/ihub/daksha_venv/bin/python -m harness.launcher prepare --name s1 --episodes 24 --robot daksha_sim_ollama

# 2. watch it live (open http://localhost:8090)
/home/ihub/daksha_venv/bin/python scripts/viewer.py --name s1 &

# 3. the agent (interactive TUI; add --non-interactive for a headless run)
/home/ihub/daksha_venv/bin/python -m harness.launcher launch --name s1

# 4. evaluate the controller it wrote on held-out layouts
/home/ihub/daksha_venv/bin/python scripts/evaluate_controller.py --name s1 --project <project>
```

The baseline controller can be evaluated the same way:

```bash
cp -r examples/baseline_controller runtime/sessions/s1/agent-workspace/code/baseline
/home/ihub/daksha_venv/bin/python scripts/evaluate_controller.py --name s1 --project baseline
```

Prepare a clean sorting session without replacing a completed fruit-only session:

```bash
/home/ihub/daksha_venv/bin/python -m harness.launcher prepare --name sort1 --task fruit_toy_sorting
/home/ihub/daksha_venv/bin/python scripts/viewer.py --name sort1 --host 0.0.0.0 --port 8090
```

## Full isolation (bubblewrap)

This host restricts unprivileged user namespaces (Ubuntu 24.04 default), so bubblewrap cannot run.
That blocks Claude Code's Bash sandbox and the evaluator's sandbox. `launch` refuses to start
until you either enable bubblewrap or pass `--allow-unsandboxed`. With that flag only
permission rules apply: the Read/Edit tools are confined to the workspace, but shell commands
could read other files. To enable bubblewrap once (needs sudo):

```bash
sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>
profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo apparmor_parser -r /etc/apparmor.d/bwrap
```

## Camera calibration

Same format and conventions as RAX `real/franka` (`services/franka/calibration.py`). Each camera's
extrinsics is a YAML file with a 4×4 `matrix`; the calibration utility also records the camera intrinsics and solve report:

- `world` (head RealSense, mount `fixed`): `T_cam_base`, P_cam = T · P_base. Daksha has no neck joint.
- `left`/`right` (wrist RealSenses, mount `tcp`): `T_cam_tcp` against `<arm>_tcp`, P_cam = T · P_tcp.

Camera frame: OpenCV optical (x right, y down, z forward). The cameras are the backend option `cameras`:

```json
{"cameras": [
  {"name": "world", "mount": "fixed", "extrinsics": "world.yaml"},
  {"name": "left", "mount": "tcp", "arm": "left", "extrinsics": "left.yaml"},
  {"name": "right", "mount": "tcp", "arm": "right", "extrinsics": "right.yaml"}]}
```

Relative paths resolve under `$DAKSHA_CALIBRATION_DIR`. A camera spec may also include `intrinsics`
(`width`, `height`, `fx`, `fy`, `cx`, `cy`); the observation metadata then uses this calibrated K, scaled to
the output resolution. In simulation, listed cameras report the pose their calibration gives (as on the real
robot) and unlisted ones the simulator's exact pose. `daksha_observe` saves `K` and `T_base_cam` per camera in
`meta.json`.

**Checking a calibration** (RAX: "verify it by projecting known FK points into the image"):
`scripts/check_calibration.py` moves each arm through 5 poses above the table. At each pose it renders
the gripper's URDF meshes from the calibrated camera pose, using the measured joints and gripper opening.
Each image gets an overlay: green is the predicted outline, orange is where the depth puts it, and a magenta
cross marks the TCP. Where there is depth on the fingers, the check measures the image shift and the depth
offset, both in mm. A camera passes when the median error is ≤ `--max-offset-mm` (default 4).
Without depth (for example, wrist fingers closer than the RealSense minimum range), judge the overlays by eye.

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
/home/ihub/daksha_venv/bin/python scripts/export_sim_calibration.py --out runtime/calibration/sim      # exact
/home/ihub/daksha_venv/bin/python scripts/check_calibration.py --backend-options runtime/calibration/sim/backend_options.json
/home/ihub/daksha_venv/bin/python scripts/export_sim_calibration.py --out runtime/calibration/bad --perturb-mm 15 --perturb-deg 2
/home/ihub/daksha_venv/bin/python scripts/check_calibration.py --backend-options runtime/calibration/bad/backend_options.json  # FAIL
# real robot, once a daksha_ros backend is registered: --robot daksha_ros [--no-move] [--speed ...]
```

Results go to `runtime/calibration_checks/<stamp>/`, and the exit code is 1 if any camera fails.

**Producing a camera calibration**: `scripts/calibrate_cameras.py` creates an ArUco target, captures images
and TCP poses from the active MCP robot service, solves the fixed world camera from a marker at a measured
robot-frame pose, and solves each wrist camera with OpenCV hand-eye calibration. It writes `world.yaml`,
`left.yaml`, `right.yaml`, and a compatible `backend_options.json` under `runtime/calibration/real/`.
Print the marker at its specified black-square size. Keep the marker fixed for all wrist samples; move each
arm to at least 8 varied poses and capture a sample at each pose. The world marker position is its centre in
the robot frame, including table height.
The marker solver requires the OpenCV contrib package (`uv pip install --python /home/ihub/daksha_venv/bin/python opencv-contrib-python`).

**Importing real hand-eye results from ROS `aruco_hand_eye`**: the ROS package saves
`eye_in_hand_{left,right}_wrist.yaml` and `eye_to_hand_world_{left,right}.yaml` under
`/home/ihub/robot/src/aruco_hand_eye/config/`. Save each camera's ROS `CameraInfo` message as YAML;
the importer reads its `width`, `height`, and `k` fields. The right/left wrist transform is converted
from `*_gripper_mount` to the simulator's `*_tcp` using the fixed joint in `assets/daksha/robot.urdf`.
The world transform is inverted to the local backend's `T_camera_base` convention. Only import one
world result after comparing the two hand-held-target solves.

```bash
# Source the ROS workspace where aruco_hand_eye is built, then start the cameras and hand-eye UI.
source /opt/ros/jazzy/setup.bash
source /home/ihub/robot/install/setup.bash
# ros2 launch camera_launch start_all_cameras.launch.py
# ros2 launch aruco_hand_eye daksha_hand_eye.launch.py

# With ROS and camera bringup running:
cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
mkdir -p runtime/calibration/captures
export PYTHONPATH=$PWD MUJOCO_GL=egl
ros2 topic echo --once /right/camera/color/camera_info > runtime/calibration/captures/right_camera_info.yaml
ros2 topic echo --once /left/camera/color/camera_info > runtime/calibration/captures/left_camera_info.yaml
ros2 topic echo --once /camera/world/color/camera_info > runtime/calibration/captures/world_camera_info.yaml

# Import the saved ROS results into runtime/calibration/real/.
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py import-hand-eye \
  --setup right_wrist --result /home/ihub/robot/src/aruco_hand_eye/config/eye_in_hand_right_wrist.yaml \
  --intrinsics runtime/calibration/captures/right_camera_info.yaml
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py import-hand-eye \
  --setup left_wrist --result /home/ihub/robot/src/aruco_hand_eye/config/eye_in_hand_left_wrist.yaml \
  --intrinsics runtime/calibration/captures/left_camera_info.yaml
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py import-hand-eye \
  --setup world_right --result /home/ihub/robot/src/aruco_hand_eye/config/eye_to_hand_world_right.yaml \
  --intrinsics runtime/calibration/captures/world_camera_info.yaml

# Review the resulting camera transforms against the rendered camera/depth data.
/home/ihub/daksha_venv/bin/python scripts/check_calibration.py \
  --backend-options runtime/calibration/real/backend_options.json
```

The importer rejects ROS solves above 10 mm translation RMS or 2° rotation RMS by default. These
limits can be changed with `--max-t-rms-mm` and `--max-r-rms-deg`; review the ROS residuals before
raising them. Importing updates only the named camera in `backend_options.json` and writes its
matrix/intrinsics to `runtime/calibration/real/<camera>.yaml`.

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py marker --out runtime/calibration/aruco_4x4_id0.png --size-mm 80

# With the robot MCP service running, capture the world camera frame at a known marker pose.
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py capture --camera world --count 1 --out runtime/calibration/captures
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py fixed \
  --image runtime/calibration/captures/world_00.png \
  --intrinsics runtime/calibration/captures/world_intrinsics.json \
  --marker-size 0.08 --marker-position 0.40 0.0 0.85 --marker-yaw-deg 0

# For each wrist, keep the marker stationary and reposition that arm between captures.
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py capture --camera left --count 12 --out runtime/calibration/captures
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py wrist --arm left \
  --samples runtime/calibration/captures/left_samples.json \
  --intrinsics runtime/calibration/captures/left_intrinsics.json
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py capture --camera right --count 12 --out runtime/calibration/captures
/home/ihub/daksha_venv/bin/python scripts/calibrate_cameras.py wrist --arm right \
  --samples runtime/calibration/captures/right_samples.json \
  --intrinsics runtime/calibration/captures/right_intrinsics.json

# Check the resulting calibration before using its backend options.
/home/ihub/daksha_venv/bin/python scripts/check_calibration.py --backend-options runtime/calibration/real/backend_options.json

# Start a new session using these calibrated transforms and intrinsics.
/home/ihub/daksha_venv/bin/python -m harness.launcher prepare --name calibrated1 \
  --backend-options runtime/calibration/real/backend_options.json
```

`capture` is read-only: it saves the current image, intrinsics and TCP pose; the operator moves the arm
between wrist samples. The shipped backend registry currently includes `daksha_sim`; connect a real robot
service before capturing real-camera data. A simulated calibration can still exercise the workflow.

## Simulation choices (differences from the real robot)

- The base is fixed. The table top is at 0.85 m and its front edge 4 cm from the torso,
  because Daksha's arms reach about 0.65 m from shoulders 1.28 m up.
- Joint position actuators with gravity compensation stand in for the PosVelEff/MIT
  controllers plus gravity feed-forward.
- The `left`/`right` cameras are mounted on the wrists. The real `left`/`right` RealSense
  placement wasn't checked, so change `scene.py` if it differs. The `world` camera is on the face.
- The finger collision meshes are replaced by box pads of the same size, for stable grasps.

## Task `tray_transport` (two tables, bimanual tray carry)

Two identical tables 5 m apart (centre to centre, the second along -y, the robot's right). A blue
10"x7"x3" lab tray holding the orange, apple and banana sits on the pick table, and a white outline marks
the drop-off zone on the place table. For this task only, Daksha gets a planar mobile base: joints and
position actuators `base_x`, `base_y`, `base_yaw`, home at x = -0.03 so the torso clears the table edge.
The `overview`, `pick_view` and `place_view` cameras are for viewing only.

Reference grasp (operator-side, used to keep layouts solvable): one hand per short end. Each hand is
half-closed to a 60 mm gap and comes straight down (tilt up to 20°), with the fingertips 15 mm below the
rim, 25 mm off centre on the banana side, straddling the end wall and handle lip. Success means the tray
rests upright inside the outline (±6 cm, ±15°), all three fruits are still in it, and no gripper touches it.

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
/home/ihub/daksha_venv/bin/python scripts/check_tray_scene.py --out runtime/tray_check      # settle, reach, clearance, drive, success check
/home/ihub/daksha_venv/bin/python -m harness.launcher prepare --name tray1 --task tray_transport --model claude-opus-5-5
```

Not done yet (step 2): base navigation tools for the agent (`daksha_move_base`), and making the
backend's workspace and table checks follow the base pose.
