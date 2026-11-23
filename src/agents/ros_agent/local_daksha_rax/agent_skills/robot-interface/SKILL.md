---
name: robot-interface
description: Daksha's tools, frames, units, limits, measured motion, grippers, cameras and failure handling in this MuJoCo session. Read before composing any robot action or writing robot code.
---

# Daksha robot interface

Daksha is a humanoid torso on a wheeled base with **two 7-joint arms**, each ending in a
parallel gripper with two sliding fingers. It stands at a table. Three cameras: `world`
on the face looking down at the table, and `left` / `right` on the wrists (called
`left_wrist` / `right_wrist` on the real robot; both namings are accepted there). This session
is a MuJoCo simulation of the real robot (same URDF, joint limits and gripper travel);
the robot reaches you only through these tools, and the world only changes while a
motion tool runs.

## Frames and units

- Robot frame: origin on the floor under the robot, **x forward, y left, z up**. Metres,
  radians. Quaternions are **[x, y, z, w]**. It moves with the base: every arm target, TCP
  pose, camera pose and `daksha_pixel_to_position` point is in the robot frame *at the time
  of the call or frame*.
- **TCP** (the pose every Cartesian tool uses): the fingertip end of the gripper,
  centred between the fingers. Its axes:
  - x = **approach** (wrist toward fingertips),
  - y = **finger_axis** (the direction the fingers open and close),
  - z = x cross y.
- Finger pads: 6 cm wide (along TCP z), 3.2 cm thick, and they span from the TCP to
  **9 cm behind it** along the approach, centred 4.5 cm behind the TCP. A straight-down
  grasp with the TCP 1-2 cm above the table therefore holds an object between about
  1 cm and 10 cm above the table.
- The table top is at **z = 0.85 m**; its front edge is at x = 0.17 m, about 4 cm in
  front of the torso. The base is fixed unless the tool list has `daksha_move_base` (see
  "Driving the base" below).
- Shoulders are at z = 1.28 m, y = +0.13 (left) and -0.13 (right). Each arm reaches
  about 0.65 m from its shoulder.

## Tools by level

| Level | Tool | Use it for | Moves |
|---|---|---|---|
| Session | `status` | episode budget, current episode, live robot summary | no |
| Episode | `start_episode`, `evaluate`, `finish` | a fresh layout; the trusted binary result; closing an episode | `start_episode` resets |
| Sense | `daksha_status` | joints, TCP poses, gripper opening and force, limits, home | no |
| | `daksha_observe` | fresh RGB (+ depth, calibration) from the cameras; saved under `frames/<frame_id>/` | no |
| | `daksha_pixel_to_position` | pixels of a saved frame to robot-frame 3-D points | no |
| Dry run | `daksha_check` | IK and path check for a TCP target, or FK for joints (real robot: IK + limits only) | no |
| Task space | `daksha_move_eef` | one TCP to a position (+ orientation); straight line or joint-space | yes |
| | `daksha_move_eef_both` | both TCPs together in one synchronized motion (lift, carry, set down with two hands); real robot: two moves started together | yes |
| Joint space | `daksha_move_joints` | one arm to 7 joint values (postures, recovery) | yes |
| Gripper | `daksha_gripper` | open, close with a force limit, or go to a width | yes |
| Posture | `daksha_home` | one or both arms to the home posture | yes |
| Base (if listed) | `daksha_move_base` | drive: forward / left / turn_deg relative to the robot | yes |

Choose the highest level that expresses the intent: `daksha_move_eef` for hand
positions, `daksha_home` or `daksha_move_joints` for postures and recovery. There is no
collision-aware planner: long motions are sequences of checked moves through clear,
high poses.

## Orientation: ask for "roughly", not "exactly"

Daksha's wrist is narrow (joint 6 is limited to +-45 deg, joint 5 turns one way only), so
an exact orientation is often unreachable. Give `approach` (and `finger_axis` when the
fingers must straddle something elongated) with `tolerance_deg` 20-35. The solver tries
6 deg, then 12 deg, then your tolerance, and returns the first that works, so you get
the most accurate orientation the arm can reach. `finger_tolerance_deg` sets the finger
axis tolerance separately. Without any orientation, a move keeps the current orientation
within 10 deg. The reply's `tcp.approach` and `tcp.finger_axis` say what you got.

`finger_axis` is symmetric: [1, 0, 0] and [-1, 0, 0] are the same grasp.

## Which tool for which motion (measured, docs/characterization.json)

| Motion | Tool and arguments | Measured accuracy | Measured time |
|---|---|---|---|
| Reposition over the table (> 10 cm, or orientation change) | `daksha_move_eef`, `path: "joint"` | 0.8 mm median, 1.8 mm p95 TCP error in free space | ~1.4 s sim (median), ~0.25 s wall |
| Straight approach or retreat near objects | `daksha_move_eef`, default `path: "linear"`, `speed` 0.05-0.10, often `relative: true` | same | distance / speed x 1.9 (smooth profile) |
| Posture change, recovery | `daksha_home`, `daksha_move_joints` | joint tracking < 0.001 rad in free space | peak 0.6 rad/s default |
| Gripper | `daksha_gripper` | opening read back from the fingers | < 0.5 s |

- A joint-space path does not move the TCP in a straight line: it can swing low between
  start and goal. The path sweep refuses it if any sample would hit the table, so choose
  start and goal poses that are high enough, or go through an intermediate point.
- A straight line can be refused at a wrist flip ("joint jump"); then use `path: "joint"`
  from a higher point, or split the line.
- When the fingers press on something, the TCP stops short of the target: the reply's
  `contacts` lists it and `position_error_m` shows how far.
- Per call: at most 0.45 m of straight-line travel and 3.2 rad per joint.

## Limits and refusals

Checked **before** anything moves (a refusal is a wall, not a fault; choose another
target or approach, never search around it):

- joint limits (URDF, minus 0.005 rad). `daksha_status` lists them;
- the TCP may not go below the table top, and must stay in x -0.2..0.85, y -0.65..0.65,
  z 0.3..1.7;
- predicted collisions along the whole path: arm links with the table, the torso or the
  other arm; fingers more than 4 mm into the table;
- IK failure, with the best error and the distance from the shoulder.

Checked **during** motion: the motion stops and the arm holds where it stopped (error
type `MotionStopped`) if an arm link touches the table, the torso or the other arm, or
the fingers press on the table with more than 60 N. Read the error, observe, then move
from the measured state.

The arms can collide with each other near the middle of the table. When one arm works
near the centre, keep the other arm high or at home.

## Two hands on one object

An object held in both hands needs both hands to move **together**: use
`daksha_move_eef_both` (both start and finish at the same time). Moving one hand at a time
drags and twists the object. With `relative: true`, `{"left": {"position": [0, 0, 0.10]},
"right": {"position": [0, 0, 0.10]}}` lifts both hands 10 cm, each keeping its orientation.
The two-arm path is checked as a whole (including arm vs arm) before anything moves.

## Driving the base

Only when `daksha_move_base` is in the tool list (`daksha_status` then has `base.odom`).

- `forward`, `left` (m) and `turn_deg` (counter-clockwise) are relative to where the robot
  stands now, done together in a straight line with a smooth start and stop (default peak
  0.25 m/s; per call at most 6 m).
- The arms hold their joints while driving, so an object held in the grippers travels
  along. Lift it clear of the table first: a held object still resting on a table is
  refused.
- Checked first over the whole drive: the torso, the arms and any held object must not hit
  a table. Stops and holds on contact (`MotionStopped`).
- The torso is about 4 cm from the table edge at the start: back away (`forward` -0.3 or
  more) before driving sideways or turning near a table.
- After a drive the robot frame has moved: positions from older frames are stale. Observe
  again before using any position. `base.odom` (room frame, fixed at the start pose) is
  saved in each frame's `meta.json` if you need to relate frames.
- Odometry is exact in this simulation; the real robot's wheel odometry drifts, so always
  re-find the target with the cameras when you arrive.

## Grippers

- `open`: about 0.09 m between the pads. `close`: drives the fingers together with a
  force limit (`force`, default 25 N, max 40 N) and stops when they stall.
- The reply gives `width_m` (measured opening), `force_n` and `stalled_before_target`.
  A stalled close with a width matching the object means something is between the
  fingers; an empty close reads about 0.003-0.006 m.
- A held object stays held while the arm moves (the gripper keeps its command).

## Cameras

| Camera | Mount | Resolution | Notes |
|---|---|---|---|
| `world` | face, 1.53 m up, tilted 62 deg down | 640 x 480, vertical FOV 66 deg | sees the whole table; the arms and grippers occlude parts of it, especially at home |
| `left`, `right` | wrist, beside the fingers, looking along the approach | 640 x 480, vertical FOV 58 deg | close-ups; move with the arm; the finger tips appear at the image edge |

Depth is in millimetres (uint16 PNG, 0 = invalid), intrinsics and camera pose are saved
in `frames/<frame_id>/meta.json` (OpenCV convention). See
[object-localization](../object-localization/SKILL.md) for accuracy.

## Home and neutral state

`daksha_home` puts both TCPs about 0.20 m above the table at x = 0.27, y = +-0.24,
pointing down. Every `start_episode` starts there with both grippers open. Home is a
safe place to return to before switching arms or after an error.

## Errors

Every error reply carries `type`, `operation`, `stage` and the complete `reason`; a
stopped motion also carries `robot_state`. A tool error in Python raises
`api.runtime.ToolError` with `.error` holding the same object.

## On the real robot (`daksha_ros`)

The same tool names and arguments work on the real Daksha; these are the differences.

| Topic | Simulation | Real robot |
|---|---|---|
| Wrist camera names | `left`, `right` | `left_wrist`, `right_wrist` (`left` / `right` are accepted and mapped) |
| Motion safety switch | none | aruco_reach starts with **execute off**: every motion tool then returns `DRY RUN (execute is off ...)` with the plan and nothing moves. That is not a planning failure: the operator must press "Enable execute". Do not retry or choose another target |
| Table height | known (0.85 m) | must be set before any motion: `daksha_measure_table` (ArUco marker flat on the table) or the operator's config; until then every motion is refused with "the table height is not set" |
| Localising without depth | depth always rendered | the world camera has no depth: `daksha_pixel_to_position` intersects a plane just above the table; for a box use `daksha_footprint(frame_id, bbox)`. The wrist cameras have aligned depth |
| `tcp` in replies | `position`, `quat`, `approach`, `finger_axis` | the same keys (+ `orientation`, the same quaternion) |
| `finger_tolerance_deg` | separate finger-axis tolerance | accepted; shares `tolerance_deg` |
| `daksha_gripper` `force` | force-limited close; reply `force_n` | accepted but the gripper is position-controlled: reply `force_n` is `null`; judge a grasp by `stalled_before_target` and `width_m` |
| `daksha_check` | IK + path sweep | IK + joint / table / workspace limits; self-collision is checked when the move is planned |
| `daksha_move_eef_both` | one jointly planned motion | two moves started together, the call returns when both finished: give both the same distance / speed |
| `daksha_home` reply | `{"moved": [...]}` | `{"moved": [...], "<arm>": {"tcp"}}` |
| Frames | `frames/<frame_id>/<camera>.png`, `_depth.png`, `meta.json` | the same files (when the session has a workspace), plus `distortion` in `meta.json`: the real lenses are NOT distortion-free, undistort pixels before using `K` |
| Extra tools | | `daksha_measure_table`, `daksha_footprint`; the RAX UI agent also has the skills `daksha_find`, `daksha_track`, `daksha_pick_up`, `daksha_put_down` |
