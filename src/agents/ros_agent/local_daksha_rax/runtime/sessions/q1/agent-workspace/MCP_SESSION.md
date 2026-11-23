# MCP session contract

Task `fruit_to_bowl`, robot `daksha_sim_ollama`, observation profile `rgbd`, exploration budget 24 episodes. Call these as MCP tools (server `robot`) or from Python with `ctx.call(name, **arguments)`.

## `status`

Session status: task, exploration episodes used and left, the current episode, and the live robot summary.

## `start_episode`

Close the current episode (if any) and start the next one: the robot returns to home, grippers open, and the task objects and target bowls are placed in a new layout. Consumes one exploration episode. Takes no seed or layout.

## `evaluate`

The trusted binary result of the current episode: task_complete true when every TASK.md condition holds now. Does not end the episode.

## `finish`

End the current episode (records its final evaluate result). The robot stays where it is until the next start_episode.

## `daksha_status`

Measured robot state: both arms' joints (rad), TCP poses (robot frame: origin on the floor under the robot, x forward, y left, z up; quaternions [x,y,z,w]; approach = TCP x axis, finger_axis = TCP y), gripper opening between the pads (m) and grip force (N), joint limits, the home posture and the simulation clock.

## `daksha_observe`

Render fresh images from the cameras: "world" (head, looking down at the table), "left" and "right" (on the wrists, looking along the approach axis past the fingers). Returns each RGB image, saves RGB, depth (uint16 PNG, millimetres, 0 = invalid) and camera intrinsics/extrinsics (OpenCV convention: x right, y down, z forward) under frames/<frame_id>/, and returns the frame_id for daksha_pixel_to_position.

| argument | type | required | notes |
|---|---|---|---|
| `cameras` | array |  | Default: all three. (minItems 1, maxItems 3) |
| `attach` | boolean |  | Return the images in the reply (default true); false only saves them.  |

## `daksha_pixel_to_position`

Robot-frame 3-D points for pixels of a saved frame, from its depth image and calibration. pixels are [u, v] (u = column from the left, v = row from the top). window takes the median depth of a (2*window+1)^2 patch, ignoring invalid pixels. Depth is the visible surface: for a round object the point is on its near side, not its centre.

| argument | type | required | notes |
|---|---|---|---|
| `frame_id` | string | yes |   |
| `camera` | string world/left/right | yes |   |
| `pixels` | array | yes |  (minItems 1, maxItems 256) |
| `window` | integer |  |  (minimum 0, maximum 10) |

## `daksha_move_joints`

Move one arm to joint positions (rad) along a smooth joint-space path. Checked before moving: URDF limits, per-call step, and a predicted-collision sweep of the whole path. Stops and holds on contact of an arm link with the table, torso or other arm. Returns measured joints and TCP pose.

| argument | type | required | notes |
|---|---|---|---|
| `arm` | string left/right | yes |   |
| `positions` | array | yes | joint_1..joint_7 of this arm, rad. (minItems 7, maxItems 7) |
| `speed` | number |  | Peak joint speed rad/s (default 0.6). (minimum 0.05, maximum 1.2) |

## `daksha_move_eef`

Move one arm's TCP (the fingertip end of the gripper, centred between the fingers) to a position, with an optional orientation (quat, or approach [+ finger_axis] with tolerance_deg). Without orientation the current orientation is kept within tolerance_deg (default 10). path "linear" (default) moves the TCP in a straight line with interpolated orientation; "joint" solves IK at the goal and moves in joint space (use it for large repositioning). relative=true treats position as an offset from the current TCP position. Refused before moving if IK fails, the line is infeasible, or the path would collide.

| argument | type | required | notes |
|---|---|---|---|
| `arm` | string left/right | yes |   |
| `position` | array | yes |  (minItems 3, maxItems 3) |
| `quat` | array |  | TCP orientation quaternion [x, y, z, w] in the robot frame. (minItems 4, maxItems 4) |
| `approach` | array |  | Approach direction (TCP x axis, wrist to fingertips), robot frame. Roll about it is free unless finger_axis is given. (minItems 3, maxItems 3) |
| `finger_axis` | array |  | Finger-opening direction (TCP y axis); needs approach. Sign does not matter. (minItems 3, maxItems 3) |
| `tolerance_deg` | number |  | Allowed deviation of approach/finger_axis (default 0). Daksha's wrist is narrow: 15-35 deg lets IK choose a feasible tilt; the solver still prefers the smallest tilt it can reach (6, then 12 deg, then this). (minimum 0, maximum 90) |
| `finger_tolerance_deg` | number |  | Separate allowed deviation of finger_axis (default: tolerance_deg). (minimum 0, maximum 90) |
| `relative` | boolean |  |   |
| `path` | string linear/joint |  |   |
| `speed` | number |  | TCP speed m/s for linear paths (default 0.1). (minimum 0.01, maximum 0.25) |

## `daksha_move_eef_both`

Move both TCPs at the same time, in one synchronized motion: both start and finish together (use it to lift, carry and set down an object held in both hands). left and right take the daksha_move_eef targets (position + optional orientation; without orientation each hand keeps its own). relative=true treats both positions as offsets from the current TCPs (e.g. [0, 0, 0.1] for each to lift 10 cm). path "linear" (default) moves each TCP in a straight line, "joint" in joint space. The whole two-arm path is checked first (IK, workspace, collisions incl. arm vs arm) and stops on contact like daksha_move_eef.

| argument | type | required | notes |
|---|---|---|---|
| `left` | object | yes |   |
| `right` | object | yes |   |
| `relative` | boolean |  |   |
| `path` | string linear/joint |  |   |
| `speed` | number |  | TCP speed m/s of the faster hand (default 0.1). (minimum 0.01, maximum 0.25) |

## `daksha_check`

Dry run, nothing moves: solve IK for a TCP target (same arguments as daksha_move_eef) and check the path; or, with joints, compute the TCP pose (FK). Returns reachable, the joint solution, the predicted TCP pose and errors, and the first problem on the path if any.

| argument | type | required | notes |
|---|---|---|---|
| `arm` | string left/right | yes |   |
| `position` | array |  |  (minItems 3, maxItems 3) |
| `quat` | array |  | TCP orientation quaternion [x, y, z, w] in the robot frame. (minItems 4, maxItems 4) |
| `approach` | array |  | Approach direction (TCP x axis, wrist to fingertips), robot frame. Roll about it is free unless finger_axis is given. (minItems 3, maxItems 3) |
| `finger_axis` | array |  | Finger-opening direction (TCP y axis); needs approach. Sign does not matter. (minItems 3, maxItems 3) |
| `tolerance_deg` | number |  | Allowed deviation of approach/finger_axis (default 0). Daksha's wrist is narrow: 15-35 deg lets IK choose a feasible tilt; the solver still prefers the smallest tilt it can reach (6, then 12 deg, then this). (minimum 0, maximum 90) |
| `finger_tolerance_deg` | number |  | Separate allowed deviation of finger_axis (default: tolerance_deg). (minimum 0, maximum 90) |
| `relative` | boolean |  |   |
| `path` | string linear/joint |  |   |
| `joints` | array |  |  (minItems 7, maxItems 7) |

## `daksha_gripper`

Open or close one gripper. "open" opens fully (0.089 m between the pads); "close" closes with a force limit and stops when the fingers stall on an object; "width" moves to an opening in metres. Returns the measured opening, grip force and whether the fingers stalled before closing fully (something is between them).

| argument | type | required | notes |
|---|---|---|---|
| `arm` | string left/right | yes |   |
| `command` | string open/close/width | yes |   |
| `width` | number |  |  (minimum 0.0, maximum 0.089) |
| `force` | number |  | Grip force limit N (default 25.0). (minimum 2, maximum 40.0) |

## `daksha_home`

Move one or both arms to the home posture (TCPs ~0.20 m above the table at x=0.27, y=+-0.24, pointing down) along checked joint-space paths, one arm after the other. Grippers keep their state.

| argument | type | required | notes |
|---|---|---|---|
| `arm` | string left/right/both |  |   |
| `speed` | number |  |  (minimum 0.05, maximum 1.2) |

## `vlm_query`

Ask the local vision-language model (Ollama, LOCAL_VLM_MODEL) about a saved camera frame. Give the frame_id from daksha_observe and a camera. With `detect` (object names) it returns boxes and centre points in PIXELS of that image: [{"label", "bbox": [x1,y1,x2,y2], "point": [u,v]}], ready for daksha_pixel_to_position. Otherwise it answers `prompt` (set json=true for a JSON reply). A timeout, refusal or truncated reply comes back as an error: treat it as no answer, never as a negative. Takes ~5 s per call on this machine; ask between motions, not during them.

| argument | type | required | notes |
|---|---|---|---|
| `frame_id` | string | yes |   |
| `camera` | string world/left/right | yes |   |
| `prompt` | string |  |   |
| `detect` | array |  |  (maxItems 12) |
| `json` | boolean |  |   |
| `max_tokens` | integer |  |  (minimum 16, maximum 2048) |

