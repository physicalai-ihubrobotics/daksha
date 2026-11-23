---
name: object-localization
description: Locate visible object points in the robot's frame from the current observation, choosing the method the observation profile allows (depth with camera geometry, RGB with camera geometry, or RGB with measured robot motion only). Use before choosing any approach, grasp, placement or drive target, and again after anything moved.
---

# Object localization

Ground every target in the current observation. Never reuse pixels or positions after
the camera, the robot base or the object moved, and never replace missing geometry with
guessed coordinates. Fill in the robot's facts in [This robot](#this-robot); the
methods below are the same everywhere.

## Pixels

- Select pixels in the original, full-resolution image of the current observation, not
  in a downscaled preview or a composed video; convert preview pixels with the scale the
  reply gives.
- Pick several pixels inside the object's visible surface, away from edges, shadows and
  occluders: edge pixels take the depth of the background.
- Prefer features that look distinct from every view (tips, corners, marks); symmetric
  features are easy to mismatch between views.

## Choose the method

**Depth and camera geometry.** Back-project pixels with `daksha_pixel_to_position` (or with
the depth image, intrinsics and camera pose in your own code,
[references/geometry.md](references/geometry.md)). Discard points with invalid or
sparse depth, or far from the others. Cross-check from a second view or camera before a
precise action; disagreement beyond the robot's tolerance means a wrong pixel, so
reselect rather than average.

**RGB with camera geometry.** One pixel gives a ray, not a point. Either intersect the
ray with a known plane (a support surface of known height; use the object's contact
point with it, not its top), or triangulate the same physical feature from two camera
positions far enough apart that the rays meet at a wide angle with a small gap. Check an
estimate by projecting it back into a new image where the robot offers that.

**RGB only, no camera geometry.** Images give directions, not metric positions. Learn
how the image maps to robot motion by moving an empty gripper a small, known
displacement and watching its visible feature. Align the same gripper feature with the
target in fresh images from more than one view, with small motions near a safe height.
One image's overlap does not prove alignment or contact (parallax); confirm in a second
view or after a small motion. A learned mapping holds only near that height, orientation
and camera.

## Use the result carefully

- A localized point is a visible surface point, not an object centre, pose, grasp or
  clearance. Average several top-surface points for a centre, offset toward the centre
  by about half the object's thickness for a grasp, and choose approach offsets from
  current evidence.
- Close views resolve far more detail than overview cameras; get one with
  [camera-inspection](../camera-inspection/SKILL.md) for contact-critical targets, or
  confirm by touch where the robot senses force.
- Record in memory what you localized against, the views and the method, not one
  layout's coordinates.

## This robot

| Fact | Value |
|---|---|
| Cameras, resolution, what moves each | `world` (face, fixed), `left` and `right` (wrists, move with the arm; `left_wrist` / `right_wrist` on the real robot); 640 x 480 each in simulation, real wrists 848 x 480 ([robot-interface](../robot-interface/SKILL.md)) |
| Where a frame's images, depth, intrinsics and camera pose are saved | `frames/<frame_id>/<camera>.png`, `<camera>_depth.png`, `meta.json` (real robot: also `distortion`, which is not zero there: undistort pixels first, see [references/geometry.md](references/geometry.md)) |
| Depth units and invalid value; where depth is unreliable | uint16 millimetres, 0 = invalid (beyond 3 m); a pixel covered by an arm or gripper returns that occluder, not the object behind it |
| Robot frame and units of returned points | robot frame (origin on the floor under the robot, x forward, y left, z up), metres |
| Measured surface accuracy of `daksha_pixel_to_position` on unoccluded fruit surfaces | median 5 mm (world), 0.1-5 mm (wrist); p95 7-14 mm on the wrist cameras. The world camera's p95 was 0.3 m because arms at home cover some fruit: check that a point's height is plausible (docs/characterization.json) |
| Cross-view tolerance | points of the same feature from two cameras should agree within ~1 cm; more means a wrong or occluded pixel |
| Known plane heights | table top z = 0.85 m; floor z = 0 |
| Robot-body pixels to ignore | in `world`, the grippers and forearms at the left and right image edges; in the wrist cameras, the dark finger tips at the image border |

A point on a round object is on its visible surface: the centre of a sphere seen from
above is about one radius below its top point. Image right in `world` is the robot's
right (-y).
