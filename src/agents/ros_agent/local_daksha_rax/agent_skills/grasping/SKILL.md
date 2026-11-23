---
name: grasping
description: How Daksha's parallel grippers grasp, hold and release objects on the table: approach direction, TCP height, finger axis for elongated objects, checks before lifting, and releasing into containers. Read before the first grasp.
---

# Grasping with Daksha's grippers

Facts from [robot-interface](../robot-interface/SKILL.md), applied to grasping. Every
number here was measured in this simulation; check them against your own replies.

## Geometry that decides a grasp

- The TCP is at the **fingertips**; the pads reach 9 cm back from it along the approach,
  centred 4.5 cm behind it. The fingers open about 9 cm, so objects up to ~8.5 cm across
  the finger axis fit.
- From straight above, the pads' centre sits 4.5 cm above the TCP. With the TCP 1-2 cm
  above the table, the pads span 1-10 cm above the table: a small or low object sits near
  the lower end of the pads, so a grasp holds it below the pad centre.
- Tilting the approach away from vertical swings the pads' 6 cm-wide lower edge toward
  the table. In simulation keep the approach within ~12 deg of vertical near the table:
  give `approach: [0, 0, -1]` with a tolerance and read back the tilt you got.
- **Real robot:** Daksha's wrist cannot point straight down near the table (joint 6 is
  limited to +-45 deg, joint 5 turns one way): a vertical approach is refused by IK.
  Ask for 15-30 deg of tilt (e.g. `approach: [0.5, 0, -0.866]`, `tolerance_deg` 20) or a
  side grasp; the real backend also searches approaches tilted 15-30 deg by itself.
- Elongated objects: set `finger_axis` across the object's long axis (horizontal,
  perpendicular to it) with `finger_tolerance_deg` 20-30, and aim at the middle of its
  length.

## A grasp, step by step

1. Localize the object in a fresh observation and estimate its centre and extent
   ([object-localization](../object-localization/SKILL.md)).
2. Open the gripper.
3. Move above the object first (`path: "joint"`, 10-15 cm above it). Check the reply:
   the TCP position, the tilt (`tcp.approach`) and the finger axis.
4. Optional: look through that arm's wrist camera. Directly above the object, it should
   be centred between the fingers.
5. Descend in a straight line (`relative: true`, `speed` <= 0.08) to the grasp height.
   `relative: true` offsets are in the **robot frame** (x forward, y left, z up), not
   along the gripper: `[0, 0, -d]` goes straight down. With a tilted approach, descend
   ALONG it instead, `position = d * approach` (the unit `tcp.approach` from the last
   reply), or the pads scrape sideways into the object. Watch `contacts` and
   `position_error_m`: stopping short means the fingers landed on the object or the table.
6. Close with a moderate force (20-30 N; the real gripper ignores `force` and reports
   `force_n: null`). `stalled_before_target: true` with a `width_m` near the object's size
   means it is between the fingers; a width near 0 means you missed.
7. Lift straight up a few centimetres, then check again: `width_m` is unchanged and the
   object is visibly off the table in a fresh image (wrist or world).

## Carrying and releasing

- Keep a held object high (10+ cm above anything it could hit) while moving, and move
  the carrying arm with `path: "joint"` only between high poses.
- Release over a container's centre from low enough that the object cannot bounce out,
  but high enough that the fingers clear the rim. An object dropped from high can
  rebound or knock other objects.
- After releasing, raise the arm before moving sideways, so the fingers don't drag the
  object back out.
- Confirm placement in a fresh image after the scene settles.

## Two arms

Each arm reaches its own side of the table (left arm: y > ~0; right arm: y < ~0). Both
reach the table centre at x ~0.28-0.36 from above. Use the arm on the object's side, and
keep the other arm out of the way (home) when one works near the centre.
