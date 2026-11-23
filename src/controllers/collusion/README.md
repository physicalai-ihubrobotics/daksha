# collusion

Self- and environment-collision monitor for the Daksha arms, built on
[placo](https://github.com/Rhoban/placo) and coal (hpp-fcl).

It reads joint positions, measures the separation of every checked link pair
(robot link to robot link, and robot link to static obstacles like the table),
and publishes per-joint guards. One guard says a joint **has collided**
(reactive). The other says it **is about to collide** (predictive).

> This is a software monitor. It does not replace a hardware e-stop.

## Package layout

```
collusion/
├── collusion/
│   ├── collision_checker.py      # CollisionChecker: pure Python, no ROS
│   ├── collision_monitor_node.py # ROS 2 node wrapping the checker
│   ├── check_pose.py             # offline CLI: check one configuration
│   └── pydeps.py                 # PYTHONPATH helper for the isolated placo install
├── launch/
│   ├── collision_monitor.launch.py  # monitor only (for the real robot / IK pipeline)
│   └── collision_check.launch.py    # interactive: GUI sliders + RViz
└── config/
    ├── collision_monitor.yaml    # node parameters
    ├── obstacles.yaml            # static environment obstacles
    └── collision_check.rviz      # RViz layout for collision_check
```

## Installation

### 1. Install placo in its own directory

placo 0.11 is the only version with Python 3.12 wheels, and it requires
numpy 2. ROS Jazzy and the system scipy are built against numpy 1.26. To keep
them apart, install placo, coal, numpy 2 and a matching scipy into a separate
directory:

```bash
python3 -m pip install --target ~/.placo_py312 --break-system-packages \
    placo coal scipy
```

The launch files call `collusion.pydeps.placo_env()`, which puts
`~/.placo_py312` on the `PYTHONPATH` of the collision node only. Every other
process keeps the system numpy. If that directory doesn't exist, placo must be
importable some other way.

### 2. Build

```bash
cd ~/robot
colcon build --packages-select collusion
source install/setup.bash
```

Runtime dependencies from the workspace:

- `daksha_description_full_body`: `robot.urdf` (meshes) and
  `robot_collision.urdf` (primitives)
- `kinematics`: `config/collision_pairs.json` (the pair allowlist) and
  `config/joint_direction.yaml`

## Usage

### Live monitor

```bash
ros2 launch collusion collision_monitor.launch.py
ros2 launch collusion collision_monitor.launch.py joint_states_topic:=/hw/joint_states
```

This launch file loads `kinematics/config/joint_direction.yaml` too. The
monitor then divides each incoming angle by its joint direction, which turns
the angles published by `ik_node` back into URDF convention. Measured
feedback is the safer input. `/joint_states` from `ik_node` is the
*commanded* pose.

### Interactive check (sliders + RViz)

```bash
ros2 launch collusion collision_check.launch.py
ros2 launch collusion collision_check.launch.py geometry:=primitive
ros2 launch collusion collision_check.launch.py rviz:=false
```

This starts `robot_state_publisher`, `joint_state_publisher_gui`, the monitor
and RViz. Drag the joint sliders and watch:

- **Red spheres**: links in contact now (reactive)
- **Yellow spheres + line**: pairs about to collide (predictive), with the line
  across the remaining gap

New contacts are also logged to the console.

| Argument              | Default | Description                              |
|-----------------------|---------|------------------------------------------|
| `geometry`            | `mesh`  | `mesh` or `primitive` (see below)        |
| `rviz`                | `true`  | Start RViz                               |
| `contact_distance`    | `0.0`   | Reactive threshold (m)                   |
| `predictive_distance` | `0.01`  | Predictive threshold (m)                 |
| `prediction_horizon`  | `0.5`   | Look-ahead time (s)                      |

**Geometry modes**

- `mesh` uses the real STL collision meshes in `robot.urdf`. This is what RViz
  shows, so it is accurate. Each check takes about 190 ms, so the monitor runs
  at 2 Hz.
- `primitive` uses the enclosing cylinders and boxes in `robot_collision.urdf`.
  This is what the IK avoids. Each check takes about 1 ms, so the monitor runs
  at 30 Hz. The shapes are oversized: several pairs already overlap by up to
  about 22 mm at the all-zero pose.

The GUI publishes URDF-convention angles, so this launch file does **not**
load `joint_direction.yaml`.

### Offline check of one pose

No ROS graph needed:

```bash
ros2 run collusion check_pose left_joint_2=0.5 right_joint_4=-1.2
ros2 run collusion check_pose --obstacles my_cell.yaml --top 20 left_joint_1=1.0
```

Joints you don't list stay at 0. The output lists the closest pairs first and
marks overlaps with `COLLISION`. The format looks like this (the values are
made up):

```
self        left_link_5 <-> base_link                    -3.2 mm  COLLISION
env         left_link_7 <-> table                        41.0 mm
```

Options: `--urdf` (default `robot_collision.urdf`), `--pairs`, `--obstacles`,
`--top N` (default 10).

## Topics

**Subscribed**

| Topic                                   | Type                     |
|-----------------------------------------|--------------------------|
| `/joint_states` (`joint_states_topic`)  | `sensor_msgs/JointState` |

If the message has no velocities, the node estimates them from successive
positions and low-pass filters the result (`velocity_filter`).

**Published**

| Topic                          | Type                             | Meaning |
|--------------------------------|----------------------------------|---------|
| `/collision/predictive_guard`  | `sensor_msgs/JointState`         | Per joint, `position` = `1.0` if it **is going to** collide, else `0.0` |
| `/collision/reactive_guard`    | `sensor_msgs/JointState`         | Per joint, `position` = `1.0` if it **has** collided, else `0.0` |
| `/collision/joint_distance`    | `sensor_msgs/JointState`         | Per joint, `position` = closest separation (m) of the link it carries. `inf` = link in no checked pair, `nan` = stale data |
| `/collision/min_distance`      | `std_msgs/Float64`               | Closest separation over all pairs (m) |
| `/collision/markers`           | `visualization_msgs/MarkerArray` | Obstacles, contact points and predictive gaps for RViz |

```bash
ros2 topic echo /collision/predictive_guard
ros2 topic echo /collision/reactive_guard
ros2 topic echo /collision/joint_distance
```

## How it works

### Pair types

- **self**: robot link to robot link, through placo's
  `RobotWrapper.distances()`. Only pairs in the allowlist
  (`collision_pairs.json`) are checked. Parent/child pairs are always dropped,
  because they overlap at their shared joint by construction.
- **env**: robot link to a static obstacle from `obstacles.yaml`. placo has no
  notion of external objects, so these distances come straight from coal,
  using shapes rebuilt from the robot's collision objects at the current pose.

### Which joint gets flagged

A flagged pair flags only the movable joint that carries each link in contact.
The search walks up through fixed joints, so `left_gripper_mount` maps to
`left_joint_7`. Joint N moves link N+1, so `left_link_5` hitting `base_link`
flags `left_joint_4` alone. For an env pair, only the robot link's joint is
flagged. Links fixed to the root, such as `base_link`, flag nothing.

### Reactive vs predictive

On every tick:

1. **Reactive.** The node checks the current pose. Any pair within
   `contact_distance` flags its joints, and newly flagged joints are logged as
   errors.
2. **Predictive.** If any joint is moving (|v| > 1e-3 rad/s), the node
   extrapolates the pose `prediction_horizon` seconds ahead at the current
   velocity and checks again. Any pair within `predictive_distance` flags its
   joints. When nothing is moving, it reuses the reactive result, so the
   second check only costs time during motion.

### Fail-safe on stale data

If no joint state arrives for `joint_state_timeout` seconds, the node treats
the state as unknown:

- **every** joint is set in the predictive guard
- the reactive guard keeps its last value
- `joint_distance` is published as `nan`
- `min_distance` is `nan`

Missing data is treated as a possible collision.

## Configuration

### `config/collision_monitor.yaml`

| Parameter              | Default          | Description |
|------------------------|------------------|-------------|
| `urdf_file`            | `""`             | Empty: `daksha_description_full_body/urdf/robot_collision.urdf` |
| `collision_pairs_file` | `""`             | Empty: `kinematics/config/collision_pairs.json` |
| `obstacles_file`       | `""`             | Empty: `collusion/config/obstacles.yaml` |
| `joint_states_topic`   | `/joint_states`  | Input topic |
| `rate_hz`              | `50.0`           | Check rate |
| `check_self`           | `true`           | Check robot-to-robot pairs |
| `check_environment`    | `true`           | Check robot-to-obstacle pairs |
| `joint_state_timeout`  | `0.5`            | Seconds before the state counts as stale |
| `contact_distance`     | `0.0`            | Reactive threshold (m) |
| `predictive_distance`  | `0.03`           | Predictive threshold (m) |
| `prediction_horizon`   | `0.3`            | Look-ahead time (s) |
| `velocity_filter`      | `0.5`            | Low-pass factor on estimated velocity (1.0 = unfiltered) |
| `marker_frame`         | `world`          | Frame for RViz markers (placo's world = URDF root link, 0.22 m below `base_link`) |
| `joint_direction.<joint>` | `1.0`         | Sign map from `kinematics`, used to undo `ik_node`'s convention |

The thresholds and `check_*` flags are read on every tick, so you can change
them while the node runs:

```bash
ros2 param set /collision_monitor predictive_distance 0.05
```

**The URDF and the pair allowlist must come from the same geometry.**
Regenerate both together with `kinematics/scripts/gen_collision_primitives.py`
and `kinematics/scripts/gen_collision_pairs.py`. If the pair file is missing,
the node logs a warning and checks every pair. That is slow, and adjacent
geometry may trigger false collisions.

### `config/obstacles.yaml`

Poses are in the placo world frame (URDF root link `world`). `xyz` is in
metres and `rpy` in radians (extrinsic xyz).

```yaml
obstacles:
  - name: table
    type: box                 # box | sphere | cylinder
    size: [1.2, 0.8, 0.03]    # box: [x, y, z] full extents
                              # sphere: [radius]
                              # cylinder: [radius, length], axis along local z
    xyz: [0.45, 0.0, -0.35]
    rpy: [0.0, 0.0, 0.0]
    ignore_links: [base_link] # links allowed to touch this obstacle
```

The shipped `table` and `back_wall` entries are **placeholders**. Measure your
cell and replace them.

## Using the checker from Python

`CollisionChecker` has no ROS dependency, so you can call it from scripts or
tests:

```python
from collusion.collision_checker import CollisionChecker, load_obstacles

checker = CollisionChecker(
    "robot_collision.urdf",
    "collision_pairs.json",
    load_obstacles("obstacles.yaml"),
)
checker.set_joints({"left_joint_2": 0.5})   # returns names not in the URDF
pairs = checker.check()                      # list[PairDistance], closest first

for p in pairs[:5]:
    print(p.kind, p.link_a, p.link_b, p.distance)

checker.joint_distances(pairs)               # {joint: closest separation}
checker.flagged_joints(pairs, threshold=0.0) # joints in contact
```

Use the primitive URDF for anything that runs at control rate. The mesh URDF
works, but it is about 100 times slower.

## Tests

```bash
colcon test --packages-select collusion
colcon test-result --verbose
```

These run the standard ament lint checks (flake8, pep257, copyright).
