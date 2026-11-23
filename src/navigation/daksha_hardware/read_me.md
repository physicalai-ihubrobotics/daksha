# daksha_hardware

`ros2_control` hardware interface plugin for the Drax differential drive robot.  
Faithful C++ port of the Python `DifferentialDriveController` (`motor.py`).

---

## Single source of truth — how parameters flow

```
amr.urdf.xacro          ← EDIT HERE ONLY
       │
       │  xacro:property values (wheel_circumference, wheel_distance, ...)
       │
       ├─────────────────────────────────────────────────────────────────────►
       │  Substituted into <hardware><param> blocks                           │
       │                                                                      ▼
       │                                                          DraxHardware::on_init()
       │                                                          reads every param from URDF
       │
       └──► bringup.launch.py reads the same xacro properties
                  │
                  │  derives: wheel_radius = wheel_circumference / (2π)
                  │           wheel_separation = wheel_distance
                  │
                  └──► injects into diff_drive_controller at runtime
                             (overrides anything in controllers.yaml)
```

**Rule:** change a value in `amr.urdf.xacro` once → hardware plugin, controller, and
visual geometry all update on the next `ros2 launch`.

---

## Where to find and edit parameters

| What to change | File to edit |
|---|---|
| **Any robot geometry or config** | [`daksha_description/urdf/amr.urdf.xacro`](../daksha_description/urdf/amr.urdf.xacro) — `xacro:property` block at the top |
| **Controller non-geometry settings** (frame IDs, timeouts, window size) | [`daksha_description/config/controllers.yaml`](../daksha_description/config/controllers.yaml) |
| **Do NOT** manually set `wheel_radius` or `wheel_separation` in `controllers.yaml` — the launch file injects them | — |

---

## Parameter reference — `amr.urdf.xacro` properties

All properties are at the top of the file. The `<param>` values in `<hardware>` are all `${property_name}` substitutions — not typed by hand.

### Robot geometry

| Property | Default | Description |
|---|---|---|
| `wheel_circumference` | `0.575` m | Outer tyre circumference. Drives: `wheel_radius`, joint visual size, hardware param, controller `wheel_radius` |
| `wheel_distance` | `0.50` m | Centre-to-centre track width. Drives: joint y-offset, hardware param, controller `wheel_separation` |
| `ticks_per_revolution` | `4096` | Encoder ticks per full wheel revolution |

**Derived values (computed automatically):**
- `wheel_radius = wheel_circumference / (2π)` — used for wheel visual cylinder and injected into `diff_drive_controller`
- `wheel_half = wheel_distance / 2` — used for joint y-origin offset

### CAN / motor init

| Property | Default | Description |
|---|---|---|
| `can_interface` | `can0` | Linux SocketCAN interface |
| `left_node_id` | `1` | CANopen node ID — left motor |
| `right_node_id` | `2` | CANopen node ID — right motor |
| `invert_left` | `false` | Negate left motor RPM before CAN write |
| `invert_right` | `true` | Negate right motor RPM before CAN write |
| `nmt_start` | `true` | Send NMT Start on activation |
| `fix_din3` | `true` | Vendor DIN3 quirk fix (`0x2010:01/02`) |
| `motor_mode` | `-3` | CANopen operation mode (`0x6060`) |
| `timeout_sec` | `0.35` | SDO response timeout (seconds) |
| `retries` | `3` | SDO retry count |
| `cyclic_timeout_sec` | `0.03` | SDO timeout inside read()/write() (measured round trip: median 0.4 ms, max 7 ms) |
| `cyclic_retries` | `2` | SDO retries inside read()/write(); init sequences keep `timeout_sec`/`retries` |
| `feedback_resolution` | `4096` | Motor feedback counts/rev (`rpmToDec` formula) |
| `precise_speed_command` | `true` | Write both `0x2FF0:09` and `0x60FF:00`; `false` = only `0x2FF0:09` |
| `keep_enabled` | `false` | Skip `disable()` on shutdown |

### Velocity / RPM limits

| Property | Default | Description |
|---|---|---|
| `max_rpm` | `30` | Hard RPM clamp in hardware write loop |
| `max_linear_velocity` | `1.0` m/s | Injected into `diff_drive_controller` velocity limit |
| `max_angular_velocity` | `0.8` rad/s | Injected into `diff_drive_controller` velocity limit |
| `motor_command_timeout` | `0.5` s | Seconds without new wheel command before hardware zeros RPM |

### Slew rate limiter

Limits how fast the **sent** RPM can change per cycle. Prevents jerks.

| Property | Default | Description |
|---|---|---|
| `ramp_enabled` | `true` | Enable/disable |
| `accel_rpm_per_sec` | `30.0` | Max RPM increase per second |
| `decel_rpm_per_sec` | `200.0` | Max RPM decrease per second |

> **This was the root cause of motors running at different speeds:** the original default was `0.05` RPM/s — at 50 Hz, that allows only `0.001 RPM` per cycle, so the actual sent RPM never tracked the target. `30.0` RPM/s is the correct value for this robot.

### Odometry encoder corrections

| Property | Default | Description |
|---|---|---|
| `swap_odometry_wheels` | `true` | Swap raw left/right encoders before computing odometry |
| `invert_odometry_left` | `true` | Negate left wheel distance in odometry |
| `invert_odometry_right` | `false` | Negate right wheel distance |
| `invert_odometry_linear` | `false` | Negate both (reverses forward/back) |
| `invert_odometry_angular` | `false` | Negate angular delta |
| `swap_angular_linear_odometry` | `false` | Legacy mode — keep `false` |

### Balance controller

| Property | Default | Description |
|---|---|---|
| `balance_enabled` | `false` | Enable encoder + IMU heading correction |
| `balance_kp` | `0.002` | Encoder error proportional gain |
| `max_correction_rpm` | `20` | Hard cap on correction RPM |
| `safe_balance_ratio` | `0.15` | Dynamic limit fraction (0–1) |

### IMU yaw

| Property | Default | Description |
|---|---|---|
| `use_imu_yaw` | `true` | Override encoder yaw with IMU |
| `imu_topic` | `/bno055/imu` | `sensor_msgs/Imu` source topic |
| `imu_yaw_timeout` | `0.30` s | Fall back to encoders if IMU older than this |
| `use_imu_gyro_extrapolation` | `true` | Extrapolate yaw between IMU samples |
| `max_imu_extrapolation` | `0.12` s | Max extrapolation window |
| `invert_imu_yaw` | `false` | Negate IMU yaw |

### Safety topics

| Property | Default | Description |
|---|---|---|
| `emergency_stop_topic` | `/emergency_stop` | `std_msgs/Bool` — `true` = latch stop |

---

## Failure behaviour

| Event | What happens |
|---|---|
| Drive stops answering | read()/write() fail fast (30 ms x 2); after 10 consecutive failures the component returns ERROR and `on_error()` sends zero speed + disable |
| Ctrl-C / normal shutdown | `on_deactivate` / `on_shutdown` / destructor send zero speed (+ disable unless `keep_enabled`) |
| `/emergency_stop` true | Zero RPM on the next write() cycle, ramp bypassed |
| Crash, `kill -9`, hang | Handled outside the process by `tools/drax_can_watchdog.py` (below) |

## CAN safety watchdog

`tools/drax_can_watchdog.py` (systemd unit `drax_can_watchdog.service`, runs as root, no ROS dependency):

- Watches the controller's SDO requests on `can0`. If they stop for 0.5 s after having been flowing, it sends zero speed to nodes 1 and 2 (`--disable` also switches them off).
- Keeps `can0` up at 500 kbit/s with `restart-ms 100` so a bus-off recovers. Settings on a link the controller is actively using are never changed; they are applied once traffic stops.

```bash
sudo systemctl status drax_can_watchdog
journalctl -u drax_can_watchdog -f
```

---

## Build

```bash
cd /path/to/amr_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select daksha_hardware daksha_description
source install/setup.bash
```

---

## Launch

```bash
# 1. Bring up CAN bus
sudo ip link set can0 up type can bitrate 1000000

# 2. Launch robot (processes xacro → injects geometry into controller)
ros2 launch daksha_description bringup.launch.py

# 3. Send a velocity command
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}" --once

# Emergency stop
ros2 topic pub /emergency_stop std_msgs/msg/Bool "{data: true}" --once
```

---

## File structure

```
daksha_description/
├── urdf/
│   └── amr.urdf.xacro       ← SINGLE SOURCE OF TRUTH for all params
├── config/
│   └── controllers.yaml     ← non-geometry controller settings only
└── launch/
    └── bringup.launch.py    ← processes xacro, injects geometry into controller

daksha_hardware/
├── include/drax_hardware/
│   ├── socket_can.hpp
│   ├── canopen_node.hpp
│   ├── motor.hpp
│   └── drax_hardware.hpp
├── src/
│   ├── socket_can.cpp
│   ├── canopen_node.cpp
│   ├── motor.cpp
│   └── drax_hardware.cpp    ← reads all params from URDF <hardware> block
├── plugin.xml
├── CMakeLists.txt
└── package.xml
```