# actuator_configures

Web UI for scanning and configuring DM-J4340-2EC motors over SocketCAN.

This is a **local tool** — it must run on the machine with the SocketCAN interface, because browsers cannot talk to CAN hardware directly. The node opens the bus once and serves a small web page that operates it over HTTP.

## Launch

```bash
# Default: channel=can1, host=::, port=8000
ros2 launch actuator_configures motor_web.launch.py

# Custom parameters
ros2 launch actuator_configures motor_web.launch.py channel:=can0 port:=9000

# Or run the node directly
ros2 run actuator_configures motor_web --ros-args -p channel:=can1 -p port:=8000
```

Then open `http://<hostname>.local:8000` from any device on the LAN.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channel` | `can1`  | SocketCAN interface the motors are on |
| `host`    | `::`    | Bind address; use `127.0.0.1` for local-only access |
| `port`    | `8000`  | HTTP port for the web UI |

## Features

- **Scan** — discover motors on the CAN bus (configurable ID range and timeout)
- **Probe** — read position, velocity, torque, and temperature from a single motor
- **Set ID** — change a motor's CAN ID (ESC_ID) and/or feedback ID (MST_ID)
- **Zero** — set the current position as zero for one or all motors
- **Save to flash** — persist ID changes across power cycles

## Dependencies

- `rclpy` — ROS 2 Python client
- `python-can` — SocketCAN interface
- `flask` — web server

## Notes

- On UMI the bus is `can1`, created by `ros2 run umi can_bridge` (`can0` is the Jetson's on-board CAN controller).
- Do not run alongside bringup — `ros2_control` command frames would appear as fake motors during a scan.
- There is no authentication by default. Set `host:=127.0.0.1` on untrusted networks.
