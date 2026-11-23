#!/usr/bin/env python3
"""ROS 2 node that serves the bionic-hand web UI and bridges it to ROS.

One process owns a single ``BionicHands`` (the serial bus can only be opened
once), serves the existing Flask UI from ``web/app.py``, and in parallel runs a
ROS node that:

  * publishes the hands' state on ``joint_states`` and ``status``,
  * accepts the same commands over ROS (``gesture``, ``finger_command``),

so the hands can be driven from the browser *and* from ROS at the same time,
sharing the UI's single-movement runner and activity log.

Parameters
----------
serial_port      bus device, e.g. /dev/ttyACM0 (or "sim://both" for the sim)
baud             bus baud rate (default 1000000)
web_host         Flask bind address (default 0.0.0.0)
web_port         Flask port (default 5000)
auto_connect     open the bus on startup (default true)
enable_on_start  arm movement on startup, skipping the UI toggle (default false)
publish_rate_hz  joint_states / status rate (default 5.0)

Run with::

    ros2 run bionic_hand hand_web_ui
    ros2 run bionic_hand hand_web_ui --ros-args -p serial_port:=sim://both
"""

import importlib.util
import json
import logging
import os
import sys
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

FINGERS = ("thumb", "index", "middle", "ring")


def _load_webapp():
    """Load web/app.py (Flask routes + shared state) from share dir or source.

    Returns the imported module so we can reuse its ``app``, ``hands``,
    ``state``, ``runner`` and helper functions instead of duplicating them.
    """
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory("bionic_hand"), "web", "app.py"))
    except Exception:
        pass
    # Source-tree fallback: .../bionic_hand_sdk/web/app.py
    here = os.path.dirname(os.path.realpath(__file__))
    candidates.append(os.path.normpath(os.path.join(here, "..", "web", "app.py")))

    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("bionic_hand_webapp", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "web/app.py not found (looked in: " + ", ".join(candidates) + ")")


class BionicHandWebNode(Node):

    def __init__(self, webapp):
        super().__init__("bionic_hand_ui")
        self.webapp = webapp
        self.hands = webapp.hands

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baud", 1000000)
        self.declare_parameter("web_host", "0.0.0.0")
        self.declare_parameter("web_port", 5000)
        self.declare_parameter("auto_connect", True)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("publish_rate_hz", 5.0)

        # Publishers / subscribers / services on the shared hands.
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.status_pub = self.create_publisher(String, "status", 10)
        self.create_subscription(String, "gesture", self._on_gesture, 10)
        self.create_subscription(JointState, "finger_command", self._on_fingers, 10)
        self.create_service(Trigger, "stop", self._srv_stop)
        self.create_service(Trigger, "open", self._srv_open)
        self.create_service(Trigger, "close", self._srv_close)

        if self.get_parameter("auto_connect").value:
            self._connect()
        if self.get_parameter("enable_on_start").value:
            webapp.state["enabled"] = True
            webapp.add_log("Movement ENABLED (enable_on_start).", "warn")

        rate = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish)

    # -- connection -------------------------------------------------------
    def _connect(self):
        port = self.get_parameter("serial_port").value
        baud = int(self.get_parameter("baud").value)
        from .protocol import find_bionic_hand_port
        if port == "auto" or not port:
            detected = find_bionic_hand_port()
            if detected:
                port = detected
                self.get_logger().info(f"auto-detected bionic hand on {port}")
            else:
                self.get_logger().warn("bionic hand not detected on USB; skipping connect")
                self.webapp.add_log("Bionic hand not detected. Check USB.", "error")
                return
        else:
            detected = find_bionic_hand_port()
            if detected:
                port = detected
                self.get_logger().info(f"auto-detected bionic hand on {port}")
        try:
            self.hands.connect(port, baud)
            self.webapp.add_log(f"Connected to {port} at {baud} baud (ROS).", "ok")
            self.get_logger().info(f"connected on {port}")
        except Exception as exc:
            self.get_logger().warn(f"could not connect to {port}: {exc}")

    # -- command helpers --------------------------------------------------
    def _submit(self, label, fn):
        """Run through the web UI's single-movement runner (shared with the UI)."""
        if not self.hands.connected:
            self.get_logger().warn(f"ignoring '{label}': not connected")
            return
        try:
            self.webapp.runner.start(label, fn)
        except Exception as exc:  # ApiError when busy
            self.get_logger().warn(f"'{label}' rejected: {exc}")

    # -- subscriptions ----------------------------------------------------
    def _on_gesture(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        hand, name = (text.split(":", 1) if ":" in text else ("both", text))
        hand, name = hand.strip().lower(), name.strip()
        if hand not in ("left", "right", "both"):
            self.get_logger().warn(f"unknown hand {hand!r}")
            return
        if name.upper() not in self.webapp.BionicHands.gestures():
            self.get_logger().warn(f"unknown gesture {name!r}")
            return
        selected = self.webapp.selected_hands(hand)
        self._submit(
            f"{name} ({hand}) [ros]",
            lambda: self.webapp.run_hands(
                selected, True, lambda h, m: h.gesture(name, motion=m), None),
        )

    def _on_fingers(self, msg: JointState):
        grouped = {"left": {}, "right": {}}
        for name, pos in zip(msg.name, msg.position):
            name = name.strip().lower()
            if "_" not in name:
                continue
            hand, finger = name.split("_", 1)
            if hand in grouped and finger in FINGERS:
                grouped[hand][finger] = max(0.0, min(1.0, float(pos)))
        for hand, fingers in grouped.items():
            if not fingers:
                continue
            selected = self.webapp.selected_hands(hand)
            self._submit(
                f"fingers {hand} [ros]",
                lambda sel=selected, f=dict(fingers): self.webapp.run_hands(
                    sel, False, lambda h, m: h.set_fingers(**f, motion=m), None),
            )

    # -- services ---------------------------------------------------------
    def _srv_stop(self, _req, resp):
        self.hands.stop()
        self.webapp.add_log("STOP requested (ROS).", "warn")
        resp.success, resp.message = True, "stopped"
        return resp

    def _srv_open(self, _req, resp):
        self._submit("OPEN HAND (both) [ros]",
                     lambda: self.webapp.run_hands(
                         self.webapp.selected_hands("both"), True,
                         lambda h, m: h.gesture("OPEN HAND", motion=m), None))
        resp.success, resp.message = True, "opening"
        return resp

    def _srv_close(self, _req, resp):
        self._submit("CLOSE / FIST (both) [ros]",
                     lambda: self.webapp.run_hands(
                         self.webapp.selected_hands("both"), True,
                         lambda h, m: h.gesture("CLOSE / FIST", motion=m), None))
        resp.success, resp.message = True, "closing"
        return resp

    # -- publishing -------------------------------------------------------
    def _publish(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        for hand in (self.hands.left, self.hands.right):
            for sid in hand.ids:
                js.name.append(f"motor_{sid}")
                js.position.append(float(hand.pose.get(sid, 0.0)))
        self.joint_pub.publish(js)

        status = {
            "connected": self.hands.connected,
            "enabled": self.webapp.state.get("enabled", False),
            "busy": self.webapp.runner.busy,
            "left_moving": self.hands.left.is_moving,
            "right_moving": self.hands.right.is_moving,
        }
        self.status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    # Silence werkzeug's per-request access log (GET /api/status ... spam).
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    webapp = _load_webapp()

    # daksha_web_access.py is installed by the automation package into its lib dir.
    from ament_index_python.packages import get_package_prefix
    _automation_dir = os.path.join(get_package_prefix("automation"), "lib", "automation")
    if _automation_dir not in sys.path:
        sys.path.insert(0, _automation_dir)
    from daksha_web_access import setup_flask_gate
    setup_flask_gate(webapp.app, "bionic_hand")

    rclpy.init(args=args)
    node = BionicHandWebNode(webapp)

    # ROS spins in the background; Flask owns the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    host = node.get_parameter("web_host").value
    port = int(node.get_parameter("web_port").value)
    node.get_logger().info(f"serving web UI on http://{host}:{port}")
    try:
        webapp.app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            webapp.hands.disconnect()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
