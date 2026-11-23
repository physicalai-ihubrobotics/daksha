#!/usr/bin/env python3
"""ROS 2 node wrapping the bionic-hand SDK.

Exposes the two SCS0009 hands on ROS. Movement calls in the SDK block until
they finish, so all hardware commands run on a single background worker thread;
the ROS callbacks only enqueue work and never block the executor.

Parameters
----------
port              serial device, e.g. /dev/ttyACM0 (or "sim://both" for the
                  built-in simulator, no hardware needed)
baud              bus baud rate (default 1000000)
calibration_file  optional JSON calibration to load on start ("" = none)
assume_open       if true, call confirm_open() after connecting so smooth/
                  staged motion is allowed without a read
auto_connect      connect on startup (default true)
publish_rate_hz   joint_states / status publish rate (default 5.0)

Topics
------
subscribe  gesture         std_msgs/String        "[hand:]GESTURE"
subscribe  finger_command  sensor_msgs/JointState name "<hand>_<finger>",
                                                  position = closure 0..1
publish    joint_states    sensor_msgs/JointState motor_1..16, position=degrees
publish    status          std_msgs/String        connection / motion state

Services (all std_srvs/Trigger)
-------------------------------
open, close, stop, confirm_open, connect, disconnect, ping
"""

import json
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from . import BionicHands, BusyError, PoseUnknownError

FINGERS = ("thumb", "index", "middle", "ring")


class _CommandRunner:
    """Single worker thread that serializes blocking SDK calls.

    Jobs with the same ``key`` coalesce (latest wins), so a high-rate stream of
    finger commands never backs up behind itself.
    """

    def __init__(self, logger):
        self._logger = logger
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._jobs = []           # list[(key, callable)]
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, fn, key=None):
        with self._cv:
            if key is not None:
                self._jobs = [(k, f) for (k, f) in self._jobs if k != key]
            self._jobs.append((key, fn))
            self._cv.notify()

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._jobs and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                _key, fn = self._jobs.pop(0)
            try:
                fn()
            except (BusyError, PoseUnknownError) as exc:
                self._logger.warn(str(exc))
            except Exception as exc:  # keep the worker alive
                self._logger.error(f"command failed: {exc}")


class BionicHandNode(Node):

    def __init__(self):
        super().__init__("bionic_hand")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 1000000)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("assume_open", False)
        self.declare_parameter("auto_connect", True)
        self.declare_parameter("publish_rate_hz", 5.0)

        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)
        calib = self.get_parameter("calibration_file").value or None

        self.hands = BionicHands(port=port, baud=baud, calibration_file=calib)
        self.runner = _CommandRunner(self.get_logger())

        # Publishers
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.status_pub = self.create_publisher(String, "status", 10)

        # Subscribers
        self.create_subscription(String, "gesture", self._on_gesture, 10)
        self.create_subscription(JointState, "finger_command", self._on_fingers, 10)

        # Services
        self._srv = [
            self.create_service(Trigger, "open", self._srv_open),
            self.create_service(Trigger, "close", self._srv_close),
            self.create_service(Trigger, "stop", self._srv_stop),
            self.create_service(Trigger, "confirm_open", self._srv_confirm_open),
            self.create_service(Trigger, "connect", self._srv_connect),
            self.create_service(Trigger, "disconnect", self._srv_disconnect),
            self.create_service(Trigger, "ping", self._srv_ping),
        ]

        if self.get_parameter("auto_connect").value:
            self._connect(self.get_parameter("assume_open").value)

        rate = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"bionic_hand node up (port={port}, baud={baud}); "
            "topics: gesture, finger_command -> joint_states, status"
        )

    # -- connection -------------------------------------------------------
    def _connect(self, assume_open=False):
        try:
            self.hands.connect()
        except Exception as exc:
            self.get_logger().warn(f"could not connect to {self.hands.bus.port}: {exc}")
            return False
        if assume_open:
            self.hands.confirm_open()
        self.get_logger().info(f"connected on {self.hands.bus.port}")
        return True

    # -- subscriptions ----------------------------------------------------
    def _on_gesture(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        if ":" in text:
            hand, name = text.split(":", 1)
            hand = hand.strip().lower()
        else:
            hand, name = "both", text
        name = name.strip()
        if hand not in ("left", "right", "both"):
            self.get_logger().warn(f"ignoring gesture: unknown hand {hand!r}")
            return
        self.runner.submit(lambda: self.hands.gesture(name, hand=hand, parallel=True),
                           key=f"gesture:{hand}")

    def _on_fingers(self, msg: JointState):
        # Group {hand: {finger: closure}} from names like "left_index".
        grouped = {"left": {}, "right": {}}
        for name, pos in zip(msg.name, msg.position):
            name = name.strip().lower()
            if "_" not in name:
                continue
            hand, finger = name.split("_", 1)
            if hand in grouped and finger in FINGERS:
                grouped[hand][finger] = max(0.0, min(1.0, float(pos)))
        for hand, fingers in grouped.items():
            if fingers:
                h = self.hands.hands[hand]
                self.runner.submit(lambda h=h, f=dict(fingers): h.set_fingers(**f),
                                   key=f"fingers:{hand}")

    # -- services ---------------------------------------------------------
    def _ok(self, resp, message=""):
        resp.success = True
        resp.message = message
        return resp

    def _srv_open(self, _req, resp):
        self.runner.submit(lambda: self.hands.open(parallel=True), key="gesture:both")
        return self._ok(resp, "opening")

    def _srv_close(self, _req, resp):
        self.runner.submit(lambda: self.hands.close(parallel=True), key="gesture:both")
        return self._ok(resp, "closing")

    def _srv_stop(self, _req, resp):
        self.hands.stop()  # thread-safe, interrupts the running sequence now
        return self._ok(resp, "stopped")

    def _srv_confirm_open(self, _req, resp):
        self.hands.confirm_open()
        return self._ok(resp, "pose set to OPEN")

    def _srv_connect(self, _req, resp):
        ok = self._connect(self.get_parameter("assume_open").value)
        resp.success = ok
        resp.message = "connected" if ok else "connect failed"
        return resp

    def _srv_disconnect(self, _req, resp):
        self.hands.disconnect()
        return self._ok(resp, "disconnected")

    def _srv_ping(self, _req, resp):
        try:
            result = self.hands.ping_all()
        except Exception as exc:
            resp.success = False
            resp.message = f"ping failed: {exc}"
            return resp
        alive = sorted(sid for sid, ok in result.items() if ok)
        return self._ok(resp, f"alive {len(alive)}/16: {alive}")

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
            "left_moving": self.hands.left.is_moving,
            "right_moving": self.hands.right.is_moving,
            "left_pose_known": self.hands.left.pose_known,
            "right_pose_known": self.hands.right.pose_known,
        }
        self.status_pub.publish(String(data=json.dumps(status)))

    def destroy_node(self):
        try:
            self.runner.shutdown()
            self.hands.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BionicHandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
