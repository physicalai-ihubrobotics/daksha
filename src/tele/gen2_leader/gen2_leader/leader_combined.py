#!/usr/bin/env python3
"""Leader -> /joint_cmd in normal or mirror mode, switchable at runtime.

Combines main_with_service_wifi (normal) and main_with_service_mirror_wifi
(mirror) in one node, so the teleop mode changes through a service call
instead of killing one launch and starting the other.

Services:
  ~/set_mirror   std_srvs/SetBool   true -> mirror, false -> normal
  ~/toggle_mode  std_srvs/Trigger   normal <-> mirror
Topic:
  ~/mode         std_msgs/String    current mode (transient local)

Parameters (per mode, live-settable):
  <mode>.right_direction / <mode>.left_direction           7 x (+1/-1)
  <mode>.right_gripper_direction / <mode>.left_gripper_direction   +1/-1
"""

import math
import os

import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from std_msgs.msg import Int32MultiArray, String
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

from gen2_leader.main_with_service_mirror_wifi import (
    DEFAULT_LEFT_DIRECTION as MIRROR_DEFAULT_LEFT,
    DEFAULT_RIGHT_DIRECTION as MIRROR_DEFAULT_RIGHT,
    load_joint_limits,
)


MODES = ("normal", "mirror")

# Fallbacks when joint_direction.yaml leaves a key out; same values the
# standalone normal/mirror nodes use.
DEFAULTS = {
    "normal": {
        "right_direction": [-1, -1, 1, -1, 1, -1, 1],
        "left_direction": [-1, 1, -1, 1, 1, 1, 1],
        "right_gripper_direction": 1,
        "left_gripper_direction": -1,
    },
    "mirror": {
        "right_direction": MIRROR_DEFAULT_RIGHT,
        "left_direction": MIRROR_DEFAULT_LEFT,
        "right_gripper_direction": -1,
        "left_gripper_direction": 1,
    },
}

MODE_QOS = QoSProfile(
    depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def load_direction_config(yaml_path):
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class LeaderCombinedController(Node):

    def __init__(self):

        super().__init__('leader_combined')

        share = get_package_share_directory("gen2_leader")

        self.declare_parameter("joint_limits_path", "")
        joint_limits_path = self.get_parameter(
            "joint_limits_path").get_parameter_value().string_value
        if not joint_limits_path:
            joint_limits_path = os.path.join(
                share, "config", "leader_limit_limit.yaml")
        self.joint_limits = load_joint_limits(joint_limits_path)

        self.CENTER = 2048
        self.RAD_PER_COUNT = 2 * math.pi / 4096

        self.right_ids = list(range(1, 8))
        self.left_ids  = list(range(9, 16))

        self.right_gripper_id = 8
        self.left_gripper_id  = 16

        self.GRIPPER_CLOSED    = {8: 2065, 16: 2039}
        self.GRIPPER_DIR       = {8: 1,   16: -1}
        self.GRIPPER_OPEN_DIST = 0.044
        self.GRIPPER_RANGE_DEG = 45.0

        self.GRIPPER_SPAN = round(
            self.GRIPPER_RANGE_DEG / 360.0 * 4096
        )

        # Per-mode direction settings, mirrored from the "<mode>.<key>"
        # parameters so the hot path doesn't query parameters.
        config = load_direction_config(
            os.path.join(share, "config", "joint_direction.yaml"))
        self.dirs = {}
        for mode in MODES:
            section = config.get(mode) or {}
            self.dirs[mode] = {}
            for key, default in DEFAULTS[mode].items():
                val = section.get(key)
                if not self._valid(key, val):
                    val = default
                self.declare_parameter(f"{mode}.{key}", val)
                self.dirs[mode][key] = val

        self.declare_parameter("mode", "normal")
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        if self.mode not in MODES:
            self.mode = "normal"

        self.add_on_set_parameters_callback(self._on_param_change)

        self.left_joints = [
            "left_joint_1",
            "left_joint_2",
            "left_joint_3",
            "left_joint_4",
            "left_joint_5",
            "left_joint_6",
            "left_joint_7",
            "left_gripper_left_joint"
        ]

        self.right_joints = [
            "right_joint_1",
            "right_joint_2",
            "right_joint_3",
            "right_joint_4",
            "right_joint_5",
            "right_joint_6",
            "right_joint_7",
            "right_gripper_right_joint"
        ]

        self.pub = self.create_publisher(JointState, '/joint_cmd', 10)
        self.left_pub = self.create_publisher(
            JointState, '/leader/left_joint_states', 10)
        self.right_pub = self.create_publisher(
            JointState, '/leader/right_joint_states', 10)
        self.mode_pub = self.create_publisher(String, '~/mode', MODE_QOS)

        self.create_service(SetBool, '~/set_mirror', self._on_set_mirror)
        self.create_service(Trigger, '~/toggle_mode', self._on_toggle_mode)

        self.create_subscription(
            Int32MultiArray,
            '/leader_motor_values',
            self.motor_callback,
            10
        )

        self._publish_mode()
        self.get_logger().info(
            f"Leader combined started in '{self.mode}' mode; "
            f"normal={self.dirs['normal']} mirror={self.dirs['mirror']}"
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.get_logger().info(f"Mode switched to '{mode}'")
        self._publish_mode()

    def _publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))

    def _on_set_mirror(self, request, response):
        self._set_mode("mirror" if request.data else "normal")
        response.success = True
        response.message = self.mode
        return response

    def _on_toggle_mode(self, request, response):
        self._set_mode("normal" if self.mode == "mirror" else "mirror")
        response.success = True
        response.message = self.mode
        return response

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _valid(key, val):
        if key.endswith("gripper_direction"):
            return val in (1, -1)
        return (val is not None and len(val) == 7
                and all(v in (1, -1) for v in val))

    def _on_param_change(self, params):
        updates = []
        for p in params:
            if p.name == "mode":
                if p.value not in MODES:
                    return SetParametersResult(
                        successful=False, reason="mode must be 'normal' or 'mirror'")
                updates.append(("mode", None, p.value))
                continue
            mode, _, key = p.name.partition(".")
            if mode not in MODES or key not in DEFAULTS[mode]:
                continue
            val = p.value if key.endswith("gripper_direction") else list(p.value)
            if not self._valid(key, val):
                return SetParametersResult(
                    successful=False,
                    reason=f"{p.name} must be "
                    + ("1 or -1" if key.endswith("gripper_direction")
                       else "7 values of 1 or -1"))
            updates.append((mode, key, val))

        for mode, key, val in updates:
            if mode == "mode":
                self._set_mode(val)
            else:
                self.dirs[mode][key] = val
                self.get_logger().info(f"{mode}.{key} updated: {val}")
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def gripper_to_meters(self, servo_id, raw, sign):

        closed = self.GRIPPER_CLOSED[servo_id]
        d      = self.GRIPPER_DIR[servo_id]
        span   = self.GRIPPER_SPAN

        travel = (d * (raw - closed)) % 4096

        if travel <= span:
            frac = travel / span
        else:
            dead = 4096 - span
            frac = 1.0 if (
                (travel - span) < dead / 2
            ) else 0.0

        frac = max(0.0, min(1.0, frac))

        return frac * self.GRIPPER_OPEN_DIST * sign

    def filter_positions(self, names, positions):
        # Clamps each commanded position into its leader_limit_limit.yaml
        # range before it goes out on /joint_cmd (mirror mode only, as in
        # main_with_service_mirror_wifi).
        filtered = []
        for name, pos in zip(names, positions):
            limit = self.joint_limits.get(name)
            if limit is not None:
                pos = max(limit[0], min(limit[1], pos))
            filtered.append(pos)
        return filtered

    def mirror_positions(self, positions):
        # right leader drives the left follower and vice versa
        mirror_left = list(positions[:8])
        mirror_right = list(positions[8:])

        # Joint 4 limits are one-sided and opposite per arm (left [-2.09, 0],
        # right [0, 2.09]), so the mirrored joint 4 must be flipped on both
        # sides - otherwise filter_positions clamps it to 0 and it never moves.
        mirror_left[3] = -mirror_left[3]
        mirror_right[3] = -mirror_right[3]

        return mirror_left + mirror_right

    def arm_positions(self, motors, ids, direction, gripper_id, gripper_sign):
        positions = [
            direction[i] * (motors.get(sid, self.CENTER) - self.CENTER)
            * self.RAD_PER_COUNT
            for i, sid in enumerate(ids)
        ]
        positions.append(self.gripper_to_meters(
            gripper_id, motors.get(gripper_id, self.CENTER), gripper_sign))
        return positions

    def motor_callback(self, msg):

        data = msg.data
        motors = {data[i]: data[i + 1] for i in range(0, len(data) - 1, 2)}

        mode = self.mode
        cfg = self.dirs[mode]

        positions = (
            self.arm_positions(
                motors, self.right_ids, cfg["right_direction"],
                self.right_gripper_id, cfg["right_gripper_direction"])
            + self.arm_positions(
                motors, self.left_ids, cfg["left_direction"],
                self.left_gripper_id, cfg["left_gripper_direction"])
        )

        stamp = self.get_clock().now().to_msg()

        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        if mode == "mirror":
            joint_msg.name = self.left_joints + self.right_joints
            joint_msg.position = self.filter_positions(
                joint_msg.name, self.mirror_positions(positions))
        else:
            joint_msg.name = self.right_joints + self.left_joints
            joint_msg.position = positions

        self.pub.publish(joint_msg)

        right_msg = JointState()
        right_msg.header.stamp = stamp
        right_msg.name = self.right_joints
        right_msg.position = positions[:8]
        self.right_pub.publish(right_msg)

        left_msg = JointState()
        left_msg.header.stamp = stamp
        left_msg.name = self.left_joints
        left_msg.position = positions[8:]
        self.left_pub.publish(left_msg)


def main(args=None):

    rclpy.init(args=args)

    node = LeaderCombinedController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
