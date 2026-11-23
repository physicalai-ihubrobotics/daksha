#!/usr/bin/env python3

import math
import os

import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import JointState


# Used when joint_direction.yaml has no value for a key. Module-level so
# leader_controller_ui can show the same fallbacks the node actually runs with.
DEFAULT_RIGHT_DIRECTION = [-1, -1, -1, 1, -1, 1, -1]
DEFAULT_LEFT_DIRECTION = [-1, 1, 1, -1, -1, 1, -1]


def load_joint_limits(yaml_path):
    """name -> (lower, upper), loaded from leader_limit_limit.yaml.

    Same layout as gen2/config/arm_limit.yaml: joints grouped per
    controller, each controller a ros__parameters block mapping
    joint name -> {lower, upper}. Flattened into a single dict here."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    limits = {}
    for controller in data.values():
        for name, bounds in controller.get("ros__parameters", {}).items():
            limits[name] = (float(bounds["lower"]), float(bounds["upper"]))
    return limits


def load_joint_directions(yaml_path, mode):
    """Load direction arrays from joint_direction.yaml for the given mode."""
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        section = data.get(mode, {})
        return (section.get("right_direction"),
                section.get("left_direction"),
                section.get("right_gripper_direction"),
                section.get("left_gripper_direction"))
    except Exception:
        return None, None, None, None


class LeaderToPositionController(Node):


    def __init__(self):

        super().__init__('leader_to_position_controller')

        self.declare_parameter("joint_limits_path", "")
        joint_limits_path = self.get_parameter(
            "joint_limits_path").get_parameter_value().string_value
        if not joint_limits_path:
            joint_limits_path = os.path.join(
                get_package_share_directory("gen2_leader"),
                "config", "leader_limit_limit.yaml")

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
        # right_gripper_right_joint opens toward +0.044, but
        # left_gripper_left_joint opens toward -0.044 (opposite sign
        # convention confirmed from a working record/replay session) -
        # using one constant for both drove the left gripper into its
        # closed-side hard stop instead of opening it. Both signs flipped
        # from that baseline to reverse gripper direction on both sides.
        self.GRIPPER_OPEN_M    = {8: 0.044, 16: -0.044}

        self.GRIPPER_SPAN = round(
            self.GRIPPER_RANGE_DEG / 360.0 * 4096
        )

        config_path = os.path.join(
            get_package_share_directory("gen2_leader"),
            "config", "joint_direction.yaml")
        right_dir, left_dir, right_grip, left_grip = load_joint_directions(config_path, "mirror")
        init_right = right_dir if right_dir else DEFAULT_RIGHT_DIRECTION
        init_left = left_dir if left_dir else DEFAULT_LEFT_DIRECTION

        self.declare_parameter("right_direction", init_right)
        self.declare_parameter("left_direction", init_left)
        # Sign of the published gripper value when the leader gripper opens
        # (GRIPPER_DIR is the leader encoder's own direction and stays fixed;
        # flipping it only pushes readings into the dead zone).
        self.declare_parameter(
            "right_gripper_direction",
            right_grip if right_grip in (1, -1) else -1)
        self.declare_parameter(
            "left_gripper_direction",
            left_grip if left_grip in (1, -1) else 1)
        self.GRIPPER_OPEN_M = {
            8: self.GRIPPER_OPEN_DIST
            * self.get_parameter("right_gripper_direction").get_parameter_value().integer_value,
            16: self.GRIPPER_OPEN_DIST
            * self.get_parameter("left_gripper_direction").get_parameter_value().integer_value,
        }

        self.right_direction = list(
            self.get_parameter("right_direction").get_parameter_value().integer_array_value)
        self.left_direction = list(
            self.get_parameter("left_direction").get_parameter_value().integer_array_value)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"Mirror directions R={self.right_direction} L={self.left_direction}"
        )

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

        self.joint_names = (
            self.right_joints +
            self.left_joints
        )

        self.pub = self.create_publisher(
            JointState,
            '/joint_cmd',
            10
        )

        self.left_pub = self.create_publisher(
            JointState,
            '/leader/left_joint_states',
            10
        )

        self.right_pub = self.create_publisher(
            JointState,
            '/leader/right_joint_states',
            10
        )

        self.create_subscription(
            Int32MultiArray,
            '/leader_motor_values',
            self.motor_callback,
            10
        )

        self.get_logger().info(
            "Leader motor subscriber started"
        )

    def _on_param_change(self, params):
        for p in params:
            if p.name == "right_direction":
                vals = list(p.value)
                if len(vals) != 7 or not all(v in (1, -1) for v in vals):
                    return SetParametersResult(
                        successful=False, reason="right_direction must be 7 values of 1 or -1")
                self.right_direction = vals
                self.get_logger().info(f"right_direction updated: {vals}")
            elif p.name == "left_direction":
                vals = list(p.value)
                if len(vals) != 7 or not all(v in (1, -1) for v in vals):
                    return SetParametersResult(
                        successful=False, reason="left_direction must be 7 values of 1 or -1")
                self.left_direction = vals
                self.get_logger().info(f"left_direction updated: {vals}")
            elif p.name in ("right_gripper_direction", "left_gripper_direction"):
                if p.value not in (1, -1):
                    return SetParametersResult(
                        successful=False, reason=f"{p.name} must be 1 or -1")
                sid = self.right_gripper_id if p.name.startswith("right") else self.left_gripper_id
                self.GRIPPER_OPEN_M[sid] = self.GRIPPER_OPEN_DIST * p.value
                self.get_logger().info(f"{p.name} updated: {p.value}")
        return SetParametersResult(successful=True)

    def gripper_to_meters(self, servo_id, raw):

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

        return frac * self.GRIPPER_OPEN_M[servo_id]

    def filter_positions(self, names, positions):
        # Clamps each commanded position into its leader_limit_limit.yaml
        # range before it goes out on /joint_cmd, so the leader never asks
        # for a position past the joint's limit in the first place (rather
        # than relying solely on joint_command_limiter downstream to reject
        # it and hold the last valid target).
        filtered = []
        for name, pos in zip(names, positions):
            limit = self.joint_limits.get(name)
            if limit is not None:
                pos = max(limit[0], min(limit[1], pos))
            filtered.append(pos)
        return filtered

    def mirror_positions(self, positions):
        # positions are built as:
        # right arm + right gripper, then left arm + left gripper.
        # The right leader drives the left follower and vice versa; per-joint
        # sign is set by right_direction / left_direction.
        mirror_left = list(positions[:8])
        mirror_right = list(positions[8:])

        # Joint 4 limits are one-sided and opposite per arm (left [-2.09, 0],
        # right [0, 2.09]), so the mirrored joint 4 must be flipped on both
        # sides - otherwise filter_positions clamps it to 0 and it never moves.
        mirror_left[3] = -mirror_left[3]
        mirror_right[3] = -mirror_right[3]

        return mirror_left + mirror_right

    def motor_callback(self, msg):

        motors = {}

        data = msg.data

        for i in range(0, len(data), 2):

            sid = data[i]
            pos = data[i + 1]

            motors[sid] = pos

        positions = []

        # RIGHT ARM
        for i, sid in enumerate(self.right_ids):

            raw = motors.get(sid, self.CENTER)

            zero = raw - self.CENTER

            rad = (
                self.right_direction[i]
                * zero
                * self.RAD_PER_COUNT
            )

            positions.append(rad)

        # RIGHT GRIPPER
        positions.append(
            self.gripper_to_meters(
                self.right_gripper_id,
                motors.get(
                    self.right_gripper_id,
                    self.CENTER
                )
            )
        )

        # LEFT ARM
        for i, sid in enumerate(self.left_ids):

            raw = motors.get(sid, self.CENTER)

            zero = raw - self.CENTER

            rad = (
                self.left_direction[i]
                * zero
                * self.RAD_PER_COUNT
            )

            positions.append(rad)

        # LEFT GRIPPER
        positions.append(
            self.gripper_to_meters(
                self.left_gripper_id,
                motors.get(
                    self.left_gripper_id,
                    self.CENTER
                )
            )
        )

        # self.get_logger().info(
        #     f"Right gripper: {positions[7]:.4f}, "
        #     f"Left gripper: {positions[15]:.4f}"
        # )

        stamp = self.get_clock().now().to_msg()

        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.name = (
            self.left_joints +
            self.right_joints
        )
        joint_msg.position = self.filter_positions(
            joint_msg.name, self.mirror_positions(positions))

        self.pub.publish(joint_msg)

        right_positions = list(positions[:8])
        # right_positions[7] = -right_positions[7]

        right_msg = JointState()
        right_msg.header.stamp = stamp
        right_msg.name = self.right_joints
        right_msg.position = right_positions

        self.right_pub.publish(right_msg)

        left_positions = list(positions[8:])
        # left_positions[7] = -left_positions[7]

        left_msg = JointState()
        left_msg.header.stamp = stamp
        left_msg.name = self.left_joints
        left_msg.position = left_positions

        self.left_pub.publish(left_msg)


def main(args=None):

    rclpy.init(args=args)

    node = LeaderToPositionController()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
