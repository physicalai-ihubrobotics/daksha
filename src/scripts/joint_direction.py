#!/usr/bin/env python3
"""ROS 2 node that publishes the arm joint directions from /robot_description.

Subscribes to the URDF published by robot_state_publisher (std_msgs/String,
transient local) and reads <param name="direction"> from each <ros2_control>
block. The list is positional: entry i belongs to the i-th <joint> in the same
block (the same pairing arm_interface.cpp uses for motor_ids).

Publishes sensor_msgs/JointState:
    name     = joint names (left arm, then right arm)
    position = direction of each joint (+1.0 / -1.0)

Parameters:
    robot_description_topic  (string, "/robot_description")  URDF source topic
    topic                    (string, "/joint_directions")   output topic
    rate                     (double, 1.0)                   publish rate in Hz; <= 0 publishes once per URDF

    python3 joint_direction.py
    python3 joint_direction.py --ros-args -p rate:=0.0
    ros2 topic echo /joint_directions
"""

import xml.etree.ElementTree as ET

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def read_directions(urdf_xml):
    """{system name: {"can": iface, "joints": [(joint, motor_id, direction), ...]}}"""
    # Encode so a leading <?xml ... encoding="..."?> declaration is accepted.
    root = ET.fromstring(urdf_xml.encode("utf-8"))
    systems = {}
    for system in root.findall("ros2_control"):
        hw = system.find("hardware")
        if hw is None:
            continue
        params = {p.get("name"): (p.text or "").strip() for p in hw.findall("param")}
        if "direction" not in params:
            continue
        direction = [int(x) for x in params["direction"].split(",")]
        ids = [int(x) for x in params.get("motor_ids", "").split(",") if x.strip()]
        joints = [j.get("name") for j in system.findall("joint")]
        rows = [(joints[i] if i < len(joints) else "?",
                 ids[i] if i < len(ids) else "?",
                 d) for i, d in enumerate(direction)]
        systems[system.get("name")] = {"can": params.get("can_interface", "?"), "joints": rows}
    return systems


def system_order(systems):
    """Left first, then right, then anything else."""
    return sorted(systems, key=lambda n: (0 if "left" in n.lower() else 1 if "right" in n.lower() else 2, n))


class JointDirectionPublisher(Node):

    def __init__(self):
        super().__init__("joint_direction_publisher")

        self.declare_parameter("robot_description_topic", "/robot_description")
        self.declare_parameter("topic", "/joint_directions")
        self.declare_parameter("rate", 1.0)

        desc_topic = self.get_parameter("robot_description_topic").value
        self.topic = self.get_parameter("topic").value
        rate = float(self.get_parameter("rate").value)

        self.names, self.directions = [], []

        # Transient local on both sides: we get the URDF even if robot_state_publisher
        # started first, and late subscribers to our topic still get the last message.
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(JointState, self.topic, latched)
        self.create_subscription(String, desc_topic, self.on_robot_description, latched)

        if rate > 0.0:
            self.create_timer(1.0 / rate, self.publish_directions)

        self.get_logger().info(f"Waiting for URDF on {desc_topic} ...")

    def on_robot_description(self, msg):
        try:
            systems = read_directions(msg.data)
        except (ET.ParseError, ValueError) as e:
            self.get_logger().error(f"Failed to parse robot_description: {e}")
            return
        if not systems:
            self.get_logger().error("No <ros2_control> block with a direction param in robot_description")
            return

        names, directions = [], []
        lines = ["Joint directions from robot_description:"]
        for sys_name in system_order(systems):
            info = systems[sys_name]
            dirs = [d for _, _, d in info["joints"]]
            lines.append(f"{sys_name} ({info['can']})  direction = {','.join(map(str, dirs))}")
            for joint, motor_id, d in info["joints"]:
                lines.append(f"  motor {motor_id}  {joint:<28} {d:+d}")
                names.append(joint)
                directions.append(float(d))
        self.names, self.directions = names, directions
        self.get_logger().info("\n" + "\n".join(lines))
        self.get_logger().info(f"Publishing {len(names)} joint directions on {self.topic}")

        self.publish_directions()

    def publish_directions(self):
        if not self.names:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = self.directions
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = JointDirectionPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
