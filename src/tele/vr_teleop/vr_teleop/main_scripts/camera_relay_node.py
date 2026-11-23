#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Daksha cameras (see camera/camera_launch/config/cameras.yaml). All three are
# compressed color feeds published by the per-camera image_transport republish
# process; depth is disabled on every camera, so only color is relayed here.
#   world -> camera1   left -> camera2   right -> camera3
CAMERA_TOPICS = {
    1: {"in": "/camera/world/color/image_raw/compressed", "out": "/quest/camera1", "compressed": True},
    2: {"in": "/left/camera/color/image_raw/compressed",  "out": "/quest/camera2", "compressed": True},
    3: {"in": "/right/camera/color/image_raw/compressed", "out": "/quest/camera3", "compressed": True},
}

TARGET_FPS = 8
STALE_THRESHOLD = 1.0  # seconds — frames older than this are dropped before publishing


class CameraRelay(Node):
    def __init__(self):
        super().__init__("camera_relay")

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._latest: dict[int, tuple | None] = {cam_id: None for cam_id in CAMERA_TOPICS}

        for cam_id, topics in CAMERA_TOPICS.items():
            msg_type = CompressedImage if topics["compressed"] else Image

            pub = self.create_publisher(msg_type, topics["out"], pub_qos)

            self.create_subscription(
                msg_type,
                topics["in"],
                lambda msg, cid=cam_id: self._store(cid, msg),
                sub_qos,
            )

            self.create_timer(
                1.0 / TARGET_FPS,
                lambda p=pub, cid=cam_id: self._publish(p, cid),
            )

            self.get_logger().info(
                f"[Camera {cam_id}]  {topics['in']}  →  {topics['out']}  @ {TARGET_FPS}fps"
            )

    def _store(self, cam_id: int, msg):
        self._latest[cam_id] = (msg, time.monotonic())

    def _publish(self, pub, cam_id: int):
        entry = self._latest[cam_id]
        if entry is None:
            return

        msg, received_at = entry
        if time.monotonic() - received_at > STALE_THRESHOLD:
            self._latest[cam_id] = None
            return

        pub.publish(msg)
        self._latest[cam_id] = None


def main():
    rclpy.init()
    node = CameraRelay()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
