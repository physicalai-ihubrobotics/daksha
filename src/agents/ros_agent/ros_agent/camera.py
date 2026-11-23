"""Latest-frame cache for the robot's cameras (sensor_msgs/CompressedImage).

The camera set comes from automation's config/camera_topics.yaml -- the same
file the bring_up Cameras panel uses -- so there is one place to add, rename
or remove a camera.
"""

import base64
import os
import threading
import time

import yaml
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

try:  # optional: used only to shrink frames before they go to the LLM
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None


def default_camera_config():
    """automation installs config/ into lib/automation (next to bring_up.py)."""
    try:
        from ament_index_python.packages import get_package_prefix
        return os.path.join(get_package_prefix("automation"), "lib", "automation",
                            "config", "camera_topics.yaml")
    except Exception:
        return ""


def load_cameras(path):
    """[{"id", "display_name", "topic"}, ...]; rows missing a key are dropped."""
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return [c for c in doc.get("cameras", [])
            if isinstance(c, dict) and {"id", "display_name", "topic"} <= c.keys()]


class CameraFeed:
    def __init__(self, node, cameras):
        self.cameras = {c["id"]: c for c in cameras}
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frames = {}  # id -> (bytes, format, monotonic time, frame count)
        for cam_id, cam in self.cameras.items():
            node.create_subscription(CompressedImage, cam["topic"],
                                     lambda m, i=cam_id: self._store(i, m),
                                     qos_profile_sensor_data)

    def _store(self, cam_id, msg):
        with self._lock:
            n = self._frames.get(cam_id, (None, None, None, 0))[3] + 1
            self._frames[cam_id] = (bytes(msg.data), msg.format, time.monotonic(), n)
            self._new_frame.notify_all()

    def wait_frame(self, cam_id, after=0, timeout=1.0):
        """Block until camera `cam_id` has a frame numbered > `after`.
        Returns (jpeg_bytes, frame_number) or (None, after) on timeout."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                f = self._frames.get(cam_id)
                if f is not None and f[3] > after:
                    return f[0], f[3]
                left = deadline - time.monotonic()
                if left <= 0:
                    return None, after
                self._new_frame.wait(left)

    def latest(self, cam_id):
        """(jpeg_bytes, format, age_s) or (None, None, None)."""
        with self._lock:
            f = self._frames.get(cam_id)
        if f is None:
            return None, None, None
        return f[0], f[1], time.monotonic() - f[2]

    def status(self):
        out = []
        for cam_id, cam in self.cameras.items():
            data, _, age = self.latest(cam_id)
            out.append({"id": cam_id, "display_name": cam["display_name"],
                        "topic": cam["topic"], "available": data is not None,
                        "age_s": None if age is None else round(age, 2)})
        return out

    def snapshot(self, cam_id, max_side=768, quality=80):
        """Downscaled JPEG for the LLM: (bytes, width, height, age_s) or Nones."""
        data, fmt, age = self.latest(cam_id)
        if data is None:
            return None, None, None, None
        if cv2 is None:
            return data, None, None, age
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None, age
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, None, None, age
        return buf.tobytes(), img.shape[1], img.shape[0], age


def data_uri(jpeg):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
