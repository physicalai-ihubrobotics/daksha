"""Thread-safe cache of live robot data, with per-item freshness."""

import math
import threading
import time

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import tf2_ros


class Stamped:
    __slots__ = ("value", "t")

    def __init__(self, value):
        self.value, self.t = value, time.monotonic()

    def age(self):
        return time.monotonic() - self.t


def load_joint_directions(path):
    """{joint: +-1.0} from a ros__parameters YAML with a `joint_direction` map."""
    import yaml
    if not path:
        return {}
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    for section in doc.values():
        params = (section or {}).get("ros__parameters", {})
        if "joint_direction" in params:
            return {k: float(v) for k, v in params["joint_direction"].items()}
    return {}


class LiveState:
    def __init__(self, node, robot_cfg, joint_directions=None):
        self._lock = threading.Lock()
        self._data = {}
        self.node = node
        self.joint_directions = joint_directions or {}

        node.create_subscription(JointState, robot_cfg.get("joint_state_topic", "/joint_states"),
                                 self._on_joint_states, 10)

        topics = robot_cfg.get("collision", {}) or {}
        kinds = {"self_collision": Bool, "world_collision": Bool, "min_distance": Float64,
                 "speed_scale": Float64, "joint_distance": JointState, "speed_limits": JointState}
        for key, msg_type in kinds.items():
            if topics.get(key):
                node.create_subscription(msg_type, topics[key],
                                         lambda m, k=key: self._store(k, m), 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

    def _store(self, key, msg):
        if isinstance(msg, (Bool, Float64)):
            value = msg.data
        else:
            value = {n: (msg.position[i] if i < len(msg.position) else None,
                         msg.velocity[i] if i < len(msg.velocity) else None,
                         msg.effort[i] if i < len(msg.effort) else None)
                     for i, n in enumerate(msg.name)}
        with self._lock:
            self._data[key] = Stamped(value)

    def _on_joint_states(self, msg):
        with self._lock:
            prev = self._data.get("joint_states")
            merged = dict(prev.value) if prev else {}
        for i, n in enumerate(msg.name):
            sign = self.joint_directions.get(n, 1.0)
            vel = msg.velocity[i] if i < len(msg.velocity) else None
            merged[n] = {
                "position": sign * msg.position[i] if i < len(msg.position) else math.nan,
                "velocity": None if vel is None else sign * vel,
            }
        with self._lock:
            self._data["joint_states"] = Stamped(merged)

    def get(self, key):
        """(value, age_seconds) or (None, None) if never received."""
        with self._lock:
            s = self._data.get(key)
        return (None, None) if s is None else (s.value, s.age())
