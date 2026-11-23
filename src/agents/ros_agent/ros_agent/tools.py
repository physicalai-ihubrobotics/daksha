"""Read-only tools the agent (or an operator) can call.

Every tool returns JSON-serialisable data. None of them commands motion.
"""

import math
from dataclasses import dataclass, field

import rclpy.time

from .safety import joint_limit_report


@dataclass
class Tool:
    name: str
    description: str
    fn: callable
    # arg -> (type, required, description); type in string|number|integer|boolean
    args: dict = field(default_factory=dict)

    def spec(self):
        return {"name": self.name, "description": self.description,
                "args": {k: {"type": t, "required": r, "description": d}
                         for k, (t, r, d) in self.args.items()}}


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


class ToolBox:
    def __init__(self, model, live, robot_cfg, max_state_age, cameras=None,
                 camera_max_age=2.0):
        self.model, self.live, self.cfg = model, live, robot_cfg
        self.max_age = max_state_age
        self.cameras, self.camera_max_age = cameras, camera_max_age
        camera_tools = [] if not (cameras and cameras.cameras) else [
            Tool("list_cameras", "Cameras on the robot, and whether each is streaming.",
                 lambda a: {"cameras": self.cameras.status()}),
            Tool("look_at_camera", "Take the latest frame from a camera and LOOK at it. The "
                 "image is attached to the next message. Use 'world' to see the workspace, "
                 "objects and the arms; use a wrist camera for a close-up of a gripper.",
                 self.look_at_camera,
                 {"camera": ("string", False, "camera id from list_cameras (default "
                                              "'world')")}),
        ]
        self.registry = {t.name: t for t in camera_tools + [
            Tool("describe_robot", "Structural summary of the robot from URDF/SRDF: links, "
                 "joints, DOF, groups, end effectors, limits, collision geometry.",
                 lambda a: self.model.describe()),
            Tool("get_robot_tree", "Kinematic tree (links and joints) as indented text.",
                 lambda a: {"tree": self.model.tree_text()}),
            Tool("get_joint_info", "Type, parent/child link, axis, limits, mimic and groups "
                 "of one joint.", lambda a: self.model.joint_info(a["joint"]),
                 {"joint": ("string", True, "joint name from describe_robot")}),
            Tool("get_joint_states", "Live joint positions/velocities (URDF convention) with "
                 "limit status. Optionally restricted to one group.", self.joint_states,
                 {"group": ("string", False, "group name, e.g. left_arm")}),
            Tool("check_joint_limits", "Joints that are near (5% of range) or beyond their "
                 "URDF limits right now.", self.check_limits),
            Tool("get_link_pose", "Pose of a link via TF, in base_frame unless 'frame' given.",
                 self.link_pose,
                 {"link": ("string", True, "link name"),
                  "frame": ("string", False, "reference frame")}),
            Tool("check_collision", "Live self/world collision status, minimum distance and the "
                 "closest joints, from collision_scene.", self.collision),
            Tool("get_speed_limits", "Live speed-governor scale and allowed per-joint "
                 "velocity/acceleration.", self.speed_limits),
        ]}

    def specs(self):
        return [t.spec() for t in self.registry.values()]

    def call(self, name, args):
        return self.registry[name].fn(args)

    # ------------------------------------------------------------------ helpers
    def _fresh(self, key):
        value, age = self.live.get(key)
        if value is None:
            return None, {"available": False, "reason": f"no data received for {key}"}
        meta = {"available": True, "age_s": round(age, 3), "stale": age > self.max_age}
        return value, meta

    def _positions(self):
        js, meta = self._fresh("joint_states")
        return ({n: d["position"] for n, d in js.items()} if js else {}), js, meta

    # -------------------------------------------------------------------- tools
    def joint_states(self, a):
        positions, js, meta = self._positions()
        if js is None:
            return meta
        names = self.model.movable_joints()
        group = a.get("group")
        if group:
            if group not in self.model.groups:
                raise KeyError(f"unknown group '{group}'; known: {sorted(self.model.groups)}")
            names = self.model.groups[group]["joints"]
        limits = joint_limit_report(self.model, positions)
        out = {}
        for n in names:
            if n not in js:
                out[n] = {"status": "not_published"}
                continue
            out[n] = {"position": _finite(js[n]["position"]),
                      "velocity": _finite(js[n]["velocity"]),
                      "limit_status": limits.get(n, {}).get("status", "no_limit")}
        return {"joints": out, **meta}

    def check_limits(self, a):
        positions, js, meta = self._positions()
        if js is None:
            return meta
        rep = joint_limit_report(self.model, positions)
        flagged = {n: r for n, r in rep.items() if r["status"] != "ok"}
        return {"all_ok": not flagged, "flagged": flagged, "checked": len(rep), **meta}

    def link_pose(self, a):
        link = a["link"]
        frame = a.get("frame") or self.cfg.get("base_frame", self.model.root)
        if link not in self.model.links:
            raise KeyError(f"unknown link '{link}'")
        try:
            tf = self.live.tf_buffer.lookup_transform(frame, link, rclpy.time.Time())
        except Exception as e:  # tf2 raises several exception types
            return {"available": False, "reason": f"TF {frame} -> {link}: {e}"}
        t, q = tf.transform.translation, tf.transform.rotation
        return {"frame": frame, "link": link,
                "xyz": [round(t.x, 4), round(t.y, 4), round(t.z, 4)],
                "quat_xyzw": [round(q.x, 4), round(q.y, 4), round(q.z, 4), round(q.w, 4)]}

    def collision(self, a):
        selfc, m1 = self._fresh("self_collision")
        world, _ = self._fresh("world_collision")
        dmin, _ = self._fresh("min_distance")
        jd, m2 = self._fresh("joint_distance")
        if selfc is None and jd is None:
            return {"available": False,
                    "reason": "collision_scene is not publishing (is collision_scene_node running?)"}
        closest = []
        if jd:
            closest = sorted(((n, v[0]) for n, v in jd.items() if v[0] is not None
                              and math.isfinite(v[0])), key=lambda x: x[1])[:6]
        return {"self_collision": selfc, "world_collision": world,
                "min_distance_m": _finite(dmin),
                "closest_joints": [{"joint": n, "distance_m": round(d, 4)} for n, d in closest],
                "stale": bool(m1.get("stale") or m2.get("stale")),
                "note": "negative distance = penetration of collision geometry"}

    def look_at_camera(self, a):
        """Returns metadata plus "_image" (JPEG bytes); run_agent strips "_image" and
        attaches it to the conversation instead of serialising it as text."""
        cam = a.get("camera") or "world"
        if cam not in self.cameras.cameras:
            raise KeyError(f"unknown camera '{cam}'; known: {sorted(self.cameras.cameras)}")
        jpeg, w, h, age = self.cameras.snapshot(cam)
        if jpeg is None:
            return {"available": False, "camera": cam,
                    "reason": f"no frame received on {self.cameras.cameras[cam]['topic']}"}
        return {"available": True, "camera": cam,
                "display_name": self.cameras.cameras[cam]["display_name"],
                "width": w, "height": h, "age_s": round(age, 2),
                "stale": age > self.camera_max_age, "_image": jpeg}

    def speed_limits(self, a):
        scale, meta = self._fresh("speed_scale")
        lim, _ = self._fresh("speed_limits")
        if scale is None:
            return {"available": False,
                    "reason": "speed_governor is not publishing (is speed_governor_node running?)"}
        joints = {}
        for n, (s, v, acc) in (lim or {}).items():
            joints[n] = {"scale": _finite(s), "max_velocity": _finite(v),
                         "max_acceleration": _finite(acc)}
        return {"global_scale": scale, "joints": joints, **meta}
