"""Self- and environment-collision checking on top of placo.

No ROS in here: CollisionChecker takes joint positions and returns the
separation of every checked pair, so it can be driven by the monitor node,
an offline script, or a test.

Two kinds of pair are checked:

  self   robot link <-> robot link, via placo's RobotWrapper.distances().
         Only the pairs in the allowlist JSON are checked (see
         kinematics/scripts/gen_collision_pairs.py); parent/child pairs are
         dropped regardless, since they overlap at their joint by
         construction and would read as permanently colliding.

  env    robot link <-> static obstacle (table, wall, ...) described in
         obstacles.yaml. placo has no notion of external objects, so these
         are computed directly with coal against shapes rebuilt from the
         robot's collision objects, placed at the current configuration.

Use the primitive-geometry URDF (robot_collision.urdf). The full mesh URDF
works but is ~100x slower - far too slow to monitor at control rate.

Written against placo 0.11 (pip; pulls numpy 2), whose collision_model
exposes .objects (type/size/placement) rather than pinocchio's
.geometryObjects.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import placo
import yaml
from scipy.spatial.transform import Rotation as R

try:
    import coal as fcl          # pinocchio >= 3.x
except ImportError:
    import hppfcl as fcl

# Renamed Transform3f -> Transform3s in coal 3.
_Transform = getattr(fcl, "Transform3s", None) or fcl.Transform3f


@dataclass
class PairDistance:
    kind: str                   # "self" or "env"
    link_a: str
    link_b: str                 # robot link, or obstacle name for "env"
    distance: float             # metres; <= 0 means contact / penetration
    point_a: np.ndarray         # closest points, world frame
    point_b: np.ndarray


@dataclass
class Obstacle:
    name: str
    shape: object               # fcl collision geometry
    transform: np.ndarray       # 4x4, world frame
    ignore_links: set = field(default_factory=set)
    kind: str = "box"
    size: list = field(default_factory=list)


def _link(name):
    """placo suffixes collision object names with _<index>; strip it back
    to the URDF link name so pairs match what the URDF declares."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() else name


def _joint_tree(urdf_path):
    """child link -> (joint name, parent link), for every URDF joint."""
    root = ET.parse(urdf_path).getroot()
    tree = {}
    for joint in root.iter("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            tree[child.get("link")] = (joint.get("name"), parent.get("link"))
    return tree


def _homogeneous(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _ensure_ros_package_path():
    """placo's bundled pinocchio resolves package:// mesh URIs through
    ROS_PACKAGE_PATH only, which ROS 2 no longer sets. Derive it from
    AMENT_PREFIX_PATH so the URDF's mesh references load."""
    if os.environ.get("ROS_PACKAGE_PATH"):
        return
    prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join(
        os.path.join(p, "share") for p in prefixes)


def _robot_shape(obj):
    """coal shape from a pinocchio GeometryObject.

    pinocchio >= 2.6: GeometryObject.geometry is already the hppfcl/coal shape.
    Older placo-bundled pinocchio: reconstruct from .type/.size/.mesh_path/.mesh_scale.
    """
    # New API: geometry attribute holds the shape directly.
    if hasattr(obj, 'geometry'):
        return obj.geometry
    # Old placo API: reconstruct from explicit type/size fields.
    import numpy as _np
    size = _np.asarray(obj.size, dtype=float)
    if obj.type == "box":
        return fcl.Box(*size[:3])
    if obj.type == "cylinder":
        return fcl.Cylinder(size[0], size[1])
    if obj.type == "sphere":
        return fcl.Sphere(size[0])
    if obj.type == "mesh":
        scale = _np.asarray(obj.mesh_scale, dtype=float)
        return fcl.MeshLoader().load(obj.mesh_path, scale)
    raise ValueError(f"collision object '{obj.name}': unsupported type '{obj.type}'")


def _fcl_transform(T):
    return _Transform(np.ascontiguousarray(T[:3, :3]), np.ascontiguousarray(T[:3, 3]))


def load_obstacles(path):
    """Read obstacles.yaml. Poses are in the placo world frame, which is
    the URDF root link ("world") at the origin."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    obstacles = []
    for o in data.get("obstacles") or []:
        kind = o["type"]
        size = [float(v) for v in o["size"]]
        if kind == "box":
            shape = fcl.Box(*size)                    # full x, y, z extents
        elif kind == "sphere":
            shape = fcl.Sphere(size[0])               # radius
        elif kind == "cylinder":
            shape = fcl.Cylinder(size[0], size[1])    # radius, length (z)
        else:
            raise ValueError(f"obstacle '{o['name']}': unknown type '{kind}'")
        obstacles.append(Obstacle(
            name=o["name"],
            shape=shape,
            transform=_homogeneous(o.get("xyz", [0, 0, 0]), o.get("rpy", [0, 0, 0])),
            ignore_links=set(o.get("ignore_links") or []),
            kind=kind,
            size=size,
        ))
    return obstacles


class CollisionChecker:

    def __init__(self, urdf_path, collision_pairs_path=None, obstacles=None):
        _ensure_ros_package_path()
        self.robot = placo.RobotWrapper(urdf_path)
        if collision_pairs_path:
            self.robot.load_collision_pairs(collision_pairs_path)
        self.robot.update_kinematics()

        self.joint_order = list(self.robot.joint_names())   # movable, model order
        self.joint_names = set(self.joint_order)
        self.obstacles = list(obstacles or [])
        self._tree = _joint_tree(urdf_path)
        self._adjacent = {frozenset((c, p)) for c, (_, p) in self._tree.items()}
        self._link_joint = {}

        # pinocchio >= 2.6 renamed .objects → .geometryObjects; handle both.
        _gm = self.robot.collision_model
        objects = _gm.geometryObjects if hasattr(_gm, 'geometryObjects') else _gm.objects
        self._geom_link = [_link(o.name) for o in objects]
        # Shapes are only needed for environment checks.
        self._shapes = [_robot_shape(o) for o in objects] if self.obstacles else []

        self._req = fcl.DistanceRequest()
        self._req.enable_nearest_points = True

    def set_joints(self, positions):
        """Apply {joint_name: rad}. Unknown names are ignored and returned."""
        unknown = []
        for name, value in positions.items():
            if name in self.joint_names:
                self.robot.set_joint(name, float(value))
            else:
                unknown.append(name)
        self.robot.update_kinematics()
        return unknown

    def self_distances(self):
        out = []
        for d in self.robot.distances():
            a, b = self._geom_link[d.objA], self._geom_link[d.objB]
            if a == b or frozenset((a, b)) in self._adjacent:
                continue
            out.append(PairDistance("self", a, b, float(d.min_distance),
                                    np.array(d.pointA), np.array(d.pointB)))
        return out

    def environment_distances(self):
        if not self.obstacles:
            return []
        out = []
        placements = self.robot.collision_placements(self.robot.state.q)
        for link, geometry, T in zip(self._geom_link, self._shapes, placements):
            T_geom = _fcl_transform(np.asarray(T))
            for obs in self.obstacles:
                if link in obs.ignore_links:
                    continue
                res = fcl.DistanceResult()
                dist = fcl.distance(geometry, T_geom, obs.shape,
                                    _fcl_transform(obs.transform), self._req, res)
                out.append(PairDistance("env", link, obs.name, float(dist),
                                        np.array(res.getNearestPoint1()),
                                        np.array(res.getNearestPoint2())))
        return out

    def link_joint(self, link):
        """The movable joint that carries link: the joint whose child it is,
        walking up through fixed joints (left_gripper_mount -> left_joint_7).
        None for links fixed to the root (base_link, left_link_1)."""
        if link not in self._link_joint:
            joint, current = None, link
            while current in self._tree:
                name, current = self._tree[current]
                if name in self.joint_names:
                    joint = name
                    break
            self._link_joint[link] = joint
        return self._link_joint[link]

    def pair_joints(self, pair):
        """Joints of the links actually in contact - both links for a self
        pair, the robot link only for an obstacle."""
        links = (pair.link_a,) if pair.kind == "env" else (pair.link_a, pair.link_b)
        return {j for j in map(self.link_joint, links) if j}

    def joint_distances(self, pairs):
        """{joint: closest separation of any pair involving its link}.
        Joints whose links are in no checked pair are absent."""
        out = {}
        for c in pairs:
            for j in self.pair_joints(c):
                if c.distance < out.get(j, float("inf")):
                    out[j] = c.distance
        return out

    def flagged_joints(self, pairs, threshold):
        """Joints carrying a link in any pair closer than threshold."""
        joints = set()
        for c in pairs:
            if c.distance > threshold:
                break                       # pairs are sorted, closest first
            joints |= self.pair_joints(c)
        return joints

    def check(self, self_collision=True, environment=True):
        """All checked pairs, closest first."""
        pairs = []
        if self_collision:
            pairs += self.self_distances()
        if environment:
            pairs += self.environment_distances()
        pairs.sort(key=lambda p: p.distance)
        return pairs
