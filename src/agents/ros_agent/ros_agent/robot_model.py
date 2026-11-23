"""Structured robot description built from URDF (+ optional SRDF).

Pure Python, no ROS dependency, so it can be unit-tested and reused.
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter


def resolve_path(path):
    """Resolve 'package://pkg/rel' via ament_index; return other paths as-is."""
    if not path or not path.startswith("package://"):
        return path
    from ament_index_python.packages import get_package_share_directory
    pkg, _, rel = path[len("package://"):].partition("/")
    return os.path.join(get_package_share_directory(pkg), rel)


def _floats(text, default):
    return [float(v) for v in text.split()] if text else default


def _geometry(item):
    geom = item.find("geometry")
    if geom is None or len(geom) == 0:
        return None
    shape = geom[0]
    info = {"type": shape.tag}
    if shape.tag == "mesh":
        info["file"] = shape.get("filename", "")
    else:
        for k, v in shape.attrib.items():
            vals = _floats(v, [])
            info[k] = vals[0] if len(vals) == 1 else vals
    origin = item.find("origin")
    info["xyz"] = _floats(origin.get("xyz") if origin is not None else None, [0.0] * 3)
    info["rpy"] = _floats(origin.get("rpy") if origin is not None else None, [0.0] * 3)
    return info


class RobotModel:
    def __init__(self, urdf_path, srdf_path=None, groups=None, end_effectors=None):
        self.urdf_path = urdf_path
        root = ET.parse(urdf_path).getroot()
        self.name = root.get("name", "")

        self.links = {}
        for link in root.findall("link"):
            self.links[link.get("name")] = {
                "collision": [g for g in map(_geometry, link.findall("collision")) if g],
                "visual": [g for g in map(_geometry, link.findall("visual")) if g],
            }

        self.joints = {}
        for j in root.findall("joint"):
            limit, mimic, axis = j.find("limit"), j.find("mimic"), j.find("axis")
            self.joints[j.get("name")] = {
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "axis": _floats(axis.get("xyz") if axis is not None else None, [1.0, 0.0, 0.0]),
                "limit": None if limit is None else {
                    k: float(limit.get(k)) for k in ("lower", "upper", "velocity", "effort")
                    if limit.get(k) is not None},
                "mimic": None if mimic is None else {
                    "joint": mimic.get("joint"),
                    "multiplier": float(mimic.get("multiplier", 1.0)),
                    "offset": float(mimic.get("offset", 0.0))},
            }

        self.parent_joint = {j["child"]: n for n, j in self.joints.items()}
        self.children = {}
        for n, j in self.joints.items():
            self.children.setdefault(j["parent"], []).append(n)
        roots = [l for l in self.links if l not in self.parent_joint]
        self.root = roots[0] if roots else None

        self.disabled_pairs = []
        self.srdf_groups = {}
        if srdf_path:
            self._load_srdf(srdf_path)

        self.groups = {}
        for gname, spec in (groups or {}).items():
            self.groups[gname] = {"base": spec["base"], "tip": spec["tip"],
                                  "joints": self.chain_joints(spec["base"], spec["tip"],
                                                              movable_only=True)}
        for gname, joints in self.srdf_groups.items():
            self.groups.setdefault(gname, {"joints": joints})
        self.end_effectors = dict(end_effectors or {})

    def _load_srdf(self, path):
        root = ET.parse(path).getroot()
        for el in root.findall("disable_collisions"):
            self.disabled_pairs.append(
                (el.get("link1"), el.get("link2"), el.get("reason", "")))
        for g in root.findall("group"):
            self.srdf_groups[g.get("name")] = [j.get("name") for j in g.findall("joint")]

    # ------------------------------------------------------------------ queries
    def movable_joints(self):
        return [n for n, j in self.joints.items() if j["type"] not in ("fixed", "floating")]

    def actuated_joints(self):
        return [n for n in self.movable_joints() if not self.joints[n]["mimic"]]

    def path_to_root(self, link):
        path, cur = [], link
        while cur in self.parent_joint:
            j = self.parent_joint[cur]
            path.append(j)
            cur = self.joints[j]["parent"]
        return path

    def chain_joints(self, base, tip, movable_only=False):
        """Joints from base to tip (base must be an ancestor of tip)."""
        if base not in self.links or tip not in self.links:
            raise KeyError(f"unknown link in chain {base} -> {tip}")
        chain = []
        for j in self.path_to_root(tip):
            if self.joints[j]["child"] == base:
                break
            chain.append(j)
            if self.joints[j]["parent"] == base:
                break
        else:
            if base != self.root:
                raise ValueError(f"{base} is not an ancestor of {tip}")
        chain.reverse()
        if movable_only:
            chain = [j for j in chain if self.joints[j]["type"] not in ("fixed", "floating")]
        return chain

    def joint_link_owner(self, link):
        """Nearest movable joint at or above `link`, or None."""
        for j in self.path_to_root(link):
            if self.joints[j]["type"] not in ("fixed", "floating"):
                return j
        return None

    def links_without_collision(self):
        return sorted(l for l, d in self.links.items() if not d["collision"])

    def tree_text(self, link=None, indent=""):
        link = link or self.root
        lines = [f"{indent}{link}"]
        for j in sorted(self.children.get(link, [])):
            jt = self.joints[j]["type"]
            lines.append(f"{indent}  └─[{j} : {jt}]")
            lines.extend(self.tree_text(self.joints[j]["child"], indent + "    ").splitlines())
        return "\n".join(lines)

    def joint_info(self, name):
        if name not in self.joints:
            raise KeyError(f"unknown joint '{name}'")
        j = dict(self.joints[name])
        j["name"] = name
        j["groups"] = [g for g, d in self.groups.items() if name in d["joints"]]
        return j

    def describe(self):
        reasons = Counter(r for _, _, r in self.disabled_pairs)
        coll_types = Counter(g["type"] for d in self.links.values() for g in d["collision"])
        return {
            "name": self.name,
            "urdf": self.urdf_path,
            "root_link": self.root,
            "num_links": len(self.links),
            "num_joints": len(self.joints),
            "joint_types": dict(Counter(j["type"] for j in self.joints.values())),
            "dof": len(self.actuated_joints()),
            "actuated_joints": self.actuated_joints(),
            "mimic_joints": {n: j["mimic"] for n, j in self.joints.items() if j["mimic"]},
            "groups": {g: d["joints"] for g, d in self.groups.items()},
            "end_effectors": self.end_effectors,
            "joint_limits": {n: self.joints[n]["limit"] for n in self.movable_joints()},
            "collision_geometry_types": dict(coll_types),
            "links_without_collision": self.links_without_collision(),
            "srdf_disabled_pairs": dict(reasons),
        }
