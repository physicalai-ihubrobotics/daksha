"""Convert Daksha's MuJoCo model (assets/daksha/daksha_raw.xml) into a URDF that placo can load.

placo (and the `kinematics/` package built on it) reads URDF only; daksha_raw.xml is MJCF. This keeps
everything the IK needs from the XML: the body tree, body poses, joint axes, ranges and effort limits,
inertials, and mesh geometry (visual group 1 + collision geoms, with their mesh scale and offsets).

Self-collision geometry (`collision=`):
  'primitives' (default)  the boxes/cylinders of assets/daksha/robot_collision.urdf, matched by link name
                          (same link names and frames). ~2 ms per IK solve with self-collision on.
  'mesh'                  the XML's own collision meshes (exact, but ~100+ ms per solve in placo)
  'none'                  no collision elements

    python -m local_pipeline.mjcf_to_urdf                 # -> runtime/ik/daksha_raw.urdf
    python -m local_pipeline.mjcf_to_urdf --collision mesh --out /tmp/x.urdf
"""
from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT/'assets'/'daksha'/'daksha_raw.xml'
PRIMITIVES_URDF = PROJECT/'assets'/'daksha'/'robot_collision.urdf'


def _f(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def _rpy(quat_wxyz):
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler('xyz')


def _origin(parent, pos, quat_wxyz):
    rpy = _rpy(quat_wxyz)
    ET.SubElement(parent, 'origin', xyz=' '.join(f'{v:.9g}' for v in pos), rpy=' '.join(f'{v:.9g}' for v in rpy))


def _primitive_collisions():
    """{link name: [<collision> elements]} from robot_collision.urdf."""
    if not PRIMITIVES_URDF.exists():
        return {}
    root = ET.parse(PRIMITIVES_URDF).getroot()
    return {link.get('name'): link.findall('collision') for link in root.findall('link')}


def convert(xml_path=DEFAULT_XML, out_path=None, collision='primitives'):
    xml_path = pathlib.Path(xml_path)
    mj = ET.parse(xml_path).getroot()
    compiler = mj.find('compiler')
    meshdir = (xml_path.parent/(compiler.get('meshdir', '') if compiler is not None else '')).resolve()
    meshes = {}
    for m in mj.iter('mesh'):
        meshes[m.get('name')] = (str(meshdir/m.get('file')), m.get('scale', '1 1 1'))
    prims = _primitive_collisions() if collision == 'primitives' else {}

    robot = ET.Element('robot', name=mj.get('model', 'robot'))
    ET.SubElement(robot, 'link', name='world')

    def add_body(body, parent_name):
        name = body.get('name')
        link = ET.SubElement(robot, 'link', name=name)
        inert = body.find('inertial')
        if inert is not None:
            el = ET.SubElement(link, 'inertial')
            _origin(el, _f(inert.get('pos'), [0, 0, 0]), _f(inert.get('quat'), [1, 0, 0, 0]))
            ET.SubElement(el, 'mass', value=inert.get('mass', '0.001'))
            ixx, iyy, izz = _f(inert.get('diaginertia'), [1e-6, 1e-6, 1e-6])
            ET.SubElement(el, 'inertia', ixx=f'{ixx:.9g}', iyy=f'{iyy:.9g}', izz=f'{izz:.9g}', ixy='0', ixz='0', iyz='0')
        for g in body.findall('geom'):
            if g.get('type', 'sphere') != 'mesh':
                continue
            is_visual = g.get('contype') == '0' and g.get('conaffinity') == '0'
            if not is_visual and collision != 'mesh':
                continue
            el = ET.SubElement(link, 'visual' if is_visual else 'collision')
            _origin(el, _f(g.get('pos'), [0, 0, 0]), _f(g.get('quat'), [1, 0, 0, 0]))
            geo = ET.SubElement(el, 'geometry')
            fname, scale = meshes[g.get('mesh')]
            ET.SubElement(geo, 'mesh', filename=fname, scale=scale)
            if is_visual and g.get('rgba'):
                mat = ET.SubElement(el, 'material', name=f'{name}_mat')
                ET.SubElement(mat, 'color', rgba=g.get('rgba'))
        for c in prims.get(name, []):
            link.append(c)

        joints = body.findall('joint')
        jtype, jname = 'fixed', f'{name}_fixed'
        jel_attrs = {}
        if joints:
            j = joints[0]
            jname = j.get('name')
            jtype = {'slide': 'prismatic', 'hinge': 'revolute', None: 'revolute'}[j.get('type')]
            jel_attrs = j.attrib
        joint = ET.SubElement(robot, 'joint', name=jname, type=jtype)
        ET.SubElement(joint, 'parent', link=parent_name)
        ET.SubElement(joint, 'child', link=name)
        _origin(joint, _f(body.get('pos'), [0, 0, 0]), _f(body.get('quat'), [1, 0, 0, 0]))
        if jtype != 'fixed':
            ET.SubElement(joint, 'axis', xyz=jel_attrs.get('axis', '0 0 1'))
            lo, hi = _f(jel_attrs.get('range'), [-3.14159, 3.14159])
            effort = abs(_f(jel_attrs.get('actuatorfrcrange'), [-100, 100])[1])
            ET.SubElement(joint, 'limit', lower=f'{lo:.9g}', upper=f'{hi:.9g}', effort=f'{effort:g}', velocity='3.0')
        for child in body.findall('body'):
            add_body(child, name)

    for top in mj.find('worldbody').findall('body'):
        add_body(top, 'world')
    ET.indent(robot)
    text = '<?xml version="1.0"?>\n<!-- generated from ' + str(xml_path) + ' by local_pipeline/mjcf_to_urdf.py -->\n' \
           + ET.tostring(robot, encoding='unicode') + '\n'
    if out_path is None:
        out_path = PROJECT/'runtime'/'ik'/f'{xml_path.stem}{"" if collision == "primitives" else "_" + collision}.urdf'
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or out_path.read_text() != text:
        out_path.write_text(text)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', default=str(DEFAULT_XML))
    ap.add_argument('--out')
    ap.add_argument('--collision', choices=['primitives', 'mesh', 'none'], default='primitives')
    args = ap.parse_args()
    print(convert(args.xml, args.out, args.collision))


if __name__ == '__main__':
    main()
