"""Seeded dual-bowl sorting layouts and trusted completion checks."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import BOWL, FRUITS, TABLE, TOYS

OBJECTS = {**FRUITS, **TOYS}
FRUIT_NAMES = tuple(FRUITS)
TOY_NAMES = tuple(TOYS)
ALL_OBJECTS = tuple(OBJECTS)
BOWL_NAMES = ('bowl', 'toy_bowl')
TARGET_BOWL = {**{name: 'bowl' for name in FRUIT_NAMES},
               **{name: 'toy_bowl' for name in TOY_NAMES}}

TASK_TEXT = """\
# Task: sort fruit and toys into separate bowls

On the table are three fruit objects (orange, apple, banana), three small toys (blue
block, toy car, purple ball), an orange fruit bowl on the left, and a blue toy bowl on the
right. Pick up each fruit and put it inside the orange bowl. Put each toy inside the blue
bowl. Keep the two groups separate.

The task is complete when all six objects are resting inside their matching bowl, both
bowls are upright and still on the table, and neither gripper is holding an object.

Each episode moves the objects and bowls slightly and changes the banana and car angles.
Find the objects and bowls from fresh camera images; no poses or layout are given.

Use `evaluate` to get the trusted binary result of the current episode.
"""

# Positive robot y appears on the viewer's right because the world camera faces Daksha.
BOWL_CENTERS = {'bowl': (0.30, -0.42), 'toy_bowl': (0.30, 0.42)}
OBJECT_SLOTS = {
    'orange': (0.22, -0.27), 'apple': (0.39, -0.27), 'banana': (0.32, -0.11),
    'toy_block': (0.24, 0.24), 'toy_car': (0.32, 0.16), 'toy_ball': (0.39, 0.24),
}
FOOTPRINT = {
    'orange': 0.036, 'apple': 0.041, 'banana': 0.105,
    'toy_block': 0.038, 'toy_car': 0.060, 'toy_ball': 0.027,
}


def object_rest_height(name):
    item = OBJECTS[name]
    if item['kind'] == 'sphere':
        return item['radius']
    if item['kind'] == 'ellipsoid':
        return item['size'][2]
    if item['kind'] == 'box':
        return item['size'][2]
    if item['kind'] == 'car':
        return 0.012  # wheel axes are 12 mm above the tabletop
    return item['radius']  # banana


def _yaw_quat(yaw):
    return np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])


def sample_layout(seed, reachable=None, max_tries=40):
    """Return a new, separated layout; optionally reject IK-unreachable poses."""
    rng = np.random.default_rng(seed)
    outer_radius = BOWL['inner_radius'] + BOWL['wall']
    for _ in range(max_tries):
        layout = {}
        for bowl_name, (x0, y0) in BOWL_CENTERS.items():
            x = x0 + rng.uniform(-0.008, 0.008)
            y = y0 + rng.uniform(-0.008, 0.008)
            yaw = rng.uniform(-0.12, 0.12)
            layout[bowl_name] = (x, y, yaw)
        if reachable and not all(reachable(name, *pose) for name, pose in layout.items()):
            continue

        ok = True
        for name in ALL_OBJECTS:
            bx, by = OBJECT_SLOTS[name]
            bowl_name = TARGET_BOWL[name]
            placed = False
            for _ in range(8):
                x = bx + rng.uniform(-0.005, 0.005)
                y = by + rng.uniform(-0.005, 0.005)
                yaw_limit = 0.50 if name == 'banana' else 0.35 if name == 'toy_car' else math.pi
                yaw = rng.uniform(-yaw_limit, yaw_limit)
                clear_objects = all(
                    math.hypot(x - px, y - py) > FOOTPRINT[name] + FOOTPRINT[other] + 0.008
                    for other, (px, py, _) in layout.items() if other in OBJECTS
                )
                clear_bowls = all(
                    math.hypot(x - layout[b][0], y - layout[b][1]) > outer_radius + FOOTPRINT[name] + 0.018
                    for b in BOWL_NAMES
                )
                if (clear_objects and clear_bowls and
                        (reachable is None or reachable(name, x, y, yaw))):
                    layout[name] = (x, y, yaw)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return layout
    raise RuntimeError(f'No feasible fruit-and-toy layout for seed {seed}')


def apply_layout(model, data, layout):
    for name, (x, y, yaw) in layout.items():
        joint = model.joint(f'{name}_free')
        adr = model.jnt_qposadr[joint.id]
        z = TABLE['top'] + (0.001 if name in BOWL_NAMES else object_rest_height(name) + 0.002)
        data.qpos[adr:adr + 3] = (x, y, z)
        data.qpos[adr + 3:adr + 7] = _yaw_quat(yaw)
        dof = model.jnt_dofadr[joint.id]
        data.qvel[dof:dof + 6] = 0


def task_state(model, data):
    """Check that every item rests inside its assigned bowl and is no longer held."""
    bowl_pose = {name: data.body(name) for name in BOWL_NAMES}
    bowl_state = {}
    for name, bowl in bowl_pose.items():
        upright = float(bowl.xmat.reshape(3, 3)[2, 2]) > 0.95
        on_table = abs(float(bowl.xpos[2]) - TABLE['top']) < 0.02
        still = float(np.linalg.norm(data.joint(f'{name}_free').qvel[:3])) < 0.05
        bowl_state[name] = {'upright': upright, 'on_table': on_table, 'still': still}

    pads = {model.geom(f'{f}_pad').id for arm in ('left', 'right')
            for f in (f'{arm}_left_finger', f'{arm}_right_finger')}
    object_geoms = {
        name: {g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id}
        for name in ALL_OBJECTS
    }
    held = {name: False for name in ALL_OBJECTS}
    for contact in data.contact[:data.ncon]:
        for name, geoms in object_geoms.items():
            if ((contact.geom1 in geoms and contact.geom2 in pads) or
                    (contact.geom2 in geoms and contact.geom1 in pads)):
                held[name] = True

    inside = {}
    for name in ALL_OBJECTS:
        item = data.body(name).xpos
        bowl = bowl_pose[TARGET_BOWL[name]].xpos
        radius = math.hypot(float(item[0] - bowl[0]), float(item[1] - bowl[1]))
        dz = float(item[2] - bowl[2])
        still = float(np.linalg.norm(data.joint(f'{name}_free').qvel[:3])) < 0.05
        inside[name] = bool(radius < BOWL['inner_radius'] - 0.005 and
                             0.0 < dz < BOWL['height'] + 0.06 and still and not held[name])

    complete = all(inside.values()) and all(
        all(bowl_state[name].values()) for name in BOWL_NAMES
    )
    return {'task_complete': bool(complete), 'inside': inside, 'held': held,
            'bowls': bowl_state}
