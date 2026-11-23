"""The fruit-to-bowl task: seeded layouts and the trusted success check.

Layouts are deterministic per seed. Exploration episodes use seeds 1000+i, formal
evaluation uses seeds 0..N-1, so formal layouts are never seen during exploration (the
RAX simulation rule). Every sampled layout is checked for reachability with IK before it
is accepted, so each episode is solvable by Daksha's arms.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import BOWL, FRUITS, TABLE

TASK_TEXT = """\
# Task: put the fruit in the bowl

Daksha stands at a table. On the table are a bowl and three fruits: an **orange**, an
**apple** and a **banana**. Pick up each fruit and place it **inside the bowl**.

The task is complete when, all at the same time:

1. the orange, the apple and the banana each rest inside the bowl (within its rim, not
   on the table and not balanced on the rim);
2. the bowl is upright and still on the table;
3. neither gripper is holding a fruit (the fruit stays in the bowl on its own).

Every episode starts with a new layout: the bowl and the fruits are in different places,
and the banana lies at a different angle. Nothing in the layout is told to you; find
everything with the cameras.

Use `evaluate` to get the trusted binary result of the current episode.
"""

# Work areas (TCP-reachable, see docs/DESIGN.md "Reach"): each arm works on its own side.
SIDE_REGION = {  # x range, |y| range for a fruit centre
    'x': (0.24, 0.39),
    'y': (0.16, 0.35),
}
BOWL_REGION = {'x': (0.29, 0.36), 'y': (-0.04, 0.04)}


def _yaw_quat(yaw):
    return np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])  # MuJoCo [w, x, y, z]


def fruit_rest_height(name):
    f = FRUITS[name]
    if f['kind'] == 'sphere':
        return f['radius']
    if f['kind'] == 'ellipsoid':
        return f['size'][2]
    return f['radius']


def sample_layout(seed, reachable=None, max_tries=200):
    """{'bowl': (x, y, yaw), name: (x, y, yaw), ...} for this seed.

    reachable(name, x, y, yaw) -> bool optionally vetoes a fruit placement.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        layout = {'bowl': (rng.uniform(*BOWL_REGION['x']), rng.uniform(*BOWL_REGION['y']), rng.uniform(-math.pi, math.pi))}
        # Each fruit goes to a side; at least one per side.
        names = list(FRUITS)
        sides = rng.permutation([1, -1, int(rng.choice([1, -1]))])
        ok = True
        for name, side in zip(names, sides):
            for _ in range(60):
                x = rng.uniform(*SIDE_REGION['x'])
                y = side*rng.uniform(*SIDE_REGION['y'])
                yaw = rng.uniform(-math.pi, math.pi)
                clear = all(math.hypot(x - p[0], y - p[1]) > (0.13 if 'banana' in (name, other) else 0.10)
                            for other, p in layout.items() if other != 'bowl')
                bowl_clear = math.hypot(x - layout['bowl'][0], y - layout['bowl'][1]) > BOWL['inner_radius'] + 0.075 + (0.06 if name == 'banana' else 0)
                if clear and bowl_clear and (reachable is None or reachable(name, x, y, yaw)):
                    layout[name] = (x, y, yaw)
                    break
            else:
                ok = False
                break
        if ok:
            return layout
    raise RuntimeError(f'No feasible layout for seed {seed}')


def apply_layout(model, data, layout):
    for name, (x, y, yaw) in layout.items():
        j = model.joint(f'{name}_free')
        a = model.jnt_qposadr[j.id]
        z = TABLE['top'] + (0.001 if name == 'bowl' else fruit_rest_height(name) + 0.002)
        data.qpos[a:a + 3] = (x, y, z)
        data.qpos[a + 3:a + 7] = _yaw_quat(yaw)
        data.qvel[model.jnt_dofadr[j.id]:model.jnt_dofadr[j.id] + 6] = 0


def task_state(model, data):
    """Privileged success check (never shown to the agent except as the binary result)."""
    bowl = data.body('bowl')
    bowl_up = float(bowl.xmat.reshape(3, 3)[2, 2])
    bowl_pos = bowl.xpos
    on_table = abs(bowl_pos[2] - TABLE['top']) < 0.02
    bowl_still = float(np.linalg.norm(data.joint('bowl_free').qvel[:3])) < 0.05
    pads = {model.geom(f'{f}_pad').id for f in ('left_left_finger', 'left_right_finger', 'right_left_finger', 'right_right_finger')}
    fruit_geoms = {name: {g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id} for name in FRUITS}
    held = {name: False for name in FRUITS}
    for c in data.contact[:data.ncon]:
        for name, geoms in fruit_geoms.items():
            if (c.geom1 in geoms and c.geom2 in pads) or (c.geom2 in geoms and c.geom1 in pads):
                held[name] = True
    inside = {}
    for name in FRUITS:
        p = data.body(name).xpos
        r = math.hypot(p[0] - bowl_pos[0], p[1] - bowl_pos[1])
        dz = p[2] - bowl_pos[2]
        still = float(np.linalg.norm(data.joint(f'{name}_free').qvel[:3])) < 0.05
        inside[name] = bool(r < BOWL['inner_radius'] - 0.005 and 0.0 < dz < BOWL['height'] + 0.06 and still and not held[name])
    complete = all(inside.values()) and bowl_up > 0.95 and on_table and bowl_still
    return {'task_complete': bool(complete), 'inside': inside, 'held': held, 'bowl_upright': bowl_up > 0.95,
            'bowl_on_table': bool(on_table)}
