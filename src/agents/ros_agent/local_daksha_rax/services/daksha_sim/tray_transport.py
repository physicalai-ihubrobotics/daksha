"""tray_transport: carry a fruit tray with both hands from the pick table to the place table.

Scene (scene.py, task ``tray_transport``): the pick table in front of Daksha, a second identical
table TABLE_SPACING (5 m) away along -y, a blue 10"x7"x3" lab tray holding the orange, apple and
banana on the pick table, and a white outline marking the drop-off zone on the place table.
Daksha has a planar mobile base (base_x, base_y, base_yaw) to drive between the tables.
"""
from __future__ import annotations

import math

import numpy as np

from .scene import (FRUITS, HANDLE_GRASP_DEPTH, HANDLE_PRECLOSE_GAP, HANDLE_Y, PLACE_TABLE, TABLE, TRAY,
                    TRAY_FRUIT_SLOTS, TRAY_GOAL, TRAY_HOME, _tray_to_world)

FRUIT_NAMES = tuple(FRUITS)
TRAY_NAME = 'tray'
OBJECTS = FRUITS             # backend: privileged_poses lists these ...
BOWL_NAMES = (TRAY_NAME,)    # ... plus these containers

TASK_TEXT = """\
# Task: carry the fruit tray to the other table

A blue plastic tray (10" x 7" x 3") holding an orange, an apple and a banana sits on the table
in front of you. A second table stands 5 m away, to your right. Pick up the tray with both
hands, one on each short end (the end rims are the handles), keep it level so the fruit stays
in, drive to the second table (`daksha_move_base`) and set the tray down inside the white
outline on that table. Move both hands together (`daksha_move_eef_both`) while the tray is
held.

The task is complete when the tray rests upright on the second table inside the outline, all
three fruits are still inside the tray, and neither gripper is touching the tray.

Each episode moves and turns the tray slightly and shifts the fruit inside it. Find the tray,
the fruit and the outline from fresh camera images; no poses are given.

Use `evaluate` to get the trusted binary result of the current episode.
"""

GOAL_TOLERANCE = {'xy': 0.06, 'yaw_deg': 15.0}
PREGRASP_HEIGHT = 0.10   # above the grasp
LIFT_HEIGHT = 0.15       # carry height above the grasp


def handle_grasps(x, y, yaw, z_table=TABLE['top']):
    """Operator reference for the two-handed grasp of a tray at (x, y, yaw):
    {arm: (tcp_position, finger_axis)}; approach straight down (+-20 deg), pads pre-closed to
    HANDLE_PRECLOSE_GAP. Used to check layouts are solvable; never given to the agent."""
    axis = (math.cos(yaw), math.sin(yaw), 0.0)
    out = {}
    for side in (1, -1):
        lx = side*(TRAY['length']/2 - TRAY['wall']/2)
        gx, gy, _ = _tray_to_world((x, y, yaw), (lx, HANDLE_Y, 0.0))
        arm = 'left' if gy > y else 'right'
        z = z_table + 0.001 + TRAY['height'] - HANDLE_GRASP_DEPTH
        out[arm] = (np.array([gx, gy, z]), np.array(axis))
    return out


def _yaw_quat(yaw):
    return np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])


def _yaw_of(xmat):
    R = np.asarray(xmat).reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def fruit_rest_height(name):
    f = FRUITS[name]
    if f['kind'] == 'sphere':
        return f['radius']
    if f['kind'] == 'ellipsoid':
        return f['size'][2]
    return f['radius']


def sample_layout(seed, reachable=None, max_tries=40):
    """Tray pose on the pick table and fruit poses inside it: {name: (x, y, yaw)}."""
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        tray = (TRAY_HOME[0] + rng.uniform(-0.02, 0.02), TRAY_HOME[1] + rng.uniform(-0.02, 0.02),
                TRAY_HOME[2] + rng.uniform(-0.12, 0.12))
        if reachable and not reachable(TRAY_NAME, *tray):
            continue
        layout = {TRAY_NAME: tray}
        for name in FRUIT_NAMES:
            lx, ly, _ = TRAY_FRUIT_SLOTS[name]
            jitter = 0.008 if name != 'banana' else 0.004
            local = (lx + rng.uniform(-jitter, jitter), ly + rng.uniform(-jitter, jitter),
                     rng.uniform(-0.15, 0.15) if name == 'banana' else rng.uniform(-math.pi, math.pi))
            layout[name] = _tray_to_world(tray, local)
        return layout
    raise RuntimeError(f'No feasible tray layout for seed {seed}')


def apply_layout(model, data, layout):
    for name, (x, y, yaw) in layout.items():
        joint = model.joint(f'{name}_free')
        adr = model.jnt_qposadr[joint.id]
        if name == TRAY_NAME:
            z = TABLE['top'] + 0.001
        else:
            z = TABLE['top'] + TRAY['bottom'] + fruit_rest_height(name) + 0.002
        data.qpos[adr:adr + 3] = (x, y, z)
        data.qpos[adr + 3:adr + 7] = _yaw_quat(yaw)
        dof = model.jnt_dofadr[joint.id]
        data.qvel[dof:dof + 6] = 0


def _inside_tray(tray_pos, tray_R, point, margin=0.004):
    """Is a world point within the tray's inner footprint and between its floor and rim?"""
    local = tray_R.T @ (np.asarray(point) - tray_pos)
    half_l = TRAY['length']/2 - TRAY['wall'] - margin
    half_w = TRAY['width']/2 - TRAY['wall'] - margin
    return bool(abs(local[0]) < half_l and abs(local[1]) < half_w and
                0.0 < local[2] < TRAY['height'] + 0.03)


def task_state(model, data):
    """Trusted check: tray upright, still and inside the goal outline; fruit in it; not held."""
    tray = data.body(TRAY_NAME)
    tray_pos, tray_R = tray.xpos.copy(), tray.xmat.reshape(3, 3).copy()
    upright = float(tray_R[2, 2]) > 0.97
    on_place_table = (PLACE_TABLE['x0'] < tray_pos[0] < PLACE_TABLE['x1'] and
                      PLACE_TABLE['y0'] < tray_pos[1] < PLACE_TABLE['y1'] and
                      abs(float(tray_pos[2]) - PLACE_TABLE['top']) < 0.015)
    still = float(np.linalg.norm(data.joint(f'{TRAY_NAME}_free').qvel[:3])) < 0.05
    goal_dxy = math.hypot(tray_pos[0] - TRAY_GOAL[0], tray_pos[1] - TRAY_GOAL[1])
    # The tray is symmetric under a half turn.
    dyaw = (_yaw_of(tray_R) - TRAY_GOAL[2] + math.pi/2) % math.pi - math.pi/2
    in_goal = goal_dxy < GOAL_TOLERANCE['xy'] and abs(math.degrees(dyaw)) < GOAL_TOLERANCE['yaw_deg']

    pads = {model.geom(f'{f}_pad').id for arm in ('left', 'right')
            for f in (f'{arm}_left_finger', f'{arm}_right_finger')}
    tray_body = model.body(TRAY_NAME).id
    held = False
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 == tray_body and c.geom2 in pads) or (b2 == tray_body and c.geom1 in pads):
            held = True
            break
    fruit_in_tray = {name: _inside_tray(tray_pos, tray_R, data.body(name).xpos) for name in FRUIT_NAMES}
    complete = (upright and on_place_table and still and in_goal and not held and
                all(fruit_in_tray.values()))
    return {'task_complete': bool(complete),
            'tray': {'upright': upright, 'on_place_table': on_place_table, 'still': still,
                     'in_goal': in_goal, 'goal_offset_m': round(goal_dxy, 3),
                     'goal_yaw_err_deg': round(math.degrees(dyaw), 1), 'held': held},
            'fruit_in_tray': fruit_in_tray}
