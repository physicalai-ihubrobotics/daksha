"""Feasibility check for tray_transport with privileged poses, through the agent's own tools:
pre-close both hands, synchronized two-hand grasp and lift, drive 5 m to the place table,
set the tray down in the outline, release, evaluate.

    PYTHONPATH=$PWD MUJOCO_GL=egl /home/ihub/daksha_venv/bin/python local_tests/oracle_tray_transport.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.daksha_sim import tray_transport as task  # noqa: E402
from services.daksha_sim.backend import DakshaSimBackend  # noqa: E402
from services.daksha_sim.scene import HANDLE_PRECLOSE_GAP, TABLE_SPACING, TRAY_GOAL  # noqa: E402


ROUTE = 'strafe'


def call(b, name, **args):
    b.validate(name, args)
    out, _ = b.call(name, args)
    return out


def tray_pose_robot(b):
    """Tray centre (x, y) and yaw in the robot frame (privileged)."""
    tray = b.d.body('tray')
    p = b._to_base(tray.xpos)
    yaw = task._yaw_of(tray.xmat) - b._base_pose()[2]
    return p, yaw


def run(b, seed, log):
    b.reset_episode(seed)
    p, yaw = tray_pose_robot(b)
    # Handle grasps in the robot frame (the base is at home, yaw 0: robot frame = room frame shifted).
    x0, y0, _ = b._base_pose()
    grasps = {arm: (b._to_base(pos), axis) for arm, (pos, axis) in
              task.handle_grasps(*(b._to_world(p)[:2]), yaw).items()}
    for arm in ('left', 'right'):
        call(b, 'daksha_gripper', arm=arm, command='width', width=HANDLE_PRECLOSE_GAP)
    hand = lambda arm, dz, tol: {'position': (grasps[arm][0] + [0, 0, dz]).tolist(), 'approach': [0, 0, -1],  # noqa: E731
                                 'finger_axis': grasps[arm][1].tolist(), 'tolerance_deg': tol, 'finger_tolerance_deg': 10}
    call(b, 'daksha_move_eef_both', left=hand('left', 0.10, 30), right=hand('right', 0.10, 30), path='joint')
    try:  # straight down; if the line passes a wrist limit, descend in joint space (10 cm, clear above the walls)
        call(b, 'daksha_move_eef_both', left=hand('left', 0.0, 20), right=hand('right', 0.0, 20), speed=0.05)
    except ValueError as exc:
        log.append(f'linear descent refused ({str(exc)[:60]}...), joint descent')
        call(b, 'daksha_move_eef_both', left=hand('left', 0.0, 20), right=hand('right', 0.0, 20), path='joint')
    for arm in ('left', 'right'):
        r = call(b, 'daksha_gripper', arm=arm, command='close', force=30)
        log.append(f'grip {arm}: width {r["width_m"]} stalled {r["stalled_before_target"]}')
    lift = call(b, 'daksha_move_eef_both', left={'position': [0, 0, 0.10]}, right={'position': [0, 0, 0.10]},
                relative=True, speed=0.05)
    log.append(f'lift ok, tray z {b.d.body("tray").xpos[2]:.3f}')
    # Drive: back away from the table, along the aisle, up to the place table.
    legs = {'strafe': ({'forward': -0.35}, {'left': -TABLE_SPACING}, {'forward': 0.35}),
            # Like a person: step back, turn to face the aisle, walk, turn to face the table, step in.
            'turn': ({'forward': -0.35}, {'turn_deg': -90}, {'forward': TABLE_SPACING}, {'turn_deg': 90},
                     {'forward': 0.35})}[ROUTE]
    for leg in legs:
        r = call(b, 'daksha_move_base', **leg)
        log.append(f'drive {leg}: odom {r["odom"]} err {r["tracking_error"]} holding {r["fingers_touching_object"]}')
    # Set down: shift so the tray centre lands on the outline centre (outline in the robot frame).
    p_now, yaw_now = tray_pose_robot(b)
    goal = b._to_base(np.array([TRAY_GOAL[0], TRAY_GOAL[1], 0.0]))
    dx, dy = goal[0] - p_now[0], goal[1] - p_now[1]
    log.append(f'tray before set-down: offset to goal {dx:+.3f}, {dy:+.3f} m')
    call(b, 'daksha_move_eef_both', left={'position': [dx, dy, -0.095]}, right={'position': [dx, dy, -0.095]},
         relative=True, speed=0.04)
    for arm in ('left', 'right'):
        call(b, 'daksha_gripper', arm=arm, command='width', width=HANDLE_PRECLOSE_GAP)
    call(b, 'daksha_move_eef_both', left={'position': [0, 0, 0.10]}, right={'position': [0, 0, 0.10]},
         relative=True, speed=0.06)
    call(b, 'daksha_home', arm='both')
    st = b.task_state()
    log.append(json.dumps(st))
    return st['task_complete']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--route', choices=['strafe', 'turn'], default='strafe')
    args = ap.parse_args()
    global ROUTE
    ROUTE = args.route
    root = pathlib.Path(tempfile.mkdtemp(prefix='tray_oracle_'))
    (root/'ws').mkdir()
    b = DakshaSimBackend(types.SimpleNamespace(task='tray_transport', backend_options={}, observation_profile='rgbd',
                                               workspace=root/'ws', root=root))
    ok = 0
    for seed in range(args.seeds):
        log, t = [], time.time()
        try:
            success = run(b, seed, log)
        except Exception as exc:  # report and continue
            success = False
            log.append(f'ERROR {type(exc).__name__}: {exc}')
        ok += success
        print(f'--- seed {seed}: {"SUCCESS" if success else "FAIL"} ({time.time() - t:.0f} s wall, sim {b.d.time:.0f} s)')
        print('\n'.join('   ' + line for line in log))
    print(f'{ok}/{args.seeds} succeeded')
    b.close()


if __name__ == '__main__':
    main()
