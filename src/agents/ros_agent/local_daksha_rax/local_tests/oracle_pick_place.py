"""Operator-side feasibility check: solve the task with privileged object poses through the
same robot tools the agent uses. Never shipped to the agent's workspace.

    python -m local_tests.oracle_pick_place --seeds 1000 1001 1002
"""
import argparse
import json
import math
import pathlib
import tempfile
import time
import types

import numpy as np

from services.daksha_sim.backend import DakshaSimBackend
from services.daksha_sim.scene import TABLE, BOWL


def call(b, name, **args):
    b.validate(name, args)
    value, _ = b.call(name, args)
    return value


def pick_and_place(b, fruit, verbose=True):
    poses = b.privileged_poses()
    p = np.array(poses[fruit]); bowl = np.array(poses['bowl'])
    arm = 'left' if p[1] > 0 else 'right'
    top = TABLE['top']
    if fruit == 'banana':
        yaw = poses['banana_yaw']
        axis = np.array([math.cos(yaw), math.sin(yaw), 0])
        finger = np.cross([0, 0, 1], axis).tolist()
        orient = {'approach': [0, 0, -1], 'finger_axis': finger, 'tolerance_deg': 30, 'finger_tolerance_deg': 25}
        grasp_z = top + 0.014
    else:
        orient = {'approach': [0, 0, -1], 'tolerance_deg': 30}
        grasp_z = top + 0.012
    log = []
    def step(name, **a):
        t = time.time()
        v = call(b, name, **a)
        if verbose:
            print(f'  {name} {json.dumps({k: (np.round(x, 3).tolist() if isinstance(x, (list, np.ndarray)) else x) for k, x in a.items()})} '
                  f'-> tcp {v.get("tcp", {}).get("position")} width {v.get("width_m")} stalled {v.get("stalled_before_target")} '
                  f'contacts {v.get("contacts")} ({time.time()-t:.1f}s wall)')
        return v
    step('daksha_gripper', arm=arm, command='open')
    def move(**a):
        try:
            return step('daksha_move_eef', **a)
        except ValueError as exc:
            if a.get('path') == 'joint':
                raise
            print('   linear refused:', str(exc)[:120], '-> joint')
            return step('daksha_move_eef', **{**a, 'path': 'joint'})
    move(arm=arm, position=[p[0], p[1], top + 0.14], **orient)
    for dz in (0.0, 0.006, 0.012):          # straight down (joint path if not); a bit higher if a finger hits the table
        try:
            move(arm=arm, position=[p[0], p[1], grasp_z + dz], speed=0.08, **orient)
            break
        except ValueError as exc:
            if dz == 0.012 or 'into_table' not in str(exc):
                raise
            print(f'   finger touches the table at +{dz * 1000:.0f} mm -> {(dz + 0.006) * 1000:.0f} mm higher')
    g = step('daksha_gripper', arm=arm, command='close', force=25)
    step('daksha_move_eef', arm=arm, position=[0, 0, 0.14], relative=True, speed=0.08)
    side = 1 if arm == 'left' else -1
    move(arm=arm, position=[bowl[0], bowl[1] + side*0.03, top + 0.17], approach=[0, 0, -1], tolerance_deg=50)
    step('daksha_gripper', arm=arm, command='open')
    step('daksha_home', arm=arm)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1000])
    args = ap.parse_args()
    root = pathlib.Path(tempfile.mkdtemp(prefix='daksha_oracle_'))
    cfg = types.SimpleNamespace(root=root, workspace=root/'ws', backend_options={}, observation_profile='rgbd')
    b = DakshaSimBackend(cfg)
    wins = 0
    for seed in args.seeds:
        t = time.time()
        b.reset_episode(seed)
        print('seed', seed, b.privileged_poses())
        call(b, 'daksha_observe')
        for fruit in ('orange', 'apple', 'banana'):
            print(' ', fruit)
            try:
                pick_and_place(b, fruit)
            except Exception as exc:
                print('   FAILED:', type(exc).__name__, exc)
                call(b, 'daksha_home', arm='both')
            print('   state', b.task_state()['inside'])
        s = b.task_state()
        wins += s['task_complete']
        print('seed', seed, 'complete', s['task_complete'], s, f'{time.time()-t:.0f}s wall, sim {b.d.time:.1f}s')
    print(f'{wins}/{len(args.seeds)} complete; frames in {root}')
    b.close()


if __name__ == '__main__':
    main()
