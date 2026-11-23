"""Check the tray_transport environment: physics settle, reachability of the two-handed tray
grasp, base drive between the tables, and rendered views.

    PYTHONPATH=$PWD MUJOCO_GL=egl /home/ihub/daksha_venv/bin/python scripts/check_tray_scene.py --out runtime/tray_check
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.daksha_sim import tray_transport as task  # noqa: E402
from services.daksha_sim.kinematics import Kinematics, Target  # noqa: E402
from services.daksha_sim.scene import (GRIPPER_CLOSED, GRIPPER_JOINT, GRIPPER_MIMIC, HANDLE_PRECLOSE_GAP,  # noqa: E402
                                       PAD_GAP_OPEN, ARM_JOINTS, ARMS, BASE_HOME, BASE_JOINTS, HOME, PLACE_TABLE, TRAY_GOAL, TABLE_SPACING, TRAY,  # noqa: E402
                                       TRAY_TASK, build_model)

def step(m, d, seconds):
    for _ in range(int(seconds/m.opt.timestep)):
        mujoco.mj_step(m, d)


def check_reach(m, d, kin):
    tray = d.body('tray')
    grasps = task.handle_grasps(float(tray.xpos[0]), float(tray.xpos[1]), task._yaw_of(tray.xmat))
    results = {}
    qfull = d.qpos.copy()
    for arm, (p, axis) in grasps.items():
        res = {}
        for label, dz in (('pregrasp', task.PREGRASP_HEIGHT), ('grasp', 0.0), ('lift', task.LIFT_HEIGHT)):
            tgt = Target(p + [0, 0, dz], approach=[0, 0, -1], finger_axis=axis, tolerance_deg=20,
                         finger_tolerance_deg=10)
            q, pe, re, ok = kin.solve(qfull, arm, tgt)
            res[label] = {'ok': bool(ok), 'pos_err_mm': round(pe*1000, 1), 'rot_err_deg': round(math.degrees(re), 1)}
            if label == 'grasp':
                res['q_grasp'] = [round(float(v), 3) for v in q]
        results[arm] = res
    return results


def clearance(m, d, geoms_a, body_names):
    """Smallest signed distance (mm, negative = overlap) between geoms_a and the bodies' collision geoms."""
    ids = {m.body(n).id for n in body_names}
    geoms_b = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_contype[g]]
    ft = np.zeros(6)
    best = min(mujoco.mj_geomDistance(m, d, a, b, 0.1, ft) for a in geoms_a for b in geoms_b)
    return round(float(best)*1000, 1)


def set_preclose(m, d):
    """Half-close both hands to HANDLE_PRECLOSE_GAP (kinematic)."""
    travel = (PAD_GAP_OPEN - HANDLE_PRECLOSE_GAP)/2
    for arm in ARMS:
        sign = 1 if GRIPPER_CLOSED[arm] > 0 else -1
        d.joint(GRIPPER_JOINT[arm]).qpos = sign*travel
        d.joint(GRIPPER_MIMIC[arm]).qpos = sign*travel


def grasp_clearance(m, d, reach):
    """Arms at the grasp IK solution, hands pre-closed: clearance of every arm geom to tray and fruit."""
    saved = d.qpos.copy()
    for arm in ARMS:
        for j, v in zip(ARM_JOINTS[arm], reach[arm]['q_grasp']):
            d.joint(j).qpos = v
    set_preclose(m, d)
    mujoco.mj_forward(m, d)
    arm_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and
                 m.body(m.geom_bodyid[g]).name.startswith(('left_', 'right_'))]
    out = {n: clearance(m, d, arm_geoms, [n]) for n in ('tray', *task.FRUIT_NAMES)}
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return out


def grasp_pose_check(m, d, reach, path):
    """Put both arms in their grasp IK solution (grippers open, kinematics only) and look for
    finger/arm penetration into the tray; render close-ups."""
    saved = d.qpos.copy()
    for arm in ARMS:
        for j, v in zip(ARM_JOINTS[arm], reach[arm]['q_grasp']):
            d.joint(j).qpos = v
    set_preclose(m, d)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.body('tray').xpos + [0, 0, 0.06]
    imgs = []
    for az, el, dist in ((180, -35, 0.85), (90, -20, 0.6), (270, -20, 0.6), (180, -80, 0.7)):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam)
        imgs.append(r.render())
    r.close()
    Image.fromarray(np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)).save(path)
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return {'image': str(path)}


def contacts(m, d, bodies_a, bodies_b):
    ida = {m.body(b).id for b in bodies_a}
    idb = {m.body(b).id for b in bodies_b}
    hits = []
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in ida and b2 in idb) or (b2 in ida and b1 in idb):
            hits.append((m.body(b1).name, m.body(b2).name, round(float(c.dist), 4)))
    return hits


def drive_base(m, d, target, seconds, max_speed=0.5):
    """Move base actuators to target (x, y, yaw) along a straight, speed-limited line."""
    act = [m.actuator(n).id for n in BASE_JOINTS]
    start = np.array([d.ctrl[a] for a in act])
    target = np.asarray(target, float)
    dist = np.linalg.norm(target[:2] - start[:2])
    dur = max(seconds, dist/max_speed, abs(target[2] - start[2])/0.6)
    n = int(dur/m.opt.timestep)
    worst = []
    for i in range(n):
        s = (i + 1)/n
        s = s*s*(3 - 2*s)  # smooth start/stop
        d.ctrl[act] = start + s*(target - start)
        mujoco.mj_step(m, d)
        if i % 50 == 0:
            worst += contacts(m, d, ['base_link', 'wheel_base'], ['table', 'place_table'])
    step(m, d, 1.0)
    return worst


def render(m, d, cams, path, size=(480, 640)):
    r = mujoco.Renderer(m, *size)
    imgs = []
    for c in cams:
        r.update_scene(d, c)
        imgs.append(r.render())
    while len(imgs) % 2:
        imgs.append(np.zeros_like(imgs[0]))
    rows = [np.concatenate(imgs[i:i + 2], 1) for i in range(0, len(imgs), 2)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    r.close()
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runtime/tray_check')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m, _ = build_model(TRAY_TASK)
    d = mujoco.MjData(m)
    kin = Kinematics(m)
    report = {'model': {'nq': m.nq, 'nu': m.nu, 'nbody': m.nbody, 'cameras': [m.camera(i).name for i in range(m.ncam)],
                        'table_spacing_m': TABLE_SPACING, 'tray_mm': [TRAY['length']*1000, TRAY['width']*1000,
                                                                       TRAY['height']*1000]},
              'seeds': {}}

    # 1. Settle each seeded layout: tray flat on the pick table, fruit inside, nothing touching the robot.
    for seed in range(args.seeds):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        task.apply_layout(m, d, task.sample_layout(seed))
        mujoco.mj_forward(m, d)
        step(m, d, 2.0)
        tray = d.body('tray')
        R = tray.xmat.reshape(3, 3)
        fruit_in = {n: task._inside_tray(tray.xpos, R, d.body(n).xpos) for n in task.FRUIT_NAMES}
        robot_touch = contacts(m, d, ['tray', *task.FRUIT_NAMES],
                               [m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith(('left_', 'right_'))])
        report['seeds'][seed] = {
            'tray_z_above_table_mm': round((float(tray.xpos[2]) - 0.85)*1000, 1),
            'tray_upright': round(float(R[2, 2]), 4),
            'tray_speed': round(float(np.linalg.norm(d.joint('tray_free').qvel[:3])), 4),
            'fruit_in_tray': fruit_in, 'robot_touching': robot_touch,
            'reach': (reach := check_reach(m, d, kin)),
            'grasp_clearance_mm': grasp_clearance(m, d, reach),
            'task_complete_at_start': task.task_state(m, d)['task_complete'],
        }
        if seed == 0:
            report['views_start'] = render(m, d, ['overview', 'pick_view', 'world', 'place_view'], out/'start.png')
            report['grasp_pose'] = grasp_pose_check(m, d, report['seeds'][seed]['reach'], out/'grasp_pose.png')

    # 2. Drive: back off from the pick table, slide 5 m to the right, approach the place table.
    mujoco.mj_resetDataKeyframe(m, d, 0)
    task.apply_layout(m, d, task.sample_layout(0))
    mujoco.mj_forward(m, d)
    step(m, d, 1.0)
    bx = BASE_HOME[0]
    report['base_table_contacts_at_home'] = contacts(m, d, ['base_link', 'wheel_base'], ['table'])
    path = [(bx - 0.35, 0.0, 0.0), (bx - 0.35, -TABLE_SPACING, 0.0), (bx, -TABLE_SPACING, 0.0)]
    hits = []
    for wp in path:
        hits += drive_base(m, d, wp, 2.0)
    base = [float(d.joint(n).qpos[0]) for n in BASE_JOINTS]
    report['drive'] = {'waypoints': path, 'final_base': [round(v, 3) for v in base],
                       'base_table_contacts': hits[:5], 'n_base_table_contacts': len(hits),
                       'arm_tracking_err_rad': round(float(max(abs(d.joint(j).qpos[0] - HOME[j])
                                                               for a in ARMS for j in ARM_JOINTS[a])), 4)}
    report['views_at_place_table'] = render(m, d, ['overview', 'place_view', 'world', 'pick_view'], out/'at_place.png')
    # 3. Success check: the tray (with its fruit) moved rigidly into the goal outline must pass;
    #    the same with the banana dropped on the table must fail.
    for label, spill in (('tray_in_goal', False), ('banana_spilled', True)):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        lay = task.sample_layout(0)
        tx, ty, tyaw = lay['tray']
        gx, gy, gyaw = TRAY_GOAL
        moved = {}
        for n, (x, y, yaw) in lay.items():
            c, s_ = math.cos(gyaw - tyaw), math.sin(gyaw - tyaw)
            moved[n] = (gx + c*(x - tx) - s_*(y - ty), gy + s_*(x - tx) + c*(y - ty), yaw + gyaw - tyaw)
        task.apply_layout(m, d, moved)
        if spill:
            adr = m.jnt_qposadr[m.joint('banana_free').id]
            d.qpos[adr:adr + 3] = (gx + 0.0, gy + 0.30, PLACE_TABLE['top'] + 0.02)
        mujoco.mj_forward(m, d)
        step(m, d, 2.0)
        report.setdefault('success_check', {})[label] = task.task_state(m, d)
    (out/'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
