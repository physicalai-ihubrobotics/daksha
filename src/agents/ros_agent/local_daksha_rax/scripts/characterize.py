"""Characterization protocol (RAX docs/REAL_ROBOT_ADAPTATION.md, Principle 2) for the
simulated Daksha: measure the motion and perception numbers the skills quote.

    MUJOCO_GL=egl PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/characterize.py > docs/characterization.json
"""
import json
import math
import pathlib
import sys
import tempfile
import time
import types

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.daksha_sim.backend import DakshaSimBackend  # noqa: E402
from services.daksha_sim.scene import TABLE  # noqa: E402


def call(b, name, **a):
    b.validate(name, a)
    return b.call(name, a)[0]


def project(b, cam, p):
    info = b._camera_info(cam)
    R, t = np.array(info['rotation_camera_to_robot']), np.array(info['position'])
    pc = R.T @ (np.asarray(p) - t)
    if pc[2] <= 0:
        return None
    u = info['fx']*pc[0]/pc[2] + info['cx'] - 0.5
    v = info['fy']*pc[1]/pc[2] + info['cy'] - 0.5
    return (u, v) if 0 <= u < info['width'] and 0 <= v < info['height'] else None


def main():
    root = pathlib.Path(tempfile.mkdtemp(prefix='daksha_char_'))
    b = DakshaSimBackend(types.SimpleNamespace(root=root, workspace=root/'ws', backend_options={}, observation_profile='rgbd'))
    out = {'measured_at': time.strftime('%Y-%m-%d'), 'simulator': f'MuJoCo {mujoco.__version__}'}
    top = TABLE['top']
    # --- motion accuracy and time over random reachable targets
    rng = np.random.default_rng(0)
    eef_err, eef_time, joint_err, wall = [], [], [], []
    blocked = 0
    b.reset_episode(5000)
    for i in range(30):
        arm = 'left' if i % 2 == 0 else 'right'
        s = 1 if arm == 'left' else -1
        p = [rng.uniform(0.24, 0.38), s*rng.uniform(0.12, 0.36), top + rng.uniform(0.05, 0.20)]
        t0 = time.time()
        try:
            r = call(b, 'daksha_move_eef', arm=arm, position=p, approach=[0, 0, -1], tolerance_deg=25, path='joint')
        except ValueError:
            continue
        wall.append(time.time() - t0)
        if r['contacts']:
            blocked += 1
            continue
        eef_err.append(r['position_error_m'])
        eef_time.append(r['duration_s'])
        joint_err.append(r['joint_tracking_error_max'])
        try:  # short straight moves
            r = call(b, 'daksha_move_eef', arm=arm, position=[0, 0, -0.04], relative=True, speed=0.08)
            if r['contacts']:
                blocked += 1
            else:
                eef_err.append(r['position_error_m'])
        except ValueError:
            pass
    call(b, 'daksha_home', arm='both')
    out['move_eef'] = {'samples_free_space': len(eef_err), 'samples_blocked_by_contact': blocked,
                       'note': 'free-space moves only; when fingers press on an object the TCP stops short and '
                               'the reply lists the contact', 'position_error_mm_median': round(1000*float(np.median(eef_err)), 2),
                       'position_error_mm_p95': round(1000*float(np.percentile(eef_err, 95)), 2),
                       'joint_tracking_error_rad_max': round(float(np.max(joint_err)), 4),
                       'joint_path_duration_s_median': round(float(np.median(eef_time)), 2),
                       'wall_seconds_per_call_median': round(float(np.median(wall)), 2)}
    # --- pixel_to_position accuracy on fruit tops (world camera) and from the wrist cameras
    errs = {'world': [], 'left': [], 'right': []}
    for seed in range(5001, 5009):
        b.reset_episode(seed)
        poses = b.privileged_poses()
        obs = call(b, 'daksha_observe', attach=False)
        for name in ('orange', 'apple'):
            c = np.array(poses[name])
            radius = 0.036 if name == 'orange' else 0.034
            topp = c + [0, 0, radius]
            for cam in ('world', 'left', 'right'):
                uv = project(b, cam, topp)
                if uv is None:
                    continue
                r = call(b, 'daksha_pixel_to_position', frame_id=obs['frame_id'], camera=cam, pixels=[list(uv)], window=1)
                pt = r['points'][0]
                if pt['valid']:
                    # The visible surface point nearest the camera along the ray: compare with the sphere surface.
                    errs[cam].append(abs(np.linalg.norm(np.array(pt['position']) - c) - radius))
    out['pixel_to_position_surface_error_mm'] = {
        cam: {'median': round(1000*float(np.median(v)), 2), 'p95': round(1000*float(np.percentile(v, 95)), 2), 'samples': len(v)}
        for cam, v in errs.items() if v}
    # --- gripper widths
    b.reset_episode(5100)
    out['gripper'] = {}
    for arm in ('left', 'right'):
        out['gripper'][arm] = {'open_width_m': call(b, 'daksha_gripper', arm=arm, command='open')['width_m'],
                               'closed_empty_width_m': call(b, 'daksha_gripper', arm=arm, command='close')['width_m']}
        call(b, 'daksha_gripper', arm=arm, command='open')
    out['fruit_size_m'] = {'orange_diameter': 0.072, 'apple_diameter': 0.076, 'apple_height': 0.068,
                           'banana_thickness': 0.036, 'banana_length': 0.20}
    out['bowl'] = {'inner_radius_m': 0.095, 'rim_height_above_table_m': 0.073}
    out['table_top_z_m'] = top
    b.close()
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
