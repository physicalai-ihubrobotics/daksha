"""Write the simulated cameras' exact extrinsics as calibration files (RAX real/franka format,
services/calibration.py), plus the backend option list that loads them.

    PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/export_sim_calibration.py --out runtime/calibration/sim

--perturb-mm / --perturb-deg write a deliberately wrong calibration (each camera moved and
rotated by that much in a random direction) to show that scripts/check_calibration.py
catches a bad one. The files are also the template for the real robot's calibration.
"""
import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.calibration import pose_matrix, write_matrix  # noqa: E402
from services.daksha_sim.backend import true_camera_pose  # noqa: E402
from services.daksha_sim.scene import build_model  # noqa: E402

CAMERAS = {'world': ('fixed', None), 'left': ('tcp', 'left'), 'right': ('tcp', 'right')}


def perturbation(rng, mm, deg):
    axis = rng.normal(size=3)
    direction = rng.normal(size=3)
    R = Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(deg)).as_matrix()
    return pose_matrix(R, direction/np.linalg.norm(direction)*mm/1000.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('runtime/calibration/sim'))
    ap.add_argument('--perturb-mm', type=float, default=0.0)
    ap.add_argument('--perturb-deg', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m, _ = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(args.seed)
    specs = []
    for name, (mount, arm) in CAMERAS.items():
        T_base_cam = true_camera_pose(m, d, name)
        if mount == 'fixed':
            matrix, what = np.linalg.inv(T_base_cam), 'T_cam_base (P_cam = T @ P_base), base = robot frame'
        else:
            s = m.site(f'{arm}_tcp').id
            T_base_tcp = pose_matrix(d.site_xmat[s].reshape(3, 3), d.site_xpos[s])
            matrix, what = np.linalg.inv(T_base_cam) @ T_base_tcp, f'T_cam_tcp (P_cam = T @ P_tcp), against {arm}_tcp'
        note = 'exact simulator extrinsics'
        if args.perturb_mm or args.perturb_deg:
            matrix = perturbation(rng, args.perturb_mm, args.perturb_deg) @ matrix
            note = f'simulator extrinsics PERTURBED by {args.perturb_mm} mm / {args.perturb_deg} deg (seed {args.seed})'
        path = args.out/f'{name}.yaml'
        write_matrix(path, matrix, f'{name} camera: {what}\n{note}')
        specs.append({'name': name, 'mount': mount, **({'arm': arm} if arm else {}), 'extrinsics': str(path.resolve())})
    options = args.out/'backend_options.json'
    options.write_text(json.dumps({'cameras': specs}, indent=1) + '\n')
    print(f'wrote {", ".join(s["name"] for s in specs)} extrinsics and {options}')


if __name__ == '__main__':
    main()
