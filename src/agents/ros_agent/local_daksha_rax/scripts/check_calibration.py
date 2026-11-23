"""Check the camera calibration (RAX docs/REAL_ROBOT_ADAPTATION.md: "verify it by projecting
known FK points into the image").

Moves each arm through a few poses above the table. At each pose it captures every camera,
poses the robot model with the measured joints and gripper opening, renders the gripper from
the calibrated camera pose with the frame's K and T_base_cam (from the calibration files) and:

- saves an overlay per image: green = gripper outline as FK + calibration predict it, orange =
  where the measured depth puts it, magenta cross = TCP. Green on the real fingers = calibrated;
- with depth on the fingers, measures the shift (pixels -> mm) and depth offset between the
  predicted and the measured gripper. PASS: median combined error <= --max-offset-mm.

    # simulation, exact calibration (expect ~0 px)
    PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/export_sim_calibration.py --out runtime/calibration/sim
    MUJOCO_GL=egl PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/check_calibration.py \\
        --backend-options runtime/calibration/sim/backend_options.json

    # a deliberately wrong one (expect FAIL)
    PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/export_sim_calibration.py --out runtime/calibration/bad --perturb-mm 15 --perturb-deg 2
    MUJOCO_GL=egl PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/check_calibration.py \\
        --backend-options runtime/calibration/bad/backend_options.json

    # the real robot (once services/backends.py registers it): --robot daksha_ros
    # --no-move checks the current pose only.

Output: runtime/calibration_checks/<stamp>/report.json and the overlay PNGs.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import types

import numpy as np
from PIL import Image

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MUJOCO_GL', 'egl')
from services.backends import make_backend  # noqa: E402
from services.calibration_check import FingerModel, check_frame  # noqa: E402
from services.daksha_sim.scene import ARMS, PAD_GAP_OPEN  # noqa: E402

DEFAULT_CAMERAS = ('world', 'left', 'right')


def poses(arm, top):
    s = 1 if arm == 'left' else -1
    return [(0.30, s*0.22, top + 0.18), (0.40, s*0.15, top + 0.14), (0.40, s*0.32, top + 0.20),
            (0.30, s*0.34, top + 0.12), (0.48, s*0.24, top + 0.16)]


def rotation_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1)/2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def summarise(rows, max_offset_mm):
    out = {}
    for cam in sorted({r['camera'] for r in rows}):
        rs = [r for r in rows if r['camera'] == cam]
        measured = [r for r in rs if r['verdict'] == 'measured']
        entry = {'images': len(rs), 'measured': len(measured)}
        if measured:
            med = lambda key: round(float(np.median([r[key] for r in measured])), 2)
            mm = np.array([r['offset_estimate_mm'] for r in measured])
            entry.update({'median_offset_px': med('offset_norm_px'), 'median_lateral_mm': med('lateral_offset_mm'),
                          'median_depth_offset_mm': med('depth_offset_mm'),
                          'median_offset_mm': round(float(np.median(mm)), 1), 'max_offset_mm': round(float(mm.max()), 1),
                          'median_agreement_at_zero': med('agreement_at_zero'),
                          'verdict': 'PASS' if np.median(mm) <= max_offset_mm else 'FAIL'})
        else:
            entry['verdict'] = 'CHECK_OVERLAYS'  # no depth on the fingers: judge the PNGs by eye
        truth = [r['truth'] for r in rs if 'truth' in r]
        if truth:
            entry['sim_truth_error'] = {'position_mm': round(float(np.median([t['position_mm'] for t in truth])), 1),
                                        'rotation_deg': round(float(np.median([t['rotation_deg'] for t in truth])), 2)}
        out[cam] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', help='take robot and backend options from runtime/sessions/<name>/configuration.json')
    ap.add_argument('--robot', help='backend registry key (default daksha_sim)')
    ap.add_argument('--backend-options', type=pathlib.Path, help='JSON object merged over the backend options')
    ap.add_argument('--cameras', nargs='+', default=list(DEFAULT_CAMERAS))
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    ap.add_argument('--no-move', action='store_true', help='check the current pose only')
    ap.add_argument('--speed', type=float, help='passed to daksha_move_eef')
    ap.add_argument('--max-offset-mm', type=float, default=4.0,
                    help='median error at the fingers (image-plane shift and depth offset combined) that still passes')
    ap.add_argument('--search-px', type=int, default=40)
    ap.add_argument('--seed', type=int, default=4242, help='simulation layout (sim only)')
    ap.add_argument('--out', type=pathlib.Path)
    ap.add_argument('--yes', action='store_true', help='do not ask before moving a real robot')
    args = ap.parse_args()

    robot, options = 'daksha_sim', {}
    if args.name:
        cfg = json.loads((PROJECT/'runtime'/'sessions'/args.name/'configuration.json').read_text())
        robot, options = cfg.get('robot', robot), dict(cfg.get('backend_options') or {})
    robot = args.robot or robot
    if args.backend_options:
        options.update(json.loads(args.backend_options.read_text()))
    out = args.out or PROJECT/'runtime'/'calibration_checks'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    root = out/'session'
    config = types.SimpleNamespace(root=root, workspace=root/'ws', robot=robot, task='fruit_to_bowl',
                                   backend_options=options, observation_profile='rgbd')
    if robot != 'daksha_sim' and not args.no_move and not args.yes:
        if input(f'{robot}: the arms will move to {len(poses("left", 0.85))} poses above the table. '
                 'Clear the workspace and keep the e-stop in reach. Type yes: ').strip() != 'yes':
            raise SystemExit('aborted')

    backend = make_backend(config)
    simulated = hasattr(backend, 'camera_truth')

    def call(name, **a):
        backend.validate(name, a)
        return backend.call(name, a)[0]

    fingers = FingerModel()
    rows = []
    try:
        if hasattr(backend, 'reset_episode'):
            backend.reset_episode(args.seed)
        status = call('daksha_status')
        top = status.get('table_top_z', 0.85)
        plan = [(None, None)] if args.no_move else [(arm, p) for arm in args.arms for p in poses(arm, top)]
        if not args.no_move:
            call('daksha_home', arm='both')
            for arm in ARMS:
                call('daksha_gripper', arm=arm, command='open')
        for i, (arm, target) in enumerate(plan):
            label = f'p{i:02d}' + (f'_{arm}' if arm else '')
            if target is not None:
                move = {'arm': arm, 'position': list(target), 'approach': [0, 0, -1], 'tolerance_deg': 25, 'path': 'joint'}
                if args.speed:
                    move['speed'] = args.speed
                try:
                    call('daksha_move_eef', **move)
                except Exception as error:  # unreachable or refused: skip the pose
                    print(f'{label}: skipped ({error})')
                    continue
            state = call('daksha_status')['arms']
            arms_state = {a: (np.array(state[a]['joints']), state[a]['gripper'].get('width_m', PAD_GAP_OPEN)) for a in ARMS}
            frame = call('daksha_observe', cameras=args.cameras, attach=False)
            for cam in args.cameras:
                meta = frame['cameras'][cam]
                rgb = np.asarray(Image.open(config.workspace/meta['rgb']).convert('RGB'))
                depth = None
                if meta.get('depth'):
                    depth = np.asarray(Image.open(config.workspace/meta['depth']), dtype=np.float64)/1000.0
                K, T = meta['intrinsics']['K'], meta['extrinsics']['T_base_cam']
                project = arms_state if cam == 'world' else {cam: arms_state[cam]}
                result, overlay = check_frame(fingers, project, K, T, rgb, depth, search_px=args.search_px)
                row = {'pose': label, 'camera': cam, **result, 'extrinsics_source': meta['extrinsics'].get('source')}
                if simulated:
                    true = backend.camera_truth(cam)
                    T = np.asarray(T)
                    row['truth'] = {'position_mm': round(1000*float(np.linalg.norm(T[:3, 3] - true[:3, 3])), 1),
                                    'rotation_deg': round(rotation_deg(T[:3, :3], true[:3, :3]), 2)}
                overlay.save(out/f'{label}_{cam}.png')
                rows.append(row)
                shift = (f"{row['offset_norm_px']} px, depth {row['depth_offset_mm']} mm -> {row['offset_estimate_mm']} mm"
                         if result['verdict'] == 'measured' else row.get('reason', ''))
                print(f"{label} {cam:5s} {result['verdict']:12s} {shift}")
        if not args.no_move:
            call('daksha_home', arm='both')
    finally:
        if hasattr(backend, 'end_episode'):
            backend.end_episode()
        backend.close()
        fingers.close()

    summary = summarise(rows, args.max_offset_mm)
    report = {'robot': robot, 'cameras': options.get('cameras'), 'max_offset_mm': args.max_offset_mm,
              'summary': summary, 'images': rows}
    (out/'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    print(f'overlays and report: {out}')
    return 0 if all(s['verdict'] != 'FAIL' for s in summary.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
