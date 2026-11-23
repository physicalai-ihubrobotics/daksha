"""Calibrate Daksha's world and wrist cameras with an ArUco marker.

Intrinsics are read from JSON (OpenCV convention):
  {"width":640,"height":480,"fx":...,"fy":...,"cx":...,"cy":...}

World camera: put a measured, flat marker at a known pose in the robot frame, then solve
its pose from one image. Wrist cameras: keep a marker fixed in the robot frame, move the
arm through at least 8 varied poses, and provide each image with that pose's T_base_tcp.
The wrist solver uses OpenCV hand-eye calibration and does not need the marker's robot pose.

Outputs are YAML extrinsics plus backend_options.json, compatible with services/calibration.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT = Path(__file__).resolve().parents[1]
DICT_NAMES = ('4x4_50', '4x4_100', '5x5_50', '5x5_100', '5x5_250', '6x6_50', '6x6_250', '7x7_50')


def dictionary(name):
    require_opencv()
    key = 'DICT_' + name.upper().replace('DICT_', '')
    if not hasattr(cv2.aruco, key):
        raise SystemExit(f'Unsupported ArUco dictionary {name!r}; choose from {", ".join(DICT_NAMES)}')
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def require_opencv():
    if cv2 is None or not hasattr(cv2, 'aruco'):
        raise SystemExit('Camera calibration needs OpenCV ArUco. Install it with: uv pip install --python /home/ihub/daksha_venv/bin/python opencv-contrib-python')


def make_marker(args):
    image = cv2.aruco.generateImageMarker(dictionary(args.dictionary), args.marker_id, 800)
    image = cv2.copyMakeBorder(image, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=255)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    print(f'Wrote {args.out}; print the black square exactly {args.size_mm:g} mm wide.')


def capture(args):
    """Capture images, intrinsics and TCP poses from an active Daksha MCP session."""
    sys.path.insert(0, str(PROJECT/'auto_research_agent'))
    from api.runtime import Context  # noqa: PLC0415
    args.out.mkdir(parents=True, exist_ok=True)
    arm = args.camera if args.camera in ('left', 'right') else None
    samples = []
    with Context(args.socket) as ctx:
        for i in range(args.count):
            answer = input(f'Place the marker in view of {args.camera}; move {arm or "the marker"} to pose {i+1}/{args.count}, then press Enter (q to stop): ')
            if answer.strip().lower() == 'q':
                break
            state = ctx.value('daksha_status')
            result = ctx.call('daksha_observe', cameras=[args.camera], attach=True)
            payload = next(json.loads(b['text']) for b in result['content'] if b.get('type') == 'text')
            images = Context.images(result)
            if not images:
                raise RuntimeError(f'{args.camera}: MCP observe returned no RGB image')
            meta = payload['cameras'][args.camera]['intrinsics']
            intrinsics = {k: meta[k] for k in ('width', 'height')}
            intrinsics['K'] = meta['K']
            (args.out/f'{args.camera}_intrinsics.json').write_text(json.dumps(intrinsics, indent=1)+'\n')
            image_path = args.out/f'{args.camera}_{i:02d}.png'
            image_path.write_bytes(images[0])
            sample = {'image': str(image_path.resolve())}
            if arm:
                tcp = state['arms'][arm]['tcp']
                sample['T_base_tcp'] = matrix_from_pose(tcp['position'], tcp['quat']).tolist()
            samples.append(sample)
            print(f'Saved {image_path}')
    manifest = args.out/f'{args.camera}_samples.json'
    manifest.write_text(json.dumps({'samples': samples}, indent=1)+'\n')
    print(f'Saved samples: {manifest}')


def read_intrinsics(path):
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = yaml.safe_load(text)
    raw_K = data.get('K', data.get('k'))
    if raw_K is not None:
        K = np.asarray(raw_K, dtype=float).reshape(3, 3)
        width, height = int(data['width']), int(data['height'])
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        width, height = int(data['width']), int(data['height'])
        fx, fy = float(data['fx']), float(data['fy'])
        cx, cy = float(data['cx']), float(data['cy'])
    if width <= 0 or height <= 0 or min(fx, fy) <= 0:
        raise ValueError(f'{path}: invalid camera intrinsics')
    return {'width': width, 'height': height, 'fx': float(fx), 'fy': float(fy),
            'cx': float(cx), 'cy': float(cy)}


def urdf_fixed_transform(parent, child):
    """Return T_parent_child for a fixed joint in this project's robot URDF."""
    root = ET.parse(PROJECT/'assets'/'daksha'/'robot.urdf').getroot()
    for joint in root.findall('joint'):
        p = joint.find('parent')
        c = joint.find('child')
        if p is None or c is None or p.get('link') != parent or c.get('link') != child:
            continue
        if joint.get('type') != 'fixed':
            raise ValueError(f'{parent} -> {child} must be a fixed URDF joint')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(f'invalid origin on fixed joint {joint.get("name")}')
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        T[:3, 3] = xyz
        return T
    raise ValueError(f'no fixed URDF joint from {parent!r} to {child!r}')


def import_hand_eye(args):
    """Import a ROS aruco_hand_eye result into the local backend format."""
    setups = {
        'right_wrist': ('right', 'tcp', 'right', 'eye_in_hand', 'right_gripper_mount'),
        'left_wrist': ('left', 'tcp', 'left', 'eye_in_hand', 'left_gripper_mount'),
        'world_right': ('world', 'fixed', None, 'eye_to_hand', 'base_link'),
        'world_left': ('world', 'fixed', None, 'eye_to_hand', 'base_link'),
    }
    name, mount, arm, mode, parent = setups[args.setup]
    result_path = args.result.resolve()
    result = yaml.safe_load(result_path.read_text())
    if result.get('setup') != args.setup or result.get('mode') != mode:
        raise ValueError(f'{result_path}: expected setup={args.setup!r}, mode={mode!r}')
    if result.get('parent_frame') != parent:
        raise ValueError(f'{result_path}: expected parent_frame={parent!r}, got {result.get("parent_frame")!r}')

    t_rms = float(result['t_rms_mm'])
    r_rms = float(result['r_rms_deg'])
    if t_rms > args.max_t_rms_mm or r_rms > args.max_r_rms_deg:
        raise ValueError(f'calibration residuals are {t_rms:.2f} mm / {r_rms:.2f} deg; limits are '
                         f'{args.max_t_rms_mm:.2f} mm / {args.max_r_rms_deg:.2f} deg. Re-solve or '
                         'explicitly raise --max-t-rms-mm / --max-r-rms-deg after reviewing the residuals.')

    translation, quaternion = result['translation'], result['rotation']
    T_parent_camera = np.eye(4)
    T_parent_camera[:3, :3] = Rotation.from_quat([quaternion[k] for k in ('x', 'y', 'z', 'w')]).as_matrix()
    T_parent_camera[:3, 3] = [translation[k] for k in ('x', 'y', 'z')]
    if mount == 'tcp':
        # ROS saves T_gripper_mount_camera. The local backend needs T_camera_tcp.
        T_mount_tcp = urdf_fixed_transform(parent, f'{arm}_tcp')
        transform = np.linalg.inv(T_parent_camera) @ T_mount_tcp
    else:
        # ROS saves T_base_camera; the local fixed-camera format stores T_camera_base.
        transform = np.linalg.inv(T_parent_camera)

    intrinsics = read_intrinsics(args.intrinsics)
    report = {'source': 'aruco_hand_eye', 'setup': args.setup,
              'source_result': str(result_path), 'parent_frame': parent,
              'child_frame': result.get('child_frame'), 'method': result.get('method'),
              'samples': result.get('n_samples'), 't_rms_mm': t_rms, 'r_rms_deg': r_rms}
    save_camera(args.out, name, mount, arm, transform, intrinsics,
                f'aruco_hand_eye_{result.get("method", "unknown")}', report)


def camera_matrix(K):
    return np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]], float)


def marker_pose(image_path, K, marker_size, marker_id, dict_name):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot read image: {image_path}')
    h, w = image.shape[:2]
    if (w, h) != (K['width'], K['height']):
        raise ValueError(f'{image_path}: image is {w}x{h}, intrinsics are {K["width"]}x{K["height"]}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dictionary(dict_name), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or marker_id not in ids.reshape(-1):
        raise ValueError(f'marker id {marker_id} not found in {image_path}')
    found_ids = ids.reshape(-1).tolist()
    corner = corners[found_ids.index(marker_id)].reshape(4, 2)
    half = marker_size / 2
    # ArUco corner order is top-left, top-right, bottom-right, bottom-left.
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], float)
    # Use iterative PnP: square IPPE can select the wrong planar solution for a nearly
    # front-facing marker, despite reporting success.
    ok, rvec, tvec = cv2.solvePnP(obj, corner, camera_matrix(K), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError(f'PnP failed for marker in {image_path}')
    R, _ = cv2.Rodrigues(rvec)
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = R
    T_cam_marker[:3, 3] = tvec.reshape(3)
    projected = cv2.projectPoints(obj, rvec, tvec, camera_matrix(K), None)[0].reshape(4, 2)
    error = float(np.mean(np.linalg.norm(projected - corner, axis=1)))
    if error > 3.0:
        raise ValueError(f'{image_path}: marker reprojection error {error:.2f}px exceeds 3px')
    return T_cam_marker, error


def marker_pose_base(position, yaw_deg):
    # Marker x points to robot-right; marker y (its printed top edge) points at yaw from +x.
    yaw = math.radians(yaw_deg)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    T[:3, 3] = np.asarray(position, float)
    return T


def matrix_from_pose(position, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def save_camera(out_dir, name, mount, arm, transform, K, method, report):
    out_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = out_dir / f'{name}.yaml'
    spec = {'name': name, 'mount': mount, 'extrinsics': str(extrinsics.resolve()),
            'intrinsics': K, 'calibration_method': method}
    if arm:
        spec['arm'] = arm
    entry = {'matrix': np.asarray(transform).round(9).tolist(),
             'intrinsics': K, 'method': method, 'report': report}
    extrinsics.write_text(yaml.safe_dump(entry, sort_keys=False))
    config_path = out_dir / 'backend_options.json'
    options = json.loads(config_path.read_text()) if config_path.exists() else {'cameras': []}
    cameras = [x for x in options.get('cameras', []) if x.get('name') != name]
    cameras.append(spec)
    options['cameras'] = sorted(cameras, key=lambda x: ('world', 'left', 'right').index(x['name']))
    config_path.write_text(json.dumps(options, indent=1) + '\n')
    print(f'Saved {name} calibration to {extrinsics}')
    print(f'Updated backend options: {config_path}')


def calibrate_fixed(args):
    K = read_intrinsics(args.intrinsics)
    T_cam_marker, error = marker_pose(args.image, K, args.marker_size, args.marker_id, args.dictionary)
    T_base_marker = marker_pose_base(args.marker_position, args.marker_yaw_deg)
    T_base_camera = T_base_marker @ np.linalg.inv(T_cam_marker)
    if T_base_camera[2, 3] < 0.05:
        raise ValueError(f'computed world camera height {T_base_camera[2, 3]:.3f}m is implausible')
    print(f'world camera position in robot frame: {T_base_camera[:3, 3].round(4).tolist()}')
    print(f'marker reprojection error: {error:.2f}px')
    save_camera(args.out, 'world', 'fixed', None, np.linalg.inv(T_base_camera), K,
                'aruco_fixed_pose', {'reprojection_error_px': round(error, 3),
                                     'marker_pose_base': T_base_marker.tolist()})


def calibrate_wrist(args):
    K = read_intrinsics(args.intrinsics)
    manifest_path = Path(args.samples).resolve()
    samples = json.loads(manifest_path.read_text())
    if isinstance(samples, dict):
        samples = samples.get('samples', [])
    if len(samples) < 8:
        raise ValueError(f'{args.samples}: need at least 8 varied poses, got {len(samples)}')
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = [], [], [], []
    marker_poses = []
    errors = []
    for i, sample in enumerate(samples):
        image = Path(sample['image'])
        if not image.is_absolute():
            image = manifest_path.parent / image
        T_base_tcp = np.asarray(sample.get('T_base_tcp'), float)
        if T_base_tcp.shape != (4, 4):
            raise ValueError(f'sample {i}: T_base_tcp must be 4x4')
        T_cam_marker, err = marker_pose(image, K, args.marker_size, args.marker_id, args.dictionary)
        R_gripper2base.append(T_base_tcp[:3, :3])
        t_gripper2base.append(T_base_tcp[:3, 3].reshape(3, 1))
        R_target2cam.append(T_cam_marker[:3, :3])
        t_target2cam.append(T_cam_marker[:3, 3].reshape(3, 1))
        marker_poses.append(T_cam_marker)
        errors.append(err)

    R_cam2tcp, t_cam2tcp = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK)
    T_tcp_cam = np.eye(4)
    T_tcp_cam[:3, :3] = R_cam2tcp
    T_tcp_cam[:3, 3] = np.asarray(t_cam2tcp).reshape(3)
    # Daksha stores T_cam_tcp; CameraModel computes T_base_cam = T_base_tcp @ inverse(T_cam_tcp).
    T_cam_tcp = np.linalg.inv(T_tcp_cam)

    # Each sample should reconstruct the same fixed marker pose. Report residuals as a quality check.
    reconstructed = [tcp @ T_tcp_cam @ cam_marker for tcp, cam_marker in
                     ((np.asarray(s['T_base_tcp'], float), m) for s, m in zip(samples, marker_poses))]
    centre = np.median(np.array([T[:3, 3] for T in reconstructed]), axis=0)
    R_mean = Rotation.from_matrix(np.array([T[:3, :3] for T in reconstructed])).mean().as_matrix()
    pos_res = [float(np.linalg.norm(T[:3, 3] - centre)) * 1000 for T in reconstructed]
    rot_res = [float(np.degrees(Rotation.from_matrix(R_mean.T @ T[:3, :3]).magnitude())) for T in reconstructed]
    report = {'samples': len(samples), 'mean_marker_reprojection_error_px': round(float(np.mean(errors)), 3),
              'median_marker_position_residual_mm': round(float(np.median(pos_res)), 2),
              'median_marker_rotation_residual_deg': round(float(np.median(rot_res)), 2)}
    print(f'{args.arm} wrist hand-eye result: {report}')
    save_camera(args.out, args.arm, 'tcp', args.arm, T_cam_tcp, K, 'aruco_hand_eye_park', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-hand-eye', help='import a saved ROS aruco_hand_eye result and CameraInfo intrinsics')
    p.add_argument('--setup', choices=('right_wrist', 'left_wrist', 'world_right', 'world_left'), required=True)
    p.add_argument('--result', type=Path, required=True, help='aruco_hand_eye result YAML')
    p.add_argument('--intrinsics', type=Path, required=True, help='JSON or ROS CameraInfo YAML')
    p.add_argument('--max-t-rms-mm', type=float, default=10.0)
    p.add_argument('--max-r-rms-deg', type=float, default=2.0)
    p.add_argument('--out', type=Path, default=PROJECT/'runtime'/'calibration'/'real')
    p.set_defaults(func=import_hand_eye)

    p = sub.add_parser('marker', help='generate a printable ArUco marker')
    p.add_argument('--out', type=Path, default=PROJECT/'runtime'/'calibration'/'aruco_4x4_id0.png')
    p.add_argument('--size-mm', type=float, default=80, help='physical black-square size when printed')
    p.add_argument('--marker-id', type=int, default=0)
    p.add_argument('--dictionary', default='4x4_50')
    p.set_defaults(func=make_marker)

    p = sub.add_parser('capture', help='capture RGB, intrinsics and TCP poses from the active robot MCP service')
    p.add_argument('--camera', choices=('world', 'left', 'right'), required=True)
    p.add_argument('--count', type=int, default=1, help='number of images (use 8 or more for a wrist camera)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--socket', help='robot MCP socket (defaults to RAX_MCP_SOCKET)')
    p.set_defaults(func=capture)

    p = sub.add_parser('fixed', help='calibrate the fixed world camera from a marker at a known robot-frame pose')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, required=True, help='JSON with width,height and fx,fy,cx,cy or K')
    p.add_argument('--marker-size', type=float, default=0.08, help='printed black marker side, metres')
    p.add_argument('--marker-id', type=int, default=0)
    p.add_argument('--dictionary', default='4x4_50')
    p.add_argument('--marker-position', type=float, nargs=3, default=[0.40, 0.0, 0.85], metavar=('X', 'Y', 'Z'),
                   help='marker centre in robot frame, metres')
    p.add_argument('--marker-yaw-deg', type=float, default=0.0,
                   help='marker top edge direction: 0=robot +x, 90=+y, 180=-x')
    p.add_argument('--out', type=Path, default=PROJECT/'runtime'/'calibration'/'real')
    p.set_defaults(func=calibrate_fixed)

    p = sub.add_parser('wrist', help='hand-eye calibrate one wrist camera from a sample manifest')
    p.add_argument('--arm', choices=('left', 'right'), required=True)
    p.add_argument('--samples', type=Path, required=True, help='JSON list of image paths and T_base_tcp matrices')
    p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--marker-size', type=float, default=0.08)
    p.add_argument('--marker-id', type=int, default=0)
    p.add_argument('--dictionary', default='4x4_50')
    p.add_argument('--out', type=Path, default=PROJECT/'runtime'/'calibration'/'real')
    p.set_defaults(func=calibrate_wrist)

    args = parser.parse_args()
    if hasattr(args, 'marker_id') and not 0 <= args.marker_id:
        parser.error('--marker-id must be non-negative')
    if args.command == 'capture' and args.count < 1:
        parser.error('--count must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
