"""Camera extrinsics in the Daksha robot frame.

Adapted from RAX real/franka ``services/franka/calibration.py`` (same file format and
conventions), with one addition: Daksha has two arms, so a wrist camera names its arm.

Each camera's extrinsics is a YAML file with one 4x4 homogeneous ``matrix``:

- mount ``fixed`` (the head/world RealSense): ``T_cam_base``, P_cam = T @ P_base. The base
  frame is the one FK reports (services/daksha_sim/kinematics.py, the URDF root: origin on
  the floor under the robot, x forward, y left, z up). Daksha has no neck joint, so the head
  camera is fixed relative to it.
- mount ``tcp`` (the left/right wrist RealSenses): ``T_cam_tcp`` against ``<arm>_tcp``,
  P_cam = T @ P_tcp. The TCP is rigid on the gripper mount, so this transform is constant.

The camera frame is the OpenCV optical frame (x right, y down, z forward). ``K`` uses the
OpenCV pixel convention (pixel centres at integer coordinates), as in ``camera_info``.

The cameras are the backend option ``cameras``: one entry per camera with its ``name``
(world/left/right), ``mount`` ('fixed' or 'tcp'), ``arm`` (tcp only) and ``extrinsics``
YAML. A relative extrinsics path resolves under $DAKSHA_CALIBRATION_DIR.
"""
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import yaml

CALIBRATION_ENVIRONMENT = 'DAKSHA_CALIBRATION_DIR'
MOUNTS = ('fixed', 'tcp')
ARMS = ('left', 'right')


def calibration_dir():
    value = os.environ.get(CALIBRATION_ENVIRONMENT)
    return Path(value).expanduser() if value else None


def extrinsics_path(spec):
    path = Path(spec['extrinsics']).expanduser()
    if path.is_absolute():
        return path
    base = calibration_dir()
    if base is None:
        raise ValueError(f"camera {spec['name']}: the relative extrinsics path {path} needs "
                         f"{CALIBRATION_ENVIRONMENT} (or give an absolute path)")
    return base/path


def load_matrix(path):
    with open(path) as handle:
        matrix = np.asarray(yaml.safe_load(handle)['matrix'], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f'{path}: expected a 4x4 homogeneous matrix')
    R = matrix[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
        raise ValueError(f'{path}: the rotation block is not a proper rotation')
    return matrix


def write_matrix(path, matrix, comment=None):
    """Write a 4x4 matrix in the format load_matrix reads."""
    matrix = np.asarray(matrix, dtype=np.float64)
    text = ''.join(f'# {line}\n' for line in (comment or '').splitlines())
    text += yaml.safe_dump({'matrix': [[round(float(x), 9) for x in row] for row in matrix]},
                           default_flow_style=None, sort_keys=False)
    Path(path).write_text(text)


def pose_matrix(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


@dataclass(frozen=True)
class CameraModel:
    name: str
    mount: str            # 'fixed' (T_cam_base) or 'tcp' (T_cam_tcp, eye in hand)
    arm: str | None       # the arm a tcp camera rides on
    matrix: np.ndarray
    intrinsics: dict | None = None

    @classmethod
    def from_spec(cls, spec):
        if spec.get('mount') not in MOUNTS:
            raise ValueError(f"camera {spec.get('name')}: mount must be 'fixed' or 'tcp'")
        arm = spec.get('arm')
        if spec['mount'] == 'tcp' and arm not in ARMS:
            raise ValueError(f"camera {spec['name']}: a tcp camera needs arm 'left' or 'right'")
        if spec['mount'] == 'fixed' and arm is not None:
            raise ValueError(f"camera {spec['name']}: a fixed camera has no arm")
        intrinsics = spec.get('intrinsics')
        if intrinsics is not None:
            required = ('width', 'height', 'fx', 'fy', 'cx', 'cy')
            if any(k not in intrinsics for k in required):
                raise ValueError(f"camera {spec['name']}: intrinsics need {', '.join(required)}")
            intrinsics = {k: float(intrinsics[k]) for k in required}
            if (not np.isfinite(list(intrinsics.values())).all() or intrinsics['width'] <= 0 or
                    intrinsics['height'] <= 0 or min(intrinsics['fx'], intrinsics['fy']) <= 0):
                raise ValueError(f"camera {spec['name']}: invalid image size or focal length")
        return cls(spec['name'], spec['mount'], arm, load_matrix(extrinsics_path(spec)), intrinsics)

    def base_from_camera(self, T_base_tcp=None):
        """T_base_cam: the camera's pose in the base frame at the given TCP pose."""
        if self.mount == 'fixed':
            return np.linalg.inv(self.matrix)
        if T_base_tcp is None:
            raise ValueError(f'{self.name} is eye-in-hand; its pose needs the {self.arm} TCP pose')
        return np.asarray(T_base_tcp) @ np.linalg.inv(self.matrix)


def load_cameras(specs, allowed=None):
    """{name: CameraModel} from the backend option ``cameras`` (None or [] -> {})."""
    models = {}
    for spec in specs or []:
        if not isinstance(spec, dict) or 'name' not in spec or 'extrinsics' not in spec:
            raise ValueError('each camera needs at least name, mount and extrinsics')
        if allowed is not None and spec['name'] not in allowed:
            raise ValueError(f"camera {spec['name']!r}: choose one of {', '.join(allowed)}")
        if spec['name'] in models:
            raise ValueError(f"camera {spec['name']!r} is listed twice")
        models[spec['name']] = CameraModel.from_spec(spec)
    return models


def backproject(u, v, depth_m, K):
    """Pixel + metric depth -> camera-frame point (pinhole, OpenCV pixel convention)."""
    return np.array([(u - K[0][2]) / K[0][0] * depth_m, (v - K[1][2]) / K[1][1] * depth_m, depth_m])


def project(point_base, T_base_cam, K):
    p = np.linalg.inv(T_base_cam) @ np.append(np.asarray(point_base, dtype=np.float64), 1.0)
    if p[2] <= 0:
        return None
    uv = np.asarray(K) @ p[:3]
    return (uv[:2] / uv[2]).tolist()
