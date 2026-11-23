"""Calibration check (RAX docs/REAL_ROBOT_ADAPTATION.md: verify a camera pose "by projecting
known FK points into the image").

For one camera image, the robot model (the same URDF meshes kinematics.py plans with) is
posed with the measured arm joints and gripper opening and rendered from a virtual camera
placed at the calibrated pose (T_base_cam) with the camera's intrinsics (K). Only the gripper
(the gripper mount and both fingers) is kept, using segmentation, so self-occlusion by the
wrist is handled. Then:

1. overlay: the predicted gripper outline is drawn on the image (green). With a good
   calibration it sits on the real fingers. A person or the agent judges this;
2. depth match (when the image has depth on the fingers): the predicted depth is slid over
   the measured depth; the shift with the best agreement is the calibration error in pixels
   at the fingers (drawn in orange when non-zero). Shift 0 with high agreement = calibrated.

Nothing here is simulator-specific: it needs an RGB image, optionally a metric depth image,
K, T_base_cam and the measured joints, which a real backend provides too. Rendering needs an
OpenGL context (MUJOCO_GL=egl works headless, including on the Jetson).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from PIL import Image

from .daksha_sim.scene import (ARM_JOINTS, FINGER_BODIES, GRIPPER_CLOSED, GRIPPER_JOINT, GRIPPER_MIMIC, MOUNT_BODY,
                               PAD_GAP_OPEN, _robot, _urdf_text)

PAD_HALF_THICKNESS = 0.016  # the gripper opening is measured between the pad faces
VIRTUAL_CAMERA = 'calibration_check'
ROBOT_VISUAL_GROUP = 1      # scene._robot puts every visual mesh in group 1
MAX_IMAGE = (1920, 1200)


def _robot_model():
    spec = mujoco.MjSpec.from_string(_urdf_text())
    spec.compiler.degree = False
    _robot(spec)
    cam = spec.worldbody.add_camera()
    cam.name = VIRTUAL_CAMERA
    spec.visual.global_.offwidth, spec.visual.global_.offheight = MAX_IMAGE
    return spec.compile()


class FingerModel:
    """The robot posed from measured joints: finger-pad boxes, TCP pose, and the gripper as a
    calibrated camera would see it."""

    def __init__(self):
        self.m = _robot_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {arm: np.array([m.jnt_qposadr[m.joint(j).id] for j in js]) for arm, js in ARM_JOINTS.items()}
        self.grip = {arm: (m.jnt_qposadr[m.joint(GRIPPER_JOINT[arm]).id], m.jnt_qposadr[m.joint(GRIPPER_MIMIC[arm]).id])
                     for arm in ARM_JOINTS}
        self.pads = {arm: [m.geom(f'{f}_pad').id for f in FINGER_BODIES[arm]] for arm in ARM_JOINTS}
        self.tcp = {arm: m.site(f'{arm}_tcp').id for arm in ARM_JOINTS}
        self.cam = m.camera(VIRTUAL_CAMERA).id
        # Bodies that make up each gripper: the mount and everything below it.
        self.gripper_bodies = {}
        for arm in ARM_JOINTS:
            root = m.body(MOUNT_BODY[arm]).id
            self.gripper_bodies[arm] = {b for b in range(m.nbody) if self._descends(b, root)}
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[ROBOT_VISUAL_GROUP] = 1
        self._renderer = None
        # Pad-face gap with the gripper fully open, from FK (differs from the nominal
        # PAD_GAP_OPEN by a few mm per arm), so a measured width maps to the right finger joint.
        self.gap_open = {arm: PAD_GAP_OPEN for arm in ARM_JOINTS}
        for arm in ARM_JOINTS:
            self.gap_open[arm] = self.opening(arm, np.zeros(7), PAD_GAP_OPEN)

    def _descends(self, body, root):
        while True:
            if body == root:
                return True
            if body == 0:
                return False
            body = self.m.body_parentid[body]

    def _finger_q(self, arm, width):
        closed = GRIPPER_CLOSED[arm]
        return math.copysign(min(abs(closed), max(0.0, (self.gap_open[arm] - float(width))/2)), closed)

    def set_state(self, arms):
        """arms: {arm: (joints7, gripper_width_m)}; other arms stay at zero."""
        self.d.qpos[:] = self.m.qpos0
        for arm, (joints, width) in arms.items():
            self.d.qpos[self.qadr[arm]] = joints
            drive, mimic = self.grip[arm]
            self.d.qpos[drive] = self.d.qpos[mimic] = self._finger_q(arm, width)  # scene mimic: mimic = drive
        mujoco.mj_kinematics(self.m, self.d)

    def boxes(self, arm, joints, width):
        """[(centre, R, half_sizes)] of the arm's two finger pads in the base frame."""
        self.set_state({arm: (joints, width)})
        return [(self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy(), self.m.geom_size[g].copy())
                for g in self.pads[arm]]

    def opening(self, arm, joints, width):
        """Pad-face gap FK gives for a measured width (self-test of the gripper mapping)."""
        (a, Ra, _), (b, _, _) = self.boxes(arm, joints, width)
        return max(0.0, abs(float(np.dot(a - b, Ra[:, 1]))) - 2*PAD_HALF_THICKNESS)

    def tcp_pose(self, arm, joints):
        self.set_state({arm: (joints, PAD_GAP_OPEN)})
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp[arm]].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp[arm]]
        return T

    def expected_depth(self, arms, K, T_base_cam, shape):
        """Depth (m, camera z) of the grippers as the calibrated camera would see them, NaN elsewhere."""
        h, w = shape
        if w > MAX_IMAGE[0] or h > MAX_IMAGE[1]:
            raise ValueError(f'image {w}x{h} is larger than {MAX_IMAGE}')
        K = np.asarray(K, float)
        T_base_cam = np.asarray(T_base_cam, float)
        self.set_state(arms)
        # MuJoCo cameras: square pixels, centred principal point, x right / y up / -z forward.
        # Render with focal fy, then resample to the real K (same centre: an exact 2-D mapping).
        f = K[1][1]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (T_base_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])).flatten())
        self.m.cam_pos[self.cam], self.m.cam_quat[self.cam] = T_base_cam[:3, 3], q
        self.m.cam_fovy[self.cam] = math.degrees(2*math.atan((h/2)/f))
        mujoco.mj_forward(self.m, self.d)
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (h, w):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.m, h, w)
        r = self._renderer
        r.enable_depth_rendering()
        r.update_scene(self.d, self.cam, self.option)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.d, self.cam, self.option)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        bodies = set().union(*(self.gripper_bodies[a] for a in arms))
        geom_body = self.m.geom_bodyid
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ids = np.where(is_geom, seg[..., 0], -1)
        keep = is_geom & np.isin(np.where(ids >= 0, geom_body[np.clip(ids, 0, None)], -1), list(bodies))
        rendered = np.where(keep, depth, np.nan)
        # Resample from the render's intrinsics to K.
        vv, uu = np.mgrid[0:h, 0:w]
        ur = np.rint(f*(uu - K[0][2])/K[0][0] + (w/2 - 0.5)).astype(int)
        vr = np.rint(f*(vv - K[1][2])/K[1][1] + (h/2 - 0.5)).astype(int)
        inside = (ur >= 0) & (ur < w) & (vr >= 0) & (vr < h)
        out = np.full((h, w), np.nan)
        out[inside] = rendered[vr[inside], ur[inside]]
        return out if np.isfinite(out).any() else None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def to_camera(points, T_base_cam):
    T = np.linalg.inv(T_base_cam)
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def match_depth(expected, measured, search_px=40, tolerance_m=0.01, max_bias_m=0.05):
    """Slide the predicted gripper depth over the measured depth: find the image shift and the
    depth bias (error along the viewing direction, which a shift cannot show) that agree best.

    agreement = fraction of predicted gripper pixels whose measured depth, minus the bias, is
    within tolerance."""
    vs, us = np.nonzero(np.isfinite(expected))
    z = expected[vs, us]
    h, w = measured.shape

    def score(du, dv, bias=None):
        u, v = us + du, vs + dv
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        obs = np.zeros_like(z)
        obs[inside] = measured[v[inside], u[inside]]
        valid = obs > 0
        r = obs - z
        if bias is None:
            near = valid & (np.abs(r) < max_bias_m)
            bias = float(np.median(r[near])) if near.sum() > 0.1*z.size else 0.0
        return (valid & (np.abs(r - bias) < tolerance_m)).mean(), valid.mean(), bias

    def search(centre, radius, step):
        best = None
        for dv in range(centre[1] - radius, centre[1] + radius + 1, step):
            for du in range(centre[0] - radius, centre[0] + radius + 1, step):
                s, _, bias = score(du, dv)
                key = s - 5e-4*math.hypot(du, dv)  # a flat optimum must not drift: prefer the smaller shift
                if best is None or key > best[0]:
                    best = (key, (du, dv), s, bias)
        return best

    _, coarse, _, _ = search((0, 0), search_px, 2)
    _, shift, agreement, bias = search(coarse, 2, 1)
    zero, coverage, _ = score(0, 0, 0.0)
    return {'pixels': int(z.size), 'offset_px': [int(shift[0]), int(shift[1])],
            'offset_norm_px': round(math.hypot(*shift), 2), 'depth_offset_mm': round(1000*bias, 1),
            'agreement_best': round(float(agreement), 3), 'agreement_at_zero': round(float(zero), 3),
            'depth_coverage': round(float(coverage), 3), 'median_depth_m': round(float(np.median(z)), 3)}


def check_frame(fingers, arms, K, T_base_cam, rgb, depth_m=None, *, search_px=40, tolerance_m=0.01,
                min_agreement=0.5, min_pixels=150):
    """Check one camera image. ``arms``: {arm: (joints7, gripper_width_m)} for the grippers to project.

    Returns (result dict, overlay PIL image)."""
    K = np.asarray(K, float)
    T_base_cam = np.asarray(T_base_cam, float)
    h, w = rgb.shape[:2]
    expected = fingers.expected_depth(arms, K, T_base_cam, (h, w))
    pixels = 0 if expected is None else int(np.isfinite(expected).sum())
    result = {'gripper_pixels': pixels}
    shift = None
    if pixels < min_pixels:
        result['verdict'] = 'not_in_view'
    elif depth_m is None:
        result['verdict'] = 'overlay_only'
        result['reason'] = 'no depth image'
    else:
        result.update(match_depth(expected, depth_m, search_px, tolerance_m))
        if result['depth_coverage'] < 0.3:
            result['verdict'] = 'overlay_only'
            result['reason'] = 'little valid depth on the gripper (too close for the depth sensor?)'
        elif result['agreement_best'] < min_agreement:
            result['verdict'] = 'overlay_only'
            result['reason'] = ('no shift makes the measured depth agree with the gripper (occluded, or the '
                                'calibration is off by more than the search radius)')
        else:
            shift = result['offset_px']
            f = (K[0][0] + K[1][1])/2
            lateral = 1000*result['offset_norm_px']*result['median_depth_m']/f
            result['lateral_offset_mm'] = round(lateral, 1)
            result['offset_estimate_mm'] = round(math.hypot(lateral, result['depth_offset_mm']), 1)
            result['verdict'] = 'measured'
    tcps = [to_camera([fingers.tcp_pose(arm, q)[:3, 3]], T_base_cam)[0] for arm, (q, _) in arms.items()]
    return result, draw_overlay(rgb, expected, tcps, K, shift)


def _outline(mask, thickness=1):
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    for _ in range(thickness - 1):
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    return edge


def draw_overlay(rgb, expected, tcps_cam, K, shift=None):
    """Green: gripper outline as FK + calibration predict it. Orange: where the measured depth
    puts it (only when shifted). Magenta cross: TCP."""
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    h, w = img.shape[:2]
    if expected is not None:
        mask = np.isfinite(expected)
        if shift is not None and tuple(shift) != (0, 0):
            moved = np.zeros_like(mask)
            vs, us = np.nonzero(mask)
            u, v = us + shift[0], vs + shift[1]
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            moved[v[ok], u[ok]] = True
            img[_outline(moved, 2)] = (255, 150, 0)
        img[_outline(mask, 2)] = (0, 255, 0)
    for p in tcps_cam:
        if p[2] <= 1e-3:
            continue
        u = int(round(K[0][0]*p[0]/p[2] + K[0][2]))
        v = int(round(K[1][1]*p[1]/p[2] + K[1][2]))
        for d in range(-6, 7):
            for uu, vv in ((u + d, v), (u, v + d)):
                if 0 <= uu < w and 0 <= vv < h:
                    img[vv, uu] = (255, 0, 255)
    return Image.fromarray(img)
