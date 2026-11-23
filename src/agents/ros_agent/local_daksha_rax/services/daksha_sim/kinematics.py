"""Forward and inverse kinematics for Daksha's arms on the compiled MuJoCo model.

Uses a private MjData so planning never disturbs the simulation. The tool point is the
URDF's ``<arm>_tcp`` frame: its x axis is the approach direction (wrist to fingertips),
its y axis the finger-opening axis. Quaternions are [x, y, z, w] at the API boundary.

Orientation targets (``Target``):
- ``quat``: the full TCP orientation;
- ``approach`` (+ optional ``finger_axis``): the approach direction, with the roll about
  it free unless a finger axis is given (the finger axis is made perpendicular to the
  approach);
- ``tolerance_deg``: the approach (and finger axis) may deviate this much; IK stops
  pulling once inside the cone. This is what makes Daksha's narrow wrist usable: ask for
  "roughly down" and let the solver pick a feasible tilt.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .scene import ARM_JOINTS, HOME


def quat_xyzw_to_mat(q):
    return Rotation.from_quat(q).as_matrix()


def mat_to_quat_xyzw(m):
    return Rotation.from_matrix(m).as_quat()


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError('Direction vector has zero length')
    return v/n


def _axis_error(current, target, tolerance):
    """Rotation vector turning unit `current` toward unit `target`, minus the tolerance cone."""
    c = np.cross(current, target)
    s = np.linalg.norm(c)
    angle = float(np.arctan2(s, np.dot(current, target)))
    excess = max(0.0, angle - tolerance)
    if excess <= 0:
        return np.zeros(3), angle
    if s < 1e-9:  # opposite: any perpendicular axis
        perp = np.cross(current, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(current, [0, 1, 0])
        return _unit(perp)*excess, angle
    return c/s*excess, angle


@dataclass
class Target:
    position: np.ndarray
    quat: np.ndarray | None = None            # [x, y, z, w]
    approach: np.ndarray | None = None
    finger_axis: np.ndarray | None = None
    tolerance_deg: float = 0.0
    finger_tolerance_deg: float | None = None  # default: tolerance_deg

    def __post_init__(self):
        self.position = np.asarray(self.position, float)
        if self.quat is not None:
            R = quat_xyzw_to_mat(self.quat)
            self.approach, self.finger_axis = R[:, 0], R[:, 1]
        if self.approach is not None:
            self.approach = _unit(self.approach)
        if self.finger_axis is not None:
            if self.approach is None:
                raise ValueError('finger_axis needs an approach direction')
            f = np.asarray(self.finger_axis, float)
            f = f - np.dot(f, self.approach)*self.approach
            self.finger_axis = _unit(f)

    @property
    def tolerance(self):
        return np.radians(self.tolerance_deg)

    def errors(self, p, R):
        """(6-vector error, mask of active rotation rows, pos err, rot err rad)."""
        e_pos = self.position - p
        e_rot = np.zeros(3)
        rot_err = 0.0
        if self.approach is not None:
            ea, rot_err = _axis_error(R[:, 0], self.approach, self.tolerance)
            e_rot += ea
            if self.finger_axis is not None:
                # The finger axis is symmetric (a gripper turned 180 degrees is the same grasp).
                f = self.finger_axis if np.dot(R[:, 1], self.finger_axis) >= 0 else -self.finger_axis
                ftol = self.tolerance if self.finger_tolerance_deg is None else np.radians(self.finger_tolerance_deg)
                ef, fa = _axis_error(R[:, 1], f, ftol)
                fa = max(0.0, fa - ftol) + self.tolerance if fa > ftol else 0.0
                e_rot += ef
                rot_err = max(rot_err, fa)
        return e_pos, e_rot, float(np.linalg.norm(e_pos)), rot_err


class Kinematics:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = {arm: np.array([model.jnt_qposadr[model.joint(j).id] for j in js]) for arm, js in ARM_JOINTS.items()}
        self.dadr = {arm: np.array([model.jnt_dofadr[model.joint(j).id] for j in js]) for arm, js in ARM_JOINTS.items()}
        self.lower = {arm: np.array([model.jnt_range[model.joint(j).id][0] for j in js]) for arm, js in ARM_JOINTS.items()}
        self.upper = {arm: np.array([model.jnt_range[model.joint(j).id][1] for j in js]) for arm, js in ARM_JOINTS.items()}
        self.site = {arm: model.site(f'{arm}_tcp').id for arm in ARM_JOINTS}
        self.home = {arm: np.array([HOME[j] for j in js]) for arm, js in ARM_JOINTS.items()}
        self._rng = np.random.default_rng(7)
        self.margin = 0.008  # stay inside the backend's joint margin (0.005 rad)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def _set(self, qpos_full, arm=None, q=None):
        self.d.qpos[:] = qpos_full
        if arm is not None and q is not None:
            self.d.qpos[self.qadr[arm]] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def fk(self, qpos_full, arm, q=None):
        """(position, rotation matrix) of the arm's TCP."""
        self._set(qpos_full, arm, q)
        s = self.site[arm]
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def ik(self, qpos_full, arm, target: Target, seed=None, *, iterations=300, pos_tol=2e-3,
           rot_tol=np.radians(1.5), rot_weight=0.3, posture_weight=0.01):
        """Damped least squares with joint limits. Returns (q, pos_err_m, rot_err_rad, converged)."""
        q = np.array(seed if seed is not None else qpos_full[self.qadr[arm]], dtype=float)
        lo, hi = self.lower[arm] + self.margin, self.upper[arm] - self.margin
        q = np.clip(q, lo, hi)
        cols = self.dadr[arm]
        s = self.site[arm]
        best = None
        tol = target.tolerance + rot_tol
        for _ in range(iterations):
            self._set(qpos_full, arm, q)
            p = self.d.site_xpos[s]
            R = self.d.site_xmat[s].reshape(3, 3)
            e_pos, e_rot, pe, re = target.errors(p, R)
            score = pe + 0.05*max(0.0, re - target.tolerance)
            if best is None or score < best[0]:
                best = (score, q.copy(), pe, re)
            if pe < pos_tol and (target.approach is None or re <= tol):
                return q, pe, re, True
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, s)
            Jp, Jr = self._jp[:, cols], self._jr[:, cols]
            if target.approach is not None:
                if target.finger_axis is None:  # roll about the approach is free
                    a = R[:, 0]
                    Jr = (np.eye(3) - np.outer(a, a)) @ Jr
                J = np.vstack([Jp, rot_weight*Jr])
                e = np.concatenate([e_pos, rot_weight*e_rot])
            else:
                J, e = Jp, e_pos
            damping = 1e-3 + 0.05*pe**2
            JJt = J @ J.T + damping*np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (posture_weight*(self.home[arm] - q))
            step = np.max(np.abs(dq))
            if step > 0.25:
                dq *= 0.25/step
            q = np.clip(q + dq, lo, hi)
        _, q, pe, re = best
        return q, pe, re, False

    def solve(self, qpos_full, arm, target: Target, *, restarts=8, current_first=True):
        """IK that prefers the most accurate orientation the arm can reach: with a tolerance cone it
        tries 6 deg, then 12 deg, then the full tolerance, and returns the first that converges.
        Returns (q, pos_err, rot_err, converged)."""
        if target.approach is None or target.tolerance_deg <= 6:
            return self._solve(qpos_full, arm, target, restarts=restarts, current_first=current_first)
        result = None
        for tol in [t for t in (6.0, 12.0) if t < target.tolerance_deg] + [target.tolerance_deg]:
            rung = Target(target.position, approach=target.approach, finger_axis=target.finger_axis, tolerance_deg=tol,
                          finger_tolerance_deg=target.finger_tolerance_deg)
            result = self._solve(qpos_full, arm, rung, restarts=restarts if tol == target.tolerance_deg else 2,
                                 current_first=current_first)
            if result[3]:
                return result
        return result

    def _solve(self, qpos_full, arm, target: Target, *, restarts=8, current_first=True):
        """IK from the current posture, nearby seeds, home, then random restarts; the converged
        solution nearest the current posture, else the best failure."""
        current = qpos_full[self.qadr[arm]]
        seeds = [current, self.home[arm]] if current_first else [self.home[arm], current]
        # Nearby seeds first (solutions close to the current posture give short, predictable paths).
        seeds[1:1] = [current + self._rng.normal(0, 0.3, 7) for _ in range(3)]
        seeds += [self._rng.uniform(self.lower[arm], self.upper[arm]) for _ in range(restarts)]
        best_ok, best_fail = None, None
        for i, seed in enumerate(seeds):
            q, pe, re, ok = self.ik(qpos_full, arm, target, seed)
            if ok:
                dist = float(np.max(np.abs(q - current)))
                if best_ok is None or dist < best_ok[0]:
                    best_ok = (dist, (q, pe, re, True))
                if best_ok[0] < 0.6:
                    break
            elif best_fail is None or pe < best_fail[0]:
                best_fail = (pe, (q, pe, re, False))
        return best_ok[1] if best_ok else best_fail[1]
