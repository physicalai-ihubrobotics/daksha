"""Human-posture IK for Daksha, built on the `kinematics` package (placo) in this folder.

The solver is `kinematics/kinematics/arm_ik_solver.py:RobotKinematics` (placo frame task, joint
limits, posture task, self-collision constraint from `kinematics/config/collision_pairs.json`).
Robot model: assets/daksha/daksha_raw.xml (MuJoCo MJCF), converted to URDF for placo by
local_pipeline/mjcf_to_urdf.py (FK identical to MuJoCo on that XML); `robot_model='urdf'` uses
assets/daksha/robot_collision.urdf instead. Additions on top of RobotKinematics,
with three additions so the arm looks like a human arm while it reaches:

  1. approach task      the TCP x axis (wrist -> fingertips) aligned with a direction (e.g. down);
                        optional finger axis; or a full quaternion through the frame task
  2. elbow task (soft)  the elbow (link_5) pulled below and slightly in front of the shoulder,
                        close to the torso: upper arm hanging, no "chicken wing"
  3. posture task       redundant joints relax toward a human reference posture, computed once
                        from (2) at a ready pose (instead of the all-zero configuration)

Several seeds are solved (current joints, human reference, sim home) and the solution that reaches
the target with the most human-like elbow wins. Only the chosen arm's 7 joints move; the other arm
and the grippers are frozen in the model (but still collision-checked).

The placo world frame equals the simulator's robot frame (FK matches the sim TCP to 0.1 mm).

CLI (separate from the pipeline):
    cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
    PYTHONPATH=$PWD MUJOCO_GL=egl /home/ihub/daksha_venv/bin/python -m local_pipeline.human_ik \\
        --arm left --pos 0.30 0.20 0.90 --approach 0 0 -1
    ... -m local_pipeline.human_ik --demo          # sim IK vs human IK, side-by-side images
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

PROJECT = pathlib.Path(__file__).resolve().parents[1]
KIN_PKG = PROJECT/'kinematics'
if str(KIN_PKG) not in sys.path:
    sys.path.insert(0, str(KIN_PKG))

from kinematics.arm_ik_solver import RobotKinematics  # noqa: E402  (the user's placo solver)

ARMS = ('left', 'right')
JOINTS = {arm: [f'{arm}_joint_{i}' for i in range(1, 8)] for arm in ARMS}
GRIPPER_JOINTS = ['left_gripper_left_joint', 'left_gripper_right_joint',
                  'right_gripper_left_joint', 'right_gripper_right_joint']
SHOULDER, ELBOW, WRIST = '{arm}_link_3', '{arm}_link_5', '{arm}_link_7'
SIM_HOME = {  # services/daksha_sim/scene.py HOME (kept here so this module needs no MuJoCo)
    'left_joint_1': -1.253, 'left_joint_2': -0.974, 'left_joint_3': 1.279, 'left_joint_4': -1.694,
    'left_joint_5': -0.291, 'left_joint_6': 0.687, 'left_joint_7': 1.309,
    'right_joint_1': 1.315, 'right_joint_2': -0.947, 'right_joint_3': 1.363, 'right_joint_4': 1.799,
    'right_joint_5': 0.372, 'right_joint_6': 0.583, 'right_joint_7': -1.379,
}
READY_TCP = {'left': [0.30, 0.22, 1.05], 'right': [0.30, -0.22, 1.05]}  # hands well above the fruit tops
# At all-zero joints Daksha's arm hangs straight down (elbow 0.22 m below the shoulder). joint_1 swings the
# upper arm forward, joint_2 lifts it sideways (the sim HOME has it at ~-0.97 rad: the "chicken wing"),
# joint_4 bends the elbow. A relaxed human arm: joint_2 ~ 0, upper arm a little forward, elbow bent ~75 deg.
HANGING = {'left': [-0.40, 0.0, 0.0, -1.30, 0.0, 0.0, 0.0], 'right': [0.40, 0.0, 0.0, 1.30, 0.0, 0.0, 0.0]}


def model_urdf(robot_model='xml', xml_path=None):
    """URDF path placo loads: 'xml' = daksha_raw.xml converted (default), 'urdf' = robot_collision.urdf."""
    if robot_model == 'xml':
        from .mjcf_to_urdf import DEFAULT_XML, convert
        return convert(xml_path or DEFAULT_XML, collision='primitives')
    if robot_model == 'urdf':
        return resolved_urdf()
    raise ValueError("robot_model must be 'xml' or 'urdf'")


def resolved_urdf(name='robot_collision.urdf'):
    """A copy of assets/daksha/<name> with package:// mesh paths made absolute (placo has no ROS)."""
    src = PROJECT/'assets'/'daksha'/name
    out = PROJECT/'runtime'/'ik'/name
    out.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text().replace('package://daksha_description_full_body/', str(PROJECT/'assets'/'daksha') + '/')
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return str(out)


def _unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v)


def _cone_target(current, desired, tolerance, margin=0.8):
    """The axis to pull toward: `current` itself while inside the tolerance cone around `desired`, otherwise the
    cone's edge (at margin*tolerance) on the way from desired to current."""
    from scipy.spatial.transform import Rotation
    c = float(np.clip(np.dot(current, desired), -1.0, 1.0))
    angle = math.acos(c)
    edge = margin*tolerance
    if angle <= edge:
        return np.asarray(current, float)
    axis = np.cross(desired, current)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.asarray(desired, float)
    return Rotation.from_rotvec(axis/n*edge).apply(desired)


class HumanPostureIK:
    """One solver per arm; `solve()` returns joint angles (rad) for a TCP target in the robot frame."""

    def __init__(self, arm, *, collision=True, elbow_weight=0.3, posture_weight=0.05, joint_margin=0.01,
                 adduct_weight=0.15, wrist_weight=0.05, upright_weight=0.2,
                 elbow_drop=0.17, elbow_forward=0.05, elbow_out=0.04, quiet=True, robot_model='xml', xml_path=None):
        if arm not in ARMS:
            raise ValueError(f'arm must be one of {ARMS}')
        self.arm, self.side = arm, (1.0 if arm == 'left' else -1.0)
        self.joints = JOINTS[arm]
        self.elbow_drop, self.elbow_forward, self.elbow_out = elbow_drop, elbow_forward, elbow_out
        pairs = KIN_PKG/'config'/'collision_pairs.json'
        with _Silence(quiet):
            self.kin = RobotKinematics(
                model_urdf(robot_model, xml_path), target_frame_name=f'{arm}_tcp', joint_names=self.joints,
                posture_joints_deg=None, posture_weight=posture_weight,
                collision_pairs_path=str(pairs) if collision and pairs.exists() else None)
        self.robot, self.solver = self.kin.robot, self.kin.solver
        # Same margins as the simulator, so a solution is never refused for a limit graze.
        for j in self.joints:
            lo, hi = self.robot.get_joint_limits(j)
            self.robot.set_joint_limits(j, lo + joint_margin, hi - joint_margin)
        self.limits = np.array([self.robot.get_joint_limits(j) for j in self.joints])
        for j in JOINTS['right' if arm == 'left' else 'left'] + GRIPPER_JOINTS:
            if j in self.robot.joint_names():
                self.solver.mask_dof(j)  # frozen, but still part of the collision model
        tcp = f'{arm}_tcp'
        # TCP position is the hard task (exact when feasible); everything else is a soft preference.
        self.kin.tip_frame.configure(tcp, 'soft', 0.0, 0.0)  # used only for full-quaternion targets
        self.pos_task = self.solver.add_position_task(tcp, np.zeros(3))
        self.pos_task.configure('tcp_position', 'hard', 1.0)
        self._pos_hard = True
        self.approach_task = self.solver.add_axisalign_task(tcp, np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
        self.approach_task.configure('approach', 'soft', 1.0)
        self.finger_task = self.solver.add_axisalign_task(tcp, np.array([0, 1.0, 0]), np.array([0, 1.0, 0]))
        self.finger_task.configure('finger', 'soft', 0.0)
        self.elbow_frame, self.shoulder_frame = ELBOW.format(arm=arm), SHOULDER.format(arm=arm)
        self.elbow_task = self.solver.add_position_task(self.elbow_frame, np.zeros(3))
        self.elbow_weight = elbow_weight
        self.elbow_task.configure('elbow_down', 'soft', elbow_weight)
        self.posture_weight = posture_weight
        # Human style, as direct joint preferences: upper arm close to the body (joint_2 ~ 0, no sideways lift),
        # wrist not bent sideways (joint_6 ~ 0).
        self.adduct_task = self.solver.add_joints_task()
        self.adduct_task.set_joints({f'{arm}_joint_2': 0.0})
        self.adduct_weight, self.wrist_weight = adduct_weight, wrist_weight
        self.adduct_task.configure('adduct', 'soft', adduct_weight)
        self.wrist_task = self.solver.add_joints_task()
        self.wrist_task.set_joints({f'{arm}_joint_6': 0.0})
        self.wrist_task.configure('wrist_neutral', 'soft', wrist_weight)
        # A weak, constant pull toward the exact wanted approach: inside the cone the gripper still prefers
        # straight, but only as much as the human-style tasks allow.
        self.upright_task = self.solver.add_axisalign_task(tcp, np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
        self.upright_weight = upright_weight
        self.upright_task.configure('upright', 'soft', 0.0)
        self._cone = (math.radians(10), math.radians(10))
        self.reference = self._human_reference()
        self.kin.posture_task.set_joints({j: float(v) for j, v in zip(self.joints, self.reference)})

    # ------------------------------------------------------------------ model helpers
    def set_state(self, q_all: dict):
        """Set every joint we know (both arms, grippers) from {name: rad}; unknown names are ignored."""
        names = set(self.robot.joint_names())
        for j, v in q_all.items():
            if j in names:
                self.robot.set_joint(j, float(v))
        self.robot.update_kinematics()

    def q(self):
        return np.array([self.robot.get_joint(j) for j in self.joints])

    def _set_q(self, q):
        for j, v in zip(self.joints, q):
            self.robot.set_joint(j, float(v))
        self.robot.update_kinematics()

    def frame(self, name):
        return self.robot.get_T_world_frame(name)

    def fk(self, q=None):
        if q is not None:
            self._set_q(q)
        return self.frame(f'{self.arm}_tcp')

    def elbow_goal(self, target=None):
        """Below the shoulder, a little forward and out; leaning slightly toward the target."""
        s = self.frame(self.shoulder_frame)[:3, 3]
        goal = s + np.array([self.elbow_forward, self.side*self.elbow_out, -self.elbow_drop])
        if target is not None:  # a far reach lifts the elbow forward a bit, like a person does
            reach = np.asarray(target) - s
            goal[0] += 0.25*max(0.0, reach[0] - 0.25)
        return goal

    def human_cost(self, q=None):
        """0 = elbow exactly where a relaxed human elbow would be; grows with height and splay."""
        if q is not None:
            self._set_q(q)
        s = self.frame(self.shoulder_frame)[:3, 3]
        e = self.frame(self.elbow_frame)[:3, 3]
        lift = max(0.0, e[2] - (s[2] - self.elbow_drop))           # elbow above its relaxed height
        splay = max(0.0, self.side*(e[1] - s[1]) - self.elbow_out)  # elbow out to the side
        return float(lift + 1.5*splay)

    # ------------------------------------------------------------------ solve
    def _iterate(self, target, approach, finger_axis, quat, iterations, tol):
        T = np.eye(4)
        T[:3, 3] = target
        if quat is not None:
            from scipy.spatial.transform import Rotation
            T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        self.kin.tip_frame.T_world_frame = T
        self.kin.tip_frame.configure(f'{self.arm}_tcp', 'soft', 0.0, 1.0 if quat is not None else 0.0)
        self.pos_task.target_world = np.asarray(target, float)
        self._set_pos_priority(True)
        use_axis = quat is None and approach is not None
        self.approach_task.configure('approach', 'soft', 0.5 if use_axis else 0.0)
        use_finger = quat is None and finger_axis is not None
        self.finger_task.configure('finger', 'soft', 0.3 if use_finger else 0.0)
        self.upright_task.configure('upright', 'soft', self.upright_weight if use_axis else 0.0)
        if use_axis:
            self.upright_task.targetAxis_world = _unit(approach)
        prev = None
        for _ in range(iterations):
            self.elbow_task.target_world = self.elbow_goal(target)
            # Orientation as a cone, like a person: no pull while the gripper is within the tolerance of the
            # wanted direction, so the wrist stays free and the elbow can stay low.
            if use_axis or use_finger:
                R = self.fk()[:3, :3]
                if use_axis:
                    self.approach_task.targetAxis_world = _cone_target(R[:, 0], _unit(approach), self._cone[0])
                if use_finger:
                    f = _unit(finger_axis)
                    f = -f if np.dot(f, R[:, 1]) < 0 else f  # fingers are symmetric: either sign grips the same
                    self.finger_task.targetAxis_world = _cone_target(R[:, 1], f, self._cone[1])
            try:
                self.solver.solve(True)
            except RuntimeError:  # hard position infeasible here (limits/collision): soft, heavily weighted
                if not self._pos_hard:
                    raise
                self._set_pos_priority(False)
                continue
            self.robot.update_kinematics()
            q = self.q()
            if prev is not None and np.max(np.abs(q - prev)) < tol:
                break
            prev = q
        return self.q()

    def _style(self, scale):
        """Scale every human-style preference (elbow, posture, tucked arm, neutral wrist) together."""
        self.elbow_task.configure('elbow_down', 'soft', self.elbow_weight*scale)
        self.kin.set_posture_weight(self.posture_weight*scale)
        self.adduct_task.configure('adduct', 'soft', self.adduct_weight*scale)
        self.wrist_task.configure('wrist_neutral', 'soft', self.wrist_weight*scale)

    def _set_pos_priority(self, hard):
        if hard != self._pos_hard:
            self.pos_task.configure('tcp_position', 'hard' if hard else 'soft', 1.0 if hard else 100.0)
            self._pos_hard = hard

    def errors(self, target, approach=None, finger_axis=None, quat=None):
        T = self.fk()
        pe = float(np.linalg.norm(T[:3, 3] - np.asarray(target)))
        if quat is not None:
            from scipy.spatial.transform import Rotation
            re = float(np.linalg.norm((Rotation.from_matrix(T[:3, :3]).inv()*Rotation.from_quat(quat)).as_rotvec()))
        elif approach is not None:
            re = float(math.acos(np.clip(np.dot(T[:3, 0], _unit(approach)), -1, 1)))
        else:
            re = 0.0
        fe = 0.0
        if quat is None and finger_axis is not None:
            fe = float(math.acos(min(1.0, abs(np.dot(T[:3, 1], _unit(finger_axis))))))
        return pe, re, fe

    def solve(self, target, approach=(0, 0, -1), *, finger_axis=None, quat=None, q_current=None, state=None,
              tolerance_deg=30.0, finger_tolerance_deg=None, pos_tol=0.004, iterations=150, seeds=None):
        """Joint angles (rad, 7) for the TCP at `target` [m, robot frame].

        Returns dict(q, ok, position_error_m, approach_error_deg, finger_error_deg, human_cost, seed).
        `ok` = position within pos_tol and approach/finger within their tolerances.
        """
        target = np.asarray(target, float)
        if state:
            self.set_state(state)
        start = self.q() if q_current is None else np.asarray(q_current, float)
        if seeds is None:
            seeds = {'current': start, 'human': self.reference, 'hanging': np.array(HANGING[self.arm]),
                     'home': np.array([SIM_HOME[j] for j in self.joints])}
        tol_r = math.radians(tolerance_deg)
        tol_f = math.radians(finger_tolerance_deg if finger_tolerance_deg is not None else tolerance_deg)
        self._cone = (tol_r, tol_f)
        best = None
        stages = [1.0]
        if finger_axis is not None or quat is not None:
            stages += [0.4, 0.1]  # orientation-critical grasps: relax the human style only as far as needed
        for scale in stages:
            self._style(scale)
            stage_seeds = dict(seeds)
            if scale < 1.0:  # wrist pre-rolled seeds help the narrow wrist reach a finger direction
                for k, roll in (('roll+', 1.2), ('roll-', -1.2)):
                    q = np.array(self.reference, float)
                    q[6] = roll
                    stage_seeds[k] = q
            for name, seed in stage_seeds.items():
                self._set_q(np.clip(seed, self.limits[:, 0], self.limits[:, 1]))
                q = self._iterate(target, approach, finger_axis, quat, iterations, 1e-5)
                pe, re, fe = self.errors(target, approach, finger_axis, quat)
                ok = pe < pos_tol and re <= tol_r + 1e-6 and fe <= tol_f + 1e-6
                hc = self.human_cost(q)
                jump = float(np.max(np.abs(q - start)))
                # Reach the target first; then the most human-like; then the smallest move from now.
                score = (0 if ok else 1, round(hc, 3) + 0.02*jump + 10*pe)
                cand = {'q': q, 'ok': ok, 'position_error_m': round(pe, 4), 'approach_error_deg': round(math.degrees(re), 1),
                        'finger_error_deg': round(math.degrees(fe), 1), 'human_cost': round(hc, 3),
                        'max_joint_move_rad': round(jump, 3), 'seed': name if scale == 1.0 else f'{name}@style{scale}',
                        '_score': score}
                if best is None or cand['_score'] < best['_score']:
                    best = cand
            if best['ok']:
                break
        self._style(1.0)
        self._set_q(best['q'])
        best.pop('_score')
        return best

    def solve_line(self, start_q, target, approach=(0, 0, -1), *, finger_axis=None, step=0.01, **kw):
        """Straight TCP line from FK(start_q) to target, IK per `step` m, warm-started (posture stays continuous)."""
        self._set_q(start_q)
        p0 = self.fk()[:3, 3]
        dist = float(np.linalg.norm(np.asarray(target) - p0))
        n = max(1, int(math.ceil(dist/step)))
        qs, q = [np.asarray(start_q, float)], np.asarray(start_q, float)
        for i in range(1, n + 1):
            wp = p0 + (np.asarray(target) - p0)*i/n
            r = self.solve(wp, approach, finger_axis=finger_axis, q_current=q, seeds={'current': q}, **kw)
            if not r['ok'] or np.max(np.abs(r['q'] - q)) > 0.3:
                raise ValueError(f'human-posture line infeasible at {100*i/n:.0f}% '
                                 f'(pos err {r["position_error_m"]*1000:.1f} mm, approach err {r["approach_error_deg"]} deg)')
            q = r['q']
            qs.append(q)
        return np.array(qs)

    # ------------------------------------------------------------------ reference posture
    def _human_reference(self):
        """Ready-pose solution with a strong elbow-down pull: the posture the arm relaxes toward."""
        self.kin.set_posture_weight(0.0)
        self.elbow_task.configure('elbow_down', 'soft', 1.0)
        self.set_state(SIM_HOME)
        self._set_q(np.clip(HANGING[self.arm], self.limits[:, 0], self.limits[:, 1]))
        q = self._iterate(np.array(READY_TCP[self.arm]), (0, 0, -1), None, None, 300, 1e-6)
        self.elbow_task.configure('elbow_down', 'soft', self.elbow_weight)
        self.kin.set_posture_weight(self.posture_weight)
        return q


class _Silence:
    """placo prints a self-collision warning list at load; keep the console readable."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        if self.on:
            import os
            sys.stdout.flush()
            sys.stderr.flush()
            self._fds = (os.dup(1), os.dup(2))
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 1)
            os.dup2(self._null, 2)

    def __exit__(self, *a):
        if self.on:
            import os
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self._fds[0], 1)
            os.dup2(self._fds[1], 2)
            os.close(self._null)
            for fd in self._fds:
                os.close(fd)


class DualHumanIK:
    def __init__(self, **kw):
        self.arms = {arm: HumanPostureIK(arm, **kw) for arm in ARMS}

    def __getitem__(self, arm):
        return self.arms[arm]


# ---------------------------------------------------------------------- CLI
def _demo(args):
    """Same targets, sim IK vs human-posture IK: images + elbow numbers."""
    import tempfile
    import types

    from PIL import Image, ImageDraw

    from services.daksha_sim.backend import DakshaSimBackend
    out = PROJECT/'runtime'/'ik'/'demo'
    out.mkdir(parents=True, exist_ok=True)
    r = pathlib.Path(tempfile.mkdtemp(prefix='hik_'))
    b = DakshaSimBackend(types.SimpleNamespace(root=r, workspace=r/'ws', backend_options={}, observation_profile='rgbd'))
    b.reset_episode(args.seed)
    ik = DualHumanIK(robot_model=args.robot_model, xml_path=args.xml)
    targets = [('left', [0.30, 0.22, 0.90]), ('left', [0.33, 0.05, 0.95]), ('right', [0.30, -0.25, 0.90]),
               ('right', [0.38, -0.10, 0.92])]
    rows = []
    import mujoco
    ren = mujoco.Renderer(b.m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.15, 0.0, 1.05)
    cam.distance, cam.azimuth, cam.elevation = 1.6, 160.0, -12.0

    def shot(label):
        mujoco.mj_forward(b.m, b.d)
        ren.update_scene(b.d, cam)
        im = Image.fromarray(ren.render())
        ImageDraw.Draw(im).text((6, 6), label, fill=(255, 255, 255))
        return im
    for arm, p in targets:
        ik_state = {j: float(b.d.qpos[b.kin.qadr[a][i]]) for a in ARMS for i, j in enumerate(JOINTS[a])}
        h = ik[arm].solve(p, (0, 0, -1), state=ik_state, tolerance_deg=args.tolerance)
        from services.daksha_sim.kinematics import Target
        qs, pe, re, ok = b.kin.solve(b.d.qpos.copy(), arm,
                                     Target(b._to_world(np.array(p)), approach=b._R_wb() @ np.array([0, 0, -1.0]),
                                            tolerance_deg=args.tolerance))
        imgs = []
        for name, q in (('sim IK', qs), ('human IK', h['q'])):
            saved = b.d.qpos.copy()
            b.d.qpos[b.kin.qadr[arm]] = q
            elbow_cost = ik[arm].human_cost(q)
            imgs.append(shot(f'{name}: {arm} -> {p}  elbow cost {elbow_cost:.3f}'))
            b.d.qpos[:] = saved
        row = Image.new('RGB', (960, 360))
        row.paste(imgs[0], (0, 0))
        row.paste(imgs[1], (480, 0))
        rows.append(row)
        print(json.dumps({'arm': arm, 'target': p, 'sim_ik': {'ok': bool(ok), 'elbow_cost': round(ik[arm].human_cost(qs), 3)},
                          'human_ik': {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()}}))
    sheet = Image.new('RGB', (960, 360*len(rows)))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, 360*i))
    sheet.save(out/'sim_vs_human_ik.png')
    print('saved', out/'sim_vs_human_ik.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=ARMS, default='left')
    ap.add_argument('--pos', type=float, nargs=3, help='TCP target x y z (m, robot frame)')
    ap.add_argument('--approach', type=float, nargs=3, default=[0, 0, -1])
    ap.add_argument('--finger-axis', type=float, nargs=3)
    ap.add_argument('--tolerance', type=float, default=30.0, help='approach tolerance (deg)')
    ap.add_argument('--no-collision', action='store_true')
    ap.add_argument('--robot-model', choices=['xml', 'urdf'], default='xml',
                    help='xml = assets/daksha/daksha_raw.xml (default), urdf = assets/daksha/robot_collision.urdf')
    ap.add_argument('--xml', help='another MJCF file for --robot-model xml')
    ap.add_argument('--demo', action='store_true', help='compare with the sim IK on sample targets, save images')
    ap.add_argument('--seed', type=int, default=1000)
    args = ap.parse_args()
    if args.demo:
        return _demo(args)
    if args.pos is None:
        ap.error('--pos is required (or use --demo)')
    ik = HumanPostureIK(args.arm, collision=not args.no_collision, robot_model=args.robot_model, xml_path=args.xml)
    print('robot model:', model_urdf(args.robot_model, args.xml))
    print('human reference posture (rad):', np.round(ik.reference, 3).tolist())
    r = ik.solve(args.pos, args.approach, finger_axis=args.finger_axis, state=SIM_HOME, tolerance_deg=args.tolerance)
    r['q'] = np.round(r['q'], 4).tolist()
    T = ik.fk(np.array(r['q']))
    r['tcp'] = np.round(T[:3, 3], 4).tolist()
    r['elbow'] = np.round(ik.frame(ik.elbow_frame)[:3, 3], 3).tolist()
    r['shoulder'] = np.round(ik.frame(ik.shoulder_frame)[:3, 3], 3).tolist()
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
