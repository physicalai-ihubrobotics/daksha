"""Interruptible Daksha sim + VLM-guided manipulation skills for the local pipeline.

The robot is the same DakshaSimBackend the RAX harness uses (same `daksha_*` tools), with two
additions that only this pipeline needs:
  - an operator interrupt: `robot.interrupt.set()` stops any arm motion at the next 100 Hz
    control tick (raised as MotionStopped; the arm holds where it stopped);
  - real-time pacing, so motions are watchable in the browser and can be interrupted in time.

Perception: the local VLM (Ollama: qwen3.5:35b / qwen3.5:9b / qwen3-vl:8b) names and boxes objects in the head camera; depth back-projection
(`daksha_pixel_to_position`) turns the box into a 3-D position. Colour masks are kept as a
fallback when the VLM gives no answer.
"""
from __future__ import annotations

import io
import math
import threading
import time

import numpy as np
from PIL import Image

import sys


class MotionStopped(RuntimeError):
    """A motion was stopped (operator interrupt or a safety check). Raised by non-sim
    backends such as the ROS 2 real-robot backend; the MuJoCo backend raises its own
    class of the same name, and Skills accepts both (see _stop_types)."""


class DryRun(RuntimeError):
    """A real-robot backend planned a motion but execute is off. Deliberately not a
    ValueError/StepFailed, so skills don't retry or detour around it: it ends the command
    and the operator sees the plan."""


def _stop_types():
    sim = sys.modules.get('services.daksha_sim.backend')
    return (MotionStopped,) + ((sim.MotionStopped,) if sim is not None else ())

TABLE_Z = 0.85
CLEAR_POSE = {'left': [0.12, 0.46, 1.12], 'right': [0.12, -0.46, 1.12]}  # arms out of the head camera's view
GRASP_TCP_Z = TABLE_Z + 0.014
PREGRASP_DZ = 0.13
RELEASE_TCP_Z = TABLE_Z + 0.17
REACH = {'x': (0.22, 0.42), 'y': 0.37}      # where a hand can put things down on the table (robot frame, m)
HANDOVER_POSE = {'left': [0.36, 0.18, 1.10], 'right': [0.36, -0.18, 1.10]}  # held up in front, toward a person
BOWL_CLEAR = 0.15     # a put-down spot keeps this far from a container centre (bowl outer radius ~0.105 + fruit)
OBJECT_CLEAR = 0.09   # ... and this far from other objects
REGIONS = {  # named table spots (x, y)
    'center': (0.32, 0.0), 'middle': (0.32, 0.0), 'left': (0.31, 0.26), 'right': (0.31, -0.26),
    'front': (0.40, 0.0), 'far': (0.40, 0.0), 'back': (0.25, 0.0), 'near': (0.25, 0.0),
    'front-left': (0.39, 0.22), 'front-right': (0.39, -0.22), 'back-left': (0.25, 0.24), 'back-right': (0.25, -0.24),
}
CONTAINERS = ('bowl', 'toy_bowl', 'fruit_bowl', 'tray')
# TCP approach direction (TCP x axis) for grasp / place / hover, and the grasp's tolerance.
# Straight down suits the sim; the real Daksha's narrow wrist can't point straight down
# with any joint margin, so the real-robot config tilts it forward (see apply_scene).
GRASP_APPROACH = [0.0, 0.0, -1.0]
GRASP_TOL_DEG = 15.0


def apply_scene(table_z=None, grasp_dz=0.014, pregrasp_dz=None, release_dz=0.17, clear_pose=None,
                handover_pose=None, reach=None, regions=None, grasp_approach=None, grasp_tolerance_deg=None):
    """Set the table geometry the skills use (robot frame, metres). The defaults above are the
    MuJoCo scene; a real-robot backend calls this with measured values. Derived heights are
    recomputed so they stay consistent with table_z."""
    global TABLE_Z, GRASP_TCP_Z, PREGRASP_DZ, RELEASE_TCP_Z, CLEAR_POSE, HANDOVER_POSE, REACH, REGIONS
    global GRASP_APPROACH, GRASP_TOL_DEG
    if grasp_approach is not None:
        v = np.asarray(grasp_approach, float)
        GRASP_APPROACH = (v / np.linalg.norm(v)).tolist()
    if grasp_tolerance_deg is not None:
        GRASP_TOL_DEG = float(grasp_tolerance_deg)
    if table_z is not None:
        TABLE_Z = float(table_z)
    if pregrasp_dz is not None:
        PREGRASP_DZ = float(pregrasp_dz)
    GRASP_TCP_Z = TABLE_Z + float(grasp_dz)
    RELEASE_TCP_Z = TABLE_Z + float(release_dz)
    if clear_pose:
        CLEAR_POSE = {k: [float(v) for v in p] for k, p in clear_pose.items()}
    if handover_pose:
        HANDOVER_POSE = {k: [float(v) for v in p] for k, p in handover_pose.items()}
    if reach:
        REACH = {'x': tuple(reach['x']), 'y': float(reach['y'])}
    if regions:
        REGIONS = {k: tuple(v) for k, v in regions.items()}
    return {'table_z': TABLE_Z, 'grasp_tcp_z': GRASP_TCP_Z, 'release_tcp_z': RELEASE_TCP_Z}


class Interrupted(RuntimeError):
    """The operator interrupted the current command."""


class StepFailed(RuntimeError):
    pass


class _InterruptibleSim:  # mixin; combined with DakshaSimBackend on first use
    def __init__(self, config, realtime=1.0):
        self.interrupt = threading.Event()
        self.realtime = realtime            # 1.0 = wall clock, 0 = as fast as possible
        self._pace_wall = None
        self._pace_sim = None
        super().__init__(config)

    def _monitor(self, arms):
        if self.interrupt.is_set():
            return 'operator interrupt'
        return super()._monitor(arms)

    def follow_waypoints(self, arm, qs, *, joint_speed=0.6, min_duration=0.25):
        """Execute joint waypoints (N, 7) from the current joints: same joint-limit check, collision sweep,
        contact monitor and operator interrupt as the built-in tools. Min-jerk timing over the whole path."""
        qs = np.vstack([self._q(arm), np.asarray(qs, float).reshape(-1, 7)])
        for q in qs[1:]:
            self._limits(arm, q)
        span = float(np.abs(np.diff(qs, axis=0)).max(axis=1).sum())
        if span < 1e-4:
            return {'arm': arm, 'duration_s': 0.0, 'tcp': self._tcp(arm), 'contacts': self._contact_summary(arm)}
        duration = max(min_duration, 1.875*span/min(joint_speed, self.o['joint_speed_max']))
        steps = max(2, int(math.ceil(duration*self.o['control_hz'])))
        s = np.linspace(0, 1, steps)
        s = 10*s**3 - 15*s**4 + 6*s**5
        idx = s*(len(qs) - 1)
        lo = np.floor(idx).astype(int).clip(0, len(qs) - 1)
        hi = np.minimum(lo + 1, len(qs) - 1)
        path = qs[lo]*(1 - (idx - lo)[:, None]) + qs[hi]*(idx - lo)[:, None]
        reason = self._sweep(arm, path)
        if reason:
            raise ValueError('Refused: ' + reason)
        return self._execute(arm, path)

    def joint_state(self):
        """{joint name: rad} for both arms (what the human-posture IK needs to model the robot)."""
        return {f'{a}_joint_{i + 1}': float(v) for a in ('left', 'right') for i, v in enumerate(self._q(a))}

    def _tick(self):
        super()._tick()
        if self.realtime <= 0:
            return
        now = time.monotonic()
        if self._pace_wall is None or now - self._pace_wall > 0.5 or self.d.time < self._pace_sim:
            self._pace_wall, self._pace_sim = now, float(self.d.time)  # re-anchor after idle time
            return
        ahead = (self.d.time - self._pace_sim)/self.realtime - (now - self._pace_wall)
        if ahead > 0:
            time.sleep(ahead)


_SIM_CLASS = None


def InterruptibleBackend(config, realtime=1.0):
    """The interruptible MuJoCo backend. MuJoCo is imported only here, so the skills and
    agent can run on another backend (e.g. the real robot over ROS 2) without it."""
    global _SIM_CLASS
    if _SIM_CLASS is None:
        from services.daksha_sim.backend import DakshaSimBackend
        _SIM_CLASS = type('InterruptibleBackend', (_InterruptibleSim, DakshaSimBackend), {})
    return _SIM_CLASS(config, realtime=realtime)


# What a planner could do differently when a sub-task keeps failing (shown with the failure).
SUBTASK_HINTS = {
    'locate': 'make sure it is in view (daksha_observe), or look with a wrist camera (daksha_find camera=...), '
              'or ask the operator where it is',
    'open the gripper': 'check the gripper state (daksha_status grippers); if it holds something put it down first',
    'go near': "that approach is out of reach here: try another grasp style (style='top' or 'tilted') or the "
               "other arm, or ask the operator to move the object closer",
    'move in': "the hand could not get onto the object: try style='tilted' or 'top', or re-find the object "
               "(it may have been pushed) and pick again",
    'close the gripper': "nothing ended up between the fingers: the object may have moved or be wider than "
                         "the 8.8 cm opening; re-find it (daksha_find) and pick again, or try style='top' "
                         "/ the other arm",
    'lift': 'it slipped: pick again (maybe style=\'top\' for a better hold)',
    'check what the hand holds': 'nothing is held: pick the object first',
    'choose a free spot': 'ask for a different location (left / right / center / near another object)',
    'go above the spot': 'choose a different location closer to that arm',
    'lower it onto the table': 'choose a different, flatter / free location',
    'move the hand away': 'call daksha_home for that arm',
    'go home': 'call daksha_home again, or STOP and let the operator check the arm',
    'check where it is': 'look again (daksha_find); it may be placed but hidden from the camera',
}


class Skills:
    """Perception + pick/place through the robot tools. Every motion checks the interrupt."""

    def __init__(self, robot, vlm, log=print, on_detections=None, human_ik=None):
        self.r, self.vlm, self.log = robot, vlm, log
        self.on_detections = on_detections or (lambda *a: None)
        self.hik = human_ik  # local_pipeline.human_ik.DualHumanIK, or None for the simulator's own IK
        self.held = {'left': None, 'right': None}  # what each hand holds (best knowledge)
        self.grasp_pose = {}                        # arm -> {'style', 'tcp_above_base'}: how it holds it (for put-down)
        self.last_grasp = None                      # pick feedback (see _grasp)
        self.on_grasp = None
        self.on_step = None                         # (label, status, detail) per sub-task -> task log
        self.on_plan = None                         # ('subs', labels=[...]) / ('sub', label, status, ...) -> plan card
        self.all_labels = []  # every object + container name of the task (set by the agent)
        self.containers = ['bowl']
        self.memory = {}  # last perceived {label: entry}, for the planner's scene summary

    # ------------------------------------------------------------------ tool plumbing
    def check(self):
        if self.r.interrupt.is_set():
            raise Interrupted('interrupted by the operator')

    def tool(self, name, **args):
        self.check()
        self.r.validate(name, args)
        try:
            value, images = self.r.call(name, args)
        except _stop_types() as exc:
            if self.r.interrupt.is_set():
                raise Interrupted('interrupted by the operator (arm holds where it stopped)') from None
            raise StepFailed(str(exc)) from None
        return value, images

    def abortable(self, fn, *args, **kw):
        """Run a (slow) VLM call in a helper thread; an interrupt abandons the wait immediately."""
        box = {}
        def run():
            try:
                box['value'] = fn(*args, **kw)
            except BaseException as exc:
                box['error'] = exc
        t = threading.Thread(target=run, daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.05)
            self.check()
        if 'error' in box:
            raise box['error']
        return box['value']

    def motion(self, fn, *args, **kw):
        """Run a backend motion; map an operator stop / contact stop to Interrupted / StepFailed."""
        self.check()
        try:
            return fn(*args, **kw)
        except _stop_types() as exc:
            if self.r.interrupt.is_set():
                raise Interrupted('interrupted by the operator (arm holds where it stopped)') from None
            raise StepFailed(str(exc)) from None

    def human_move(self, arm, position, approach=None, finger_axis=None, tolerance_deg=None,
                   finger_tolerance_deg=None, relative=False, speed=None, path='linear', joint_speed=None, **_):
        """move_eef through the human-posture IK (local_pipeline/human_ik.py). Raises ValueError if it can't."""
        ik = self.hik[arm]
        state = self.r.joint_state()
        ik.set_state(state)
        q_now = np.array([state[j] for j in ik.joints])
        T = ik.fk(q_now)
        target = np.asarray(position, float) + (T[:3, 3] if relative else 0.0)
        if approach is None:  # keep the current approach direction, like the sim's move_eef does
            approach, tolerance_deg = T[:3, 0], 20.0 if tolerance_deg is None else tolerance_deg
        tol = max(5.0, float(tolerance_deg or 0.0))  # soft axis-align + elbow pull leaves a few degrees
        ftol = finger_tolerance_deg if finger_tolerance_deg is not None else tol
        dist = float(np.linalg.norm(target - T[:3, 3]))
        if path == 'linear' and dist <= 0.25:
            qs = ik.solve_line(q_now, target, approach, finger_axis=finger_axis, step=0.01, tolerance_deg=tol,
                               finger_tolerance_deg=ftol)
            v = float(speed or self.r.o['cartesian_speed_default'])
            return self.motion(self.r.follow_waypoints, arm, qs[1:], joint_speed=self.r.o['joint_speed_max'],
                               min_duration=1.875*dist/v)
        r = ik.solve(target, approach, finger_axis=finger_axis, q_current=q_now, tolerance_deg=tol,
                     finger_tolerance_deg=ftol)
        if not r['ok']:
            raise ValueError(f'human-posture IK: no solution (pos err {r["position_error_m"]*1000:.0f} mm, '
                             f'approach err {r["approach_error_deg"]} deg)')
        return self.motion(self.r.follow_waypoints, arm, [r['q']],
                           joint_speed=float(joint_speed or self.r.o['joint_speed_default']))

    def move(self, **kw):
        if self.hik is not None:
            try:
                return self.human_move(**kw)
            except ValueError as exc:
                self.log(f'human-posture IK fallback to sim IK: {str(exc)[:120]}')
        try:
            return self.tool('daksha_move_eef', **kw)[0]
        except (ValueError, StepFailed) as exc:
            if kw.get('path') == 'joint' or isinstance(exc, Interrupted):
                raise
            return self.tool('daksha_move_eef', **{**kw, 'path': 'joint'})[0]

    def home_arm(self, arm):
        """One arm to its home: the human ready posture with human IK, otherwise the sim HOME."""
        if self.hik is not None:
            try:
                return self.motion(self.r.follow_waypoints, arm, [self.hik[arm].reference], joint_speed=0.6)
            except ValueError as exc:
                self.log(f'human home of the {arm} arm refused ({str(exc)[:90]}); using sim home')
        return self.tool('daksha_home', arm=arm)

    def home(self):
        """Both arms home. After an interrupt an arm may hold where the straight joint path home would hit
        the torso; then go out to the clear pose (or lift) first and home from there."""
        for arm in ('left', 'right'):
            try:
                self.home_arm(arm)
                continue
            except ValueError as exc:
                self.log(f'direct homing of the {arm} arm refused ({str(exc)[:90]}); detouring')
            for detour in ({'position': CLEAR_POSE[arm], 'approach': [0, 0, -1], 'tolerance_deg': 60, 'speed': 0.15},
                           {'position': [0, 0, 0.08], 'relative': True, 'speed': 0.1},
                           {'position': CLEAR_POSE[arm], 'tolerance_deg': 90, 'path': 'joint', 'speed': 0.15}):
                try:
                    self.move(arm=arm, **detour)
                    self.home_arm(arm)
                    break
                except (ValueError, StepFailed):
                    continue
            else:
                raise StepFailed(f'cannot bring the {arm} arm home safely; try "home" again or "reset"')

    def open_grippers(self):
        for arm in ('left', 'right'):
            self.tool('daksha_gripper', arm=arm, command='open')

    @property
    def real(self):
        """True on the real robot (ros_agent's RosRobotBackend)."""
        return hasattr(self.r, 'n')

    def clear_view_real(self):
        self._held_from_gripper()                  # never move a hand that is holding something
        """Real robot: an arm hanging over the table hides objects from the world camera (and a box
        partly hidden behind the gripper is localised wrongly, so a retry went to the same wrong
        spot). Home every arm that is over the table and not holding anything."""
        tz = self.r.n.table_z if self.r.n.table_z is not None else 0.9
        for arm in ('left', 'right'):
            if self.held.get(arm) or (hasattr(self.r, 'at_home') and self.r.at_home(arm)):
                continue                    # holding something, or already parked at home
            try:
                x, y, z = self.r._tcp(arm)['position']
            except Exception:
                continue
            if x > 0.20 and z < tz + 0.45:
                self.log(f'{arm} arm is over the table: moving it out of the world camera view first')
                try:
                    self.home_arm(arm)
                except (ValueError, StepFailed, DryRun) as exc:   # dry run: look from where the arms are
                    self.log(f'could not clear the {arm} arm ({str(exc)[:100]}); the view may be blocked')

    def clear_view(self):
        """Arms out of the head camera's view. Always the simulator's IK: no human-like pose clears the view
        (Daksha's reach does not include hands-at-chest poses, and a hanging arm would cross the table edge),
        so this one brief gesture uses the sideways pose; every working motion uses human posture."""
        for arm in ('left', 'right'):
            kw = dict(arm=arm, position=CLEAR_POSE[arm], approach=[0, 0, -1], tolerance_deg=60, speed=0.2)
            try:
                try:
                    self.tool('daksha_move_eef', **kw)
                except (ValueError, StepFailed):
                    self.tool('daksha_move_eef', **{**kw, 'path': 'joint'})
            except (ValueError, StepFailed):
                pass  # an arm in view makes perception worse, not impossible

    # ------------------------------------------------------------------ perception
    def observe(self):
        meta, images = self.tool('daksha_observe', cameras=['world'])
        import base64
        png = base64.b64decode(images[0]['data'])
        return meta['frame_id'], np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))

    def _backproject(self, frame_id, pixels):
        if not pixels:
            return np.zeros((0, 3))
        reply, _ = self.tool('daksha_pixel_to_position', frame_id=frame_id, camera='world', pixels=pixels, window=1)
        return np.array([p['position'] for p in reply['points'] if p['valid']]).reshape(-1, 3)

    def _localize(self, frame_id, label, box, point):
        """3-D centre (and banana yaw) from a VLM box: depth samples above the table, near the VLM point.
        (Without depth the samples are the camera rays hitting a plane just above the table: within ~1 cm
        for objects up to ~8 cm tall in a synthetic check; the wrist camera then corrects the grasp.)"""
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1)*0.12, (y2 - y1)*0.12
        us = np.linspace(x1 + mx, x2 - mx, 11)
        vs = np.linspace(y1 + my, y2 - my, 11)
        px = [[float(u), float(v)] for v in vs for u in us]
        pts = self._backproject(frame_id, px)
        pts = pts[(pts[:, 2] > TABLE_Z + 0.003) & (pts[:, 2] < TABLE_Z + 0.25)] if len(pts) else pts
        if len(pts) < 6:
            return None
        container = label in CONTAINERS
        if not container:
            anchor = self._backproject(frame_id, [list(map(float, point))])
            if len(anchor):
                radius = 0.12 if label == 'banana' else 0.06
                near = pts[np.linalg.norm(pts[:, :2] - anchor[0, :2], axis=1) < radius]
                if len(near) >= 6:
                    pts = near
        if container:
            ring = pts[pts[:, 2] > TABLE_Z + 0.02]
            if len(ring) < 6:
                return None
            xy = (ring[:, :2].min(0) + ring[:, :2].max(0))/2
        else:
            xy = np.median(pts[:, :2], axis=0)
        entry = {'centre': [float(xy[0]), float(xy[1]), float(np.max(pts[:, 2]))], 'n': int(len(pts))}
        if label == 'banana':
            c = pts[:, :2] - pts[:, :2].mean(0)
            _, v = np.linalg.eigh(c.T @ c)
            entry['yaw'] = float(math.atan2(v[1, -1], v[0, -1]))
            entry['centre'][:2] = pts[:, :2].mean(0).tolist()
        return entry

    def perceive(self, labels, clear='auto'):
        """{label: {'centre', 'yaw'?, 'bbox', 'source'}} from a fresh head-camera frame.

        clear='auto' (default): look from where the arms are; only if a needed object is not found, move the
        arms out of the camera's view and look again. True: always clear first. False: never move."""
        if self.real and clear is not False:
            self.clear_view_real()              # cheap when the arms are already out of the way
        elif clear is True:
            self.clear_view()
        found = self._perceive_once(labels)
        missing = [l for l in labels if l not in found]
        if clear == 'auto' and missing and not self.real:
            self.log(f'{", ".join(missing)} not visible from here; moving the arms out of the camera view')
            self.clear_view()
            found.update(self._perceive_once(labels))
        return found

    def _drop_robot_parts(self, dets):
        """Real robot: drop world-camera detections that are the robot's own hand (e.g. "black cube" on the
        black wrist camera housing). Hands holding something are not checked."""
        part = getattr(getattr(self.r, 'n', None), 'robot_part', None) if self.real else None
        if part is None:
            return dets
        skip = [a for a, o in self.held.items() if o]
        keep = []
        for d in dets:
            arm = part(d['bbox'], skip=skip)
            if arm:
                self.log(f'ignored "{d["label"]}" at {[round(v) for v in d["bbox"]]}: that is the {arm} '
                         'gripper / wrist camera in the image, not an object')
            else:
                keep.append(d)
        return keep

    def perceive_all(self):
        """Open-vocabulary 2-D scene scan; localization is added when table geometry is available."""
        frame_id, rgb = self.observe()
        try:
            t = time.time()
            dets = self.abortable(self.vlm.detect, rgb, None)
            self.log(f'{self.vlm.model} found {[d["label"] for d in dets]} in the live camera ({time.time() - t:.1f}s)')
            dets = self._drop_robot_parts(dets)
        except Interrupted:
            raise
        except Exception as exc:
            self.log(f'VLM gave no scene scan ({exc})')
            dets = []
        self.check()
        found = {}
        real_table_z = getattr(getattr(self.r, 'n', None), 'table_z', None)
        for d in dets:
            label = d['label']
            if label in found:
                continue
            e = {'bbox': d['bbox'], 'point': d['point'], 'source': 'vlm'}
            # A camera-only scan should still render useful detections before the operator
            # has measured the table. Robot actions still require 3-D localization.
            if not self.real or real_table_z is not None:
                try:
                    e.update(self._localize(frame_id, label, d['bbox'], d['point']) or {})
                except Exception as exc:
                    self.log(f'{label}: 2-D detection only ({exc})')
            found[label] = e
        self.on_detections(rgb, dets, found, frame_id)
        self.memory.update({k: v for k, v in found.items() if 'centre' in v})
        return found

    def perceive_camera(self, labels, camera):
        """Detect objects in ONE camera ('world', 'right_wrist', 'left_wrist') and localise them from
        that frame's aligned depth (the wrist cameras have it; without depth: the table plane).
        Returns ({label: entry}, annotated RGB). Wrist detections are not drawn on the operator's
        world view (their boxes are in wrist-image pixels)."""
        if camera == 'world':
            found = self.perceive(labels)
            return found, None
        meta, images = self.tool('daksha_observe', cameras=[camera])
        import base64
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[0]['data']))).convert('RGB'))
        fid = meta['frames'][camera]
        dets = self.abortable(self.vlm.detect, rgb, list(labels))
        self.log(f'{self.vlm.model} saw {[d["label"] for d in dets]} in the {camera} camera'
                 f'{" (with depth)" if meta.get("depth") else ""}')
        found = {}
        for d in dets:
            if d['label'] in found:
                continue
            e = self._localize(fid, d['label'], d['bbox'], d['point'])
            if e:
                found[d['label']] = {**e, 'bbox': d['bbox'], 'point': d['point'], 'source': f'vlm+{camera}',
                                     'depth': bool(meta.get('depth'))}
        from PIL import ImageDraw
        im = Image.fromarray(rgb.astype(np.uint8)).convert('RGB')
        dr = ImageDraw.Draw(im)
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            dr.rectangle([x1, y1, x2, y2], outline=(60, 220, 60), width=3)
            e = found.get(d['label'])
            tag = d['label'] + (' ({:.3f}, {:.3f}, {:.3f})'.format(*e['centre']) if e else ' (no 3-D)')
            dr.rectangle([x1, y1 - 14, x1 + dr.textlength(tag) + 6, y1], fill=(0, 0, 0))
            dr.text((x1 + 3, y1 - 13), tag, fill=(60, 220, 60))
        self.memory.update({k: v for k, v in found.items() if 'centre' in v})
        return found, np.asarray(im)

    def _perceive_once(self, labels):
        frame_id, rgb = self.observe()
        found, dets = {}, []
        try:
            t = time.time()
            dets = self.abortable(self.vlm.detect, rgb, list(labels))
            self.log(f'{self.vlm.model} saw {[d["label"] for d in dets]} ({time.time() - t:.1f}s)')
            dets = self._drop_robot_parts(dets)
        except Interrupted:
            raise
        except Exception as exc:  # VLMError or transport: no answer, not "absent"
            self.log(f'VLM gave no answer ({exc}); using colour fallback')
        self.check()
        for d in dets:
            if d['label'] in found:
                continue
            e = self._localize(frame_id, d['label'], d['bbox'], d['point'])
            if e:
                found[d['label']] = {**e, 'bbox': d['bbox'], 'point': d['point'], 'source': 'vlm'}
        for label in labels:
            if label not in found:
                e = self._colour_fallback(frame_id, rgb, label)
                if e:
                    found[label] = {**e, 'source': 'colour-fallback'}
        self.on_detections(rgb, dets, found, frame_id)
        self.memory.update({k: v for k, v in found.items() if 'centre' in v})
        return found

    def _colour_fallback(self, frame_id, img, label):
        img = img.astype(int)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        mx, mn = img.max(-1), img.min(-1)
        sat = (mx - mn)/np.maximum(mx, 1)
        masks = {'banana': (r > 170) & (g > 150) & (b < 90) & (abs(r - g) < 70),
                 'orange': (r > 170) & (g > 70) & (g < 160) & (b < 60) & (r - g > 60),
                 'apple': (r > 120) & (g < 60) & (b < 60),
                 'bowl': (mn > 175) & (sat < 0.12)}
        if label not in masks:
            return None
        vs, us = np.nonzero(masks[label])
        if len(us) < 15:
            return None
        ys, xs = vs.min(), us.min()
        ye, xe = vs.max(), us.max()
        e = self._localize(frame_id, label, [xs, ys, xe, ye], [float(np.median(us)), float(np.median(vs))])
        if e:
            e['bbox'] = [float(xs), float(ys), float(xe), float(ye)]
        return e

    # ------------------------------------------------------------------ manipulation
    # ------------------------------------------------------------------ sub-tasks
    def subtask(self, label, fn, retries=None, recover=None):
        """Run one named step of a skill; on failure run `recover(exc)` and try again, up to `retries`
        extra times (default: subtask_retries in the robot config, else 2). STOP / dry run are never
        retried. Each step's outcome goes to the task log (on_step)."""
        retries = int(self.r.c.get('subtask_retries', 2) if self.real else 2) if retries is None else retries
        last = None
        for attempt in range(1, retries + 2):
            self.check()
            self.log(f'▶ {label}' + (f'  (retry {attempt - 1}/{retries})' if attempt > 1 else ''))
            self._plan('sub', label=label, status='running' if attempt == 1 else 'retry', attempt=attempt)
            try:
                out = fn()
                self._step(label, 'done', f'{attempt} tries' if attempt > 1 else '')
                self._plan('sub', label=label, status='done', attempt=attempt)
                return out
            except (Interrupted, DryRun, MotionStopped) as exc:
                self._plan('sub', label=label, status='stopped',
                           detail='dry run (execute is off)' if isinstance(exc, DryRun) else 'stopped', attempt=attempt)
                raise
            except (StepFailed, ValueError) as exc:
                last = exc
                self.log(f'✗ {label}: {str(exc)[:160]}')
                if attempt <= retries and recover is not None:
                    try:
                        recover(exc)
                    except (Interrupted, DryRun, MotionStopped):
                        raise
                    except (StepFailed, ValueError) as rexc:
                        self.log(f'  recovery for "{label}" failed: {str(rexc)[:120]}')
        self._step(label, 'failed', str(last)[:200])
        self._plan('sub', label=label, status='failed', detail=str(last), attempt=retries + 1)
        hint = next((v for k, v in SUBTASK_HINTS.items() if label.startswith(k)), '')
        raise StepFailed(f'sub-task "{label}" failed after {retries + 1} tries: {last}.'
                         + (f' What could work instead: {hint}' if hint else ''))

    def _plan(self, kind, **kw):
        cb = getattr(self, 'on_plan', None)
        if cb is not None:
            try:
                cb(kind, **kw)
            except Exception:
                pass

    def _step(self, label, status, detail=''):
        cb = getattr(self, 'on_step', None)
        if cb is not None:
            try:
                cb(label, status, detail)
            except Exception:
                pass

    def _gripper_checked(self, arm, command):
        """Open / close and confirm from the finger width; raises StepFailed when not confirmed."""
        g, _ = self.tool('daksha_gripper', arm=arm, command=command)
        if command == 'open' and g['width_m'] < 0.06:
            raise StepFailed(f'gripper did not open (width {g["width_m"] * 100:.1f} cm)')
        return g

    # ------------------------------------------------------------------ pick (sub-tasks)
    def pick(self, name, obj, arm=None, style=None, locate=None):
        """Pick as sub-tasks, each retried on its own:
            locate -> open the gripper -> go near -> move in (wrist camera) -> close the gripper
            -> lift -> check the grasp
        style: 'side' (default on the real robot), 'top', 'tilted'. locate(): re-find the object
        (e.g. from the live tracker) for a retry of "go near"."""
        arm = arm or ('left' if obj['centre'][1] > 0 else 'right')
        if self.real:
            self._held_from_gripper()
            if self.held.get(arm):
                raise StepFailed(f'the {arm} hand is already holding the {self.held[arm]} (gripper part-closed): '
                                 'put it down first (opening it to pick would drop it)')
        style = style or (self.r.c.get('grasp_style') if self.real else None) or 'tilted'
        cfg = self.r.c if self.real else {}
        ctx = {'obj': obj}
        gripper_cam = self.real and style == 'side' and cfg.get('wrist_approach', True) and hasattr(self.r, 'wrist_measure')
        self._plan('subs', labels=[f'see the {name} with the world camera', 'open the gripper', f'go near the {name}']
                   + ([f'see the {name} with the gripper camera'] if gripper_cam else [])
                   + [f'move in to the {name}', 'close the gripper', f'lift the {name}', 'check the grasp'])

        def where():
            o = ctx['obj']
            x, y = o['centre'][:2]
            if self.real and hasattr(self.r, 'remap_to_arm'):
                x, y, _ = self.r.remap_to_arm(o['centre'], arm)
            top = float(o['centre'][2]) if len(o['centre']) > 2 else TABLE_Z + 0.06
            return x, y, top

        def relocate(_exc=None):
            if locate is not None:
                o = locate()
                if o:
                    ctx['obj'] = o

        # 1. locate
        x, y, top = self.subtask(f'see the {name} with the world camera', where)
        self.log(f'pick {name} with the {arm} arm at ({x:.3f}, {y:.3f}), {style} grasp')
        # 2. open the gripper
        self.subtask('open the gripper', lambda: self._gripper_checked(arm, 'open'))

        # 3. go near: the pre-grasp pose (side: tries the reachable approach angles)
        def go_near():
            x, y, top = where()
            if style == 'side':
                cands = cfg.get('side_grasp_candidates') or [[15, 20], [30, 20], [15, 40], [30, 40], [45, 20], [0, 20]]
                # start above anything the object may stand on; the gripper camera then steers the hand
                # down (or up) onto the object itself during "move in"
                height = max(TABLE_Z + float(cfg.get('side_grasp_start_height_m', 0.10)),
                             min(TABLE_Z + (top - TABLE_Z) * 0.5, TABLE_Z + 0.08))
                reach = float(cfg.get('side_grasp_reach_m', 0.02))
                standoff = float(cfg.get('side_grasp_standoff_m', 0.10))
                inward = 1.0 if arm == 'right' else -1.0
                last = None
                k = ctx.get('cand_shift', 0) % len(cands)
                for yaw_deg, pitch_deg in cands[k:] + cands[:k]:
                    yw, pt = math.radians(inward * yaw_deg), math.radians(pitch_deg)
                    a = np.array([math.cos(yw) * math.cos(pt), math.sin(yw) * math.cos(pt), -math.sin(pt)])
                    fa = [-math.sin(yw), math.cos(yw), 0.0]
                    g = np.array([x, y, height]) + a * reach
                    pre = g - a * standoff
                    orient = {'approach': a.tolist(), 'finger_axis': fa, 'tolerance_deg': 10, 'finger_tolerance_deg': 20}
                    try:
                        self.move(arm=arm, position=pre.tolist(), **orient)
                    except (ValueError, StepFailed) as exc:
                        last = exc
                        self.log(f'  approach {yaw_deg} deg inward / {pitch_deg} deg down not reachable')
                        continue
                    self.log(f'  near the {name}: {yaw_deg} deg inward, {pitch_deg} deg down, '
                             f'TCP {100 * (height - TABLE_Z):.0f} cm above the table')
                    self.grasp_pose[arm] = {'style': 'side', 'tcp_above_base': height - TABLE_Z}
                    return {'a': a, 'g': g, 'pre': pre, 'orient': orient, 'reach': reach}
                raise StepFailed(f'no reachable side approach ({last})')
            orient = {'approach': [0.0, 0.0, -1.0], 'tolerance_deg': 30} if style == 'top' else \
                {'approach': list(GRASP_APPROACH), 'tolerance_deg': GRASP_TOL_DEG}
            if name == 'banana' and 'yaw' in ctx['obj']:
                yaw = ctx['obj']['yaw']
                orient.update(finger_axis=[-math.sin(yaw), math.cos(yaw), 0.0], finger_tolerance_deg=25)
            pre = np.array([x, y, GRASP_TCP_Z + PREGRASP_DZ])
            self.move(arm=arm, position=pre.tolist(), **orient)
            self.grasp_pose[arm] = {'style': style}
            return {'a': np.array([0.0, 0.0, -1.0]), 'g': np.array([x, y, GRASP_TCP_Z]), 'pre': pre,
                    'orient': orient, 'reach': 0.0}
        near = self.subtask(f'go near the {name}', go_near, recover=relocate)

        # 3b. the gripper camera must see it too before moving in (one detection, then its live tracker)
        if gripper_cam:
            def gripper_sees():
                st = self.r.wrist_track_start(arm, name)      # raises if not found in the gripper camera
                self.log(f'  gripper camera sees the {name} (box {st.get("box")})')
                return st

            def look_again(_exc=None):                        # world camera again, then another approach angle
                relocate()
                ctx['cand_shift'] = ctx.get('cand_shift', 0) + 1
                near.update(go_near())
            self.subtask(f'see the {name} with the gripper camera', gripper_sees, recover=look_again)

        # 4. move in (side: wrist-camera feedback until the object is between the fingers)
        def move_in():
            if style == 'side' and self.real and cfg.get('wrist_approach', True) and hasattr(self.r, 'wrist_measure'):
                self.wrist_approach(name, arm, near['a'], near['g'], near['orient'], near['reach'])
            elif style != 'side' and self.real and cfg.get('wrist_refine', False):
                x2, y2 = self.wrist_refine(name, arm, near['g'][0], near['g'][1], near['orient'], top_z=where()[2])
                self.move(arm=arm, position=[x2, y2, GRASP_TCP_Z], speed=0.05, **near['orient'])
            else:
                self.move(arm=arm, position=near['g'].tolist(), speed=0.05, **near['orient'])

        def back_off(_exc=None):
            self.move(arm=arm, position=near['pre'].tolist(), speed=0.05, **near['orient'])
        self.subtask(f'move in to the {name}', move_in, recover=back_off)

        # 5. close the gripper: it must stop on something
        def close():
            g, _ = self.tool('daksha_gripper', arm=arm, command='close', force=25)
            ctx['g'] = g
            if not g['stalled_before_target'] or g['width_m'] < 0.015:
                self._grasp(name, arm, False, 'fingers closed fully, nothing between them'
                            if not g['stalled_before_target'] else 'fingers stopped too narrow (< 1.5 cm)', g)
                raise StepFailed(f'empty grasp (width {g["width_m"] * 100:.1f} cm)')
            return g

        def regrasp(_exc=None):                         # open, back off, come in again, then close again
            self._gripper_checked(arm, 'open')
            back_off()
            move_in()
        g = self.subtask('close the gripper', close, recover=regrasp)
        try:
            tcp = self.r._tcp(arm)['position'] if self.real else None
        except Exception:
            tcp = None
        if tcp is not None:
            gp = self.grasp_pose.setdefault(arm, {'style': style})
            gp['pick_tcp'] = [float(v) for v in tcp]          # where the hand held it: "put it back" goes here
            if gp.get('style') == 'side':
                gp['tcp_above_base'] = float(tcp[2]) - TABLE_Z

        # 6. lift
        def lift():
            self.move(arm=arm, position=[0, 0, PREGRASP_DZ], relative=True, speed=0.05)
            g2, _ = self.tool('daksha_gripper', arm=arm, command='close', force=25)
            if g2['width_m'] < 0.015:
                self._grasp(name, arm, False, 'slipped out while lifting', g2)
                raise StepFailed('slipped out while lifting')
            return g2
        g = self.subtask(f'lift the {name}', lift, retries=0)

        # 7. check the grasp (gripper width + wrist camera)
        def check():
            why = f'gripping ({g["width_m"] * 100:.1f} cm), lifted {PREGRASP_DZ * 100:.0f} cm'
            if self.real:
                seen = self.wrist_sees_held(name, arm)
                why += ('; wrist camera sees it between the fingers' if seen is True else
                        '; WARNING: the wrist camera does not see it between the fingers' if seen is False else '')
            return why
        why = self.subtask('check the grasp', check, retries=0)
        self.held[arm] = name
        self._grasp(name, arm, True, why, g)
        return arm

    # ------------------------------------------------------------------ wrist cameras (real robot)
    def wrist_approach(self, name, arm, a, g_world, orient, reach, max_steps=12, step_max=0.03):
        """From the pre-grasp pose, move toward the object under WRIST-CAMERA feedback until the
        fingertips are `reach` past the object centre: each step measures object and fingertips in
        the wrist camera (depth) and moves by their difference (sideways fully, forward <= step_max).
        Falls back to the straight world-camera approach when the wrist camera cannot see it."""
        a = np.asarray(a, float)
        # Steer in the TABLE PLANE only, at the grasp height: the vertical part of the wrist correction
        # (and of a tipped-down approach) would drive the fingertips into the table (seen 2026-11-15:
        # TCP at table height, the hand stalled 3 cm short of the box).
        z_hold = float(g_world[2])
        a = np.array([a[0], a[1], 0.0])
        a /= max(np.linalg.norm(a), 1e-6)
        try:
            self.r.wrist_track_start(arm, name)        # reuses the gripper camera's live box if it has one
        except (ValueError, RuntimeError) as exc:
            self.log(f'wrist approach: {exc}; straight approach from the world-camera estimate')
            self.move(arm=arm, position=list(g_world), speed=0.05, **orient)
            return
        try:
            m = self.r.wrist_measure(arm)
            if not m['ok']:
                self.log(f'wrist approach: {m["why"]}; straight approach from the world-camera estimate')
                self.move(arm=arm, position=list(g_world), speed=0.05, **orient)
                return
            F = m['fingers_cam']                 # fingertips: measured once, hand open, object still far
            self.log(f'wrist approach: fingertips at {F[2] * 100:.0f} cm in the wrist camera, object at '
                     f'{m["object_depth_m"] * 100:.0f} cm')
            travelled = 0.0
            for step in range(max_steps):
                self.check()
                m = self.r.wrist_measure(arm, fingers_cam=F)
                if not m['ok']:
                    self.log(f'wrist approach: {m["why"]} at step {step}; stopping here')
                    break
                e = np.array(m['err_world'], float)
                dz = float(np.clip(e[2], -0.02, 0.02))      # vertical: toward the object centre, 2 cm a step
                e[2] = 0.0
                along = float(e @ a)
                lat = e - a * along
                todo = along + reach                       # want the fingertips `reach` past the centre
                self.log(f'wrist step {step}: object {along * 100:+.1f} cm ahead of the fingertips, '
                         f'{np.linalg.norm(lat) * 100:.1f} cm sideways')
                if np.linalg.norm(lat) < 0.008 and todo < 0.01 and abs(dz) < 0.01:
                    break
                if np.linalg.norm(lat) > step_max:
                    lat = lat / np.linalg.norm(lat) * step_max
                d = lat + a * float(np.clip(todo, -0.02, step_max))
                travelled += float(np.linalg.norm(d))
                if travelled > 0.25:
                    self.log('wrist approach: moved 25 cm without reaching it; stopping')
                    break
                p = np.array(self.r._tcp(arm)['position'], float) + d
                z_hold = float(np.clip(z_hold + dz, TABLE_Z + 0.04, TABLE_Z + 0.30))   # fingers stay off the table
                p[2] = z_hold
                self.move(arm=arm, position=p.tolist(), speed=0.05, **orient)
                time.sleep(0.3)                          # tracker frames after the move
        finally:
            self.r.wrist_track_stop()

    def _wrist_frame(self, arm):
        meta, images = self.tool('daksha_observe', cameras=[f'{arm}_wrist'])
        import base64
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[0]['data']))).convert('RGB'))
        return meta['frames'][f'{arm}_wrist'], rgb

    def wrist_refine(self, name, arm, x, y, orient, top_z=None, rounds=2, max_shift=0.08, done=0.008):
        """Above the object, look with this arm's wrist camera (close range, nearly top-down) and move
        the hand over where IT sees the object. Up to `rounds` corrections; a correction larger than
        `max_shift` is distrusted (wrong object / bad view) and the world estimate is kept.
        From above the camera sees the object's TOP face, so the box centre is projected onto the
        object's top height (from the world camera's footprint estimate), not onto the table."""
        top_z = float(top_z) if top_z is not None else GRASP_TCP_Z + 0.03
        for k in range(rounds):
            try:
                fid, rgb = self._wrist_frame(arm)
                dets = self.abortable(self.vlm.detect, rgb, [name])
            except Interrupted:
                raise
            except Exception as exc:
                self.log(f'wrist camera: no look ({str(exc)[:100]}); keeping the world estimate')
                return x, y
            if not dets:
                self.log(f'wrist camera: {name} not visible from above; keeping the world estimate')
                return x, y
            b = dets[0]['bbox']
            uv = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
            r, _ = self.tool('daksha_pixel_to_position', frame_id=fid, camera='world', pixels=[uv], z=top_z)
            if not r['points'] or not r['points'][0]['valid']:
                self.log('wrist camera: the view ray misses the object height; keeping the world estimate')
                return x, y
            wx, wy = r['points'][0]['position'][:2]
            shift = math.hypot(wx - x, wy - y)
            self.log(f'wrist camera ({arm}) sees {name} at ({wx:.3f}, {wy:.3f}): '
                     f'{shift * 100:.1f} cm from the current estimate ({x:.3f}, {y:.3f})')
            if shift > max_shift:
                self.log(f'wrist correction {shift * 100:.0f} cm is too large to trust; keeping ({x:.3f}, {y:.3f})')
                return x, y
            x, y = wx, wy
            if shift < done:
                break
            self.move(arm=arm, position=[x, y, GRASP_TCP_Z + PREGRASP_DZ], speed=0.05, **orient)
        return x, y

    def gripper_cam_yes(self, arm, question):
        """A yes/no question about this arm's gripper (wrist) camera image; None when unsure."""
        try:
            _, rgb = self._wrist_frame(arm)
            r = self.abortable(self.vlm.ask_json, rgb, 'This image is from a camera on a robot gripper. ' + question +
                               ' Reply JSON only: {"answer": true or false, "reason": "<few words>"}', max_tokens=120)
            ans = r.get('answer')
            self.log(f'  gripper camera ({arm}): {question} -> {ans} ({r.get("reason", "")})')
            return ans if isinstance(ans, bool) else None
        except Interrupted:
            raise
        except Exception as exc:
            self.log(f'  gripper camera ({arm}): no answer ({str(exc)[:80]})')
            return None

    def wrist_sees_held(self, name, arm):
        """True / False: does the wrist camera see the object between the fingers; None: no answer."""
        try:
            _, rgb = self._wrist_frame(arm)
            r = self.abortable(self.vlm.ask_json, rgb,
                               f'This image is from a camera on a robot gripper; the two white fingers are at the '
                               f'bottom. Is a {name.replace("_", " ")} held between the fingers? '
                               'Reply JSON only: {"held": true or false, "reason": "<few words>"}', max_tokens=120)
            held = r.get('held')
            self.log(f'wrist camera grasp check ({arm}): held={held} ({r.get("reason", "")})')
            return held if isinstance(held, bool) else None
        except Interrupted:
            raise
        except Exception as exc:
            self.log(f'wrist camera grasp check: no answer ({str(exc)[:80]})')
            return None

    def verify_placed(self, obj, x, y, tol=0.10):
        """World camera after a put-down (arms out of view): is the object there?"""
        found = self.perceive([obj])
        e = found.get(obj)
        if not e or 'centre' not in e:
            self.log(f'place check: the world camera does not see the {obj} after placing')
            return None
        d = math.hypot(e['centre'][0] - x, e['centre'][1] - y)
        self.log(f'place check: {obj} seen {d * 100:.1f} cm from the target ({x:.3f}, {y:.3f})')
        if d > tol:
            raise StepFailed(f'{obj} put down, but the world camera sees it {d * 100:.0f} cm from the target')
        return d

    def _grasp(self, name, arm, picked, why, g):
        """Pick feedback: what the gripper measured, for the UI / chat (on_grasp)."""
        import time as _t
        self.last_grasp = {'object': name, 'arm': arm, 'picked': bool(picked), 'why': why,
                           'width_m': round(float(g.get('width_m', 0.0)), 4),
                           'stalled': bool(g.get('stalled_before_target')), 'time': _t.time()}
        self.log(f'pick feedback: {name} {"PICKED" if picked else "NOT picked"} ({arm}): {why}, '
                 f'width {self.last_grasp["width_m"] * 100:.1f} cm')
        cb = getattr(self, 'on_grasp', None)
        if cb is not None:
            try:
                cb(dict(self.last_grasp))
            except Exception:
                pass

    def _held_from_gripper(self, obj=None):
        """Real robot, nothing recorded as held (e.g. after a restart): a gripper stopped part-closed
        (1.2-8 cm) is holding something; assume the configured grasp style (side: TCP 5 cm above the
        object's base)."""
        fb = getattr(self.r.n, 'gripper_feedback', None)
        if fb is None:
            return
        for a, g in fb({}).items():
            if self.held.get(a):
                continue
            w = g.get('width_m')
            if w is not None and 0.012 < w < 0.08:
                self.held[a] = obj or 'object'
                if self.r.c.get('grasp_style', 'side') == 'side':
                    # where it was picked is NOT known after a restart (the arm may have moved since), so
                    # "put it back" is refused rather than guessed
                    self.grasp_pose.setdefault(a, {'style': 'side', 'tcp_above_base': 0.05})
                self.log(f'the {a} gripper is part-closed ({w * 100:.1f} cm): taking it as holding the {self.held[a]}')

    def place(self, arm, target_name, target):
        bx, by, _ = target['centre']
        side = 1 if arm == 'left' else -1
        self.log(f'place into {target_name} at ({bx:.3f}, {by:.3f})')
        self.move(arm=arm, position=[bx, by + side*0.025, RELEASE_TCP_Z], approach=list(GRASP_APPROACH),
                  tolerance_deg=50, path='joint')
        self.tool('daksha_gripper', arm=arm, command='open')
        self.held[arm] = None
        self.move(arm=arm, position=[0, 0, 0.06], relative=True, speed=0.1)
        self.home_arm(arm)

    def pick_place(self, obj, target='bowl', attempts=2):
        for attempt in range(1, attempts + 1):
            self.check()
            scene = self.perceive([obj, target])
            if target not in scene:
                raise StepFailed(f'cannot find the {target}')
            if obj not in scene:
                raise StepFailed(f'cannot find the {obj}')
            if inside(scene[obj], scene[target]):
                self.log(f'{obj} is already in the {target}')
                return True
            try:
                arm = self.pick(obj, scene[obj])
                self.place(arm, target, scene[target])
                return True
            except StepFailed as exc:
                self.log(f'attempt {attempt} failed: {exc}')
                self.recover()
        return False

    # ------------------------------------------------------------------ more skills for free-form commands
    def holding_arm(self, arm=None, obj=None):
        if self.real and not any(self.held.values()):
            self._held_from_gripper(obj)
        if arm and self.held.get(arm):
            return arm
        for a, o in self.held.items():
            if o and (obj is None or o == obj):
                return a
        return None

    def pick_up(self, obj, arm=None, attempts=2, style=None):
        """Pick an object and keep holding it, lifted."""
        for attempt in range(1, attempts + 1):
            scene = self.perceive([obj])
            if obj not in scene:
                raise StepFailed(f'cannot find the {obj}')
            try:
                return self.pick(obj, scene[obj], arm=arm, style=style)
            except StepFailed as exc:
                self.log(f'pick attempt {attempt} failed: {exc}')
                self.recover()
        raise StepFailed(f'could not pick up the {obj}')

    def put_in(self, container, arm=None):
        arm = self.holding_arm(arm)
        if arm is None:
            raise StepFailed('I am not holding anything to put down')
        scene = self.perceive([container], clear=False)
        if container not in scene:
            raise StepFailed(f'cannot see the {container}')
        self.place(arm, container, scene[container])

    def location_xy(self, spec, exclude=None):
        """A table spot from a planner location: {"region": "left"} | {"near": "orange", "side": "left", "distance": 0.1}
        | {"xy": [x, y]}. Clamped to where a hand can put things down."""
        spec = spec or {}
        if isinstance(spec, str):
            spec = {'region': spec}
        if 'xy' in spec:
            x, y = (float(v) for v in spec['xy'][:2])
        elif spec.get('near'):
            ref = spec['near']
            if ref not in self.memory or ref == exclude:
                scene = self.perceive([ref], clear=False)
                if ref not in scene:
                    raise StepFailed(f'cannot see the {ref}')
            cx, cy = self.memory[ref]['centre'][:2]
            d = float(spec.get('distance', 0.11))
            dx, dy = {'left': (0, d), 'right': (0, -d), 'front': (d, 0), 'far': (d, 0), 'behind': (-d, 0),
                      'back': (-d, 0), 'near': (-d, 0)}.get(str(spec.get('side', '')).lower(),
                                                       (0, d if cy < 0.05 else -d))  # default: toward the middle
            x, y = cx + dx, cy + dy
        else:
            region = str(spec.get('region', 'center')).lower().replace(' ', '-').replace('_', '-')
            region = region.replace('-side', '').replace('of-the-table', '').strip('-') or 'center'
            if region not in REGIONS:
                raise StepFailed(f'unknown table location {region!r}; try left, right, front, back, center')
            x, y = REGIONS[region]
        return x, y

    def free_spots(self, x, y, arm, exclude=None, n=4):
        """Up to n reachable put-down spots nearest (x, y) that keep clear of containers and other objects."""
        side = 1.0 if arm == 'left' else -1.0
        others = [(name, e['centre'][:2]) for name, e in self.memory.items() if name != exclude]
        spots = []
        for r in np.arange(0.0, 0.25, 0.025):
            for ang in np.linspace(0, 2*math.pi, 1 if r == 0 else 16, endpoint=False):
                px, py = x + r*math.cos(ang), y + r*math.sin(ang)
                if not (REACH['x'][0] <= px <= REACH['x'][1] and abs(py) <= REACH['y']):
                    continue
                if side*py < -0.06:  # an arm does not reach far across the robot's middle
                    continue
                mine = 0.13 if exclude == 'banana' else 0.0
                clear = all(math.hypot(px - ox, py - oy) >= (BOWL_CLEAR + mine/2 if name in self.containers else
                                                             max(OBJECT_CLEAR, mine, 0.13 if name == 'banana' else 0))
                            for name, (ox, oy) in others)
                if clear and all(math.hypot(px - sx, py - sy) > 0.04 for sx, sy in spots):
                    spots.append((float(px), float(py)))
                if len(spots) >= n:
                    return spots
        return spots

    def place_at(self, location, arm=None):
        """Put the held object down as sub-tasks, each retried on its own:
            check what the hand holds -> choose a free spot -> go above the spot -> lower it onto the
            table -> open the gripper -> move the hand away -> go home -> check where it is"""
        gcam = self.real and hasattr(self.r.n, 'wrist_snapshot')
        self._plan('subs', labels=['check what the hand holds', 'choose a free spot', 'go above the spot',
                                   'lower it onto the table', 'open the gripper', 'move the hand away']
                   + (['gripper camera: is it standing there?'] if gcam else []) + ['go home', 'check where it is'])

        def holding():
            held = self.holding_arm(arm)
            if held is None:
                raise StepFailed('I am not holding anything to put down')
            return held
        arm = self.subtask('check what the hand holds', holding, retries=0)
        obj = self.held[arm]

        back = (isinstance(location, dict) and location.get('back')) or \
            (isinstance(location, str) and location.strip().lower() in ('back', 'where it was', 'original place'))
        gp0 = self.grasp_pose.get(arm) or {}

        def choose():
            if back:
                if not gp0.get('pick_tcp'):
                    raise StepFailed('I do not know where it was picked from (no recorded grasp)')
                return [tuple(gp0['pick_tcp'][:2])]       # the hand goes back exactly where it held it
            others = [l for l in self.all_labels if l != obj]
            if others:
                self.perceive(others, clear=False)
            want = self.location_xy(location, exclude=obj)
            spots = self.free_spots(*want, arm, exclude=obj)
            if not spots:
                raise StepFailed(f'no free spot near ({want[0]:.2f}, {want[1]:.2f}) that the {arm} hand can reach')
            if math.hypot(spots[0][0] - want[0], spots[0][1] - want[1]) > 0.02:
                self.log(f'  requested spot ({want[0]:.2f}, {want[1]:.2f}) is occupied or out of reach')
            return spots
        spots = self.subtask('choose a free spot', choose, retries=1)
        gp = self.grasp_pose.get(arm) or {}
        side = gp.get('style') == 'side'
        # side grasp: keep the hand as it holds the object and put its base just above the table
        set_z = (TABLE_Z + gp['tcp_above_base'] + 0.006) if side else GRASP_TCP_Z + 0.002
        if back and gp.get('pick_tcp'):
            set_z = float(gp['pick_tcp'][2]) + 0.006     # same height as when it was picked (e.g. on a box)
        above = {} if side else {'approach': list(GRASP_APPROACH), 'tolerance_deg': 45}
        ctx = {'i': 0}

        def go_above():
            x, y = spots[ctx['i'] % len(spots)]
            self.log(f'  put the {obj} down at ({x:.3f}, {y:.3f})' + (' keeping the side grip' if side else ''))
            self.move(arm=arm, position=[x, y, set_z + 0.08], path='joint', **above)
            return x, y

        def next_spot(_exc=None):
            ctx['i'] += 1
        x, y = self.subtask('go above the spot', go_above, retries=min(3, len(spots) - 1) if len(spots) > 1 else 1,
                            recover=next_spot)
        zc = {'z': set_z}

        def lower():
            self.move(arm=arm, position=[x, y, zc['z']], speed=0.03)
            if gcam:
                on = self.gripper_cam_yes(arm, f'Is the bottom of the {obj.replace("_", " ")} held by the gripper '
                                               'touching the surface below it (the table or the object it is '
                                               'placed on)?')
                if on is False:
                    raise StepFailed('the gripper camera sees a gap under it')

        def lower_more(_exc=None):
            zc['z'] = max(TABLE_Z + 0.03, zc['z'] - 0.01)      # never into the table
            self.log(f'  lowering 1 cm more (TCP {100 * (zc["z"] - TABLE_Z):.0f} cm above the table)')
        self.subtask('lower it onto the table', lower, retries=8, recover=lower_more)
        self.subtask('open the gripper', lambda: self._gripper_checked(arm, 'open'))
        self.held[arm] = None
        self.grasp_pose.pop(arm, None)
        self.subtask('move the hand away', lambda: self.move(arm=arm, position=[0, 0, PREGRASP_DZ], relative=True,
                                                             speed=0.05))
        if gcam:
            def standing():
                ok = self.gripper_cam_yes(arm, f'Is a {obj.replace("_", " ")} standing on the table in this image?')
                if ok is False:
                    raise StepFailed(f'the gripper camera does not see the {obj} on the table')
            self.subtask('gripper camera: is it standing there?', standing, retries=0)
        self.subtask('go home', lambda: self.home_arm(arm))
        self.memory.pop(obj, None)
        if self.real:
            self.subtask('check where it is', lambda: self.verify_placed(obj, x, y), retries=0)

    def hover(self, obj, arm=None, height=0.12):
        """Point at an object: hand above it."""
        scene = self.perceive([obj])
        if obj not in scene:
            raise StepFailed(f'cannot find the {obj}')
        x, y, _ = scene[obj]['centre']
        arm = arm or ('left' if y > 0 else 'right')
        self.move(arm=arm, position=[x, y, GRASP_TCP_Z + height], approach=list(GRASP_APPROACH), tolerance_deg=40,
                  path='joint')

    def move_arm(self, arm, dx=0.0, dy=0.0, dz=0.0):
        """Relative hand move in the robot frame (x forward, y left, z up), metres."""
        step = math.sqrt(dx*dx + dy*dy + dz*dz)
        if step > 0.25:
            k = 0.25/step
            dx, dy, dz = dx*k, dy*k, dz*k
            self.log('relative move limited to 25 cm')
        self.move(arm=arm, position=[dx, dy, dz], relative=True, speed=0.1)

    def hand_over(self, obj=None, arm=None):
        """Pick (if needed) and offer the object to a person in front of the robot."""
        arm = self.holding_arm(arm, obj)
        if arm is None:
            if obj is None:
                raise StepFailed('what should I hand over?')
            arm = self.pick_up(obj, arm=None)
        self.move(arm=arm, position=HANDOVER_POSE[arm], approach=list(GRASP_APPROACH), tolerance_deg=45,
                  path='joint')
        self.log(f'holding the {self.held[arm]} out with the {arm} arm; say "release" to let go')

    def gripper(self, arm, command):
        arms = ('left', 'right') if arm in (None, 'both') else (arm,)
        for a in arms:
            self.tool('daksha_gripper', arm=a, command=command)
            if command == 'open':
                self.held[a] = None

    def recover(self):
        try:
            self.open_grippers()
            self.held = {'left': None, 'right': None}
            self.home()
        except (StepFailed, ValueError) as exc:
            self.log(f'recovery: {exc}')


def inside(obj, container, radius=0.065):
    """Conservative: the bowl's inner radius is 0.095 m, but a fruit resting on the rim reads ~0.08-0.10 m from
    the centre once VLM/depth error is added, so only well-inside counts (placed fruit lands 0.02-0.05 m out)."""
    return math.hypot(obj['centre'][0] - container['centre'][0], obj['centre'][1] - container['centre'][1]) < radius
