"""Daksha in MuJoCo: the robot backend behind the RAX MCP frontend.

Follows the RAX backend contract (services/backends.py): ``tools()``, ``validate()``,
``call()``, ``status()``, ``close()``. The frontend owns the episode lifecycle and calls
``reset_episode(seed)`` and ``task_state()``.

What the agent gets mirrors what the real Daksha stack can offer through ROS 2:
joint commands (``/joint_cmd``), Cartesian targets through the Placo IK (here MuJoCo DLS
on the same URDF), gripper position with a force limit, three RealSense-like cameras and
their calibration. What it does not get: object identities or poses, rewards, seeds, the
layout. The simulation only advances while a motion tool runs.

Safety layers (as on the RAX real-robot backends):
1. validation before motion: joint limits, workspace, a predicted-collision sweep of the
   whole path (arm vs table, torso, other arm), per-call step and speed caps;
2. a monitor during motion: the motion stops and holds on any arm-link contact with the
   table, the torso or the other arm, or a pad pressing the table harder than a limit;
3. actuator effort limits from the URDF (100 N m) and the gripper force limit.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import pathlib
import time

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from . import sorting, tasks, tray_transport
from .kinematics import Kinematics, Target, mat_to_quat_xyzw, quat_xyzw_to_mat
from .scene import (ARMS, ARM_JOINTS, BOWL, FINGER_BODIES, FRUITS, GRIPPER_CLOSED, GRIPPER_JOINT, HOME,
                    PAD_GAP_OPEN, PLACE_TABLE, SORTING_TASK, TABLE, TRAY_TASK, build_model)
from ..calibration import load_cameras, pose_matrix

DEFAULTS = {
    'control_hz': 100,              # the real controller_manager update_rate
    'image_width': 640,
    'image_height': 480,
    'joint_speed_default': 0.6,     # rad/s
    'joint_speed_max': 1.2,
    'cartesian_speed_default': 0.10,  # m/s
    'cartesian_speed_max': 0.25,
    'max_joint_step': 3.2,          # rad per call per joint
    'max_cartesian_step': 0.45,     # m per call
    'joint_margin': 0.005,          # rad inside the URDF limits
    'pad_press_limit_n': 60.0,      # pad-on-table force that stops a motion
    'settle_tolerance': 0.01,       # rad/s
    'settle_timeout': 1.5,          # s
    'gripper_force_default': 25.0,  # N
    'gripper_force_max': 40.0,
    'stream_period': 0.1,           # s of sim time between operator-viewer frames
    'frames_kept_in_memory': 64,
    'base_speed_default': 0.25,     # m/s (mobile-base tasks only)
    'base_speed_max': 0.5,
    'base_turn_speed': 0.5,         # rad/s
    'max_base_step': 6.0,           # m per call
    # Camera calibration files (services/calibration.py, RAX real/franka format). Cameras listed
    # here report the pose their calibration gives, as on the real robot; the others report the
    # simulator's exact pose. Rendering always uses the true camera.
    'cameras': None,
}

CAMERAS = ('world', 'left', 'right')


def true_camera_pose(m, d, camera):
    """T_base_cam of a MuJoCo camera in the OpenCV optical frame (x right, y down, z forward)."""
    cid = m.camera(camera).id
    return pose_matrix(d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), d.cam_xpos[cid])
STREAM_CAMERA = {'lookat': (0.30, 0.0, 0.95), 'distance': 1.75, 'azimuth': 200.0, 'elevation': -28.0}


class MotionStopped(RuntimeError):
    """A motion stopped by the monitor; the robot holds where it stopped."""
    handled_stop = True


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG', optimize=False, compress_level=3)
    return buffer.getvalue()


def _image_block(data, mime='image/png'):
    return {'type': 'image', 'mimeType': mime, 'data': base64.b64encode(data).decode()}


def _vec(v, n, name):
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        raise ValueError(f'{name} must be a list of {n} finite numbers')
    return np.array(v, float)


def _schema(props, required=()):
    return {'type': 'object', 'properties': props, 'required': list(required), 'additionalProperties': False}


ARM = {'type': 'string', 'enum': list(ARMS)}
VEC3 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
QUAT = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4,
        'description': 'TCP orientation quaternion [x, y, z, w] in the robot frame.'}
ORIENTATION_PROPS = {
    'quat': QUAT,
    'approach': {**VEC3, 'description': 'Approach direction (TCP x axis, wrist to fingertips), robot frame. '
                 'Roll about it is free unless finger_axis is given.'},
    'finger_axis': {**VEC3, 'description': 'Finger-opening direction (TCP y axis); needs approach. Sign does not matter.'},
    'tolerance_deg': {'type': 'number', 'minimum': 0, 'maximum': 90,
                      'description': 'Allowed deviation of approach/finger_axis (default 0). Daksha\'s wrist is narrow: '
                                     '15-35 deg lets IK choose a feasible tilt; the solver still prefers the '
                                     'smallest tilt it can reach (6, then 12 deg, then this).'},
    'finger_tolerance_deg': {'type': 'number', 'minimum': 0, 'maximum': 90,
                             'description': 'Separate allowed deviation of finger_axis (default: tolerance_deg).'},
}


class DakshaSimBackend:
    mutating = frozenset({'daksha_move_joints', 'daksha_move_eef', 'daksha_move_eef_both', 'daksha_gripper',
                          'daksha_home', 'daksha_move_base'})

    def __init__(self, config):
        self.task_id = getattr(config, 'task', 'fruit_to_bowl')
        task_modules = {'fruit_to_bowl': tasks, SORTING_TASK: sorting, TRAY_TASK: tray_transport}
        if self.task_id not in task_modules:
            raise ValueError(f'Unknown simulation task {self.task_id!r}')
        self.tasks = task_modules[self.task_id]
        options = dict(getattr(config, 'backend_options', {}) or {})
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown backend_options: {sorted(unknown)}')
        self.o = {**DEFAULTS, **options}
        self.profile = getattr(config, 'observation_profile', 'rgbd')
        self.frames_dir = pathlib.Path(config.workspace)/'frames'
        self.frames_rel = 'frames'
        self.stream_dir = pathlib.Path(config.root)/'stream'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        self.m, _ = build_model(self.task_id)
        self.d = mujoco.MjData(self.m)
        self.kin = Kinematics(self.m)
        self.calibration = load_cameras(self.o['cameras'], allowed=CAMERAS)
        self.substeps = max(1, round(1.0/(self.o['control_hz']*self.m.opt.timestep)))
        self._renderer = None
        self._depth = None
        self._stream_renderer = None
        self._frames = {}
        self._frame_counter = 0
        self._last_stream = -1.0
        self.episode_active = False
        self._grip_cmd = {arm: 'open' for arm in ARMS}
        self._gripper_force = {arm: self.o['gripper_force_default'] for arm in ARMS}
        self._ids()
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self._stream(force=True)

    # ------------------------------------------------------------------ ids
    def _ids(self):
        m = self.m
        self.act = {arm: np.array([m.actuator(j).id for j in ARM_JOINTS[arm]]) for arm in ARMS}
        self.grip_act = {arm: m.actuator(f'{arm}_gripper').id for arm in ARMS}
        self.grip_qadr = {arm: m.jnt_qposadr[m.joint(GRIPPER_JOINT[arm]).id] for arm in ARMS}
        self.grip_dadr = {arm: m.jnt_dofadr[m.joint(GRIPPER_JOINT[arm]).id] for arm in ARMS}
        self.pads = {arm: [m.geom(f'{f}_pad').id for f in FINGER_BODIES[arm]] for arm in ARMS}
        # Mobile base (tray_transport): planar joints base_x, base_y, base_yaw. Without them the robot frame is
        # the world frame, as before.
        self.mobile = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'base_x') >= 0
        if self.mobile:
            self.base_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ('base_x', 'base_y', 'base_yaw')])
            self.base_dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ('base_x', 'base_y', 'base_yaw')])
            self.base_act = np.array([m.actuator(n).id for n in ('base_x', 'base_y', 'base_yaw')])
        self.tables = [TABLE, PLACE_TABLE] if self.mobile else [TABLE]
        self.pad_set = {g for arm in ARMS for g in self.pads[arm]}
        body_arm = {}
        for b in range(m.nbody):
            name = m.body(b).name
            for arm in ARMS:
                if name.startswith(arm + '_'):
                    body_arm[b] = arm
        self.geom_arm = {g: body_arm.get(m.geom_bodyid[g]) for g in range(m.ngeom)}
        tables = {m.body(n).id for n in ('table', 'place_table') if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
        self.table_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in tables}
        self.free_bodies = {m.jnt_bodyid[j]: j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        torso = {m.body('base_link').id, m.body('wheel_base').id}
        self.geom_kind = {}
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if self.geom_arm[g]:
                self.geom_kind[g] = 'pad' if g in self.pad_set else 'arm'
            elif b in tables or m.geom(g).name == 'floor':
                self.geom_kind[g] = 'table'
            elif b in torso:
                self.geom_kind[g] = 'torso'
            else:
                self.geom_kind[g] = 'object'

    # ------------------------------------------------------------------ tools
    def tools(self):
        t = [
            {'name': 'daksha_status', 'description':
                'Measured robot state: both arms\' joints (rad), TCP poses (robot frame: origin on the floor under the '
                'robot, x forward, y left, z up; quaternions [x,y,z,w]; approach = TCP x axis, finger_axis = TCP y), '
                'gripper opening between the pads (m) and grip force (N), joint limits, the home posture and the '
                'simulation clock.', 'inputSchema': _schema({})},
            {'name': 'daksha_observe', 'description':
                'Render fresh images from the cameras: "world" (head, looking down at the table), "left" and "right" '
                '(on the wrists, looking along the approach axis past the fingers). Returns each RGB image, saves RGB'
                + (', depth (uint16 PNG, millimetres, 0 = invalid)' if self.profile == 'rgbd' else '')
                + ' and camera intrinsics/extrinsics (OpenCV convention: x right, y down, z forward) under '
                'frames/<frame_id>/, and returns the frame_id for daksha_pixel_to_position.',
             'inputSchema': _schema({'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': list(CAMERAS)},
                                                 'minItems': 1, 'maxItems': 3, 'uniqueItems': True,
                                                 'description': 'Default: all three.'},
                                     'attach': {'type': 'boolean', 'description': 'Return the images in the reply '
                                                '(default true); false only saves them.'}})},
            {'name': 'daksha_move_joints', 'description':
                'Move one arm to joint positions (rad) along a smooth joint-space path. Checked before moving: URDF '
                'limits, per-call step, and a predicted-collision sweep of the whole path. Stops and holds on contact '
                'of an arm link with the table, torso or other arm. Returns measured joints and TCP pose.',
             'inputSchema': _schema({'arm': ARM,
                                     'positions': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 7, 'maxItems': 7,
                                                   'description': 'joint_1..joint_7 of this arm, rad.'},
                                     'speed': {'type': 'number', 'minimum': 0.05, 'maximum': self.o['joint_speed_max'],
                                               'description': f'Peak joint speed rad/s (default {self.o["joint_speed_default"]}).'}},
                                    ['arm', 'positions'])},
            {'name': 'daksha_move_eef', 'description':
                'Move one arm\'s TCP (the fingertip end of the gripper, centred between the fingers) to a position, with '
                'an optional orientation (quat, or approach [+ finger_axis] with tolerance_deg). Without orientation '
                'the current orientation is kept within tolerance_deg (default 10). path "linear" (default) moves the TCP in a straight line with '
                'interpolated orientation; "joint" solves IK at the goal and moves in joint space (use it for large '
                'repositioning). relative=true treats position as an offset from the current TCP position. Refused '
                'before moving if IK fails, the line is infeasible, or the path would collide.',
             'inputSchema': _schema({'arm': ARM, 'position': VEC3, **ORIENTATION_PROPS,
                                     'relative': {'type': 'boolean'},
                                     'path': {'type': 'string', 'enum': ['linear', 'joint']},
                                     'speed': {'type': 'number', 'minimum': 0.01, 'maximum': self.o['cartesian_speed_max'],
                                               'description': f'TCP speed m/s for linear paths (default {self.o["cartesian_speed_default"]}).'}},
                                    ['arm', 'position'])},
            {'name': 'daksha_check', 'description':
                'Dry run, nothing moves: solve IK for a TCP target (same arguments as daksha_move_eef) and check the '
                'path; or, with joints, compute the TCP pose (FK). Returns reachable, the joint solution, the predicted '
                'TCP pose and errors, and the first problem on the path if any.',
             'inputSchema': _schema({'arm': ARM, 'position': VEC3, **ORIENTATION_PROPS,
                                     'relative': {'type': 'boolean'},
                                     'path': {'type': 'string', 'enum': ['linear', 'joint']},
                                     'joints': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 7, 'maxItems': 7}},
                                    ['arm'])},
            {'name': 'daksha_gripper', 'description':
                'Open or close one gripper. "open" opens fully (0.089 m between the pads); "close" closes with a force '
                'limit and stops when the fingers stall on an object; "width" moves to an opening in metres. Returns '
                'the measured opening, grip force and whether the fingers stalled before closing fully (something is '
                'between them).',
             'inputSchema': _schema({'arm': ARM, 'command': {'type': 'string', 'enum': ['open', 'close', 'width']},
                                     'width': {'type': 'number', 'minimum': 0.0, 'maximum': PAD_GAP_OPEN},
                                     'force': {'type': 'number', 'minimum': 2, 'maximum': self.o['gripper_force_max'],
                                               'description': f'Grip force limit N (default {self.o["gripper_force_default"]}).'}},
                                    ['arm', 'command'])},
            {'name': 'daksha_home', 'description':
                'Move one or both arms to the home posture (TCPs ~0.20 m above the table at x=0.27, y=+-0.24, '
                'pointing down) along checked joint-space paths, one arm after the other. Grippers keep their state.',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': ['left', 'right', 'both']},
                                     'speed': {'type': 'number', 'minimum': 0.05, 'maximum': self.o['joint_speed_max']}})},
        ]
        hand = _schema({'position': VEC3, **ORIENTATION_PROPS}, ['position'])
        t.insert(4, {'name': 'daksha_move_eef_both', 'description':
            'Move both TCPs at the same time, in one synchronized motion: both start and finish together (use it to '
            'lift, carry and set down an object held in both hands). left and right take the daksha_move_eef '
            'targets (position + optional orientation; without orientation each hand keeps its own). relative=true '
            'treats both positions as offsets from the current TCPs (e.g. [0, 0, 0.1] for each to lift 10 cm). '
            'path "linear" (default) moves each TCP in a straight line, "joint" in joint space. The whole two-arm '
            'path is checked first (IK, workspace, collisions incl. arm vs arm) and stops on contact like '
            'daksha_move_eef.',
         'inputSchema': _schema({'left': hand, 'right': hand, 'relative': {'type': 'boolean'},
                                 'path': {'type': 'string', 'enum': ['linear', 'joint']},
                                 'speed': {'type': 'number', 'minimum': 0.01, 'maximum': self.o['cartesian_speed_max'],
                                           'description': 'TCP speed m/s of the faster hand (default '
                                                          f'{self.o["cartesian_speed_default"]}).'}},
                                ['left', 'right'])})
        if self.mobile:
            t.append({'name': 'daksha_move_base', 'description':
                'Drive the mobile base, relative to where the robot stands now: forward (m, along robot x), left '
                '(m, along robot y) and turn_deg (counter-clockwise), done together along a straight line with a '
                'smooth start and stop. The arms hold their joint positions, so anything held in the grippers '
                'travels with the robot. Checked first over the whole drive: the torso, arms, and any object the '
                'fingers hold must not hit a table (a held object still resting on a table is refused: lift it '
                'first). Stops and holds on contact. Afterwards the robot frame (every arm and camera position) has '
                'moved with the base: observe again before using positions. Returns the odometry pose.',
             'inputSchema': _schema({'forward': {'type': 'number', 'minimum': -self.o['max_base_step'], 'maximum': self.o['max_base_step']},
                                     'left': {'type': 'number', 'minimum': -self.o['max_base_step'], 'maximum': self.o['max_base_step']},
                                     'turn_deg': {'type': 'number', 'minimum': -180, 'maximum': 180},
                                     'speed': {'type': 'number', 'minimum': 0.05, 'maximum': self.o['base_speed_max'],
                                               'description': f'Peak speed m/s (default {self.o["base_speed_default"]}).'}})})
        if self.profile == 'rgbd':
            t.insert(2, {'name': 'daksha_pixel_to_position', 'description':
                'Robot-frame 3-D points for pixels of a saved frame, from its depth image and calibration. pixels are '
                '[u, v] (u = column from the left, v = row from the top). window takes the median depth of a '
                '(2*window+1)^2 patch, ignoring invalid pixels. Depth is the visible surface: for a round object the '
                'point is on its near side, not its centre.',
             'inputSchema': _schema({'frame_id': {'type': 'string'}, 'camera': {'type': 'string', 'enum': list(CAMERAS)},
                                     'pixels': {'type': 'array', 'minItems': 1, 'maxItems': 256,
                                                'items': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}},
                                     'window': {'type': 'integer', 'minimum': 0, 'maximum': 10}},
                                    ['frame_id', 'camera', 'pixels'])})
        return t

    # ------------------------------------------------------------------ validation
    def validate(self, name, args):
        if name != 'daksha_status' and name != 'daksha_check' and not self.episode_active:
            raise PermissionError('No episode is running: call start_episode first')
        if 'arm' in args and args['arm'] not in (*ARMS, 'both'):
            raise ValueError('arm must be "left" or "right"')
        if 'speed' in args and not (isinstance(args['speed'], (int, float)) and args['speed'] > 0):
            raise ValueError('speed must be a positive number')

    # ------------------------------------------------------------------ helpers
    def _q(self, arm):
        return self.d.qpos[self.kin.qadr[arm]].copy()

    # Robot frame: on the floor under the robot, x forward, y left, z up. It moves with a mobile base; every
    # position and direction the tools take or return is in it. The simulator works in the world frame.
    def _base_pose(self, qpos=None):
        if not self.mobile:
            return 0.0, 0.0, 0.0
        q = self.d.qpos if qpos is None else qpos
        return tuple(float(v) for v in q[self.base_qadr])

    def _R_wb(self, qpos=None):
        yaw = self._base_pose(qpos)[2]
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def _T_wb(self, qpos=None):
        x, y, _ = self._base_pose(qpos)
        return pose_matrix(self._R_wb(qpos), np.array([x, y, 0.0]))

    def _to_world(self, p, qpos=None):
        x, y, _ = self._base_pose(qpos)
        return self._R_wb(qpos) @ np.asarray(p, float) + np.array([x, y, 0.0])

    def _to_base(self, p, qpos=None):
        x, y, _ = self._base_pose(qpos)
        return self._R_wb(qpos).T @ (np.asarray(p, float) - np.array([x, y, 0.0]))

    def _tcp(self, arm, qpos=None):
        qpos = self.d.qpos.copy() if qpos is None else qpos
        p, R = self.kin.fk(qpos, arm)
        p, R = self._to_base(p, qpos), self._R_wb(qpos).T @ R
        return {'position': p.round(4).tolist(), 'quat': mat_to_quat_xyzw(R).round(4).tolist(),
                'approach': R[:, 0].round(3).tolist(), 'finger_axis': R[:, 1].round(3).tolist()}

    def _gripper_width(self, arm):
        a, b = (self.d.geom_xpos[g] for g in self.pads[arm])
        axis = self.d.geom_xmat[self.pads[arm][0]].reshape(3, 3)[:, 1]
        return max(0.0, float(abs(np.dot(a - b, axis))) - 2*0.016)

    def _arm_state(self, arm):
        return {'joints': self._q(arm).round(4).tolist(), 'tcp': self._tcp(arm),
                'gripper': {'width_m': round(self._gripper_width(arm), 4),
                            'force_n': round(abs(float(self.d.actuator_force[self.grip_act[arm]])), 2),
                            'command': self._grip_cmd[arm]}}

    def status(self):
        return {'sim_time_s': round(float(self.d.time), 3), 'episode_active': self.episode_active,
                'observation_profile': self.profile,
                'arms': {arm: self._arm_state(arm) for arm in ARMS},
                'joint_limits': {arm: {'lower': self.kin.lower[arm].round(4).tolist(),
                                       'upper': self.kin.upper[arm].round(4).tolist()} for arm in ARMS},
                'home': {arm: [HOME[j] for j in ARM_JOINTS[arm]] for arm in ARMS},
                'table_top_z': TABLE['top'],
                **({'base': self._base_state()} if self.mobile else {})}

    def _base_state(self):
        x, y, yaw = self._base_pose()
        return {'odom': {'x': round(x, 4), 'y': round(y, 4), 'yaw_deg': round(math.degrees(yaw), 2)},
                'note': 'odometry in the room frame (fixed at the start pose); all arm and camera positions are in '
                        'the robot frame, which moves with the base'}

    def _target(self, arm, args):
        pos = _vec(args['position'], 3, 'position')
        if args.get('relative'):
            pos = np.array(self._tcp(arm)['position']) + pos
        pos = self._to_world(pos)          # robot frame -> world frame for the IK
        Rwb = self._R_wb()
        tol = float(args.get('tolerance_deg', 0.0))
        if 'quat' in args:
            if 'approach' in args or 'finger_axis' in args:
                raise ValueError('Give quat or approach/finger_axis, not both')
            q = _vec(args['quat'], 4, 'quat')
            if abs(np.linalg.norm(q) - 1) > 0.05:
                raise ValueError('quat must be a unit quaternion [x, y, z, w]')
            return Target(pos, quat=mat_to_quat_xyzw(Rwb @ quat_xyzw_to_mat(q/np.linalg.norm(q))), tolerance_deg=tol)
        if 'approach' in args:
            return Target(pos, approach=Rwb @ _vec(args['approach'], 3, 'approach'),
                          finger_axis=Rwb @ _vec(args['finger_axis'], 3, 'finger_axis') if 'finger_axis' in args else None,
                          tolerance_deg=tol, finger_tolerance_deg=args.get('finger_tolerance_deg'))
        if 'finger_axis' in args:
            raise ValueError('finger_axis needs approach')
        # Keep the current orientation, within 10 degrees unless told otherwise (an exact 6-D hold is often
        # infeasible for a narrow wrist over a long move).
        R = self.kin.fk(self.d.qpos.copy(), arm)[1]
        return Target(pos, quat=mat_to_quat_xyzw(R), tolerance_deg=float(args.get('tolerance_deg', 10.0)))

    def _check_workspace(self, p):
        """p in the world frame; the box is in the robot frame, the table tops are fixed in the room."""
        wx, wy, z = p
        for t in self.tables:
            if t['x0'] - 0.02 < wx < t['x1'] and t['y0'] < wy < t['y1'] and z < t['top'] - 0.003:
                return f'TCP z {z:.3f} m is below the table top ({t["top"]:.3f} m)'
        x, y, z = self._to_base(p)
        if not (-0.2 < x < 0.85 and -0.65 < y < 0.65 and 0.3 < z < 1.7):
            return (f'TCP {np.round([x, y, z], 3).tolist()} is outside the workspace box x -0.2..0.85, y -0.65..0.65, '
                    'z 0.3..1.7 (robot frame)')
        return None

    def _forbidden_contacts(self, d, arms):
        """Predicted or actual forbidden contacts in data `d` for the moving arms."""
        problems = []
        for c in d.contact[:d.ncon]:
            k1, k2 = self.geom_kind[c.geom1], self.geom_kind[c.geom2]
            a1, a2 = self.geom_arm[c.geom1], self.geom_arm[c.geom2]
            if not ((a1 in arms) or (a2 in arms)):
                continue
            kinds = {k1, k2}
            if a1 and a2 and a1 != a2:
                problems.append(('arm_vs_other_arm', c.dist))
            elif 'arm' in kinds and kinds & {'table', 'torso'}:
                problems.append(('arm_link_vs_' + ('table' if 'table' in kinds else 'torso'), c.dist))
            elif 'pad' in kinds and 'table' in kinds and c.dist < -0.004:
                problems.append(('finger_into_table', c.dist))
            elif 'pad' in kinds and 'torso' in kinds:
                problems.append(('finger_vs_torso', c.dist))
        return problems

    def _sweep(self, arm, path):
        """Kinematic collision and workspace check of a joint path (N, 7). Returns None or a reason."""
        kd = self.kin.d
        base = self.d.qpos.copy()
        stride = max(1, len(path)//60)
        for i in list(range(0, len(path), stride)) + [len(path) - 1]:
            kd.qpos[:] = base
            kd.qpos[self.kin.qadr[arm]] = path[i]
            mujoco.mj_fwdPosition(self.m, kd)
            p = kd.site_xpos[self.kin.site[arm]]
            reason = self._check_workspace(p)
            if reason:
                return f'at {100*i/max(1, len(path) - 1):.0f}% of the path: {reason}'
            problems = self._forbidden_contacts(kd, {arm})
            if problems:
                kind, dist = problems[0]
                return (f'at {100*i/max(1, len(path) - 1):.0f}% of the path the {arm} arm would collide '
                        f'({kind}, penetration {abs(dist)*1000:.0f} mm); TCP would be at '
                        f'{np.round(self._to_base(p), 3).tolist()}')
        return None

    def _limits(self, arm, q):
        lo, hi = self.kin.lower[arm] + self.o['joint_margin'], self.kin.upper[arm] - self.o['joint_margin']
        bad = [f'joint_{i + 1}={q[i]:.3f} outside [{lo[i]:.3f}, {hi[i]:.3f}]' for i in range(7) if not lo[i] <= q[i] <= hi[i]]
        if bad:
            raise ValueError('Joint limits: ' + '; '.join(bad))

    def _joint_path(self, arm, q_goal, speed):
        q0 = self._q(arm)
        dq = q_goal - q0
        span = float(np.max(np.abs(dq)))
        if span > self.o['max_joint_step']:
            raise ValueError(f'Joint step {span:.2f} rad exceeds the per-call limit {self.o["max_joint_step"]} rad')
        duration = max(0.2, 1.875*span/speed)
        n = max(2, int(math.ceil(duration*self.o['control_hz'])))
        s = np.linspace(0, 1, n)
        s = 10*s**3 - 15*s**4 + 6*s**5
        return q0 + np.outer(s, dq)

    def _linear_path(self, arm, target, speed):
        """Straight TCP line with slerped orientation, IK per 5 mm waypoint. Returns (path, goal_q) or raises."""
        qpos = self.d.qpos.copy()
        q0 = self._q(arm)
        p0, R0 = self.kin.fk(qpos, arm)
        q_goal, pe, re, ok = self.kin.solve(qpos, arm, target)
        if not ok:
            raise ValueError(self._ik_failure(arm, target, pe, re))
        pg, Rg = self.kin.fk(qpos, arm, q_goal)
        dist = float(np.linalg.norm(target.position - p0))
        if dist > self.o['max_cartesian_step']:
            raise ValueError(f'Linear move of {dist:.3f} m exceeds the per-call limit {self.o["max_cartesian_step"]} m')
        angle = float(np.linalg.norm(Rotation.from_matrix(Rg @ R0.T).as_rotvec()))
        n = max(2, int(math.ceil(max(dist/0.005, angle/math.radians(2)))) + 1)
        slerp = Slerp([0, 1], Rotation.from_matrix([R0, Rg]))
        qs = [q0]
        q = q0.copy()
        for i in range(1, n):
            s = i/(n - 1)
            wp = Target(p0 + s*(pg - p0), quat=slerp([s]).as_quat()[0])
            q_new, pe, re, ok = self.kin.ik(qpos, arm, wp, q, iterations=80, pos_tol=2e-3, rot_tol=math.radians(1.5))
            if not ok:
                raise ValueError(f'The straight line is infeasible at {100*s:.0f}% (IK error {pe*1000:.1f} mm, '
                                 f'{math.degrees(re):.1f} deg): try path "joint" or an intermediate point')
            if np.max(np.abs(q_new - q)) > 0.25:
                raise ValueError(f'The straight line passes a wrist flip or joint limit at {100*s:.0f}% '
                                 f'(joint jump {np.max(np.abs(q_new - q)):.2f} rad): try path "joint"')
            qs.append(q_new)
            q = q_new
        qs = np.array(qs)
        # Time-parameterize: min-jerk over the whole line at the requested TCP speed and joint cap.
        span = float(np.max(np.abs(np.diff(qs, axis=0)).sum(axis=0)))
        duration = max(0.25, 1.875*max(dist/speed, span/self.o['joint_speed_max']))
        steps = max(2, int(math.ceil(duration*self.o['control_hz'])))
        s = np.linspace(0, 1, steps)
        s = 10*s**3 - 15*s**4 + 6*s**5
        idx = s*(len(qs) - 1)
        lo = np.floor(idx).astype(int).clip(0, len(qs) - 1)
        hi = np.minimum(lo + 1, len(qs) - 1)
        frac = (idx - lo)[:, None]
        return qs[lo]*(1 - frac) + qs[hi]*frac, q_goal

    def _ik_failure(self, arm, target, pe, re):
        msg = f'IK for the {arm} arm did not reach the target: best position error {pe*1000:.0f} mm'
        if target.approach is not None:
            msg += f', orientation error {math.degrees(re):.0f} deg'
            if target.tolerance_deg < 10:
                msg += ' (Daksha\'s wrist is narrow: allow tolerance_deg 15-35 or tilt the approach)'
        p = self._to_base(target.position)
        shoulder = np.array([-0.022, 0.128 if arm == 'left' else -0.131, 1.278])
        msg += f'; target is {np.linalg.norm(p - shoulder):.2f} m from the {arm} shoulder (reach about 0.65 m)'
        return msg

    # ------------------------------------------------------------------ physics
    def _stream(self, force=False):
        if not force and self.d.time - self._last_stream < self.o['stream_period']:
            return
        self._last_stream = float(self.d.time)
        try:
            if self._stream_renderer is None:
                self._stream_renderer = mujoco.Renderer(self.m, 480, 640)
                self._stream_cam = mujoco.MjvCamera()
                for k, v in STREAM_CAMERA.items():
                    setattr(self._stream_cam, k, v) if k != 'lookat' else self._stream_cam.lookat.__setitem__(slice(None), v)
            self._stream_renderer.update_scene(self.d, 'overview' if self.task_id == TRAY_TASK else self._stream_cam)
            overview = self._stream_renderer.render()
            self._stream_renderer.update_scene(self.d, 'world')
            head = self._stream_renderer.render()
            frame = np.concatenate([overview, head], axis=1)
            tmp = self.stream_dir/'latest.tmp.jpg'
            Image.fromarray(frame).save(tmp, quality=80)
            os.replace(tmp, self.stream_dir/'latest.jpg')
        except Exception:
            pass  # the operator viewer must never break a motion

    def _tick(self):
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        self._stream()

    def _monitor(self, arms):
        problems = self._forbidden_contacts(self.d, arms)
        for kind, dist in problems:
            if kind != 'finger_into_table':
                return kind
        # Pad pressing on the table.
        force = np.zeros(6)
        for i, c in enumerate(self.d.contact[:self.d.ncon]):
            kinds = {self.geom_kind[c.geom1], self.geom_kind[c.geom2]}
            if kinds == {'pad', 'table'}:
                mujoco.mj_contactForce(self.m, self.d, i, force)
                if force[0] > self.o['pad_press_limit_n']:
                    return f'finger_pressing_table ({force[0]:.0f} N)'
        return None

    def _hold(self, arm):
        self.d.ctrl[self.act[arm]] = self._q(arm)

    def _settle(self, arms, timeout=None):
        timeout = self.o['settle_timeout'] if timeout is None else timeout
        t_end = self.d.time + timeout
        while self.d.time < t_end:
            self._tick()
            v = max(float(np.max(np.abs(self.d.qvel[self.kin.dadr[a] if a != 'base' else self.base_dadr]))) for a in arms)
            if v < self.o['settle_tolerance']:
                break

    def _execute(self, arm, path):
        t0 = float(self.d.time)
        for q in path:
            self.d.ctrl[self.act[arm]] = q
            self._tick()
            stop = self._monitor({arm})
            if stop:
                self._hold(arm)
                self._settle([arm], 0.5)
                self._stream(force=True)
                err = MotionStopped(f'Motion stopped by the contact monitor: {stop}. The {arm} arm holds where it '
                                    f'stopped; observe before the next motion.')
                err.robot_state = self._arm_state(arm)
                raise err
        self._settle([arm])
        self._stream(force=True)
        q_meas = self._q(arm)
        return {'arm': arm, 'duration_s': round(float(self.d.time) - t0, 3),
                'joints': q_meas.round(4).tolist(),
                'joint_tracking_error_max': round(float(np.max(np.abs(q_meas - path[-1]))), 4),
                'tcp': self._tcp(arm), 'contacts': self._contact_summary(arm)}

    def _contact_summary(self, arm):
        kinds = set()
        for c in self.d.contact[:self.d.ncon]:
            a1, a2 = self.geom_arm[c.geom1], self.geom_arm[c.geom2]
            if arm in (a1, a2):
                other = self.geom_kind[c.geom2] if a1 == arm else self.geom_kind[c.geom1]
                mine = self.geom_kind[c.geom1] if a1 == arm else self.geom_kind[c.geom2]
                if other not in ('pad', 'arm'):
                    kinds.add(f'{"fingers" if mine == "pad" else "arm link"} touching {"table" if other == "table" else "an object" if other == "object" else other}')
        return sorted(kinds)

    # ------------------------------------------------------------------ calls
    def call(self, name, args):
        handler = getattr(self, '_' + name.removeprefix('daksha_'), None)
        if handler is None:
            raise PermissionError(f'Unknown tool {name}')
        return handler(args)

    def _status(self, args):
        return self.status(), []

    def _move_joints(self, args):
        arm = args['arm']
        q_goal = _vec(args['positions'], 7, 'positions')
        self._limits(arm, q_goal)
        speed = float(args.get('speed', self.o['joint_speed_default']))
        path = self._joint_path(arm, q_goal, speed)
        reason = self._sweep(arm, path)
        if reason:
            raise ValueError('Refused: ' + reason)
        return self._execute(arm, path), []

    def _plan_eef(self, arm, args, sweep=True):
        target = self._target(arm, args)
        reason = self._check_workspace(target.position)
        if reason:
            raise ValueError('Refused: ' + reason)
        mode = args.get('path', 'linear')
        if mode == 'linear':
            path, q_goal = self._linear_path(arm, target, float(args.get('speed', self.o['cartesian_speed_default'])))
        else:
            q_goal, pe, re, ok = self.kin.solve(self.d.qpos.copy(), arm, target)
            if not ok:
                raise ValueError(self._ik_failure(arm, target, pe, re))
            self._limits(arm, q_goal)
            path = self._joint_path(arm, q_goal, float(args.get('joint_speed', self.o['joint_speed_default'])))
        if sweep:
            reason = self._sweep(arm, path)
            if reason:
                raise ValueError('Refused: ' + reason)
        return target, path, q_goal

    def _move_eef(self, args):
        arm = args['arm']
        target, path, q_goal = self._plan_eef(arm, args)
        result = self._execute(arm, path)
        target_base = self._to_base(target.position)
        result['target_position'] = np.round(target_base, 4).tolist()
        result['position_error_m'] = round(float(np.linalg.norm(np.array(result['tcp']['position']) - target_base)), 4)
        return result, []

    # ------------------------------------------------------------------ both arms together
    def _move_eef_both(self, args):
        plans = {}
        for arm in ARMS:
            if not isinstance(args.get(arm), dict):
                raise ValueError(f'{arm} must be an object with position (+ optional orientation)')
            sub = {**args[arm], 'relative': bool(args.get('relative')), 'path': args.get('path', 'linear')}
            if 'speed' in args:
                sub['speed'] = args['speed']
            try:
                plans[arm] = self._plan_eef(arm, sub, sweep=False)
            except ValueError as exc:
                raise ValueError(f'{arm} hand: {exc}') from None
        n = max(len(path) for _, path, _ in plans.values())
        paths = {arm: self._resample(path, n) for arm, (_, path, _) in plans.items()}
        reason = self._sweep_multi(paths)
        if reason:
            raise ValueError('Refused: ' + reason)
        result = self._execute_multi(paths)
        for arm, (target, _, _) in plans.items():
            tb = self._to_base(target.position)
            result[arm]['target_position'] = np.round(tb, 4).tolist()
            result[arm]['position_error_m'] = round(float(np.linalg.norm(np.array(result[arm]['tcp']['position']) - tb)), 4)
        return result, []

    @staticmethod
    def _resample(path, n):
        if len(path) == n:
            return path
        idx = np.linspace(0, len(path) - 1, n)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(path) - 1)
        frac = (idx - lo)[:, None]
        return path[lo]*(1 - frac) + path[hi]*frac

    def _sweep_multi(self, paths):
        kd = self.kin.d
        base = self.d.qpos.copy()
        n = len(next(iter(paths.values())))
        stride = max(1, n//60)
        for i in list(range(0, n, stride)) + [n - 1]:
            kd.qpos[:] = base
            for arm, path in paths.items():
                kd.qpos[self.kin.qadr[arm]] = path[i]
            mujoco.mj_fwdPosition(self.m, kd)
            for arm in paths:
                reason = self._check_workspace(kd.site_xpos[self.kin.site[arm]])
                if reason:
                    return f'at {100*i/max(1, n - 1):.0f}% of the path, {arm} hand: {reason}'
            problems = self._forbidden_contacts(kd, set(paths))
            if problems:
                kind, dist = problems[0]
                return (f'at {100*i/max(1, n - 1):.0f}% of the path the arms would collide ({kind}, penetration '
                        f'{abs(dist)*1000:.0f} mm)')
        return None

    def _execute_multi(self, paths):
        arms = list(paths)
        t0 = float(self.d.time)
        n = len(paths[arms[0]])
        for i in range(n):
            for arm in arms:
                self.d.ctrl[self.act[arm]] = paths[arm][i]
            self._tick()
            stop = self._monitor(set(arms))
            if stop:
                for arm in arms:
                    self._hold(arm)
                self._settle(arms, 0.5)
                self._stream(force=True)
                err = MotionStopped(f'Motion stopped by the contact monitor: {stop}. Both arms hold where they '
                                    'stopped; observe before the next motion.')
                err.robot_state = {arm: self._arm_state(arm) for arm in arms}
                raise err
        self._settle(arms)
        self._stream(force=True)
        out = {'duration_s': round(float(self.d.time) - t0, 3)}
        for arm in arms:
            q = self._q(arm)
            out[arm] = {'joints': q.round(4).tolist(), 'joint_tracking_error_max':
                        round(float(np.max(np.abs(q - paths[arm][-1]))), 4),
                        'tcp': self._tcp(arm), 'contacts': self._contact_summary(arm)}
        return out

    # ------------------------------------------------------------------ mobile base
    def _held_bodies(self, d=None):
        """Free bodies the finger pads touch (they travel with the robot during a drive)."""
        d = self.d if d is None else d
        held = set()
        for c in d.contact[:d.ncon]:
            for g_pad, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g_pad in self.pad_set:
                    b = self.m.geom_bodyid[g_other]
                    if b in self.free_bodies:
                        held.add(b)
        return held

    def _base_collisions(self, d, held):
        """Contacts of the robot (torso, arms, fingers) or a held object with a table, in data d."""
        out = []
        for c in d.contact[:d.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if other not in self.table_geoms or self.m.geom(other).name == 'floor':
                    continue
                kind = self.geom_kind[g]
                if kind in ('arm', 'pad', 'torso'):
                    out.append(('torso' if kind == 'torso' else 'arm', float(c.dist)))
                elif self.m.geom_bodyid[g] in held:
                    out.append(('held object', float(c.dist)))
        return out

    def _move_base(self, args):
        if not self.mobile:
            raise ValueError('The base is fixed in this task')
        fwd, left = float(args.get('forward', 0.0)), float(args.get('left', 0.0))
        turn = math.radians(float(args.get('turn_deg', 0.0)))
        dist = math.hypot(fwd, left)
        if dist > self.o['max_base_step']:
            raise ValueError(f'Drive of {dist:.2f} m exceeds the per-call limit {self.o["max_base_step"]} m')
        if dist < 1e-4 and abs(turn) < 1e-4:
            raise ValueError('Give forward, left and/or turn_deg')
        speed = float(args.get('speed', self.o['base_speed_default']))
        x0, y0, yaw0 = self._base_pose()
        goal = np.array([x0 + math.cos(yaw0)*fwd - math.sin(yaw0)*left,
                         y0 + math.sin(yaw0)*fwd + math.cos(yaw0)*left, yaw0 + turn])
        start = np.array([x0, y0, yaw0])
        duration = max(0.5, 1.875*max(dist/speed, abs(turn)/self.o['base_turn_speed']))
        n = max(2, int(math.ceil(duration*self.o['control_hz'])))
        s = np.linspace(0, 1, n)
        s = 10*s**3 - 15*s**4 + 6*s**5
        path = start + np.outer(s, goal - start)
        held = self._held_bodies()
        touching = [k for k, dd in self._base_collisions(self.d, held) if k == 'held object']
        if touching:
            raise ValueError('Refused: the object in the grippers is resting on a table; lift it before driving')
        reason = self._sweep_base(path, held)
        if reason:
            raise ValueError('Refused: ' + reason)
        t0 = float(self.d.time)
        for pose in path:
            self.d.ctrl[self.base_act] = pose
            self._tick()
            hits = [h for h in self._base_collisions(self.d, held) if h[1] < -0.001]
            if hits:
                self.d.ctrl[self.base_act] = self.d.qpos[self.base_qadr]
                self._settle(['base'], 0.5)
                self._stream(force=True)
                err = MotionStopped(f'Drive stopped by the contact monitor: {hits[0][0]} touching a table. The base '
                                    'holds where it stopped; observe before the next motion.')
                err.robot_state = self._base_state()
                raise err
        self._settle(['base', *ARMS])
        self._stream(force=True)
        meas = np.array(self._base_pose())
        return {'duration_s': round(float(self.d.time) - t0, 3), **self._base_state(),
                'tracking_error': {'position_m': round(float(np.linalg.norm(meas[:2] - goal[:2])), 4),
                                   'yaw_deg': round(math.degrees(meas[2] - goal[2]), 2)},
                'fingers_touching_object': {arm: any(self.geom_kind.get(g2) == 'object'
                                                     for c in self.d.contact[:self.d.ncon]
                                                     for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1))
                                                     if g1 in self.pads[arm]) for arm in ARMS}}, []

    def _sweep_base(self, path, held):
        """Kinematic check of a base path: the robot and the held objects (moved rigidly with it) vs the tables."""
        kd = self.kin.d
        base = self.d.qpos.copy()
        T0_inv = np.linalg.inv(self._T_wb(base))
        held_pose = {}
        for b in held:
            adr = self.m.jnt_qposadr[self.free_bodies[b]]
            held_pose[b] = (adr, pose_matrix(quat_xyzw_to_mat(np.roll(base[adr + 3:adr + 7], -1)), base[adr:adr + 3]))
        span = max(float(np.linalg.norm(path[-1, :2] - path[0, :2]))/0.02, abs(float(path[-1, 2] - path[0, 2]))/0.035)
        step = max(1, int(len(path)//max(2.0, span)))   # one check every ~2 cm or ~2 deg
        for i in list(range(0, len(path), step)) + [len(path) - 1]:
            kd.qpos[:] = base
            kd.qpos[self.base_qadr] = path[i]
            T = self._T_wb(kd.qpos) @ T0_inv
            for b, (adr, P) in held_pose.items():
                Pn = T @ P
                kd.qpos[adr:adr + 3] = Pn[:3, 3]
                kd.qpos[adr + 3:adr + 7] = np.roll(mat_to_quat_xyzw(Pn[:3, :3]), 1)
            mujoco.mj_fwdPosition(self.m, kd)
            hits = [h for h in self._base_collisions(kd, held) if h[1] < -0.001]
            if hits:
                x, y, yaw = path[i]
                return (f'at {100*i/max(1, len(path) - 1):.0f}% of the drive the {hits[0][0]} would hit a table '
                        f'(odometry there x {x:.2f}, y {y:.2f}, yaw {math.degrees(yaw):.0f} deg)')
        return None

    def _check(self, args):
        arm = args['arm']
        if 'joints' in args:
            q = _vec(args['joints'], 7, 'joints')
            qpos = self.d.qpos.copy()
            qpos[self.kin.qadr[arm]] = q
            return {'arm': arm, 'tcp': self._tcp(arm, qpos),
                    'within_limits': bool(np.all(q >= self.kin.lower[arm]) and np.all(q <= self.kin.upper[arm]))}, []
        if 'position' not in args:
            raise ValueError('daksha_check takes joints, or position (+ orientation)')
        try:
            target, path, q_goal = self._plan_eef(arm, args)
        except ValueError as exc:
            return {'arm': arm, 'reachable': False, 'problem': str(exc)}, []
        qpos = self.d.qpos.copy()
        qpos[self.kin.qadr[arm]] = q_goal
        return {'arm': arm, 'reachable': True, 'joints': q_goal.round(4).tolist(), 'tcp': self._tcp(arm, qpos),
                'path': args.get('path', 'linear'), 'estimated_duration_s': round(len(path)/self.o['control_hz'], 2)}, []

    def _home(self, args):
        which = args.get('arm', 'both')
        speed = float(args.get('speed', self.o['joint_speed_default']))
        results = []
        arms = ARMS if which == 'both' else (which,)
        # Move the arm that is lower first out of the way? Keep it simple: in order, each checked.
        for arm in arms:
            q_goal = np.array([HOME[j] for j in ARM_JOINTS[arm]])
            path = self._joint_path(arm, q_goal, speed)
            reason = self._sweep(arm, path)
            if reason:
                raise ValueError(f'Refused homing the {arm} arm: {reason}')
            results.append(self._execute(arm, path))
        return {'moved': results}, []

    def _gripper(self, args):
        arm, command = args['arm'], args['command']
        force = float(args.get('force', self._gripper_force[arm]))
        closed = GRIPPER_CLOSED[arm]
        if command == 'open':
            target = 0.0
        elif command == 'close':
            target = closed
        else:
            if 'width' not in args:
                raise ValueError('command "width" needs width (m)')
            width = float(args['width'])
            target = math.copysign(min(abs(closed), max(0.0, (PAD_GAP_OPEN - width)/2)), closed)
        a = self.grip_act[arm]
        self.m.actuator_forcerange[a] = (-force, force)
        self._gripper_force[arm] = force
        self.d.ctrl[a] = target
        self._grip_cmd[arm] = command if command != 'width' else f'width {args["width"]:.3f}'
        t0 = float(self.d.time)
        still = 0
        while self.d.time - t0 < 2.0:
            self._tick()
            v = abs(float(self.d.qvel[self.grip_dadr[arm]]))
            still = still + 1 if v < 2e-3 and self.d.time - t0 > 0.1 else 0
            if still >= 10:
                break
        self._settle([arm], 0.3)
        self._stream(force=True)
        q = float(self.d.qpos[self.grip_qadr[arm]])
        stalled = command != 'open' and abs(q - target) > 0.002
        return {'arm': arm, 'command': self._grip_cmd[arm], 'width_m': round(self._gripper_width(arm), 4),
                'force_n': round(abs(float(self.d.actuator_force[a])), 2), 'force_limit_n': force,
                'stalled_before_target': bool(stalled), 'duration_s': round(float(self.d.time) - t0, 3)}, []

    # ------------------------------------------------------------------ cameras
    def _render(self, camera, depth=False):
        h, w = self.o['image_height'], self.o['image_width']
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, h, w)
            self._depth = mujoco.Renderer(self.m, h, w)
            self._depth.enable_depth_rendering()
        r = self._depth if depth else self._renderer
        r.update_scene(self.d, camera)
        return r.render().copy()

    def _camera_info(self, camera):
        h, w = self.o['image_height'], self.o['image_width']
        cid = self.m.camera(camera).id
        model = self.calibration.get(camera)
        if model is None:
            T, source = np.linalg.inv(self._T_wb()) @ true_camera_pose(self.m, self.d, camera), 'simulator'
        else:
            T_tcp = None
            if model.mount == 'tcp':
                p, R = self.kin.fk(self.d.qpos.copy(), model.arm)
                T_tcp = pose_matrix(self._R_wb().T @ R, self._to_base(p))
            T, source = model.base_from_camera(T_tcp), f'calibration ({model.mount})'
        if model is not None and model.intrinsics is not None:
            intr = model.intrinsics
            sx, sy = w/intr['width'], h/intr['height']
            fx, fy = intr['fx']*sx, intr['fy']*sy
            cx = (intr['cx'] + 0.5)*sx - 0.5
            cy = (intr['cy'] + 0.5)*sy - 0.5
            K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        else:
            fovy = math.radians(float(self.m.cam_fovy[cid]))
            f = (h/2)/math.tan(fovy/2)
            fx = fy = f
            cx, cy = w/2 - 0.5, h/2 - 0.5
            K = [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]
        R_cv = T[:3, :3]
        return {'width': w, 'height': h, 'fx': fx, 'fy': fy, 'cx': cx + 0.5, 'cy': cy + 0.5,
                'K': K,
                'position': T[:3, 3].round(5).tolist(),
                'rotation_camera_to_robot': R_cv.round(6).tolist(),
                'quat_camera_to_robot': mat_to_quat_xyzw(R_cv).round(6).tolist(),
                'T_base_cam': T.round(6).tolist(), 'source': source,
                'convention': 'OpenCV: x right, y down, z forward; p_robot = position + R @ p_camera; '
                              'K has pixel centres at integers (cx = cx_field - 0.5)'}

    def camera_truth(self, camera):
        """Privileged (operator only, never a tool): the simulator's exact T_base_cam."""
        mujoco.mj_forward(self.m, self.d)
        return np.linalg.inv(self._T_wb()) @ true_camera_pose(self.m, self.d, camera)

    def _observe(self, args):
        cameras = args.get('cameras') or list(CAMERAS)
        attach = args.get('attach', True)
        mujoco.mj_forward(self.m, self.d)
        self._frame_counter += 1
        frame_id = f'f{self._frame_counter:05d}'
        folder = self.frames_dir/frame_id
        folder.mkdir(parents=True, exist_ok=True)
        meta, images, record = {}, [], {}
        for cam in cameras:
            rgb = self._render(cam)
            png = _png(rgb)
            (folder/f'{cam}.png').write_bytes(png)
            info = self._camera_info(cam)
            entry = {'rgb': f'{self.frames_rel}/{frame_id}/{cam}.png', 'intrinsics': {k: info[k] for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'K')},
                     'extrinsics': {k: info[k] for k in ('position', 'rotation_camera_to_robot', 'quat_camera_to_robot', 'T_base_cam', 'source', 'convention')}}
            if self.profile == 'rgbd':
                depth = self._render(cam, depth=True)
                depth[depth > 3.0] = 0.0  # beyond the RealSense-like range
                mm = np.clip(depth*1000.0, 0, 65535).astype(np.uint16)
                Image.fromarray(mm).save(folder/f'{cam}_depth.png')
                entry['depth'] = f'{self.frames_rel}/{frame_id}/{cam}_depth.png'
                record[cam] = (depth, info)
            meta[cam] = entry
            if attach:
                entry['content_index'] = len(images) + 1
                images.append(_image_block(png))
        extra = {'base_odom': self._base_state()['odom']} if self.mobile else {}
        (folder/'meta.json').write_text(json.dumps({'frame_id': frame_id, 'sim_time_s': float(self.d.time), **extra,
                                                    'cameras': meta}, indent=1))
        if record:
            self._frames[frame_id] = record
            while len(self._frames) > self.o['frames_kept_in_memory']:
                self._frames.pop(next(iter(self._frames)))
        self._stream(force=True)
        return {'frame_id': frame_id, 'sim_time_s': round(float(self.d.time), 3), **extra, 'cameras': meta}, images

    def _pixel_to_position(self, args):
        frame_id, cam = args['frame_id'], args['camera']
        if frame_id not in self._frames or cam not in self._frames[frame_id]:
            folder = self.frames_dir/frame_id
            meta_path = folder/'meta.json'
            if not (meta_path.exists() and (folder/f'{cam}_depth.png').exists()):
                raise ValueError(f'No saved depth for frame {frame_id!r} camera {cam!r}')
            meta = json.loads(meta_path.read_text())['cameras'][cam]
            depth = np.asarray(Image.open(folder/f'{cam}_depth.png'), dtype=np.float64)/1000.0
            info = {**meta['intrinsics'], **meta['extrinsics']}
        else:
            depth, info = self._frames[frame_id][cam]
        window = int(args.get('window', 2))
        R = np.array(info['rotation_camera_to_robot'])
        t = np.array(info['position'])
        h, w = depth.shape
        out = []
        for u, v in args['pixels']:
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < w and 0 <= vi < h):
                out.append({'pixel': [u, v], 'valid': False, 'reason': 'outside the image'})
                continue
            patch = depth[max(0, vi - window):vi + window + 1, max(0, ui - window):ui + window + 1]
            vals = patch[patch > 0]
            if vals.size == 0:
                out.append({'pixel': [u, v], 'valid': False, 'reason': 'no valid depth'})
                continue
            z = float(np.median(vals))
            pc = np.array([(u + 0.5 - info['cx'])/info['fx']*z, (v + 0.5 - info['cy'])/info['fy']*z, z])
            out.append({'pixel': [u, v], 'valid': True, 'depth_m': round(z, 4), 'position': (t + R @ pc).round(4).tolist()})
        return {'frame_id': frame_id, 'camera': cam, 'points': out}, []

    # ------------------------------------------------------------------ episodes (frontend only)
    def reset_episode(self, seed):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for arm in ARMS:
            self.m.actuator_forcerange[self.grip_act[arm]] = (-self.o['gripper_force_default'], self.o['gripper_force_default'])
            self._grip_cmd[arm] = 'open'
        layout = self.tasks.sample_layout(seed, reachable=self._layout_reachable)
        self.tasks.apply_layout(self.m, self.d, layout)
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.6/self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.d.time = 0.0
        self._last_stream = -1.0
        self.episode_active = True
        self._stream(force=True)

    def _layout_reachable(self, name, x, y, yaw):
        """Operator guarantee that a layout is solvable: a near-vertical grasp (<= 12 deg tilt, fingers
        across the banana within 25 deg) and a pre-grasp 0.13 m above it are both reachable."""
        if name == getattr(self.tasks, 'TRAY_NAME', None):
            return self._tray_reachable(x, y, yaw)
        arm = 'left' if y > 0 else 'right'
        if name in getattr(self.tasks, 'BOWL_NAMES', ()):
            z = TABLE['top'] + BOWL['height'] + 0.08
            poses = ((z, 25), (z + 0.13, 25))
        else:
            rest_height = getattr(self.tasks, 'object_rest_height', None)
            z = TABLE['top'] + (rest_height(name) + 0.002 if rest_height else 0.014)
            poses = ((z, 12), (z + 0.13, 20))
        orient = {'approach': [0, 0, -1]}
        if name in ('banana', 'toy_car'):
            axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            orient.update(finger_axis=np.cross([0, 0, 1], axis), finger_tolerance_deg=25)
        qpos = self.d.qpos.copy()
        for target_z, tol in poses:
            if not self.kin.solve(qpos, arm, Target([x, y, target_z], tolerance_deg=tol, **orient), restarts=4)[3]:
                return False
        return True

    def _tray_reachable(self, x, y, yaw):
        """Both hands can pinch their end wall top-down (<= 20 deg tilt, fingers along the tray's long
        axis) at grasp depth, 0.10 m above it (pre-grasp) and 0.15 m above it (lift)."""
        qpos = self.d.qpos.copy()
        for arm, (p, axis) in self.tasks.handle_grasps(x, y, yaw).items():
            for dz in (0.0, self.tasks.PREGRASP_HEIGHT, self.tasks.LIFT_HEIGHT):
                target = Target(p + [0, 0, dz], approach=[0, 0, -1], finger_axis=axis, tolerance_deg=20,
                                finger_tolerance_deg=10)
                if not self.kin.solve(qpos, arm, target, restarts=4)[3]:
                    return False
        return True

    def end_episode(self):
        self.episode_active = False

    def task_state(self):
        return self.tasks.task_state(self.m, self.d)

    def privileged_poses(self):
        """Object poses for tests and the operator only; never routed to the agent."""
        objects = getattr(self.tasks, 'OBJECTS', FRUITS)
        bowls = getattr(self.tasks, 'BOWL_NAMES', ('bowl',))
        poses = {n: self.d.body(n).xpos.round(4).tolist() for n in (*bowls, *objects)}
        for name in ('banana', 'toy_car'):
            if name in objects:
                poses[f'{name}_yaw'] = float(
                    Rotation.from_matrix(self.d.body(name).xmat.reshape(3, 3)).as_euler('xyz')[2])
        return poses

    def close(self):
        for r in (self._renderer, self._depth, self._stream_renderer):
            try:
                if r is not None:
                    r.close()
            except Exception:
                pass
        self._renderer = self._depth = self._stream_renderer = None
