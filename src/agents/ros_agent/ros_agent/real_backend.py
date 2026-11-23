"""Real-robot backend for the local VLM pipeline (local_daksha_rax/local_pipeline)
and RAX's daksha_ros backend.

Implements the `daksha_*` tools the MuJoCo backend provides, on the real Daksha:

  daksha_observe           world camera frame + its calibrated pose
  daksha_pixel_to_position pixels -> robot-frame points: aligned depth if live,
                           else the camera ray hitting a plane just above the table
  daksha_move_eef          -> /aruco_reach/move_to_pose   (aruco_hand_eye action)
  daksha_move_joints       -> /aruco_reach/move_joints
  daksha_home              -> /aruco_reach/move_joints named "home" (gen2 HomeMoveService pose)
  daksha_gripper           -> /aruco_reach/gripper

Motion is NOT done here: target positions go over ROS 2 to aruco_hand_eye's
aruco_reach, the only /joint_cmd sender, which solves IK, plans a straight-line
trajectory, checks self-collision and streams /joint_cmd under its execute switch
(dry run by default) with lag / stall / stale stops. An interrupt cancels the goal
(= STOP). This backend adds the table / workspace checks and refuses to plan
positions before the table height is known.

Robot frame for the tools = the URDF `world` link (x forward, y left, z up).
"""

import base64
import io
import math
import os
import threading
import time
import types

import numpy as np

from aruco_hand_eye import solver
from aruco_hand_eye.aruco_reach import ArmIK

ARMS = ('left', 'right')


def _arm_joints(arm):
    return [f'{arm}_joint_{i}' for i in range(1, 8)]


class RosRobotBackend:
    """See module docstring. Construct with the PipelineNode (ROS I/O) and its config."""

    is_real = True

    def __init__(self, node, cfg, config=None, realtime=1.0):
        from local_pipeline import robot as R          # skills module (table geometry, MotionStopped)
        self.R = R
        self.n = node
        self.c = cfg
        self.interrupt = threading.Event()
        m = cfg.get('motion', {})
        self.o = {'joint_speed_max': float(m.get('max_joint_speed', 0.15)),
                  'control_hz': float(m.get('rate_hz', 50.0))}
        self.tasks = types.SimpleNamespace(OBJECTS=list(cfg.get('objects', ['apple', 'orange', 'banana'])),
                                           BOWL_NAMES=list(cfg.get('containers', ['bowl'])))
        urdf = node.urdf_path
        self.ik = {arm: ArmIK(urdf, f'{arm}_tcp', _arm_joints(arm)) for arm in ARMS}   # FK only
        self.motion = node.motion            # aruco_hand_eye.motion_client.MotionClient
        self._last_approach = {}             # arm -> approach (TCP x, world) that worked last
        self.frames = {}                 # frame_id -> dict(rgb, depth, K, D, T_world_cam)
        self.frames_dir = None           # set (e.g. by the RAX agent): frames also saved on disk like the sim
        self._seq = 0

    # ------------------------------------------------------------------ pipeline API
    CAMERA_ALIASES = {'left': 'left_wrist', 'right': 'right_wrist'}   # sim names -> real camera names

    def _aliases(self, args):
        # the simulator calls the wrist cameras 'left' / 'right': accept both namings
        if isinstance(args.get('cameras'), list):
            args['cameras'] = [self.CAMERA_ALIASES.get(c, c) for c in args['cameras']]
        if isinstance(args.get('camera'), str):
            args['camera'] = self.CAMERA_ALIASES.get(args['camera'], args['camera'])
        return args

    def validate(self, name, args):
        self._aliases(args)
        if 'arm' in args and args['arm'] not in (*ARMS, 'both'):
            raise ValueError('arm must be "left" or "right"')
        if 'speed' in args and not (isinstance(args['speed'], (int, float)) and args['speed'] > 0):
            raise ValueError('speed must be a positive number')

    MOTION_TOOLS = ('daksha_move_eef', 'daksha_move_eef_both', 'daksha_move_joints', 'daksha_home', 'daksha_gripper')

    def call(self, name, args):
        args = self._aliases(dict(args))
        if name in self.MOTION_TOOLS and hasattr(self.n, 'motor_state'):
            ms = self.n.motor_state()
            if ms['stop']:
                hot = ', '.join(f"{h['arm']} motor {h['id']} {h['temp_c']:.0f} C" for h in ms['hot'])
                raise ValueError(f'Refused: motors too hot ({hot}; stop at {ms["stop_c"]:.0f} C). Let them cool down.')
        # other critical safety alerts that must stop new motion (collision alerts do not: the arm must
        # be able to move away; they are shown to the agent and the operator)
        sm = getattr(self.n, 'safety', None)
        if name in self.MOTION_TOOLS and sm is not None and not sm.c.get('ignore_warnings'):
            arms = set(ARMS) if args.get('arm') in (None, 'both') else {args['arm']}
            # a motor fault blocks only the arm that has it (a lost left gripper must not stop the right arm)
            blocking = [x for x in sm.critical if x['key'].startswith(('human', 'battery', 'data:joints'))
                        or (x['key'].startswith('fault:') and x['key'].split(':')[1] in arms)]
            if blocking:
                raise ValueError('Refused for safety: ' + '; '.join(x['msg'] for x in blocking))
        handler = getattr(self, '_' + name.removeprefix('daksha_'), None)
        if handler is None:
            raise PermissionError(f'{name} is not available on the real robot')
        return handler(args)

    def joint_state(self):
        return {j: self.n.joints.get(j) for a in ARMS for j in _arm_joints(a)}

    def reset_episode(self, seed=None):
        pass

    def task_state(self):
        return {}

    def close(self):
        self.interrupt.set()

    # ------------------------------------------------------------------ helpers
    def _table_z(self):
        z = self.n.table_z
        if z is None:
            raise ValueError('Refused: the table height is not set. Put an ArUco marker flat on the table and '
                             'press "Measure table" (or set scene.table_z in config/pipeline.yaml).')
        return z

    def _q(self, arm):
        if time.monotonic() - self.n.joint_t > 0.5:
            raise ValueError('Refused: no recent /joint_states')
        q = [self.n.joints.get(j) for j in _arm_joints(arm)]
        if any(v is None for v in q):
            raise ValueError(f'Refused: /joint_states is missing {arm} arm joints')
        return np.array(q, dtype=float)

    def remap_to_arm(self, p_world, arm):
        """A world-camera position (computed with the main calibration) as the given arm sees it:
        re-expressed with that arm's own world-camera calibration (pipeline world_calibration_<arm>)."""
        f = getattr(self.n, 'arm_calibration', None)
        T_main, T_arm = self.n.T_base_cam, (f(arm) if f else None)
        if T_main is None or T_arm is None or np.allclose(T_main, T_arm):
            return [float(v) for v in p_world]
        Twb = self.n.T_world_base
        p = Twb @ T_arm @ np.linalg.inv(T_main) @ np.linalg.inv(Twb) @ np.r_[np.asarray(p_world, float), 1.0]
        return [float(v) for v in p[:3]]

    # ------------------------------------------------------------------ wrist-camera approach
    def wrist_track_start(self, arm, name):
        """Start the wrist tracker (one vision-model detection, then CSRT) on this arm's camera."""
        tr = (getattr(self.n, 'wrist_trackers', None) or {}).get(arm)
        if tr is None:
            raise ValueError('no wrist tracker (LLM off?)')
        st = tr.state()
        if not (st.get('status') == 'tracking' and (st.get('query') or '') == name.replace('_', ' ')):
            tr.start(f'{arm}_wrist', query=name.replace('_', ' '))
        s = tr.wait_detected(timeout=60.0)
        if not s.get('box'):
            raise ValueError(f'{name} not found in the {arm} wrist camera ({s.get("message") or s.get("status")})')
        self.n.wrist_focus = {'arm': arm, 'name': name}
        return s

    def wrist_track_stop(self):
        # the boxes stay on in every view (box_everywhere); only the approach focus ends
        self.n.wrist_focus = None

    def wrist_measure(self, arm, fingers_cam=None):
        """One wrist look: object (tracker box + depth) and fingertips (depth) in the camera frame,
        their difference in the WORLD frame (rotation only), and the object's world position (for
        the dynamic TF). fingers_cam: reuse a fingertip point measured earlier."""
        from . import wrist_servo
        snap = self.n.wrist_snapshot(arm)
        if snap is None:
            raise ValueError(f'no fresh {arm} wrist frame / depth')
        tr = (getattr(self.n, 'wrist_trackers', None) or {}).get(arm)
        st = tr.state() if tr is not None else {}
        box = st.get('box') if st.get('status') == 'tracking' else None
        if box is None:
            return {'ok': False, 'why': f'wrist tracker: {st.get("status")}'}
        if fingers_cam is None:
            fingers_cam, info = wrist_servo.fingertips(snap['depth'], snap['K'], snap['D'])
            if fingers_cam is None:
                return {'ok': False, 'why': info}
        B, z = wrist_servo.object_point(snap['depth'], snap['K'], snap['D'], box)
        if B is None:
            return {'ok': False, 'why': z}
        T = snap['T_world_cam']
        err_w = T[:3, :3] @ (B - fingers_cam)
        obj_w = (T @ np.r_[B, 1.0])[:3]
        self.n.wrist_obj = {'arm': arm, 'name': self.n.wrist_focus['name'] if self.n.wrist_focus else 'object',
                            'world': obj_w.tolist(), 't': time.monotonic()}
        return {'ok': True, 'err_world': err_w, 'fingers_cam': fingers_cam, 'object_cam': B, 'box': box,
                'object_world': obj_w, 'object_depth_m': z}

    def at_home(self, arm, tol=0.10):
        """True when the arm is within tol (rad, every joint) of the gen2 home pose."""
        try:
            from gen2.HomeMoveService import DEFAULT_HOME_POSITION as H
            h = [float(v) for v in H]
            home = np.array(h[0:7] if arm == 'left' else h[8:15])
            return bool(np.max(np.abs(self._q(arm) - home)) < tol)
        except Exception:
            return False

    def _tcp(self, arm, q=None):
        T = self.ik[arm].fk(self.ik[arm].clamp(self._q(arm) if q is None else q))
        quat = np.round(solver.mat_to_quat(T[:3, :3]), 4).tolist()          # [x, y, z, w]
        # same keys as the simulator: quat (+ legacy 'orientation'), approach = TCP x, finger_axis = TCP y
        return {'position': np.round(T[:3, 3], 4).tolist(), 'quat': quat, 'orientation': quat,
                'approach': np.round(T[:3, 0], 3).tolist(), 'finger_axis': np.round(T[:3, 1], 3).tolist()}

    def _check_target(self, p):
        ws = self.c.get('workspace', {})
        tz = self._table_z()
        clear = float(ws.get('min_clearance_m', 0.005))
        x, y, z = (float(v) for v in p)
        xr, yr = ws.get('x', [0.10, 0.85]), ws.get('y', [-0.70, 0.70])
        if not (xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]):
            raise ValueError(f'Refused: target ({x:.3f}, {y:.3f}) outside the workspace x {xr} y {yr}')
        if z < tz + clear:
            raise ValueError(f'Refused: target z {z:.3f} is below the table ({tz:.3f}) + {clear:.3f} m clearance')
        if z > float(ws.get('z_max', 1.7)):
            raise ValueError(f'Refused: target z {z:.3f} above the workspace limit')

    def _target_rotation(self, arm, args, T_now):
        quat = args.get('quat', args.get('orientation'))      # sim/RAX schema uses 'quat' [x, y, z, w]
        if quat is not None:
            from scipy.spatial.transform import Rotation as Rot
            return Rot.from_quat(np.asarray(quat, float)).as_matrix(), True
        if args.get('approach') is not None:
            x = np.asarray(args['approach'], float)
            x /= np.linalg.norm(x)
            y = np.asarray(args['finger_axis'], float) if args.get('finger_axis') is not None else T_now[:3, 1]
            y = y - x * (x @ y)
            if np.linalg.norm(y) < 1e-6:
                y = np.cross(T_now[:3, 2], x)
            y /= np.linalg.norm(y)
            return np.column_stack([x, y, np.cross(x, y)]), True
        return T_now[:3, :3].copy(), False

    # ------------------------------------------------------------------ tools
    WRIST = {'right_wrist': 'right', 'left_wrist': 'left'}

    def _observe(self, args):
        """One frame per camera: 'world' (head, calibrated eye-to-hand) and 'right_wrist' /
        'left_wrist' (eye-in-hand; their pose comes from the arm's joints at capture).
        frame_id is the world frame's id; wrist frames are '<frame_id>_<camera>'."""
        cams = list(args.get('cameras') or ['world'])
        self._seq += 1
        base = f'r{self._seq:05d}'
        from PIL import Image
        meta, images, ids = {'frame_id': None, 'cameras': [], 'frames': {}}, [], {}
        for cam_name in cams:
            if cam_name == 'world':
                cam = self.n.camera_snapshot()
                if cam is None:
                    raise ValueError('Refused: no recent world camera image')
                if self.n.T_base_cam is None:
                    raise ValueError(f'Refused: {self.n.calib_problem}')
                T = self.n.T_world_base @ self.n.T_base_cam
                fid = base
            elif cam_name in self.WRIST:
                cam = self.n.wrist_snapshot(self.WRIST[cam_name]) if hasattr(self.n, 'wrist_snapshot') else None
                if cam is None:
                    raise ValueError(f'Refused: no recent {cam_name} camera image (or its calibration / joints)')
                T = cam['T_world_cam']
                fid = f'{base}_{cam_name}'
            else:
                raise ValueError(f'unknown camera {cam_name}; use world, right_wrist, left_wrist')
            rgb = cam['bgr'][:, :, ::-1].copy()
            self.frames[fid] = {'rgb': rgb, 'depth': cam.get('depth'), 'K': cam['K'], 'D': cam['D'],
                                'T_world_cam': T, 'camera': cam_name}
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format='PNG')
            images.append({'camera': cam_name, 'data': base64.b64encode(buf.getvalue()).decode()})
            meta['cameras'].append(cam_name)
            meta['frames'][cam_name] = fid
            meta['frame_id'] = meta['frame_id'] or fid
            meta.setdefault('size', [int(rgb.shape[1]), int(rgb.shape[0])])
            meta.setdefault('depth_by_camera', {})[cam_name] = cam.get('depth') is not None
            meta['depth'] = any(meta['depth_by_camera'].values())
        for old in sorted(self.frames)[:-30]:
            del self.frames[old]
        if self.frames_dir is not None:
            try:
                meta['saved'] = self._save_frames(base, meta['frames'])
            except OSError as exc:
                meta['saved_error'] = str(exc)
        return meta, images

    def _save_frames(self, base, frames):
        """frames/<base>/<camera>.png (+ <camera>_depth.png, uint16 mm, 0 = invalid) and meta.json in the
        simulator's format, so controller code reads real and simulated frames the same way. Adds
        'distortion' (OpenCV k1 k2 p1 p2 k3): NOT zero on the real cameras."""
        import json as _json
        import pathlib
        from PIL import Image
        folder = pathlib.Path(self.frames_dir) / base
        folder.mkdir(parents=True, exist_ok=True)
        cams = {}
        Twb_inv = np.linalg.inv(self.n.T_world_base)
        for cam, fid in frames.items():
            f = self.frames[fid]
            Image.fromarray(f['rgb']).save(folder / f'{cam}.png')
            K = np.asarray(f['K'], float)
            h, w = f['rgb'].shape[:2]
            T = f['T_world_cam']                                    # world (robot frame of the tools)
            entry = {'rgb': f'frames/{base}/{cam}.png', 'frame_id': fid,
                     'intrinsics': {'width': int(w), 'height': int(h), 'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
                                    'cx': float(K[0, 2]), 'cy': float(K[1, 2]), 'K': K.round(6).tolist()},
                     'distortion': np.asarray(f['D'], float).round(8).tolist(),
                     'extrinsics': {'position': T[:3, 3].round(5).tolist(),
                                    'rotation_camera_to_robot': T[:3, :3].round(6).tolist(),
                                    'quat_camera_to_robot': solver.mat_to_quat(T[:3, :3]).round(6).tolist(),
                                    'T_base_cam': T.round(6).tolist(),
                                    'T_base_link_cam': (Twb_inv @ T).round(6).tolist(),
                                    'source': 'hand-eye calibration' + (' + arm joints' if cam != 'world' else ''),
                                    'convention': 'OpenCV: x right, y down, z forward; robot frame = URDF world; '
                                                  'undistort pixels with "distortion" before using K'}}
            if f.get('depth') is not None:
                mm = np.nan_to_num(np.asarray(f['depth'], float) * 1000.0, nan=0.0)
                Image.fromarray(np.clip(mm, 0, 65535).astype(np.uint16)).save(folder / f'{cam}_depth.png')
                entry['depth'] = f'frames/{base}/{cam}_depth.png'
            cams[cam] = entry
        (folder / 'meta.json').write_text(_json.dumps({'frame_id': base, 'cameras': cams}, indent=1))
        return f'frames/{base}'

    def _ray(self, f, uv, z_world):
        """World point where pixel uv of frame f meets the plane z = z_world (None if it does not)."""
        import cv2
        xn = cv2.undistortPoints(np.array([[uv]], float), f['K'], f['D']).reshape(2)
        T = f['T_world_cam']
        d = T[:3, :3] @ np.array([xn[0], xn[1], 1.0])
        if abs(d[2]) < 1e-6:
            return None
        t = (z_world - T[2, 3]) / d[2]
        return T[:3, 3] + t * d if t > 0 else None

    def _footprint(self, args):
        """Where an object stands, from its bounding box in a frame without depth.

        Fits a box model standing on the table (centre x, y, square footprint d, height h; turned to
        face the camera) whose projection matches the detected box on all four edges. Projecting the
        box centre onto a fixed plane instead puts tall objects several cm too far away."""
        import cv2
        from scipy.optimize import least_squares
        f = self.frames.get(args['frame_id'])
        if f is None:
            raise ValueError(f'unknown frame_id {args["frame_id"]}')
        tz = self._table_z()
        box = np.array([float(v) for v in args['bbox']])
        T = f['T_world_cam']
        Tinv = np.linalg.inv(T)
        rvec, _ = cv2.Rodrigues(Tinv[:3, :3])
        tvec = Tinv[:3, 3]
        c0 = self._ray(f, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], tz + 0.03)
        if c0 is None:
            raise ValueError('the box centre does not look at the table in this view')
        view = c0[:2] - T[:2, 3]
        view /= max(np.linalg.norm(view), 1e-6)
        side = np.array([-view[1], view[0]])
        unit = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (0., 1.)])

        def bbox_of(p):
            cx, cy, d, hh = p
            pts = [np.r_[np.array([cx, cy]) + view * u[0] * d + side * u[1] * d, tz + u[2] * hh] for u in unit]
            uv, _ = cv2.projectPoints(np.array(pts), rvec, tvec, f['K'], f['D'])
            uv = uv.reshape(-1, 2)
            return np.r_[uv.min(0), uv.max(0)]
        r = least_squares(lambda p: bbox_of(p) - box, [c0[0], c0[1], 0.06, 0.06],
                          bounds=([-2, -2, 0.015, 0.01], [2, 2, 0.40, 0.40]), x_scale=[0.05, 0.05, 0.02, 0.02])
        cx, cy, d, hh = r.x
        fit_px = float(np.sqrt(np.mean(r.fun ** 2)))
        return {'centre': [round(float(cx), 4), round(float(cy), 4), round(float(tz + hh), 4)],
                'size_m': [round(float(d), 3), round(float(d), 3), round(float(hh), 3)],
                'fit_px': round(fit_px, 1), 'camera': f.get('camera', 'world')}, []

    def _pixel_to_position(self, args):
        import cv2
        f = self.frames.get(args['frame_id'])
        if f is None:
            raise ValueError(f'unknown frame_id {args["frame_id"]}')
        tz = self._table_z()
        plane_z = (float(args['z']) if args.get('z') is not None
                   else tz + float(self.c.get('scene', {}).get('plane_offset_m', 0.03)))
        T = f['T_world_cam']
        pix = np.asarray(args['pixels'], float).reshape(-1, 2)
        rays = cv2.undistortPoints(pix.reshape(-1, 1, 2), f['K'], f['D']).reshape(-1, 2)
        win = int(args.get('window', 1))
        depth = f['depth']
        pts = []
        for (u, v), (xn, yn) in zip(pix, rays):
            d_cam = np.array([xn, yn, 1.0])
            p = None
            if depth is not None:
                iu, iv = int(round(u)), int(round(v))
                patch = depth[max(0, iv - win):iv + win + 1, max(0, iu - win):iu + win + 1]
                vals = patch[np.isfinite(patch) & (patch > 0)]
                if vals.size:
                    z = float(np.median(vals))
                    p = (T @ np.r_[d_cam * z, 1.0])[:3]
            else:
                d_w = T[:3, :3] @ d_cam
                if abs(d_w[2]) > 1e-6:
                    t = (plane_z - T[2, 3]) / d_w[2]
                    if t > 0:
                        p = T[:3, 3] + t * d_w
            pts.append({'pixel': [float(u), float(v)], 'valid': p is not None,
                        'position': [] if p is None else np.round(p, 4).tolist()})
        return {'points': pts, 'source': 'depth' if depth is not None else f'table plane + {plane_z - tz:.3f} m'}, []

    @staticmethod
    def _unreachable(r):
        if r['ok'] or r['dry_run'] or r['stopped']:
            return False
        msg = r['message']
        return any(k in msg for k in ('cannot reach', 'IK gets', 'no solution', 'no straight-line path'))

    def _approach_candidates(self, arm, Rg):
        """Rotations whose approach (TCP x) is tilted 15 / 30 deg away from the requested one
        in 8 directions (finger roll kept as close as possible), plus the last approach that
        worked for this arm if it is within the allowed cone. Closest first."""
        max_dev = float(self.c.get('approach_search_deg', 30.0))
        if max_dev <= 0:
            return []
        a0 = Rg[:, 0]
        u = np.cross(a0, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(a0, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(a0, u)
        dirs = []
        last = self._last_approach.get(arm)
        if last is not None:
            dev = float(np.degrees(np.arccos(np.clip(last @ a0, -1, 1))))
            if 1.0 < dev <= max_dev:
                dirs.append((dev, last, f'{dev:.0f} deg from the requested one (as last time)'))
        for tilt in (15.0, 30.0):
            if tilt > max_dev:
                break
            t = np.radians(tilt)
            for az in range(0, 360, 45):
                w = np.radians(az)
                dirs.append((tilt, np.cos(t) * a0 + np.sin(t) * (np.cos(w) * u + np.sin(w) * v),
                             f'{tilt:.0f} deg from the requested one'))
        out = []
        for _, a, how in sorted(dirs, key=lambda d: d[0]):
            a = a / np.linalg.norm(a)
            y = Rg[:, 1] - a * (a @ Rg[:, 1])
            y /= np.linalg.norm(y)
            out.append((np.column_stack([a, y, np.cross(a, y)]), how))
        return out

    def _motion_result(self, r, what):
        """aruco_reach result -> pipeline semantics."""
        if r['dry_run']:
            raise self.R.DryRun(r['message'].replace('DRY RUN (execute is off): ', '') + ' (execute is off)')
        if r['stopped']:
            raise self.R.MotionStopped(r['message'])
        if not r['ok']:
            raise ValueError(f'Refused: {what}: {r["message"]}')
        return r

    def _move_eef(self, args):
        arm = args['arm']
        ik = self.ik[arm]
        T_now = ik.fk(ik.clamp(self._q(arm)))                     # world
        pos = np.asarray(args['position'], float)
        if args.get('relative'):
            pos = T_now[:3, 3] + pos
        self._check_target(pos)
        Rg, specified = self._target_rotation(arm, args, T_now)
        # Big wrist turn (e.g. home, hand forward -> grasp, hand tilted down)? Go to the arm's
        # ready pose first: a fixed, collision-checked pose with the hand already pointing the
        # grasp way, reachable within aruco_reach's per-move joint limit. Then plan from there.
        ready = (self.c.get('ready_joints') or {}).get(arm)
        turn = float(np.degrees(solver.rot_angle(T_now[:3, :3].T @ Rg))) if specified else 0.0
        if ready and turn > float(self.c.get('ready_trigger_deg', 45.0)):
            self.n.log(f'{arm} hand must turn {turn:.0f} deg: going to the {arm} ready pose first')
            r = self.motion.move_joints(arm, list(ready), interrupt=self.interrupt)
            self._motion_result(r, f'{arm} ready pose')
            T_now = ik.fk(ik.clamp(self._q(arm)))
            Rg, specified = self._target_rotation(arm, args, T_now)
        G = np.eye(4)
        G[:3, :3], G[:3, 3] = Rg, pos
        Gb = solver.inv(self.n.T_world_base) @ G                  # base_link for the action
        m = self.c.get('motion', {})
        floor_w = self._table_z() + float(self.c.get('workspace', {}).get('min_clearance_m', 0.005))
        floor_b = floor_w - self.n.T_world_base[2, 3]
        def attempt(Rw):
            Rb = (solver.inv(self.n.T_world_base)[:3, :3] @ Rw) if Rw is not None else None
            return self.motion.move_to_pose(
                arm, Gb[:3, 3], solver.mat_to_quat(Rb) if Rb is not None else None,
                keep_orientation=Rb is None, tolerance_deg=float(args.get('tolerance_deg', 0.0)),
                path=args.get('path', 'linear'),
                tcp_speed=min(float(args.get('speed', 1e9)), float(m.get('max_tcp_speed', 0.03))),
                joint_speed=min(float(args.get('joint_speed', 1e9)), float(m.get('max_joint_speed', 0.15))),
                min_tcp_z=floor_b, interrupt=self.interrupt)

        r = attempt(Rg if specified else None)
        if specified and self._unreachable(r):
            # The best hand angle depends on where the target is (e.g. close to the body the
            # forward tilt folds the wrist): try nearby approach directions, closest first,
            # starting with the one that worked last for this arm (keeps descents straight).
            for Rc, how in self._approach_candidates(arm, Rg):
                r2 = attempt(Rc)
                if not self._unreachable(r2):
                    r = r2
                    if r['ok'] or r['dry_run']:
                        self.n.log(f'{arm}: requested hand angle unreachable here; using one {how}')
                        self._last_approach[arm] = Rc[:, 0].copy()
                    break
        elif specified and (r['ok'] or r['dry_run']):
            self._last_approach[arm] = Rg[:, 0].copy()
        self._motion_result(r, f'{arm} TCP to {np.round(pos, 3).tolist()}')
        result = {'arm': arm, 'duration_s': round(r['duration_s'], 2), 'tcp': self._tcp(arm),
                  'target_position': np.round(pos, 4).tolist(),
                  'position_error_m': round(float(r['position_error_m']), 4)}
        return result, []

    def _move_joints(self, args):
        arm = args['arm']
        r = self.motion.move_joints(arm, list(args['positions']), joint_speed=float(
            args.get('speed', self.c.get('motion', {}).get('max_joint_speed', 0.15))), interrupt=self.interrupt)
        self._motion_result(r, f'{arm} joints')
        return {'arm': arm, 'tcp': self._tcp(arm), 'duration_s': round(r['duration_s'], 2)}, []

    def _home(self, args):
        arm = args.get('arm', 'both')
        r = self.motion.move_joints(arm, named='home', interrupt=self.interrupt)
        self._motion_result(r, f'{arm} home')
        arms = ARMS if arm == 'both' else (arm,)
        out = {a: {'tcp': self._tcp(a)} for a in arms}
        out['moved'] = [{'arm': a, 'tcp': out[a]['tcp']} for a in arms]           # simulator's key
        return out, []

    def _gripper(self, args):
        r = self.motion.gripper(args['arm'], args['command'], float(args.get('width', 0.0)),
                                interrupt=self.interrupt)
        self._motion_result(r, f'{args["arm"]} gripper')
        return {'arm': args['arm'], 'command': args['command'], 'width_m': round(r['width_m'], 4),
                'stalled_before_target': r['stalled_before_target'],
                # simulator keys: the real gripper is position-controlled and reports no force
                'force_n': None, 'force_limit_n': args.get('force'),
                'force_note': 'the real gripper does not measure force; use stalled_before_target and width_m'}, []

    # ------------------------------------------------------------------ 2.5: tools the simulator has
    def _check(self, args):
        """Dry run (no motion): can this arm's TCP reach the target? IK (position + orientation),
        joint limits, table / workspace limits. With 'positions' (7 joints): forward kinematics.
        Self-collision along the path is only checked by aruco_reach when the move is planned."""
        arm = args['arm']
        if args.get('positions') is not None:
            q = np.asarray(args['positions'], float)
            if q.shape != (7,):
                raise ValueError('positions must have 7 values')
            lim = self.ik[arm].limits
            inside = bool(np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1]))
            return {'arm': arm, 'tcp': self._tcp(arm, q), 'within_joint_limits': inside}, []
        T_now = self.ik[arm].fk(self.ik[arm].clamp(self._q(arm)))
        pos = np.asarray(args['position'], float)
        if args.get('relative'):
            pos = T_now[:3, 3] + pos
        try:
            self._check_target(pos)
        except ValueError as exc:
            return {'arm': arm, 'reachable': False, 'reason': str(exc)}, []
        Rg, specified = self._target_rotation(arm, args, T_now)
        G = np.eye(4)
        G[:3, :3], G[:3, 3] = Rg, pos
        tol = float(args.get('tolerance_deg', 5.0 if specified else 10.0))
        best = self.ik[arm].search(self._q(arm), G, 0.5 if specified else 0.2, max(tol, 1.0), 3.2,
                                   restarts=8, _no_step=True)
        if best is None:
            return {'arm': arm, 'reachable': False, 'reason': 'no IK solution'}, []
        ok = best['pos_err'] <= 0.01 and best['rot_err'] <= max(tol, 1.0)
        return {'arm': arm, 'reachable': bool(ok), 'position_error_m': round(float(best['pos_err']), 4),
                'orientation_error_deg': round(float(best['rot_err']), 2),
                'joints': np.round(best['q'], 4).tolist(), 'tcp': self._tcp(arm, best['q']),
                'note': 'IK + limits only; the path is checked for self-collision when it is planned'}, []

    def _move_eef_both(self, args):
        """Both TCPs: the two moves are started together (one thread each) and the call returns when
        both finished. Not one jointly planned trajectory as in the simulator: give both the same
        speed / distance for a matched motion (e.g. lifting an object held in two hands)."""
        import threading as _th
        common = {k: args[k] for k in ('relative', 'path', 'speed') if k in args}
        jobs = {arm: {**common, **args[arm], 'arm': arm} for arm in ARMS if args.get(arm)}
        if not jobs:
            raise ValueError('give "left" and/or "right" targets')
        out, err = {}, {}

        def run(arm, a):
            try:
                out[arm] = self._move_eef(a)[0]
            except Exception as exc:                 # keep the other arm's result
                err[arm] = exc
        ts = [_th.Thread(target=run, args=(arm, a)) for arm, a in jobs.items()]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if err:
            arm, exc = next(iter(err.items()))
            raise type(exc)(f'{arm}: {exc}' + (f' (the other arm: {out})' if out else ''))
        return {'moved': [{'arm': a, **r} for a, r in out.items()]}, []
