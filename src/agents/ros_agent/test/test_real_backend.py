"""RosRobotBackend (real_backend.py) against a fake node + fake motion client.

No robot, no ROS graph: checks perception maths, the local refusals, what goes
to aruco_reach's actions (base_link target, kept orientation, table floor) and
how dry-run / stop / refusal results map to the pipeline's exceptions.
"""

import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip('aruco_hand_eye')
pytest.importorskip('kinematics')

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'local_daksha_rax')))


class FakeMotion:
    """Records goals; returns aruco_reach-style results."""

    def __init__(self):
        self.calls, self.next = [], {}

    def _r(self, **kw):
        r = {'ok': True, 'dry_run': False, 'stopped': False, 'message': 'reached', 'reached': np.eye(4),
             'position_error_m': 0.001, 'orientation_error_deg': 0.1, 'duration_s': 1.0, 'width_m': 0.04,
             'stalled_before_target': True}
        r.update(self.next)
        r.update(kw)
        return r

    def move_to_pose(self, arm, position, quat_xyzw=None, **kw):
        self.calls.append(('move_to_pose', arm, np.asarray(position, float), quat_xyzw, kw))
        return self._r()

    def move_joints(self, arm, positions=None, named='', **kw):
        self.calls.append(('move_joints', arm, positions, named, kw))
        return self._r()

    def gripper(self, arm, command, width=0.0, **kw):
        self.calls.append(('gripper', arm, command, width, kw))
        if command == 'open':                       # an opened gripper reads ~87 mm, nothing held
            return self._r(width_m=0.087, stalled_before_target=False)
        return self._r()


class FakeNode:
    def __init__(self, table_z=0.80):
        from ament_index_python.packages import get_package_share_directory
        from aruco_hand_eye.urdf_kinematics import UrdfKinematics, rpy_to_matrix
        share = get_package_share_directory('daksha_description_full_body')
        os.environ['ROS_PACKAGE_PATH'] = os.path.dirname(share)
        self.urdf_path = os.path.join(share, 'urdf', 'robot.urdf')
        self.T_world_base = UrdfKinematics(self.urdf_path).transform('world', 'base_link', {})
        self.table_z = table_z
        self.motion = FakeMotion()
        q = {'right': [0.3, -0.5, 0.0, 0.9, 0.5, 0.2, 0.0], 'left': [-0.3, -0.5, 0.0, -0.9, -0.5, 0.2, 0.0]}
        self.joints = {f'{a}_joint_{i + 1}': v for a, vs in q.items() for i, v in enumerate(vs)}
        self.joint_t = time.monotonic() + 3600          # always "fresh"
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(-2.4, 0.0, -1.5708)
        T[:3, 3] = [0.15, 0.0, 1.10]
        self.T_base_cam = T
        self.K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])

    def camera_snapshot(self):
        return {'bgr': np.zeros((480, 640, 3), np.uint8), 'K': self.K, 'D': np.zeros(5), 'depth': None}

    def log(self, msg):
        pass


CFG = {'objects': ['apple'], 'containers': ['bowl'], 'scene': {'plane_offset_m': 0.03},
       'workspace': {'x': [0.10, 0.85], 'y': [-0.70, 0.70], 'z_max': 1.7, 'min_clearance_m': 0.005},
       'motion': {'max_tcp_speed': 0.03, 'max_joint_speed': 0.15}}


def backend(**kw):
    from ros_agent.real_backend import RosRobotBackend
    n = FakeNode(**kw)
    return RosRobotBackend(n, CFG), n


def test_pixel_to_position_hits_the_table_plane():
    b, n = backend()
    meta, imgs = b.call('daksha_observe', {'cameras': ['world']})
    p_world = np.array([0.45, -0.10, n.table_z + 0.03])
    pc = np.linalg.inv(n.T_world_base @ n.T_base_cam) @ np.r_[p_world, 1]
    u, v = (n.K @ (pc[:3] / pc[2]))[:2]
    r, _ = b.call('daksha_pixel_to_position', {'frame_id': meta['frame_id'], 'camera': 'world', 'pixels': [[u, v]]})
    assert r['points'][0]['valid'] and np.linalg.norm(np.array(r['points'][0]['position']) - p_world) < 1e-3


def test_local_refusals_never_reach_aruco_reach():
    b, n = backend()
    tcp = np.array(b._tcp('right')['position'])
    with pytest.raises(ValueError, match='below the table'):
        b.call('daksha_move_eef', {'arm': 'right', 'position': [tcp[0], tcp[1], n.table_z - 0.02]})
    with pytest.raises(ValueError, match='outside the workspace'):
        b.call('daksha_move_eef', {'arm': 'right', 'position': [1.5, 0.0, 1.0]})
    n.table_z = None
    with pytest.raises(ValueError, match='table height is not set'):
        b.call('daksha_move_eef', {'arm': 'right', 'position': [0, 0, 0.03], 'relative': True})
    assert n.motion.calls == []


def test_move_goal_is_base_link_keeps_orientation_and_carries_the_table_floor():
    b, n = backend()
    tcp_w = np.array(b._tcp('right')['position'])
    goal_w = tcp_w + [0.03, -0.02, 0.04]
    r, _ = b.call('daksha_move_eef', {'arm': 'right', 'position': goal_w.tolist()})
    kind, arm, pos_b, quat, kw = n.motion.calls[-1]
    assert kind == 'move_to_pose' and arm == 'right' and quat is None and kw['keep_orientation']
    assert np.allclose(pos_b, goal_w - n.T_world_base[:3, 3], atol=1e-6)       # world -> base_link
    assert np.isclose(kw['min_tcp_z'], n.table_z + 0.005 - n.T_world_base[2, 3])
    assert kw['interrupt'] is b.interrupt and r['position_error_m'] == 0.001
    b.call('daksha_move_eef', {'arm': 'right', 'position': goal_w.tolist(), 'approach': [0, 0, -1]})
    assert n.motion.calls[-1][3] is not None                                    # orientation sent
    b.call('daksha_move_eef', {'arm': 'right', 'position': goal_w.tolist(), 'quat': [0, 0, 0, 1]})
    q_sent = n.motion.calls[-1][3]                                              # RAX/sim 'quat' key honoured
    assert q_sent is not None and np.allclose(np.abs(q_sent), [0, 0, 0, 1], atol=1e-6)


def test_results_map_to_pipeline_outcomes():
    from local_pipeline.robot import DryRun, MotionStopped
    b, n = backend()
    rel = {'arm': 'right', 'position': [0, 0, 0.03], 'relative': True}
    n.motion.next = {'ok': False, 'dry_run': True, 'message': 'DRY RUN (execute is off): planned x'}
    with pytest.raises(DryRun):
        b.call('daksha_move_eef', rel)
    n.motion.next = {'ok': False, 'stopped': True, 'message': 'stopped (STOP / cancel)'}
    with pytest.raises(MotionStopped):
        b.call('daksha_move_eef', rel)
    n.motion.next = {'ok': False, 'message': 'refused: self-collision'}
    with pytest.raises(ValueError, match='self-collision'):
        b.call('daksha_move_eef', rel)


def test_home_and_gripper_go_to_their_actions():
    b, n = backend()
    b.call('daksha_home', {'arm': 'both'})
    assert n.motion.calls[-1][:4] == ('move_joints', 'both', None, 'home')
    g, _ = b.call('daksha_gripper', {'arm': 'left', 'command': 'close'})
    assert n.motion.calls[-1][:3] == ('gripper', 'left', 'close')
    assert g['stalled_before_target'] and g['width_m'] == 0.04


def test_big_wrist_turn_goes_to_ready_pose_first():
    from ros_agent.real_backend import RosRobotBackend
    n = FakeNode()
    ready = [0.9606, -0.9678, 0.9650, 1.7636, 0.6070, -0.0001, -1.3031]
    b = RosRobotBackend(n, {**CFG, 'ready_joints': {'right': ready}, 'ready_trigger_deg': 45})
    tcp_w = np.array(b._tcp('right')['position'])
    # small move, orientation kept: no ready step
    b.call('daksha_move_eef', {'arm': 'right', 'position': (tcp_w + [0, 0, 0.03]).tolist()})
    assert [c[0] for c in n.motion.calls] == ['move_to_pose']
    # hand must turn to point down-and-forward: ready pose first, then the move
    n.motion.calls.clear()
    b.call('daksha_move_eef', {'arm': 'right', 'position': (tcp_w + [0, 0, 0.03]).tolist(),
                               'approach': [0.5, 0.0, -0.866], 'tolerance_deg': 20})
    kinds = [c[0] for c in n.motion.calls]
    assert kinds == ['move_joints', 'move_to_pose'] and n.motion.calls[0][2] == ready


class FakeVLM:
    """Detects one object by projecting its true position into whichever image it gets
    (world 640x480 or wrist 848x480); answers the grasp check with `held`."""
    model = 'fake-vlm'

    def __init__(self, n, true_xyz, held=True):
        self.n, self.p, self.held = n, np.asarray(true_xyz, float), held
        self.wrist_T = None

    def _uv(self, T):
        pc = np.linalg.inv(T) @ np.r_[self.p, 1]
        return (self.n.K @ (pc[:3] / pc[2]))[:2]

    def detect(self, rgb, labels):
        T = self.wrist_T if rgb.shape[1] == 848 else self.n.T_world_base @ self.n.T_base_cam_true
        u, v = self._uv(T)
        return [{'label': labels[0], 'bbox': [u - 15, v - 15, u + 15, v + 15], 'point': [u, v]}]

    def ask_json(self, rgb, question, **kw):
        return {'held': self.held, 'reason': 'fake'}


def test_observe_includes_the_wrist_cameras():
    b, n = backend()
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])           # looking straight down
    T[:3, 3] = [0.45, -0.10, 1.10]
    n.wrist_snapshot = lambda arm: {'bgr': np.zeros((480, 848, 3), np.uint8), 'K': n.K, 'D': np.zeros(5),
                                    'depth': None, 'T_world_cam': T}
    meta, imgs = b.call('daksha_observe', {'cameras': ['world', 'right_wrist']})
    assert meta['cameras'] == ['world', 'right_wrist'] and len(imgs) == 2
    fid = meta['frames']['right_wrist']
    r, _ = b.call('daksha_pixel_to_position', {'frame_id': fid, 'camera': 'right_wrist',
                                               'pixels': [[320, 240]], 'z': 0.83})
    assert np.allclose(r['points'][0]['position'], [0.45, -0.10, 0.83], atol=1e-3)


def test_pick_corrects_the_world_estimate_with_the_wrist_camera():
    """World camera calibration off by ~3 cm: the wrist camera's look from above moves the grasp
    onto the object; the grasp check uses the gripper AND the wrist camera."""
    from local_pipeline import robot as R
    from local_pipeline.robot import Skills
    b, n = backend()
    R.apply_scene(table_z=n.table_z)                   # as PipelineNode._set_table does
    b.c = {**b.c, 'wrist_refine': True}                # off by default until the wrist cameras are recalibrated
    true = np.array([0.45, -0.12, n.table_z + 0.03])
    n.T_base_cam_true = n.T_base_cam.copy()
    n.T_base_cam = n.T_base_cam.copy()
    n.T_base_cam[:3, 3] += [0.03, 0.0, 0.0]            # calibration error: world estimate ~3 cm off
    vlm = FakeVLM(n, true)

    def wrist(arm):                                    # camera above the current hover target, looking down
        q = n.motion.calls[-1][2] if n.motion.calls and n.motion.calls[-1][0] == 'move_to_pose' else true
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [q[0] - 0.03, q[1], 1.15]
        vlm.wrist_T = T
        return {'bgr': np.zeros((480, 848, 3), np.uint8), 'K': n.K, 'D': np.zeros(5), 'depth': None, 'T_world_cam': T}
    n.wrist_snapshot = wrist
    sk = Skills(b, vlm, log=lambda m: None)
    found = sk.perceive(['box'])
    world_err = np.hypot(*(np.array(found['box']['centre'][:2]) - true[:2]))
    assert world_err > 0.015                           # the world camera alone is off
    n.motion.next = {'stalled_before_target': True, 'width_m': 0.05}
    assert sk.pick('box', found['box'], arm='right') == 'right'
    moves = [c for c in n.motion.calls if c[0] == 'move_to_pose']
    T_wb = n.T_world_base
    grasp_w = moves[-2][2] + T_wb[:3, 3]               # the descent (before the lift), base_link -> world
    assert np.hypot(*(grasp_w[:2] - true[:2])) < 0.006, (grasp_w, true)
    assert 'wrist camera sees it' in sk.last_grasp['why'] and sk.last_grasp['picked']


def test_unreachable_hand_angle_tries_nearby_angles_and_remembers():
    b, n = backend()
    tcp_w = np.array(b._tcp('right')['position'])
    goal = (tcp_w + [0, 0, 0.03]).tolist()
    seq = iter([{'ok': False, 'message': 'refused: right TCP cannot reach [..] (world): IK gets 19 mm / 0 deg off'},
                {'ok': False, 'message': 'refused: right TCP cannot reach: no solution'},
                {}])                                   # third try works

    real_mtp = n.motion.move_to_pose

    def mtp(*a, **kw):
        n.motion.next = next(seq, {})
        return real_mtp(*a, **kw)
    n.motion.move_to_pose = mtp
    b.call('daksha_move_eef', {'arm': 'right', 'position': goal, 'approach': [0.5, 0.0, -0.866],
                               'tolerance_deg': 20})
    calls = [c for c in n.motion.calls if c[0] == 'move_to_pose']
    assert len(calls) == 3
    from scipy.spatial.transform import Rotation as R
    a_req, a_used = (R.from_quat(calls[i][3]).as_matrix()[:, 0] for i in (0, 2))
    dev = np.degrees(np.arccos(np.clip(a_req @ a_used, -1, 1)))
    assert 14 < dev < 31                                   # a nearby angle, not something wild
    # next move (e.g. the descent) tries the remembered angle right after the requested one
    n.motion.calls.clear()
    seq = iter([{'ok': False, 'message': 'refused: cannot reach'}, {}])
    b.call('daksha_move_eef', {'arm': 'right', 'position': goal, 'approach': [0.5, 0.0, -0.866],
                               'tolerance_deg': 20})
    second = R.from_quat(n.motion.calls[1][3]).as_matrix()[:, 0]
    assert np.allclose(second, a_used, atol=1e-6)
    # refusals that are not about reach are never retried
    n.motion.calls.clear()
    seq = iter([{'ok': False, 'message': 'refused: self-collision at 40%: right_link_5 / base_link'}])
    with pytest.raises(ValueError, match='self-collision'):
        b.call('daksha_move_eef', {'arm': 'right', 'position': goal, 'approach': [0.5, 0.0, -0.866]})
    assert len(n.motion.calls) == 1


def test_wrist_servo_finds_fingertips_and_object_in_depth():
    from ros_agent import wrist_servo
    K = np.array([[435.0, 0, 424], [0, 435.0, 240], [0, 0, 1]])
    D = np.zeros(5)
    depth = np.full((480, 848), 0.60, np.float32)            # table far away
    depth[300:480, 300:400] = np.linspace(0.15, 0.09, 180)[:, None]   # left finger (tip at the top, 15 cm)
    depth[300:480, 460:560] = np.linspace(0.15, 0.09, 180)[:, None]   # right finger
    depth[180:260, 380:480] = 0.30                            # the object, 30 cm away
    F, info = wrist_servo.fingertips(depth, K, D)
    assert F is not None and abs(F[2] - 0.15) < 0.01 and abs(info['gap_m'] - 160 / 435 * 0.15) < 0.01
    B, z = wrist_servo.object_point(depth, K, D, [380, 180, 480, 260])
    assert abs(z - 0.30) < 1e-3 and abs(B[0] - (430 - 424) / 435 * 0.30) < 2e-3
    # no fingers in view -> a reason, not a crash
    F2, why = wrist_servo.fingertips(np.full((480, 848), 0.6, np.float32), K, D)
    assert F2 is None and 'fingers' in why


def test_pick_subtasks_retry_an_empty_close_and_log_each_step():
    """Close comes back empty once: the "close the gripper" sub-task opens, backs off, moves in again
    and closes again; every sub-task is reported to the task log."""
    from local_pipeline import robot as R
    from local_pipeline.robot import Skills
    b, n = backend()
    R.apply_scene(table_z=n.table_z)
    closes = iter([{'stalled_before_target': False, 'width_m': 0.001},        # 1st close: empty
                   {'stalled_before_target': True, 'width_m': 0.05},          # 2nd close: got it
                   {'stalled_before_target': True, 'width_m': 0.05}])         # re-close after lift
    real_gr = n.motion.gripper

    def gripper(arm, command, width=0.0, **kw):
        n.motion.next = next(closes) if command == 'close' else {'width_m': 0.087, 'stalled_before_target': False}
        return real_gr(arm, command, width, **kw)
    n.motion.gripper = gripper
    steps = []
    sk = Skills(b, None, log=lambda m: None)
    sk.on_step = lambda label, status, detail: steps.append((label, status))
    sk.wrist_sees_held = lambda *a: True
    sk._held_from_gripper = lambda *a: None
    assert sk.pick('box', {'centre': [0.40, -0.15, n.table_z + 0.06]}, arm='right', style='tilted') == 'right'
    labels = [s for s, _ in steps]
    assert labels == ['see the box with the world camera', 'open the gripper', 'go near the box', 'move in to the box',
                      'close the gripper', 'lift the box', 'check the grasp'], labels
    assert all(st == 'done' for _, st in steps)
    assert [c[2] for c in n.motion.calls if c[0] == 'gripper'].count('close') == 3   # empty, good, re-close


def test_visible_plan_steps_and_subtasks():
    """daksha_plan -> steps; a skill call starts the next step and lists its sub-tasks; a retried
    sub-task shows its tries; the step ends done."""
    import collections
    import threading
    from local_pipeline import robot as R
    from local_pipeline.agent import LocalAgent
    from local_pipeline.robot import Skills
    ag = LocalAgent.__new__(LocalAgent)
    ag.lock, ag.state, ag._task = threading.Lock(), {'plan': None}, {'task': 'pick and place'}
    ag.chat, ag.logs = collections.deque(), collections.deque()
    ag.plan_set(['pick up the box with the right hand', 'put it down on the table'])
    assert ag.chat[-1]['who'] == 'plan' and '1. pick up the box' in ag.chat[-1]['text']
    b, n = backend()
    R.apply_scene(table_z=n.table_z)
    closes = iter([{'stalled_before_target': False, 'width_m': 0.001}, {'stalled_before_target': True, 'width_m': 0.05},
                   {'stalled_before_target': True, 'width_m': 0.05}])
    real_gr = n.motion.gripper

    def gripper(arm, command, width=0.0, **kw):
        if command == 'close':
            n.motion.next = next(closes)
        return real_gr(arm, command, width, **kw)
    n.motion.gripper = gripper
    sk = Skills(b, None, log=lambda m: None)
    sk.on_plan, sk.wrist_sees_held, sk._held_from_gripper = ag._on_skill_plan, (lambda *a: True), (lambda *a: None)
    ag.plan_step_start('pick up box right')
    sk.pick('box', {'centre': [0.40, -0.15, n.table_z + 0.06]}, arm='right', style='tilted')
    ag.plan_step_end(True)
    p = ag.state['plan']
    s0 = p['steps'][0]
    assert s0['label'] == 'pick up the box with the right hand' and s0['status'] == 'done'
    assert [s['label'] for s in s0['subs']][:3] == ['see the box with the world camera', 'open the gripper', 'go near the box']
    assert all(s['status'] == 'done' for s in s0['subs'])
    assert next(s for s in s0['subs'] if s['label'] == 'close the gripper')['tries'] == 2
    assert p['steps'][1]['status'] == 'pending'


def test_failed_subtask_explains_alternatives_and_plan_revisions_keep_history():
    import collections
    import threading
    from local_pipeline import robot as R
    from local_pipeline.agent import LocalAgent
    from local_pipeline.robot import Skills, StepFailed
    b, n = backend()
    R.apply_scene(table_z=n.table_z)
    real_mtp = n.motion.move_to_pose

    def unreachable(*a, **kw):                          # every hand move is out of reach
        r = real_mtp(*a, **kw)
        return {**r, 'ok': False, 'message': 'refused: right TCP cannot reach [..]: no solution'}
    n.motion.move_to_pose = unreachable
    sk = Skills(b, None, log=lambda m: None)
    sk._held_from_gripper = lambda *a: None
    with pytest.raises(StepFailed) as ei:
        sk.pick('box', {'centre': [0.40, -0.15, n.table_z + 0.06]}, arm='right', style='side')
    msg = str(ei.value)
    assert 'go near the box' in msg and 'What could work instead' in msg and "style='top'" in msg
    ag = LocalAgent.__new__(LocalAgent)
    ag.lock, ag.state, ag._task = threading.Lock(), {'plan': None}, {'task': 'pick'}
    ag.chat, ag.logs = collections.deque(), collections.deque()
    ag.plan_set(['pick up the box with the right hand (side)'])
    ag.plan_step_start('pick up box right'); ag.plan_step_end(False, 'go near failed')
    ag.plan_set(['pick up the box with the right hand from the top'], reason='side approach out of reach')
    p = ag.state['plan']
    assert p['rev'] == 2 and p['reason'] == 'side approach out of reach'
    assert p['history'][0]['steps'][0]['status'] == 'failed'
    assert ag.chat[-1]['text'].startswith('Plan 2 (replanned: side approach out of reach)')
