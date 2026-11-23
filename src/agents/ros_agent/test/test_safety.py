"""SafetyMonitor (safety_monitor.py) against a fake pipeline node: every check raises the right alert,
announcements are debounced, and critical alerts block motion in the real backend."""

import time
import types

import numpy as np
import pytest

pytest.importorskip('aruco_hand_eye')
pytest.importorskip('kinematics')


def node():
    from test_real_backend import FakeNode
    from aruco_hand_eye.urdf_kinematics import UrdfKinematics
    n = FakeNode()
    n.fk = UrdfKinematics(n.urdf_path)
    n.agent = None
    n.motors = {}
    n.motor_state = lambda: {'warn_c': 80.0, 'stop_c': 90.0, 'hot': [], 'stop': False, 'arms': {}}
    n.gripper_feedback = lambda held: {}
    n.camera_snapshot = lambda: {'bgr': np.full((480, 640, 3), 255, np.uint8), 'K': n.K, 'D': np.zeros(5)}
    return n


def mon(n, **cfg):
    from ros_agent.safety_monitor import SafetyMonitor
    return SafetyMonitor(n, {'safety': {'human_detect': False, **cfg}})


def keys(m):
    return {a['key']: a['level'] for a in m.check()}


def test_all_clear_on_a_normal_pose():
    n = node()
    m = mon(n)
    k = keys(m)
    assert not any(x.startswith(('collision', 'temp', 'fault', 'battery', 'data')) for x in k), k


def test_temperature_fault_battery_and_stale_data():
    n = node()
    m = mon(n)
    n.motor_state = lambda: {'warn_c': 80.0, 'stop_c': 90.0, 'stop': True, 'arms': {},
                             'hot': [{'arm': 'left', 'id': 4, 'temp_c': 92.0}, {'arm': 'right', 'id': 2, 'temp_c': 83.0}]}
    n.motors = {'right': {'t': time.monotonic(), 'motors': [{'id': 3, 'mos_c': 40, 'rotor_c': 40, 'error': 'Overcurrent'}]}}
    m.on_battery(types.SimpleNamespace(data=8.0))
    n.joint_t = time.monotonic() - 5.0                       # stale joints
    m._t0 -= 10.0                                             # past the startup grace
    k = keys(m)
    assert k['temp:left:4'] == 'critical' and k['temp:right:2'] == 'warn'
    assert k['fault:right:3'] == 'critical' and k['battery'] == 'critical' and k['data:joints'] == 'critical'


def test_collision_awareness_table_arms_and_blocked():
    n = node()
    m = mon(n)
    tcp = n.fk.transform('world', 'right_tcp', dict(n.joints))[:3, 3]
    n.table_z = float(tcp[2]) + 0.01                         # the hand is 1 cm BELOW the table top
    m.on_reach_status(types.SimpleNamespace(data='{"arms": {"right": {"last_msg": "right: reached goal"}}}'))
    assert not any(k.startswith('collision:blocked') for k in keys(m))   # what it showed at start = history
    m.on_reach_status(types.SimpleNamespace(data='{"arms": {"right": {"last_msg": '
                                                 '"right: STOPPED, arm is not following the command. Blocked or colliding?"}}}'))
    k = keys(m)
    assert k['collision:table:right'] == 'critical' and k['collision:blocked:right'] == 'critical'
    # the same aruco_reach message again is not a new event
    t0 = m.reach_msgs['right'][1]
    m.on_reach_status(types.SimpleNamespace(data='{"arms": {"right": {"last_msg": '
                                                 '"right: STOPPED, arm is not following the command. Blocked or colliding?"}}}'))
    assert m.reach_msgs['right'][1] == t0


def test_joint_limit_warning():
    n = node()
    m = mon(n)
    m.check()                                                # loads the collision model
    lo, hi = m._collision.robot.get_joint_limits('right_joint_6')
    n.joints['right_joint_6'] = hi - np.radians(1.0)
    assert keys(m).get('limit:right_joint_6') == 'info'


def test_people_are_detected_and_announcements_debounced():
    n = node()
    m = mon(n, human_detect=True)
    m._detect_people = lambda bgr: [[100, 50, 300, 470]]   # a person in view
    k = keys(m)
    assert k['human'] == 'critical'
    new, cleared = m.to_announce()
    assert any(a['key'] == 'human' for a in new)
    new, _ = m.to_announce()
    assert not any(a['key'] == 'human' for a in new)       # not repeated within alert_repeat_s
    m._detect_people = lambda bgr: []
    m._people['t'] = 0
    m.check()
    _, cleared = m.to_announce()
    assert 'human' in cleared


def test_critical_alerts_block_motion_in_the_backend():
    from test_real_backend import CFG
    from ros_agent.real_backend import RosRobotBackend
    n = node()
    m = mon(n, human_detect=True)
    m._detect_people = lambda bgr: [[100, 50, 300, 470]]
    m.check()
    n.safety = m
    b = RosRobotBackend(n, CFG)
    with pytest.raises(ValueError, match='Refused for safety: person near the robot'):
        b.call('daksha_move_eef', {'arm': 'right', 'position': [0, 0, 0.02], 'relative': True})
    assert n.motion.calls == []
    b.call('daksha_observe', {'cameras': ['world']})       # sensing still works


def test_a_motor_fault_blocks_only_its_own_arm():
    from test_real_backend import CFG
    from ros_agent.real_backend import RosRobotBackend
    n = node()
    m = mon(n)
    n.motors = {'left': {'t': time.monotonic(), 'motors': [{'id': 8, 'mos_c': 40, 'rotor_c': 40, 'error': 'Communication Lost'}]}}
    m.check()
    n.safety = m
    b = RosRobotBackend(n, CFG)
    with pytest.raises(ValueError, match='left arm motor 8'):
        b.call('daksha_move_eef', {'arm': 'left', 'position': [0, 0, 0.02], 'relative': True})
    b.call('daksha_move_eef', {'arm': 'right', 'position': [0, 0, 0.02], 'relative': True})   # the right arm still moves
    assert len(n.motion.calls) == 1


def test_ignore_warnings_never_blocks_motion():
    from test_real_backend import CFG
    from ros_agent.real_backend import RosRobotBackend
    n = node()
    m = mon(n, human_detect=True, ignore_warnings=True)
    m._detect_people = lambda bgr: [[100, 50, 300, 470]]
    assert keys(m)['human'] == 'critical'                   # still shown
    n.safety = m
    RosRobotBackend(n, CFG).call('daksha_move_eef', {'arm': 'right', 'position': [0, 0, 0.02], 'relative': True})
    assert len(n.motion.calls) == 1
