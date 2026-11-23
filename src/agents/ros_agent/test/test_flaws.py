"""Regression tests for the issues listed in src/Flaws (agent_skills review).

Category 1: the code in agent_skills/object-localization/references/geometry.md is executed as written
and checked on the failure cases the review names. Category 2: the real backend now matches the
simulator's tool contract. Categories 3-5: the docs, runtime and launcher no longer contradict the code.
"""

import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
RAX = HERE.parent / 'local_daksha_rax'
SKILLS = RAX / 'agent_skills'
sys.path.insert(0, str(RAX))
sys.path.insert(0, str(HERE))


# ----------------------------------------------------------------------------- category 1: geometry.md
@pytest.fixture(scope='module')
def geo():
    text = (SKILLS / 'object-localization/references/geometry.md').read_text()
    ns = {}
    for block in re.findall(r'```python\n(.*?)```', text, re.S):
        exec(block, ns)
    return ns


def test_on_plane_handles_up_down_parallel_and_behind(geo):
    on_plane = geo['on_plane']
    # camera above, looking down (lists, as loaded from meta.json)
    p = on_plane([0.3, 0.0, 1.5], [0.0, 0.0, -1.0], 0.85)
    assert np.allclose(p, [0.3, 0.0, 0.85])
    # 1.1(1): camera BELOW the plane looking UP -> valid point (was rejected)
    p = on_plane([0.3, 0.0, 0.5], [0.0, 0.6, 0.8], 0.85)
    assert p is not None and np.isclose(p[2], 0.85)
    # 1.1(2): camera below, looking down -> the plane is behind it -> None (was a point behind the camera)
    assert on_plane([0.3, 0.0, 0.5], [0.0, 0.0, -1.0], 0.85) is None
    # 1.1(3): parallel -> None, no ZeroDivisionError
    assert on_plane([0.3, 0.0, 1.5], [1.0, 0.0, 0.0], 0.85) is None


def test_triangulate_rejects_parallel_and_ghost_points(geo):
    tri = geo['triangulate']
    target = np.array([0.5, 0.1, 0.9])
    o1, o2 = np.array([0.0, 0.3, 1.5]), np.array([0.0, -0.3, 1.5])
    p, gap, ang = tri(o1.tolist(), (target - o1).tolist(), o2.tolist(), (target - o2).tolist())   # lists OK
    assert np.allclose(p, target, atol=1e-9) and gap < 1e-9 and ang > 10
    # 1.2(1): parallel rays -> None, inf (no ZeroDivisionError / NaN)
    p, gap, _ = tri([0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0])
    assert p is None and gap == np.inf
    # 1.2(2): rays that only meet BEHIND the cameras -> None (was a ghost point)
    p, gap, _ = tri(o1, -(target - o1), o2, -(target - o2))
    assert p is None


def test_back_project_rejects_invalid_depth_and_handles_distortion(geo):
    bp, ray, normalized = geo['back_project'], geo['ray'], geo['normalized']
    K = [[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]
    T = np.eye(4).tolist()                                   # nested lists, as in meta.json (1.4)
    for bad in (0, 0.0, float('nan'), float('inf'), -0.1, None):
        assert bp(320, 240, bad, K, T) is None               # 1.3: never the camera's own position
    assert np.allclose(bp(320, 240, 0.5, K, T), [0, 0, 0.5])
    o, d = ray(320, 240, K, T)
    assert np.allclose(o, 0) and np.allclose(d, [0, 0, 1])
    # 1.5: with real distortion the pixel maps through cv2.undistortPoints, not K alone
    import cv2
    D = [-0.0517, 0.0529, 0.00095, -0.00025, -0.0179]
    u, v = 600.0, 400.0
    ref = cv2.undistortPoints(np.array([[[u, v]]]), np.array(K), np.array(D)).reshape(2)
    assert np.allclose(normalized(u, v, K, D)[:2], ref)
    assert not np.allclose(normalized(u, v, K, D)[:2], normalized(u, v, K)[:2])


# ----------------------------------------------------------------------------- category 2: real backend
pytest.importorskip('aruco_hand_eye')
pytest.importorskip('kinematics')


@pytest.fixture
def real(tmp_path):
    from test_real_backend import backend
    b, n = backend()
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.45, -0.10, 1.10]
    n.wrist_snapshot = lambda arm: {'bgr': np.zeros((480, 848, 3), np.uint8), 'K': n.K, 'D': np.full(5, 0.01),
                                    'depth': np.full((480, 848), 0.3, np.float32), 'T_world_cam': T}
    b.frames_dir = str(tmp_path / 'frames')
    return b, n, tmp_path


def test_sim_camera_names_are_accepted(real):
    b, n, _ = real
    meta, imgs = b.call('daksha_observe', {'cameras': ['world', 'right']})            # 2.1: 'right' -> right_wrist
    assert meta['cameras'] == ['world', 'right_wrist'] and len(imgs) == 2
    fid = meta['frames']['right_wrist']
    r, _ = b.call('daksha_pixel_to_position', {'frame_id': fid, 'camera': 'right', 'pixels': [[320, 240]]})
    assert r['points'][0]['valid']


def test_tcp_has_the_simulators_keys(real):
    b, n, _ = real
    t = b._tcp('right')                                                               # 2.2
    assert set(t) >= {'position', 'quat', 'approach', 'finger_axis'}
    a = np.array(t['approach'])
    assert abs(np.linalg.norm(a) - 1) < 1e-2


def test_gripper_force_home_moved(real):
    b, n, _ = real
    g, _ = b.call('daksha_gripper', {'arm': 'right', 'command': 'close', 'force': 25})   # 2.4
    assert 'force_n' in g and g['force_n'] is None
    h, _ = b.call('daksha_home', {'arm': 'both'})                                      # 2.7
    assert [m['arm'] for m in h['moved']] == ['left', 'right'] and 'tcp' in h['right']


def test_check_and_move_eef_both_exist_on_the_real_robot(real):
    b, n, _ = real
    tcp = np.array(b._tcp('right')['position'])
    r, _ = b.call('daksha_check', {'arm': 'right', 'position': (tcp + [0, 0, 0.02]).tolist()})   # 2.5
    assert r['reachable'] and r['position_error_m'] < 0.01 and len(r['joints']) == 7
    r, _ = b.call('daksha_check', {'arm': 'right', 'position': [1.5, 0, 1.0]})
    assert not r['reachable'] and 'workspace' in r['reason']
    fk, _ = b.call('daksha_check', {'arm': 'right', 'positions': [0.3, -0.5, 0, 0.9, 0.5, 0.2, 0]})
    assert 'tcp' in fk and fk['within_joint_limits'] in (True, False)
    n.motion.calls.clear()
    r, _ = b.call('daksha_move_eef_both', {'relative': True, 'left': {'position': [0, 0, 0.03]},
                                           'right': {'position': [0, 0, 0.03]}})
    assert sorted(m['arm'] for m in r['moved']) == ['left', 'right']
    assert sorted(c[1] for c in n.motion.calls if c[0] == 'move_to_pose') == ['left', 'right']


def test_frames_are_saved_like_the_simulator(real):
    b, n, tmp = real
    meta, _ = b.call('daksha_observe', {'cameras': ['world', 'left_wrist']})            # 2.8
    folder = tmp / meta['saved']
    m = json.loads((folder / 'meta.json').read_text())
    assert set(m['cameras']) == {'world', 'left_wrist'}
    for cam, e in m['cameras'].items():
        assert (tmp / e['rgb']).exists() and len(e['distortion']) == 5
        assert np.array(e['extrinsics']['T_base_cam']).shape == (4, 4)
        assert np.array(e['intrinsics']['K']).shape == (3, 3)
    assert (tmp / m['cameras']['left_wrist']['depth']).exists()                         # wrist depth saved
    import PIL.Image
    d = np.asarray(PIL.Image.open(tmp / m['cameras']['left_wrist']['depth']))
    assert d.dtype == np.uint16 and int(np.median(d)) == 300                            # millimetres


def test_ros_tool_schemas_accept_the_sim_arguments():
    from services.daksha_ros.backend import DakshaRosBackend
    b = DakshaRosBackend(config=None)
    tools = {t['name']: t for t in b.tools()}
    assert {'daksha_check', 'daksha_move_eef_both', 'daksha_measure_table', 'daksha_footprint'} <= set(tools)
    props = tools['daksha_move_eef']['inputSchema']['properties']
    assert 'finger_tolerance_deg' in props                                             # 2.3
    assert 'force' in tools['daksha_gripper']['inputSchema']['properties']             # 2.4
    assert 'right' in tools['daksha_observe']['inputSchema']['properties']['cameras']['items']['enum']
    hidden = DakshaRosBackend(config=None, hide_tools=('daksha_move_eef',))
    assert 'daksha_move_eef' not in {t['name'] for t in hidden.tools()}
    with pytest.raises(PermissionError):
        hidden.call('daksha_move_eef', {'arm': 'right', 'position': [0, 0, 0]})


# ----------------------------------------------------------------------------- categories 3-5
def test_runtime_finds_the_socket_without_the_env_var(tmp_path, monkeypatch):
    from auto_research_agent.api import runtime
    monkeypatch.delenv('RAX_MCP_SOCKET', raising=False)
    (tmp_path / 'robot.sock').write_text('')
    ws = tmp_path / 'agent-workspace' / 'code'
    ws.mkdir(parents=True)
    monkeypatch.chdir(ws)
    assert runtime._find_socket() == str(tmp_path / 'robot.sock')                      # 5.1: no KeyError
    monkeypatch.setenv('RAX_MCP_SOCKET', '/x/y.sock')
    assert runtime._find_socket() == '/x/y.sock'


def test_memory_headers_match_the_documented_format():
    fmt = (SKILLS / 'experience-memory/references/format.md').read_text()
    launcher = (RAX / 'harness/launcher.py').read_text()
    for header in ('| Task phase | Preconditions | What worked | Visible effect | Evidence |',
                   '| Skill | Tags | Confirmation | Last confirmed |'):
        assert header in fmt and header in launcher                                    # 5.3


def test_docs_no_longer_contradict_the_code():
    vlm = (SKILLS / 'vlm-perception/SKILL.md').read_text()
    assert 'takes image bytes' not in vlm and 'reasoning effort' not in vlm.replace('reasoning-effort', '')  # 3.1, 3.2
    assert '"detections"' in vlm and 'frame_id' in vlm                                   # 3.3
    auto = (SKILLS / 'autonomous-control/SKILL.md').read_text()
    assert 'this session has no\n  VLM tool' not in auto and 'no VLM tool' not in auto  # 3.4
    assert 'experiments/run_controller.py' not in auto and 'python -m api.runtime run' in auto   # 5.1
    assert 'json.loads(reply["content"][0]["text"])' not in auto and 'ctx.value(' in auto        # 5.2
    ri = (SKILLS / 'robot-interface/SKILL.md').read_text()
    for s in ('left_wrist', 'daksha_measure_table', 'daksha_footprint', 'DRY RUN', 'force_n'):   # 2.1, 2.6, 4.3
        assert s in ri, s
    grasp = (SKILLS / 'grasping/SKILL.md').read_text()
    assert 'robot frame' in grasp and 'd * approach' in grasp                            # 4.1
    assert '15-30 deg' in grasp                                                          # 4.2
    manual = (SKILLS / 'manual-robot-control/SKILL.md').read_text()
    assert 'DRY RUN' in manual and 'daksha_measure_table' in manual                      # 4.3, 2.6
