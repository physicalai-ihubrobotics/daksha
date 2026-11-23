"""Hand-written baseline controller for the fruit-to-bowl task, using only the agent's tools.

Operator reference, NOT given to the agent: it proves the task is solvable through the same
MCP tools and perception the agent gets (head-camera colour masks + depth back-projection; no
privileged poses), and it exercises the formal evaluator. Run it like an agent's controller:

    cp -r examples/baseline_controller runtime/sessions/<name>/agent-workspace/code/baseline
    MUJOCO_GL=egl PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/evaluate_controller.py --name <name> --project baseline
"""
import io
import json
import math

import numpy as np
from PIL import Image

TABLE_Z = 0.85
CLEAR_POSE = {'left': [0.12, 0.46, 1.12], 'right': [0.12, -0.46, 1.12]}  # arms out of the head camera's view
GRASP_TCP_Z = TABLE_Z + 0.014
PREGRASP_DZ = 0.13
RELEASE_TCP_Z = TABLE_Z + 0.17
MAX_ATTEMPTS = 3


class StepFailed(RuntimeError):
    pass


def _json(reply):
    return json.loads(next(b['text'] for b in reply['content'] if b.get('type') == 'text'))


def observe_world(ctx):
    reply = ctx.call('daksha_observe', cameras=['world'])
    meta = _json(reply)
    img = np.asarray(Image.open(io.BytesIO(ctx.images(reply)[0])).convert('RGB')).astype(int)
    return meta['frame_id'], img


def masks(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(-1), img.min(-1)
    sat = (mx - mn)/np.maximum(mx, 1)
    return {
        'banana': (r > 170) & (g > 150) & (b < 90) & (abs(r - g) < 70),
        'orange': (r > 170) & (g > 70) & (g < 160) & (b < 60) & (r - g > 60),
        'apple': (r > 120) & (g < 60) & (b < 60),
        'bowl': (mn > 175) & (sat < 0.12),
    }


def points(ctx, frame_id, mask, n=60, seed=0):
    vs, us = np.nonzero(mask)
    if len(us) < 15:
        return None
    idx = np.random.default_rng(seed).choice(len(us), size=min(n, len(us)), replace=False)
    px = [[float(us[i]), float(vs[i])] for i in idx]
    reply = ctx.value('daksha_pixel_to_position', frame_id=frame_id, camera='world', pixels=px, window=1)
    pts = np.array([p['position'] for p in reply['points'] if p['valid']])
    if len(pts) < 8:
        return None
    pts = pts[(pts[:, 2] > TABLE_Z + 0.003) & (pts[:, 2] < TABLE_Z + 0.20)]  # drop table, arm and noise
    return pts if len(pts) >= 8 else None


def perceive(ctx):
    """{name: {'centre': xyz, 'yaw': rad (banana)}} from a fresh head-camera frame."""
    frame_id, img = observe_world(ctx)
    found = {}
    for name, mask in masks(img).items():
        pts = points(ctx, frame_id, mask, n=120 if name == 'bowl' else 60)
        if pts is None:
            continue
        xy = np.median(pts[:, :2], axis=0) if name != 'bowl' else None
        if name == 'bowl':
            # Rim and inner-wall points: centre = mean of the xy extent of the white ring.
            ring = pts[pts[:, 2] > TABLE_Z + 0.02]
            if len(ring) < 8:
                continue
            xy = (ring[:, :2].min(0) + ring[:, :2].max(0))/2
        entry = {'centre': [float(xy[0]), float(xy[1]), float(np.max(pts[:, 2]))], 'n': int(len(pts)), 'frame': frame_id}
        if name == 'banana':
            c = pts[:, :2] - pts[:, :2].mean(0)
            w, v = np.linalg.eigh(c.T @ c)
            axis = v[:, -1]
            entry['yaw'] = float(math.atan2(axis[1], axis[0]))
            entry['centre'][:2] = pts[:, :2].mean(0).tolist()
        found[name] = entry
    return found


def clear_view(ctx):
    """Swing both arms outward (straight lines from home) so the head camera sees the table."""
    for arm in ('left', 'right'):
        try:
            move(ctx, arm=arm, position=CLEAR_POSE[arm], approach=[0, 0, -1], tolerance_deg=60, speed=0.2)
        except RuntimeError:
            pass  # observing with an arm in view is worse, not fatal


def move(ctx, **kw):
    try:
        return ctx.value('daksha_move_eef', **kw)
    except RuntimeError as exc:
        if kw.get('path') == 'joint':
            raise
        return ctx.value('daksha_move_eef', **{**kw, 'path': 'joint'})


def pick(ctx, name, obj):
    x, y, _ = obj['centre']
    arm = 'left' if y > 0 else 'right'
    orient = {'approach': [0, 0, -1], 'tolerance_deg': 30}
    if name == 'banana':
        yaw = obj['yaw']
        orient.update(finger_axis=[-math.sin(yaw), math.cos(yaw), 0.0], finger_tolerance_deg=25)
    ctx.call('daksha_gripper', arm=arm, command='open')
    move(ctx, arm=arm, position=[x, y, GRASP_TCP_Z + PREGRASP_DZ], **orient)
    move(ctx, arm=arm, position=[x, y, GRASP_TCP_Z], speed=0.08)
    g = ctx.value('daksha_gripper', arm=arm, command='close', force=25)
    if not g['stalled_before_target'] or g['width_m'] < 0.015:
        ctx.call('daksha_gripper', arm=arm, command='open')
        move(ctx, arm=arm, position=[0, 0, PREGRASP_DZ], relative=True, speed=0.1)
        raise StepFailed(f'{name}: empty grasp (width {g["width_m"]:.3f})')
    move(ctx, arm=arm, position=[0, 0, PREGRASP_DZ], relative=True, speed=0.08)
    g = ctx.value('daksha_gripper', arm=arm, command='close', force=25)  # re-read the grip after lifting
    if g['width_m'] < 0.015:
        raise StepFailed(f'{name}: slipped while lifting')
    return arm


def place(ctx, arm, bowl):
    bx, by, _ = bowl['centre']
    side = 1 if arm == 'left' else -1
    move(ctx, arm=arm, position=[bx, by + side*0.025, RELEASE_TCP_Z], approach=[0, 0, -1], tolerance_deg=50, path='joint')
    ctx.call('daksha_gripper', arm=arm, command='open')
    move(ctx, arm=arm, position=[0, 0, 0.06], relative=True, speed=0.1)
    ctx.call('daksha_home', arm=arm)


def in_bowl(obj, bowl):
    return math.hypot(obj['centre'][0] - bowl['centre'][0], obj['centre'][1] - bowl['centre'][1]) < 0.085


def main(ctx):
    report = {'placed': [], 'failures': []}
    ctx.call('daksha_home', arm='both')
    for _ in range(3*MAX_ATTEMPTS):
        clear_view(ctx)
        scene = perceive(ctx)
        ctx.call('daksha_home', arm='both')
        if 'bowl' not in scene:
            report['failures'].append('bowl not found')
            break
        todo = [n for n in ('orange', 'apple', 'banana') if n in scene and not in_bowl(scene[n], scene['bowl'])]
        if not todo:
            report['ok'] = True
            return report
        name = todo[0]
        try:
            arm = pick(ctx, name, scene[name])
            place(ctx, arm, scene['bowl'])
            report['placed'].append(name)
        except RuntimeError as exc:  # StepFailed and tool errors
            report['failures'].append(str(exc)[:300])
            try:
                ctx.call('daksha_gripper', arm='left', command='open')
                ctx.call('daksha_gripper', arm='right', command='open')
                ctx.call('daksha_home', arm='both')
            except RuntimeError:
                pass
    report['ok'] = False
    return report
