"""Build the Daksha MuJoCo scene: the robot, worktable and selectable sorting tasks.

The robot comes from assets/daksha/robot.urdf, a copy of
~/uni/daksha/src/daksha_description_full_body (never edited there). What this module adds,
and why:

- Position actuators on every arm joint and on each gripper's driven finger, standing in
  for Daksha's PosVelEff (MIT) controllers; ``gravcomp`` on robot bodies, standing in for
  the gravity-torque feed-forward (controllers/dynamics) that the real robot adds.
- The URDF's ``<mimic>`` finger joints become joint equality constraints.
- Finger collision meshes become box pads (the meshes' own bounding boxes) so grasps are
  stable; the visual meshes stay.
- Three cameras named like the real RealSense set (camera_launch/config/cameras.yaml):
  ``world`` (on the head, looking at the table), ``left`` and ``right`` (on the wrists,
  looking along the approach axis). The wrist placement is a simulation choice.
- Collision filter: robot geoms never collide with each other (contype 2, conaffinity 1);
  they do collide with the table, bowl and fruit (contype 1, conaffinity 1|2).
"""
from __future__ import annotations

import math
import pathlib
import re

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT/'assets'/'daksha'

ARMS = ('left', 'right')
ARM_JOINTS = {arm: [f'{arm}_joint_{i}' for i in range(1, 8)] for arm in ARMS}
# Driven finger joint per gripper (the one ros2_control commands) and its mimic partner.
GRIPPER_JOINT = {'left': 'left_gripper_left_joint', 'right': 'right_gripper_right_joint'}
GRIPPER_MIMIC = {'left': 'left_gripper_right_joint', 'right': 'right_gripper_left_joint'}
# Driven joint value when fully closed (open is 0 for both): left closes negative, right positive.
GRIPPER_CLOSED = {'left': -0.044, 'right': 0.044}
FINGER_BODIES = {'left': ('left_left_finger', 'left_right_finger'),
                 'right': ('right_left_finger', 'right_right_finger')}
TCP_BODY = {arm: f'{arm}_tcp' for arm in ARMS}
MOUNT_BODY = {arm: f'{arm}_gripper_mount' for arm in ARMS}

# Opening between the finger pads: 0.089 m with the driven joint at 0, minus 2 x travel.
PAD_GAP_OPEN = 0.089

# Counter-height table just in front of the torso (its front is at x ~0.13 m at this height):
# Daksha's arms reach ~0.65 m from shoulders 1.28 m up, so the work area is close and high.
TABLE = {'x0': 0.17, 'x1': 0.80, 'y0': -0.55, 'y1': 0.55, 'top': 0.85, 'thickness': 0.04}
BOWL = {'inner_radius': 0.095, 'wall': 0.008, 'height': 0.065, 'bottom': 0.008, 'mass': 0.35}
FRUITS = {
    # name: shape description; sizes in metres, masses in kg.
    'orange': {'kind': 'sphere', 'radius': 0.036, 'mass': 0.15, 'rgba': (1.0, 0.55, 0.05, 1)},
    'apple': {'kind': 'ellipsoid', 'size': (0.038, 0.038, 0.034), 'mass': 0.18, 'rgba': (0.80, 0.08, 0.10, 1)},
    'banana': {'kind': 'banana', 'radius': 0.018, 'length': 0.19, 'bend_deg': 55, 'mass': 0.12,
               'rgba': (0.98, 0.85, 0.15, 1)},
}
TOYS = {
    'toy_block': {'kind': 'box', 'size': (0.026, 0.026, 0.026), 'mass': 0.045,
                  'rgba': (0.12, 0.38, 0.95, 1)},
    'toy_ball': {'kind': 'sphere', 'radius': 0.027, 'mass': 0.04,
                 'rgba': (0.78, 0.16, 0.82, 1)},
    'toy_car': {'kind': 'car', 'length': 0.104, 'width': 0.064, 'height': 0.046, 'mass': 0.075,
                'rgba': (0.08, 0.78, 0.72, 1)},
}
SORTING_TASK = 'fruit_toy_sorting'
TRAY_TASK = 'tray_transport'

# --- tray_transport: carry a fruit tray from the pick table to a second table 5 m away ---
# The place table is a copy of TABLE shifted along -y (the robot's right) by TABLE_SPACING,
# centre to centre. The robot gets a planar mobile base (base_x, base_y, base_yaw) to drive there.
TABLE_SPACING = 5.0
PLACE_TABLE = {**TABLE, 'y0': TABLE['y0'] - TABLE_SPACING, 'y1': TABLE['y1'] - TABLE_SPACING}
# Blue plastic lab tray, 10" x 7" x 3" (254 x 178 x 76 mm outside at the rim). Body x is the
# long axis; the short end walls carry wide rim lips that act as handles for a two-handed carry.
TRAY = {'length': 0.254, 'width': 0.178, 'height': 0.076, 'wall': 0.005, 'bottom': 0.005,
        'flare_deg': 4.0, 'lip': 0.008, 'handle_lip': 0.020, 'lip_thickness': 0.004,
        'mass': 0.40, 'rgba': (0.20, 0.42, 0.95, 1)}
# Two-handed grasp (one hand per short end), as a person carries a full tray: the hand is
# half-closed before it descends (HANDLE_PRECLOSE_GAP between the pads, enough to straddle the end
# wall plus its handle lip), the fingertips (= TCP) go HANDLE_GRASP_DEPTH below the rim, and the grip
# sits HANDLE_Y off the tray's centre line, on the banana's side: the banana lies low, so the inner
# finger passes above it, while the orange and apple (which reach the rim) sit on the other half.
HANDLE_Y = 0.025
HANDLE_GRASP_DEPTH = 0.015
HANDLE_PRECLOSE_GAP = 0.060
# Tray pose on the pick table (x, y, yaw): long axis along world y so each hand takes one end;
# the handle grips (HANDLE_Y toward the robot) land at x ~0.37, the best-reached distance.
TRAY_HOME = (0.395, 0.0, math.pi/2)
# Drop-off zone (x, y, yaw) on the place table, mirrored from the pick pose.
TRAY_GOAL = (TRAY_HOME[0], TRAY_HOME[1] - TABLE_SPACING, math.pi/2)
# Fruit inside the tray, in the tray frame (x along the long axis, +y toward the robot): the
# banana along the near side (under the handle grips), orange and apple on the far side.
TRAY_FRUIT_SLOTS = {'orange': (-0.045, -0.037, 0.0), 'apple': (0.045, -0.037, 0.0),
                    'banana': (0.0, 0.035, 0.0)}
BASE_JOINTS = ('base_x', 'base_y', 'base_yaw')
# Base home (x, y, yaw): 3 cm back from the fixed-base pose, so the torso's convex collision hull
# clears the table edge (with a fixed base that touch was ignored; with a mobile base it pushes).
BASE_HOME = (-0.03, 0.0, 0.0)

HOME = {  # Ready posture: both TCPs ~0.20 m above the table at x=0.27, y=+-0.24, approach ~down.
    'left_joint_1': -1.253, 'left_joint_2': -0.974, 'left_joint_3': 1.279, 'left_joint_4': -1.694,
    'left_joint_5': -0.291, 'left_joint_6': 0.687, 'left_joint_7': 1.309,
    'right_joint_1': 1.315, 'right_joint_2': -0.947, 'right_joint_3': 1.363, 'right_joint_4': 1.799,
    'right_joint_5': 0.372, 'right_joint_6': 0.583, 'right_joint_7': -1.379,
}

# Collision classes (MuJoCo collides two geoms when ct1 & ca2 or ct2 & ca1):
# world and torso 1, left arm 2, right arm 4. An arm never collides with itself, but does
# with the other arm, the torso, the table, the bowl and the fruit.
WORLD_CONTYPE, WORLD_CONAFFINITY = 1, 7
ARM_CLASS = {'left': (2, 1 | 4), 'right': (4, 1 | 2)}


def _collision_class(body_name):
    for arm in ARMS:
        if body_name.startswith(arm + '_'):
            return ARM_CLASS[arm]
    return WORLD_CONTYPE, WORLD_CONAFFINITY


def _urdf_text():
    text = (ASSETS/'robot.urdf').read_text()
    text = text.replace('package://daksha_description_full_body/meshes/', '')
    compiler = (f'<mujoco><compiler meshdir="{ASSETS/"meshes"}" discardvisual="false" '
                'fusestatic="false" balanceinertia="true" strippath="false"/></mujoco>')
    return re.sub(r'(<robot[^>]*>)', r'\1\n' + compiler, text, count=1)


def _quat_from_euler(roll=0.0, pitch=0.0, yaw=0.0):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), 'xyz')
    return q


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _robot(spec):
    names = {b.name for b in spec.bodies}
    robot_bodies = [b for b in spec.bodies if b.name not in ('world',)]
    collision_dir = ASSETS/'meshes_collision'
    swapped = {}
    for body in robot_bodies:
        body.gravcomp = 1.0
        for geom in list(body.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if geom.group == 1 or geom.contype == 0:  # visual
                geom.contype = geom.conaffinity = 0
                geom.group = 1
                geom.rgba = (0.72, 0.74, 0.78, 1) if 'finger' not in body.name else (0.25, 0.25, 0.28, 1)
                continue
            if 'finger' in body.name:
                spec.delete(geom)
                continue
            # Lighter collision meshes (meshes_collision/) with the same frame.
            mesh = spec.mesh(geom.meshname)
            source = collision_dir/pathlib.Path(mesh.file).name
            if source.exists():
                if geom.meshname not in swapped:
                    m = spec.add_mesh()
                    m.name = 'col_' + geom.meshname
                    m.file = str(source)
                    m.scale = mesh.scale
                    swapped[geom.meshname] = m.name
                geom.meshname = swapped[geom.meshname]
            (geom.contype, geom.conaffinity), geom.group = _collision_class(body.name), 3
    # Finger pads: the finger mesh's bounding box (x +-0.031, y +-0.017, z +-0.045 in the finger frame).
    for arm in ARMS:
        for name in FINGER_BODIES[arm]:
            body = spec.body(name)
            pad = body.add_geom()
            pad.name = f'{name}_pad'
            pad.type = mujoco.mjtGeom.mjGEOM_BOX
            pad.size = (0.028, 0.016, 0.044)
            pad.friction = (2.0, 0.02, 0.002)
            pad.condim = 4
            pad.solref = (0.005, 1)
            pad.solimp = (0.95, 0.99, 0.001, 0.5, 2)
            (pad.contype, pad.conaffinity), pad.group = ARM_CLASS[arm], 3
            pad.rgba = (0.2, 0.2, 0.2, 0.0)
    # The shoulder links sit inside the torso's convex hull: never collide them with it.
    for arm in ARMS:
        for link in (2, 3):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = 'base_link', f'{arm}_link_{link}'
    # Joints: damping and armature like a geared actuator; position actuators per joint.
    for arm in ARMS:
        for i, joint in enumerate(ARM_JOINTS[arm]):
            j = spec.joint(joint)
            j.damping, j.armature, j.frictionloss = [3.0, 0, 0], 0.05, 0.1
            a = spec.add_actuator()
            a.name = joint
            a.target = joint
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            kp = 600.0 if i < 4 else 250.0
            a.set_to_position(kp=kp, dampratio=1.0)
            a.forcelimited = True
            a.forcerange = (-100, 100)  # URDF <limit effort="100">
            a.ctrllimited = True
            a.ctrlrange = tuple(j.range)
        drive, mimic = spec.joint(GRIPPER_JOINT[arm]), spec.joint(GRIPPER_MIMIC[arm])
        for j in (drive, mimic):
            j.damping, j.armature = [2.0, 0, 0], 0.01
        eq = spec.add_equality()
        eq.name = f'{arm}_finger_mimic'
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1, eq.name2 = mimic.name, drive.name
        eq.data[:5] = (0, 1, 0, 0, 0)
        eq.solref = (0.005, 1)
        a = spec.add_actuator()
        a.name = f'{arm}_gripper'
        a.target = drive.name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.set_to_position(kp=2000.0, dampratio=1.0)
        a.forcelimited = True
        a.forcerange = (-40, 40)
        a.ctrllimited = True
        a.ctrlrange = tuple(sorted((0.0, GRIPPER_CLOSED[arm])))
    # TCP sites (the URDF's right_tcp / left_tcp frames: x is the approach axis).
    for arm in ARMS:
        s = spec.body(TCP_BODY[arm]).add_site()
        s.name = f'{arm}_tcp'
        s.size = (0.006, 0.006, 0.006)
        s.rgba = (0, 1, 0, 0.0)
    # Wrist cameras on the gripper mounts, offset along mount x and tilted toward the fingers.
    for arm in ARMS:
        cam = spec.body(MOUNT_BODY[arm]).add_camera()
        cam.name = arm
        cam.pos = (0.055, 0.0, 0.0)
        cam.quat = _quat_from_euler(0.0, math.radians(22), 0.0)
        cam.fovy = 58
    # Head camera on the torso (base_link moves with nothing here: the base is fixed in this scene).
    cam = spec.body('base_link').add_camera()
    cam.name = 'world'
    cam.pos = (0.135, 0.0, 1.31)  # base_link is at z=0.22: 1.53 m above the floor, just in front of the face (x 0.112)
    # Look forward-down: camera -z toward +x, tilted 62 deg down.
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    tilt = math.radians(62)
    # Rotation: camera axes (x right, y up, -z forward). Forward = (cos t, 0, -sin t).
    fwd = np.array([math.cos(tilt), 0, -math.sin(tilt)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(right, fwd)
    mat = np.column_stack([right, up, -fwd])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    cam.quat = q
    cam.fovy = 66
    return names


def _world(spec, task='fruit_to_bowl'):
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.5, 0.5, 0.5)
    world = spec.worldbody
    tex = spec.add_texture()
    tex.name, tex.type, tex.builtin = 'grid', mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2, tex.width, tex.height = (0.32, 0.34, 0.38), (0.26, 0.28, 0.31), 512, 512
    mat = spec.add_material()
    mat.name = 'floor'
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    mat.texrepeat = (6, 6)
    floor = world.add_geom()
    floor.name, floor.type, floor.material = 'floor', mujoco.mjtGeom.mjGEOM_PLANE, 'floor'
    if task == TRAY_TASK:  # cover both tables and the 5 m aisle between them
        floor.size, floor.pos = (5, 5 + TABLE_SPACING/2, 0.05), (0, -TABLE_SPACING/2, 0)
        mat.texrepeat = (8, 8 + 4*TABLE_SPACING/5)
    else:
        floor.size = (4, 4, 0.05)
    floor.contype, floor.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
    light_y = (0.0, -TABLE_SPACING) if task == TRAY_TASK else (0.0,)
    for y in light_y:
        for pos, d in (((1.2, y + 1.0, 2.6), (-0.4, -0.3, -1)), ((1.2, y - 1.0, 2.6), (-0.4, 0.3, -1)),
                       ((-1.0, y, 3.0), (0.3, 0, -1))):
            light = world.add_light()
            light.pos, light.dir, light.diffuse, light.castshadow = pos, d, (0.45, 0.45, 0.45), True
    _table(world, 'table', TABLE)
    if task == TRAY_TASK:
        _table(world, 'place_table', PLACE_TABLE)
        _goal_marker(world, TRAY_GOAL)
        _tray(world, 'tray', TRAY_HOME)
        for name, f in FRUITS.items():
            x, y, yaw = _tray_to_world(TRAY_HOME, TRAY_FRUIT_SLOTS[name])
            _object(world, name, f, (x, y, yaw), base_z=TABLE['top'] + TRAY['bottom'])
        _overview_camera(world)
    elif task == SORTING_TASK:
        _bowl(world, name='bowl', position=(0.30, -0.42), rgba=(0.96, 0.48, 0.12, 1))
        _bowl(world, name='toy_bowl', position=(0.30, 0.42), rgba=(0.12, 0.55, 0.98, 1))
        defaults = {
            'orange': (0.22, -0.27, 0.0), 'apple': (0.39, -0.27, 0.0), 'banana': (0.32, -0.11, 0.0),
            'toy_block': (0.24, 0.24, 0.0), 'toy_car': (0.32, 0.16, 0.0), 'toy_ball': (0.39, 0.24, 0.0),
        }
        for name, spec_object in {**FRUITS, **TOYS}.items():
            _object(world, name, spec_object, defaults[name])
    else:
        _bowl(world)
        for name, f in FRUITS.items():
            _object(world, name, f, (0.5, 0.3, 0.0))


def _table(world, name, t):
    table = world.add_body()
    table.name = name
    table.pos = ((t['x0'] + t['x1'])/2, (t['y0'] + t['y1'])/2, 0)
    top = table.add_geom()
    top.name = f'{name}_top'
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = ((t['x1'] - t['x0'])/2, (t['y1'] - t['y0'])/2, t['thickness']/2)
    top.pos = (0, 0, t['top'] - t['thickness']/2)
    top.rgba = (0.62, 0.47, 0.32, 1)
    top.friction = (1.0, 0.01, 0.001)
    top.contype, top.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = table.add_geom()
            leg.type = mujoco.mjtGeom.mjGEOM_BOX
            leg.size = (0.025, 0.025, (t['top'] - t['thickness'])/2)
            leg.pos = (sx*((t['x1'] - t['x0'])/2 - 0.04), sy*((t['y1'] - t['y0'])/2 - 0.04), (t['top'] - t['thickness'])/2)
            leg.rgba = (0.45, 0.33, 0.22, 1)
            leg.contype, leg.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY


def _tray_to_world(tray_pose, local):
    """(x, y, yaw) of a point given in the tray frame (x along the long axis)."""
    tx, ty, tyaw = tray_pose
    lx, ly, lyaw = local
    c, s = math.cos(tyaw), math.sin(tyaw)
    return tx + c*lx - s*ly, ty + s*lx + c*ly, tyaw + lyaw


def _tray(world, name, pose):
    """Open rectangular tray: flat bottom, four outward-flared walls, a rim lip all round
    (wider on the two short ends, as handles), and visual-only ribs and floor pattern."""
    T = TRAY
    L, W, H, t, tb = T['length'], T['width'], T['height'], T['wall'], T['bottom']
    flare = math.radians(T['flare_deg'])
    inset = math.tan(flare)*H                  # rim is this much wider than the base, per side
    body = world.add_body()
    body.name = name
    body.pos = (pose[0], pose[1], TABLE['top'] + 0.001)
    body.quat = _quat_from_euler(0, 0, pose[2])
    body.add_freejoint().name = f'{name}_free'
    solid, visual = [], []

    def box(gname, size, pos, quat=(1, 0, 0, 0), collide=True):
        g = body.add_geom()
        g.name = f'{name}_{gname}'
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.quat = size, pos, quat
        (solid if collide else visual).append(g)
        return g

    Lb, Wb = L - 2*inset, W - 2*inset          # outside footprint at the base
    box('bottom', (Lb/2, Wb/2, tb/2), (0, 0, tb/2))
    # Walls: centred halfway up, tilted outward by `flare` about their own long edge.
    wall_h = H/math.cos(flare)
    mid = inset/2
    for side in (-1, 1):
        # Long side walls (normal along body y).
        box(f'side_wall_{"n" if side > 0 else "s"}', (L/2 - mid, t/2, wall_h/2),
            (0, side*(Wb/2 + mid - t/2), H/2), _quat_from_euler(side*flare, 0, 0))
        # Short end walls (normal along body x): the grasped walls.
        box(f'end_wall_{"e" if side > 0 else "w"}', (t/2, W/2 - mid, wall_h/2),
            (side*(Lb/2 + mid - t/2), 0, H/2), _quat_from_euler(0, -side*flare, 0))
    # Rim lip: flat flanges around the top outer edge; the end flanges are the handles.
    lt, lip, hl = T['lip_thickness'], T['lip'], T['handle_lip']
    z_lip = H - lt/2
    for side in (-1, 1):
        box(f'lip_{"n" if side > 0 else "s"}', (L/2 + hl, lip/2, lt/2), (0, side*(W/2 + lip/2), z_lip))
        box(f'handle_{"e" if side > 0 else "w"}', (hl/2, W/2 + lip, lt/2), (side*(L/2 + hl/2), 0, z_lip))
        # Handle stiffening rim: a short downturned edge under the handle's outer border.
        box(f'handle_rim_{"e" if side > 0 else "w"}', (0.002, W/2 + lip, 0.006),
            (side*(L/2 + hl - 0.002), 0, H - lt - 0.006))
    # Visual details: vertical ribs on the outside of the long walls, a diamond pattern on the floor.
    for side in (-1, 1):
        for i, frac in enumerate((-0.33, 0.0, 0.33)):
            x = frac*L
            y = side*(W/2 - inset/2 + t/2 + 0.001)
            box(f'rib_{side}_{i}', (0.003, 0.002, H/2*0.92), (x, y, H/2), _quat_from_euler(side*flare, 0, 0),
                collide=False)
        for i, frac in enumerate((-0.3, 0.3)):
            box(f'end_rib_{side}_{i}', (0.002, 0.003, H/2*0.92), (side*(L/2 - inset/2 + t/2 + 0.001), frac*W, H/2),
                _quat_from_euler(0, -side*flare, 0), collide=False)
    for sign in (-1, 1):
        for k in range(-1, 2):
            box(f'floor_line_{sign}_{k}', (0.06, 0.0015, 0.0006), (k*0.05, 0, tb + 0.0006),
                _quat_from_euler(0, 0, sign*math.radians(35)), collide=False)
    total_area = sum(4*g.size[0]*g.size[1] + 4*g.size[1]*g.size[2] + 4*g.size[0]*g.size[2] for g in solid)
    for g in solid:
        g.rgba = T['rgba']
        g.mass = T['mass']*(4*g.size[0]*g.size[1] + 4*g.size[1]*g.size[2] + 4*g.size[0]*g.size[2])/total_area
        g.friction = (1.0, 0.01, 0.001)
        g.condim = 4
        g.solref = (0.004, 1)
        g.contype, g.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
    for g in visual:
        g.rgba = (0.16, 0.36, 0.88, 1)
        g.contype = g.conaffinity = 0
        g.group = 1
        g.mass = 0
    site = body.add_site()
    site.name = f'{name}_center'
    site.size = (0.005, 0.005, 0.005)
    site.rgba = (0, 0, 0, 0)
    for side, end in ((1, 'e'), (-1, 'w')):
        # Handle grasp sites: on the end wall's top edge, HANDLE_Y off centre (fingers straddle the wall).
        s = body.add_site()
        s.name = f'{name}_handle_{end}'
        s.pos = (side*(L/2 - t/2), HANDLE_Y, H)
        s.size = (0.006, 0.006, 0.006)
        s.rgba = (1, 1, 0, 0)


def _goal_marker(world, pose):
    """Thin visual outline on the place table: where the tray should be set down."""
    body = world.add_body()
    body.name = 'tray_goal'
    body.pos = (pose[0], pose[1], PLACE_TABLE['top'] + 0.0005)
    body.quat = _quat_from_euler(0, 0, pose[2])
    L, W, b = TRAY['length'] + 0.03, TRAY['width'] + 0.03, 0.006
    for i, (size, pos) in enumerate((((L/2, b/2, 0.0005), (0, W/2, 0)), ((L/2, b/2, 0.0005), (0, -W/2, 0)),
                                     ((b/2, W/2, 0.0005), (L/2, 0, 0)), ((b/2, W/2, 0.0005), (-L/2, 0, 0)))):
        g = body.add_geom()
        g.name = f'tray_goal_edge_{i}'
        g.type, g.size, g.pos = mujoco.mjtGeom.mjGEOM_BOX, size, pos
        g.rgba = (0.95, 0.95, 0.95, 1)
        g.contype = g.conaffinity = 0
        g.group = 1
    s = body.add_site()
    s.name = 'tray_goal_center'
    s.size = (0.005, 0.005, 0.005)
    s.rgba = (0, 0, 0, 0)


def _look_at_quat(pos, target):
    """Camera quaternion (MuJoCo: x right, y up, looking along -z) from pos toward target."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0, 0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([right, up, -fwd]).flatten())
    return q


def _overview_camera(world):
    """Fixed room camera that sees both tables and the aisle (for viewing, not for the agent)."""
    target = (0.45, -TABLE_SPACING/2, 0.75)
    for name, pos in (('overview', (3.9, -TABLE_SPACING/2, 2.7)),
                      ('pick_view', (1.9, 1.1, 1.9)), ('place_view', (1.9, -TABLE_SPACING - 1.1, 1.9))):
        cam = world.add_camera()
        cam.name = name
        cam.pos = pos
        aim = target if name == 'overview' else (0.4, 0.0 if name == 'pick_view' else -TABLE_SPACING, 0.9)
        cam.quat = _look_at_quat(pos, aim)
        cam.fovy = 55 if name == 'overview' else 50


def _mobile_base(spec):
    """Planar base: slide x, slide y, yaw about the torso axis, each with a position actuator
    (a holonomic stand-in for the wheel base's velocity controller)."""
    base = spec.body('base_link')
    for name, kind, axis in (('base_x', mujoco.mjtJoint.mjJNT_SLIDE, (1, 0, 0)),
                             ('base_y', mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0)),
                             ('base_yaw', mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1))):
        j = base.add_joint()
        j.name, j.type, j.axis = name, kind, axis
        j.damping, j.armature = [50.0 if kind == mujoco.mjtJoint.mjJNT_SLIDE else 20.0, 0, 0], 0.5
        a = spec.add_actuator()
        a.name = name
        a.target = name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.set_to_position(kp=20000.0 if kind == mujoco.mjtJoint.mjJNT_SLIDE else 3000.0, dampratio=1.0)
        a.forcelimited = True
        a.forcerange = (-800, 800) if kind == mujoco.mjtJoint.mjJNT_SLIDE else (-300, 300)
    # The wheels ride on the floor: no contact between the base and the floor plane.
    for body in ('base_link', 'wheel_base'):
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = 'world', body


def _bowl(world, name='bowl', position=(0.5, 0.0), rgba=(0.93, 0.93, 0.95, 1)):
    b = BOWL
    body = world.add_body()
    body.name = name
    body.pos = (position[0], position[1], TABLE['top'])
    body.add_freejoint().name = f'{name}_free'
    r_out = b['inner_radius'] + b['wall']
    base = body.add_geom()
    base.name = f'{name}_bottom'
    base.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base.size = (r_out, b['bottom']/2, 0)
    base.pos = (0, 0, b['bottom']/2)
    base.rgba = rgba
    base.mass = b['mass']*0.4
    segments = 24
    for i in range(segments):
        a = 2*math.pi*i/segments
        w = body.add_geom()
        w.name = f'{name}_wall_{i}'
        w.type = mujoco.mjtGeom.mjGEOM_BOX
        half_len = math.pi*r_out/segments*1.08
        # Slightly flared wall: tilt outward by 12 degrees.
        flare = math.radians(12)
        r_mid = b['inner_radius'] + b['wall']/2 + math.sin(flare)*b['height']/2
        w.size = (b['wall']/2, half_len, b['height']/2)
        w.pos = (r_mid*math.cos(a), r_mid*math.sin(a), b['bottom'] + b['height']/2)
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([0.0, -flare, a]), 'zyx'[::-1])
        qz = _quat_from_euler(0, 0, a)
        qy = _quat_from_euler(0, -flare, 0)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, qy)
        w.quat = out
        w.rgba = rgba
        w.mass = b['mass']*0.6/segments
    for g in body.geoms:
        g.contype, g.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
        g.friction = (0.8, 0.01, 0.001)
    site = body.add_site()
    site.name = f'{name}_center'
    site.size = (0.005, 0.005, 0.005)
    site.rgba = (0, 0, 0, 0)


def _object(world, name, f, pose, base_z=None):
    body = world.add_body()
    body.name = name
    if f['kind'] == 'sphere':
        rest_height = f['radius']
    elif f['kind'] in ('ellipsoid', 'box'):
        rest_height = f['size'][2]
    elif f['kind'] == 'car':
        rest_height = 0.012
    else:
        rest_height = f['radius']
    body.pos = (pose[0], pose[1], (TABLE['top'] if base_z is None else base_z) + rest_height + 0.002)
    if pose[2]:
        body.quat = _quat_from_euler(0, 0, pose[2])
    body.add_freejoint().name = f'{name}_free'
    geoms = []
    if f['kind'] == 'sphere':
        g = body.add_geom()
        g.type, g.size = mujoco.mjtGeom.mjGEOM_SPHERE, (f['radius'], 0, 0)
        geoms.append(g)
    elif f['kind'] == 'ellipsoid':
        g = body.add_geom()
        g.type, g.size = mujoco.mjtGeom.mjGEOM_ELLIPSOID, f['size']
        geoms.append(g)
        stem = body.add_geom()
        stem.type, stem.size = mujoco.mjtGeom.mjGEOM_CAPSULE, (0.003, 0.008, 0)
        stem.pos = (0, 0, f['size'][2] + 0.004)
        stem.rgba = (0.35, 0.22, 0.1, 1)
        stem.contype = stem.conaffinity = 0
        stem.mass = 0.001
    elif f['kind'] == 'box':
        g = body.add_geom()
        g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, f['size']
        geoms.append(g)
    elif f['kind'] == 'car':
        # A palm-sized toy car: a collidable chassis and cabin, with visual-only wheels.
        chassis = body.add_geom()
        chassis.type = mujoco.mjtGeom.mjGEOM_BOX
        chassis.size = (f['length']/2, f['width']/2 - 0.006, 0.009)
        chassis.pos = (0, 0, 0.011)
        geoms.append(chassis)
        cabin = body.add_geom()
        cabin.type = mujoco.mjtGeom.mjGEOM_BOX
        cabin.size = (f['length']*0.23, f['width']*0.31, 0.012)
        cabin.pos = (-f['length']*0.08, 0, 0.032)
        geoms.append(cabin)
        wheel_index = 0
        for x in (-f['length']*0.31, f['length']*0.31):
            for y in (-f['width']*0.43, f['width']*0.43):
                wheel = body.add_geom()
                wheel.name = f'{name}_wheel_{wheel_index}'
                wheel_index += 1
                wheel.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                wheel.size = (0.012, 0.004, 0)
                wheel.pos = (x, y, 0)
                wheel.quat = _quat_from_euler(math.pi/2, 0, 0)
                wheel.rgba = (0.08, 0.08, 0.09, 1)
                (wheel.contype, wheel.conaffinity) = WORLD_CONTYPE, WORLD_CONAFFINITY
                geoms.append(wheel)
    else:  # banana: capsules along an arc in the body's xy-plane, centred on the body origin
        n, length, bend = 6, f['length'], math.radians(f['bend_deg'])
        radius_arc = length/bend
        seg = length/n
        pts = []
        for i in range(n):
            a = -bend/2 + (i + 0.5)*bend/n
            pts.append((radius_arc*math.sin(a), radius_arc*(1 - math.cos(a)), a))
        cy = sum(p[1] for p in pts)/n
        for i, (x, y, a) in enumerate(pts):
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            taper = 1.0 - 0.35*abs(i - (n - 1)/2)/((n - 1)/2)
            g.size = (f['radius']*taper, seg/2*0.95, 0)
            g.pos = (x, y - cy, 0)
            g.quat = _quat_mul(_quat_from_euler(0, 0, a), _quat_from_euler(0, math.pi/2, 0))
            geoms.append(g)
        end = pts[-1]
        tip = body.add_geom()
        tip.type, tip.size = mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0)
        tip.pos = (end[0] + seg/2*math.cos(end[2]), end[1] - cy + seg/2*math.sin(end[2]), 0)
        tip.rgba = (0.3, 0.22, 0.08, 1)
        tip.contype = tip.conaffinity = 0
        tip.mass = 0.001
    total = sum(1 for _ in geoms)
    for g in geoms:
        if not g.name.startswith(f'{name}_wheel_'):
            g.rgba = f['rgba']
        g.mass = f['mass']/total
        g.friction = (1.2, 0.01, 0.002)
        g.condim = 4
        g.contype, g.conaffinity = WORLD_CONTYPE, WORLD_CONAFFINITY
        g.solref = (0.005, 1)


def build_spec(task='fruit_to_bowl'):
    spec = mujoco.MjSpec.from_string(_urdf_text())
    spec.modelname = {SORTING_TASK: 'daksha_sorting_table', TRAY_TASK: 'daksha_tray_transport'}.get(
        task, 'daksha_fruit_table')
    spec.compiler.degree = False
    _robot(spec)
    if task == TRAY_TASK:
        _mobile_base(spec)
    _world(spec, task)
    key = spec.add_key()
    key.name = 'home'
    return spec


def build_model(task='fruit_to_bowl'):
    spec = build_spec(task)
    model = spec.compile()
    # Home keyframe: robot posture, objects at their default spots (episodes re-place them).
    data = mujoco.MjData(model)
    for name, value in HOME.items():
        data.joint(name).qpos = value
    for arm in ARMS:
        data.ctrl[model.actuator(ARM_JOINTS[arm][0]).id:model.actuator(ARM_JOINTS[arm][0]).id + 7] = [HOME[j] for j in ARM_JOINTS[arm]]
    if task == TRAY_TASK:
        for name, value in zip(BASE_JOINTS, BASE_HOME):
            data.joint(name).qpos = value
            data.ctrl[model.actuator(name).id] = value
    model.key_qpos[0] = data.qpos
    model.key_ctrl[0] = data.ctrl
    return model, spec


def export_xml(path, task='fruit_to_bowl'):
    """Write the compiled scene as MJCF (for inspection or `python -m mujoco.viewer`)."""
    model, spec = build_model(task)
    pathlib.Path(path).write_text(spec.to_xml())
    return path


if __name__ == '__main__':
    import sys
    model, _ = build_model(sys.argv[2] if len(sys.argv) > 2 else 'fruit_to_bowl')
    print('compiled:', model.nq, 'qpos,', model.nu, 'actuators,', model.ncam, 'cameras,', model.nbody, 'bodies')
    if len(sys.argv) > 1:
        print('wrote', export_xml(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'fruit_to_bowl'))
