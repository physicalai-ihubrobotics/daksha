"""RAX backend for the REAL Daksha over ROS 2 (register name: daksha_ros).

Same `daksha_*` tool names as the MuJoCo backend, so the agent, its skills and a
controller written in sim work unchanged. The robot I/O and tool logic are
ros_agent's (pipeline_node.PipelineNode in robot-I/O mode + real_backend.RosRobotBackend):

  perception  world camera + aruco_hand_eye calibration (+ table plane or aligned depth)
  motion      ROS 2 actions to aruco_hand_eye's aruco_reach (MoveToPose / MoveJoints /
              Gripper): it is the only /joint_cmd sender and owns the execute switch
              (dry run by default; motion tools then return the plan as an error).

Real-robot differences from the sim (also told to the agent in daksha_status):
  * start_episode does not rearrange anything: reset the table by hand.
  * evaluate cannot verify the task (no trusted checker): task_complete is always false.
  * only the world camera is available to daksha_observe.

Needs a sourced ROS 2 workspace (ros_agent, aruco_hand_eye) and aruco_reach running:
    ros2 launch aruco_hand_eye aruco_reach.launch.py
"""
from __future__ import annotations

import threading

ARMS = ('left', 'right')


def _schema(props, required=()):
    return {'type': 'object', 'properties': props, 'required': list(required), 'additionalProperties': False}


ARM = {'type': 'string', 'enum': list(ARMS)}
VEC3 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
ORIENTATION = {
    'quat': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4,
             'description': 'TCP orientation quaternion [x, y, z, w] in the robot frame.'},
    'approach': {**VEC3, 'description': 'Approach direction (TCP x axis, wrist to fingertips), robot frame.'},
    'finger_axis': {**VEC3, 'description': 'Finger-opening direction (TCP y axis); needs approach.'},
    'tolerance_deg': {'type': 'number', 'minimum': 0, 'maximum': 90,
                      'description': 'Allowed orientation deviation (default 5 with an orientation, 10 when kept).'},
    'finger_tolerance_deg': {'type': 'number', 'minimum': 0, 'maximum': 90,
                             'description': 'Accepted for simulator compatibility; on the real robot the finger axis '
                                            'shares tolerance_deg.'},
}
CAMERAS = ['world', 'right_wrist', 'left_wrist', 'right', 'left']   # 'right' / 'left' = the sim's wrist names


class DakshaRosBackend:
    """RAX backend contract (services/backends.py) on the real robot."""

    def __init__(self, config, node=None, real=None, extra_tools=None, hide_tools=()):
        """node / real: run inside ros_agent on its live PipelineNode and RosRobotBackend (one
        pipeline: same camera, calibration, table, interrupt) instead of starting a node here.
        extra_tools: {name: (tool definition, fn(arguments) -> (value, images))} added to the
        robot tools (ros_agent's find / track / pick / report)."""
        self.config = config
        self._node = self._real = self._executor = self._spin = None
        self._shared = node is not None
        if self._shared:
            self._node, self._real = node, real
        self._extra = dict(extra_tools or {})
        self._hidden = set(hide_tools)        # robot tools this agent may not use
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ ROS (lazy: `prepare` only lists tools)
    def _ensure(self):
        with self._lock:
            if self._real is not None:
                return self._real
            if self._shared:
                from ros_agent.real_backend import RosRobotBackend
                self._real = RosRobotBackend(self._node, self._node.cfg)
                return self._real
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from ros_agent.pipeline_node import PipelineNode
            from ros_agent.real_backend import RosRobotBackend
            if not rclpy.ok():
                rclpy.init()
            self._node = PipelineNode(name='rax_robot', agent=False)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin = threading.Thread(target=self._executor.spin, daemon=True, name='rax_ros')
            self._spin.start()
            self._real = RosRobotBackend(self._node, self._node.cfg)
            return self._real

    # ------------------------------------------------------------------ contract
    def _tools(self):
        return [
            {'name': 'daksha_status', 'description':
                'REAL robot state: joints, world camera, table height, camera calibration, aruco_reach motion '
                'server and its execute switch (motion only when execute is on), and TCP poses (robot frame).',
             'inputSchema': _schema({})},
            {'name': 'daksha_observe', 'description':
                'Grab the latest REAL camera images: "world" (head, looking at the table) and the wrist cameras '
                '"right_wrist" / "left_wrist" (on the grippers, pose from the arm joints at capture; they carry aligned '
                'DEPTH, so their pixels become measured 3-D points). Returns the '
                'images and frames {camera: frame_id} for daksha_pixel_to_position / daksha_footprint.',
             'inputSchema': _schema({'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': CAMERAS}},
                                     'attach': {'type': 'boolean'}})},
            {'name': 'daksha_footprint', 'description':
                'Where an object stands, from its bounding box [x1, y1, x2, y2] (pixels) in a saved frame without '
                'depth: the box bottom edge meets the table, so this avoids the parallax of tall objects. Returns '
                'centre [x, y, top_z] and size_m.',
             'inputSchema': _schema({'frame_id': {'type': 'string'},
                                     'bbox': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4,
                                              'maxItems': 4}}, ['frame_id', 'bbox'])},
            {'name': 'daksha_pixel_to_position', 'description':
                'Robot-frame 3-D points for pixels of a saved frame: from aligned depth when available, otherwise '
                'the camera ray hitting a horizontal plane at height z (default: just above the table). pixels are '
                '[u, v] (column, row). Works for world and wrist frames.',
             'inputSchema': _schema({'frame_id': {'type': 'string'},
                                     'camera': {'type': 'string', 'enum': CAMERAS},
                                     'z': {'type': 'number', 'description': 'plane height (robot frame, m)'},
                                     'pixels': {'type': 'array', 'minItems': 1, 'maxItems': 256,
                                                'items': {'type': 'array', 'items': {'type': 'number'},
                                                          'minItems': 2, 'maxItems': 2}},
                                     'window': {'type': 'integer', 'minimum': 0, 'maximum': 10}},
                                    ['frame_id', 'camera', 'pixels'])},
            {'name': 'daksha_move_eef', 'description':
                'Move one arm\'s TCP to a position (robot frame) with an optional orientation; without one the '
                'current orientation is kept. path "linear" (default) moves in a straight line, "joint" in joint '
                'space. relative=true offsets from the current TCP. Executed by aruco_reach (IK, self-collision '
                'check, never below the table). Refused targets come back as errors; while execute is off the '
                'error says DRY RUN and shows the plan.',
             'inputSchema': _schema({'arm': ARM, 'position': VEC3, **ORIENTATION, 'relative': {'type': 'boolean'},
                                     'path': {'type': 'string', 'enum': ['linear', 'joint']},
                                     'speed': {'type': 'number', 'exclusiveMinimum': 0}}, ['arm', 'position'])},
            {'name': 'daksha_move_joints', 'description':
                'Move one arm to 7 joint positions (rad) along a smooth, self-collision checked joint-space path.',
             'inputSchema': _schema({'arm': ARM, 'positions': {'type': 'array', 'items': {'type': 'number'},
                                                               'minItems': 7, 'maxItems': 7},
                                     'speed': {'type': 'number', 'exclusiveMinimum': 0}}, ['arm', 'positions'])},
            {'name': 'daksha_check', 'description':
                'Dry run, nothing moves: can this arm\'s TCP reach a position (+ orientation)? IK, joint and table '
                'limits; reply reachable, position_error_m, orientation_error_deg, joints, tcp. With positions '
                '(7 joints): forward kinematics (tcp) and whether they are within the joint limits.',
             'inputSchema': _schema({'arm': ARM, 'position': VEC3, **ORIENTATION, 'relative': {'type': 'boolean'},
                                     'positions': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 7,
                                                   'maxItems': 7}}, ['arm'])},
            {'name': 'daksha_move_eef_both', 'description':
                'Both TCPs: {"left": {position, orientation...}, "right": {...}} (+ relative / path / speed for both). '
                'On the real robot the two moves start together and the call ends when both finished (not one '
                'jointly planned trajectory): give both the same distance / speed for a matched motion.',
             'inputSchema': _schema({'left': {'type': 'object'}, 'right': {'type': 'object'},
                                     'relative': {'type': 'boolean'},
                                     'path': {'type': 'string', 'enum': ['linear', 'joint']},
                                     'speed': {'type': 'number', 'exclusiveMinimum': 0}})},
            {'name': 'daksha_gripper', 'description':
                'Open / close a gripper (or set a width in m, 0..0.088) and wait until the fingers stop. Returns '
                'width_m and stalled_before_target (true = closed on an object).',
             'inputSchema': _schema({'arm': ARM, 'command': {'type': 'string', 'enum': ['open', 'close', 'width']},
                                     'width': {'type': 'number', 'minimum': 0, 'maximum': 0.088},
                                     'force': {'type': 'number', 'minimum': 0, 'maximum': 40,
                                               'description': 'Accepted for simulator compatibility; the real '
                                                              'gripper is position-controlled (reply force_n is null)'}},
                                    ['arm', 'command'])},
            {'name': 'daksha_home', 'description':
                'Move an arm (or both, one after the other) to the robot\'s home pose (gen2 HomeMoveService).',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': [*ARMS, 'both']}})},
            {'name': 'daksha_measure_table', 'description':
                'Measure the table height from the 80 mm ArUco marker lying flat on the table (world camera). '
                'Positions and grasps need it; call it once at the start if daksha_status shows no table.',
             'inputSchema': _schema({})},
            *(t for t, _ in self._extra.values()),
        ]

    def tools(self):
        return [t for t in self._tools() if t['name'] not in self._hidden]

    def validate(self, name, arguments):
        if 'arm' in arguments and arguments['arm'] not in (*ARMS, 'both'):
            raise ValueError('arm must be "left" or "right"')
        if name == 'daksha_home' and arguments.get('arm', 'both') not in (*ARMS, 'both'):
            raise ValueError('arm must be left, right or both')

    def call(self, name, arguments):
        if name in self._hidden:
            raise PermissionError(f'{name} is not available to this agent')
        real = self._ensure()
        if name in self._extra:
            return self._extra[name][1](arguments)
        if name == 'daksha_status':
            return self.status(), []
        if name == 'daksha_measure_table':
            ok, msg = self._node.measure_table()
            if not ok:
                raise ValueError(msg)
            return {'table_z': self._node.table_z, 'message': msg}, []
        try:
            value, images = real.call(name, arguments)
        except Exception as exc:
            # DryRun / MotionStopped / refusals -> a clear tool error for the agent
            kind = type(exc).__name__
            if kind == 'DryRun':
                raise ValueError(f'DRY RUN (execute is off on aruco_reach): {exc}. Nothing moved.') from None
            if kind == 'MotionStopped':
                err = ValueError(f'Motion stopped: {exc}. Call daksha_status before continuing.')
                err.handled_stop = True
                raise err from None
            raise
        # MCP image blocks, like the sim backend
        blocks = [{'type': 'image', 'mimeType': 'image/png', 'data': im['data']} for im in images]
        return value, blocks

    def status(self):
        if self._real is None:
            return {'robot': 'real Daksha (ROS 2)', 'connected': False,
                    'note': 'connects to ROS on the first robot tool call'}
        n, real = self._node, self._real
        s = n.state()
        tcp = {}
        for arm in ARMS:
            try:
                tcp[arm] = real._tcp(arm)
            except Exception as exc:
                tcp[arm] = f'unavailable: {exc}'
        return {'robot': 'real Daksha (ROS 2)', 'joints_ok': s['joints_ok'], 'camera_ok': s['camera_ok'],
                'depth': s['depth'], 'table_z': s['table_z'], 'calibration': s['calibration'],
                'motion_server': s['motion_server'], 'execute': s['execute'], 'tcp': tcp,
                'grippers': s.get('grippers'),
                'safety_alerts': (s.get('safety') or {}).get('alerts'),
                'grasp_choice': s.get('grasp_choice'),
                'motor_temperatures': {'warn_c': (s.get('motors') or {}).get('warn_c'),
                                       'hot': (s.get('motors') or {}).get('hot'),
                                       'max_c': {k: v.get('max_c') for k, v in ((s.get('motors') or {}).get('arms') or {}).items()}},
                'real_robot_notes': ['start_episode does not rearrange the table: the operator resets it',
                                     'evaluate cannot verify the task on the real robot (always false)',
                                     'motion only happens while execute is on (aruco_reach)']}

    def reset_episode(self, seed):
        pass                                   # the operator resets the real table

    def end_episode(self):
        pass

    def task_state(self):
        return {'task_complete': False, 'verified': False,
                'note': 'no trusted success check on the real robot'}

    def close(self):
        if self._shared:                       # the node belongs to ros_agent
            self._real = None
            return
        if self._real is not None:
            try:
                self._real.close()
                self._node.motion.stop()
            except Exception:
                pass
            self._executor.shutdown(timeout_sec=2.0)
            self._spin.join(timeout=2.0)
            self._node.destroy_node()
            self._real = None
