"""ROS 2 node for the local VLM pick-and-place pipeline on the REAL robot.

Runs local_daksha_rax/local_pipeline's LocalAgent (planner + skills + Ollama VLM)
with RosRobotBackend (real_backend.py) instead of the MuJoCo sim.

ROS interface
    /pipeline/command        std_msgs/String   an instruction or question ("put the apple in the bowl")
    /pipeline/stop           std_srvs/Trigger  interrupt now, drop queued commands
    /pipeline/set_execute    std_srvs/SetBool  true = arm real motion (default false: dry run)
    /pipeline/measure_table  std_srvs/Trigger  table height from an ArUco marker lying on the table
    /pipeline/status         std_msgs/String   JSON state, 2 Hz
in: /joint_states, world camera image + camera_info (+ aligned depth if configured)

Motion goes to aruco_hand_eye's aruco_reach over ROS 2 actions
(aruco_hand_eye MoveToPose / MoveJoints / Gripper); aruco_reach is the
only /joint_cmd sender and owns the execute switch (this node's Execute toggles it).

Settings: config/pipeline.yaml. The web UI is ros_agent's page (port 8150), see web.py.
"""

import glob
import json
import os
import sys
import threading
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from ament_index_python.packages import get_package_share_directory


def find_rax_dir(override=''):
    """Source folder of local_daksha_rax (it is not installed: it carries its own
    runtime data and tools). Found next to this package's source, at runtime."""
    if override and os.path.isdir(override):
        return override
    here = os.path.dirname(os.path.realpath(__file__))
    candidates = [os.path.join(here, '..', 'local_daksha_rax')]
    share = get_package_share_directory('ros_agent')
    if f'{os.sep}install{os.sep}' in share:
        ws = share.split(f'{os.sep}install{os.sep}')[0]
        candidates += glob.glob(os.path.join(ws, 'src', '**', 'ros_agent', 'local_daksha_rax'), recursive=True)
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'local_pipeline')):
            return os.path.normpath(c)
    raise RuntimeError('local_daksha_rax not found next to the ros_agent sources; set pipeline.rax_dir')


WRIST_CAMERAS = {
    'right': {'image_topic': '/right/camera/color/image_raw', 'camera_info_topic': '/right/camera/color/camera_info',
              'depth_topic': '/right/camera/aligned_depth_to_color/image_raw',
              'calibration': 'eye_in_hand_right_wrist.yaml'},
    'left': {'image_topic': '/left/camera/color/image_raw', 'camera_info_topic': '/left/camera/color/camera_info',
             'depth_topic': '/left/camera/aligned_depth_to_color/image_raw',
             'calibration': 'eye_in_hand_left_wrist.yaml'},
}


class PipelineNode(Node):
    def __init__(self, overrides=None, name='pipeline', agent=True):
        """agent=False: only the robot I/O (joints, world camera, calibration, table,
        aruco_reach motion client) for another front end, e.g. RAX's daksha_ros backend;
        no VLM agent and no /pipeline/* interface."""
        from rclpy.parameter import Parameter
        super().__init__(name, parameter_overrides=[Parameter(k, value=v)
                                                    for k, v in (overrides or {}).items()])
        self.with_agent = bool(agent)
        self.agent, self.agent_error, self._log_buf = None, None, []   # before anything can log
        self.tracker, self.tracked = None, None   # ros_agent's live box tracker (set by agent_node)
        self.wrist_trackers = {}                   # arm -> tracker on that wrist (gripper) camera
        self.wrist_focus = None                    # {'arm', 'name'} while the wrist approach runs
        self.wrist_obj = None                      # last wrist measurement {'arm','name','world','t'} -> TF
        share = get_package_share_directory('ros_agent')
        cfg_path = self.declare_parameter('pipeline_config', os.path.join(share, 'config', 'pipeline.yaml')).value
        with open(cfg_path) as f:
            self.cfg = (yaml.safe_load(f) or {}).get('pipeline', {})
        # revised plans the agent may make per task (pipeline.yaml max_replans, ROS param, UI)
        self.max_replans = int(self.declare_parameter('max_replans', int(self.cfg.get('max_replans', 2))).value)
        self.grasp_choice = 'auto'                     # operator's pick style from the UI (auto = agent / config)
        self._grasp_default = self.cfg.get('grasp_style', 'side')
        c = self.cfg
        self.execute_requested = bool(self.declare_parameter('execute', bool(c.get('execute', False))).value)
        model = self.declare_parameter('model', str(c.get('model', '35b'))).value
        ollama = self.declare_parameter('ollama', str(c.get('ollama', 'http://localhost:11434'))).value

        # local_pipeline lives in the (uninstalled) local_daksha_rax folder
        self.rax_dir = find_rax_dir(c.get('rax_dir', ''))
        if self.rax_dir not in sys.path:
            sys.path.insert(0, self.rax_dir)
        from local_pipeline import robot as R
        self.R = R

        # robot model (placo resolves package:// meshes through ROS_PACKAGE_PATH)
        dshare = get_package_share_directory('daksha_description_full_body')
        os.environ['ROS_PACKAGE_PATH'] = os.pathsep.join(
            p for p in (os.path.dirname(dshare), os.environ.get('ROS_PACKAGE_PATH', '')) if p)
        self.urdf_path = os.path.join(dshare, 'urdf', 'robot.urdf')
        from aruco_hand_eye.urdf_kinematics import UrdfKinematics
        self.fk = UrdfKinematics(self.urdf_path)
        self.T_world_base = self.fk.transform('world', 'base_link', {})

        # joints: /joint_states read as-is (URDF convention, see aruco_hand_eye/joint_convention.py)
        self.joints, self.joint_t = {}, 0.0
        self.create_subscription(JointState, '/joint_states', self._joint_cb, qos_profile_sensor_data)
        # motion: aruco_reach's actions (it is the only /joint_cmd sender)
        from aruco_hand_eye.motion_client import MotionClient
        self.motion = MotionClient(self, c.get('motion_server', '/aruco_reach'))
        self._want_execute = self.execute_requested

        # world camera
        cam = c.get('camera', {})
        self._cam_lock = threading.Lock()
        self._cam = {'bgr': None, 't': 0.0, 'K': None, 'D': None, 'depth': None, 'depth_t': 0.0}
        rel = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, cam.get('image_topic', '/camera/world/color/image_raw'), self._img_cb, rel)
        self.create_subscription(CameraInfo, cam.get('camera_info_topic', '/camera/world/color/camera_info'),
                                 self._info_cb, qos_profile_sensor_data)
        if cam.get('depth_topic'):
            self.create_subscription(Image, cam['depth_topic'], self._depth_cb, rel)

        # world camera calibration (from aruco_hand_eye), with the same accuracy gate as aruco_reach
        self.T_base_cam, self.calib_problem, self.calib_info = None, 'not loaded', {}
        self._load_calibration()

        # motor temperatures (hw_interface MotorStatusArray per arm): the agent sees them, the page warns
        self.motor_temp_warn_c = float(c.get('motor_temp_warn_c', 80.0))
        self.motor_temp_stop_c = float(c.get('motor_temp_stop_c', 90.0))
        self.motors = {}                 # arm -> {'t', 'motors': [{id, mos, rotor, error}]}
        self._temp_alerted = {}          # (arm, id) -> last alert time
        try:
            from hw_interface.msg import MotorStatusArray
            for arm, topic in (('right', '/RightArmSystem/motor_status'), ('left', '/LeftArmSystem/motor_status')):
                self.create_subscription(MotorStatusArray, topic, lambda m, arm=arm: self._motor_cb(arm, m),
                                         qos_profile_sensor_data)
        except ImportError as ex:
            self.get_logger().warn(f'[pipeline] no motor temperatures ({ex})')

        # safety monitor: temperature, motor faults, collision, joint limits, people, battery, data
        from .safety_monitor import SafetyMonitor
        self.safety = SafetyMonitor(self, c)
        from std_msgs.msg import Float32
        self.create_subscription(Float32, '/battery_info', self.safety.on_battery, 10)
        try:
            from diagnostic_msgs.msg import DiagnosticArray
            self.create_subscription(DiagnosticArray, '/diagnostics', self.safety.on_diagnostics, 10)
        except ImportError:
            pass
        self.create_subscription(String, '/aruco_reach/status', self.safety.on_reach_status, 10)
        self.create_timer(1.0, self._safety_tick)

        # wrist cameras (eye-in-hand): close-range look before a grasp and grasp check after it.
        # Only the latest raw message is kept; it is decoded when a frame is actually needed.
        self._wrist = {}
        for arm, wc in (c.get('wrist_cameras') or WRIST_CAMERAS).items():
            w = self._wrist[arm] = {'msg': None, 't': 0.0, 'K': None, 'D': None, 'T_mount_cam': None,
                                    'mount': None, 'problem': None, 'calib': wc.get('calibration'),
                                    'depth_msg': None, 'depth_t': 0.0}
            if wc.get('depth_topic'):       # aligned to color (camera_launch: align_depth.enable)
                self.create_subscription(Image, wc['depth_topic'],
                                         lambda m, w=w: w.update(depth_msg=m, depth_t=time.monotonic()), rel)
            self.create_subscription(Image, wc['image_topic'], lambda m, w=w: w.update(msg=m, t=time.monotonic()), rel)
            self.create_subscription(CameraInfo, wc['camera_info_topic'],
                                     lambda m, w=w: w.update(K=np.array(m.k, float).reshape(3, 3),
                                                             D=np.array(m.d, float) if len(m.d) else np.zeros(5)),
                                     qos_profile_sensor_data)
            try:
                from aruco_hand_eye import calib_files
                from aruco_hand_eye.aruco_reach import load_calibration
                parent, _, T, doc = load_calibration(os.path.join(calib_files.default_dir(), wc['calibration']))
                w.update(T_mount_cam=T, mount=parent,
                         accuracy=f'{doc.get("t_rms_mm", 0):.0f} mm / {doc.get("r_rms_deg", 0):.1f} deg')
            except Exception as ex:
                w['problem'] = f'{wc.get("calibration")}: {ex}'

        # table height: config, or measured with /pipeline/measure_table
        self.table_z = None
        tz = c.get('scene', {}).get('table_z')
        if tz is not None:
            self._set_table(float(tz), 'config')

        self.model = model
        self.create_timer(1.0, self._apply_requested_execute)
        if not self.with_agent:
            self.get_logger().info(f'[{name}] robot I/O ready (no pipeline agent); motion via aruco_reach actions')
            return

        # ROS interface
        self.status_pub = self.create_publisher(String, '/pipeline/status', 10)
        self.create_subscription(String, '/pipeline/command', lambda m: self.submit(m.data), 10)
        self.create_service(Trigger, '/pipeline/stop', lambda q, r: self._trig(r, self.stop()))
        self.create_service(Trigger, '/pipeline/measure_table', lambda q, r: self._trig(r, *self.measure_table()))
        self.create_service(SetBool, '/pipeline/set_execute', lambda q, r: self._trig(r, self.set_execute(q.data)))
        self.create_timer(0.5, lambda: self.status_pub.publish(String(data=json.dumps(self.state(), default=str))))
        self.create_timer(0.25, self._write_live_frame)
        # detected objects -> TF (object_<name>, pipeline_goal) + /pipeline/objects (JSON, for the IK viewer)
        from tf2_ros import TransformBroadcaster
        self.tf_pub = TransformBroadcaster(self)
        self.objects_pub = self.create_publisher(String, '/pipeline/objects', 10)
        self.create_timer(0.5, self._publish_objects)

        # the agent (VLM warm-up can take a minute: start it in the background)
        threading.Thread(target=self._start_agent, args=(model, ollama), daemon=True).start()
        self.get_logger().info(f'[pipeline] ready; motion via {c.get("motion_server", "/aruco_reach")} actions '
                               f'(start aruco_hand_eye aruco_reach.launch.py); local_daksha_rax at {self.rax_dir}')

    # ------------------------------------------------------------------ startup
    def _start_agent(self, model, ollama):
        try:
            from local_pipeline.agent import LocalAgent
            from local_pipeline.ollama_vlm import OllamaVLM
            from .real_backend import RosRobotBackend
            # same context as the RAX agent's model twin (rax-*:64k), so Ollama keeps one copy loaded
            vlm = OllamaVLM(host=ollama, model=model,
                            num_ctx=65536 if str(model).startswith('rax-') else 8192)
            ok, names = vlm.available()
            if not ok:
                raise RuntimeError(f'Ollama at {ollama} does not serve {vlm.model} (has: {names or "nothing"})')
            self.model = vlm.model
            self.log(f'loading {vlm.model} on Ollama...')
            vlm.warm_up()
            root = os.path.join(self.rax_dir, 'runtime', 'ros_pipeline')
            agent = LocalAgent(root, vlm=vlm, backend=lambda cfg, realtime=1.0: RosRobotBackend(self, self.cfg, cfg))
            # questions go to ros_agent's Ask agent when agent_node wired one in (self.ask_fn)
            agent.ask_fn = lambda q: self.ask_fn(q) if getattr(self, 'ask_fn', None) else None
            agent.max_replans = self.max_replans
            self.agent = agent.start()
            for line in self._log_buf:
                self.agent.log(line)
            self.log(f'pipeline agent ready ({vlm.model})')
        except Exception as ex:
            self.agent_error = f'{type(ex).__name__}: {ex}'
            self.get_logger().error(f'[pipeline] agent failed to start: {self.agent_error}')

    GRASP_STYLES = ('auto', 'side', 'top', 'tilted')

    def set_grasp_style(self, style):
        """The operator's pick style (UI): side / top / tilted forces it for every pick; auto lets the agent
        choose (default = grasp_style in pipeline.yaml)."""
        style = str(style or '').lower()
        if style not in self.GRASP_STYLES:
            raise ValueError(f'grasp style must be one of {self.GRASP_STYLES}')
        self.grasp_choice = style
        self.cfg['grasp_style'] = style if style != 'auto' else self._grasp_default   # skills read the shared config
        self.log(f'pick style: {style}')
        return f'pick style: {style}'

    def set_max_replans(self, n):
        n = int(n)
        if not 0 <= n <= 10:
            raise ValueError('max_replans must be 0..10')
        self.max_replans = n
        if self.agent is not None:
            self.agent.max_replans = n
        self.log(f'max replans per task: {n}')
        return f'max replans: {n}'

    def set_model(self, model):
        """The UI picked a model: the VLM detector uses it too (one model for the pipeline)."""
        self.model = model
        if self.agent is not None:
            self.agent.vlm.model = model
            self.agent.vlm.num_ctx = 65536 if model.startswith('rax-') else 8192
            self.agent._set(model=model)
        self.log(f'pipeline model: {model}')

    # world camera calibration: aruco_hand_eye.world_camera (loading, accuracy gate, override)
    def _load_calibration(self):
        from aruco_hand_eye.world_camera import WorldCalibration
        self.calib = WorldCalibration(self.cfg.get('world_calibration', 'eye_to_hand_world_left.yaml'),
                                      float(self.cfg.get('max_calib_t_rms_mm', 10.0)),
                                      float(self.cfg.get('max_calib_r_rms_deg', 2.0)))
        if self.calib.T_base_cam is None and self.cfg.get('allow_inaccurate_calibration') and self.calib._T is not None:
            self.log(self.calib.allow_inaccurate() + ' (allow_inaccurate_calibration in pipeline.yaml)')
        self.T_base_cam, self.calib_problem, self.calib_info = (self.calib.T_base_cam, self.calib.problem,
                                                                self.calib.info)

    def arm_calibration(self, arm):
        """T_base_cam to use for an arm's picks: world_calibration_<arm> if configured (and usable or
        overridden together with the main one), else the main calibration."""
        name = self.cfg.get(f'world_calibration_{arm}')
        if not name:
            return self.T_base_cam
        from aruco_hand_eye.world_camera import WorldCalibration
        cache = self.__dict__.setdefault('_arm_calib', {})
        if arm not in cache:
            cache[arm] = WorldCalibration(name, float(self.cfg.get('max_calib_t_rms_mm', 10.0)),
                                          float(self.cfg.get('max_calib_r_rms_deg', 2.0)))
        c = cache[arm]
        if c.T_base_cam is None and self.calib.info.get('override') and c._T is not None:
            c.allow_inaccurate()                    # the operator accepted the world calibration
        return c.T_base_cam if c.T_base_cam is not None else self.T_base_cam

    def allow_inaccurate_calibration(self):
        msg = self.calib.allow_inaccurate()
        self.T_base_cam, self.calib_problem = self.calib.T_base_cam, self.calib.problem
        self.log(msg)
        return msg

    # ------------------------------------------------------------------ callbacks
    def _joint_cb(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = float(p)
        self.joint_t = time.monotonic()

    def _info_cb(self, msg):
        with self._cam_lock:
            self._cam['K'] = np.array(msg.k, float).reshape(3, 3)
            self._cam['D'] = np.array(msg.d, float) if len(msg.d) else np.zeros(5)

    def _img_cb(self, msg):
        from aruco_hand_eye.ros_image import image_to_bgr
        try:
            bgr = image_to_bgr(msg)
        except Exception as ex:
            self.get_logger().warn(f'[pipeline] camera: {ex}', throttle_duration_sec=10.0)
            return
        with self._cam_lock:
            self._cam['bgr'], self._cam['t'] = bgr, time.monotonic()

    def _depth_cb(self, msg):
        if msg.encoding not in ('16UC1', 'mono16', '32FC1'):
            return
        dt = np.uint16 if msg.encoding != '32FC1' else np.float32
        d = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, -1)[:, :msg.width].astype(np.float32)
        if dt == np.uint16:
            d /= 1000.0            # mm -> m
        with self._cam_lock:
            self._cam['depth'], self._cam['depth_t'] = d, time.monotonic()

    def _motor_cb(self, arm, msg):
        self.motors[arm] = {'t': time.monotonic(), 'motors': [
            {'id': int(m.id), 'mos_c': round(float(m.mos_temp), 1), 'rotor_c': round(float(m.rotor_temp), 1),
             'error': m.error_name} for m in msg.motors]}

    def motor_state(self):
        """{'warn_c', 'stop_c', 'arms': {arm: {'max_c', 'motors': [...], 'fresh'}}, 'hot': [...], 'stop': bool}"""
        now = time.monotonic()
        arms, hot = {}, []
        for arm, d in self.motors.items():
            fresh = now - d['t'] < 3.0
            mx = max((max(m['mos_c'], m['rotor_c']) for m in d['motors']), default=None)
            arms[arm] = {'max_c': mx, 'fresh': fresh, 'motors': d['motors']}
            if fresh:
                hot += [{'arm': arm, 'id': m['id'], 'temp_c': max(m['mos_c'], m['rotor_c'])}
                        for m in d['motors'] if max(m['mos_c'], m['rotor_c']) >= self.motor_temp_warn_c]
        return {'warn_c': self.motor_temp_warn_c, 'stop_c': self.motor_temp_stop_c, 'arms': arms, 'hot': hot,
                'stop': any(h['temp_c'] >= self.motor_temp_stop_c for h in hot)}

    def _safety_tick(self):
        """Run the safety checks; post new alerts to the chat (the page pops them up and speaks them);
        stop a running motion when a person is seen or the arm is in self-collision while it works."""
        alerts = self.safety.check()
        new, cleared = self.safety.to_announce()
        # alerts go to the page as small popups (state 'safety.events'), NOT into the chat
        for a in new:
            self.log(('SAFETY: ' if a['level'] == 'critical' else 'Warning: ') + a['msg'])
            self.safety.event(a['level'], a['msg'], a['key'])
        for k in cleared:
            self.safety.event('cleared', f'{k.split(":")[0]} alert cleared', k)
        busy = bool(self.agent is not None and self.agent.state.get('busy'))
        moving = busy or any(a.get('moving') for a in [])
        stop_keys = {'collision:self'} | ({'human'} if self.safety.c['human_stop'] else set())
        if busy and self.execute and not self.safety.c['ignore_warnings'] and any(a['key'] in stop_keys for a in alerts):
            why = next(a['msg'] for a in alerts if a['key'] in stop_keys)
            self.log(f'SAFETY STOP: {why}')
            self.stop()
            self.safety.event('critical', f'SAFETY STOP: {why}. Robot task stopped.', 'safety_stop')

    def _check_temps(self):
        """Hot motor (>= warn): a chat message (once a minute per motor) + the page's popup (from state);
        an all-clear when it is 5 C below the limit again."""
        st = self.motor_state()
        now = time.monotonic()
        for h in st['hot']:
            key = (h['arm'], h['id'])
            if now - self._temp_alerted.get(key, -1e9) >= 60.0:
                self._temp_alerted[key] = now
                msg = (f"WARNING: {h['arm']} arm motor {h['id']} is at {h['temp_c']:.0f} °C "
                       f"(limit {self.motor_temp_warn_c:.0f} °C)."
                       + (' Motion is stopped until it cools down.' if h['temp_c'] >= self.motor_temp_stop_c
                          else ' Let it cool down before long tasks.'))
                self.log(msg)
                if self.agent is not None:
                    self.agent._say('alert', msg)
        for key in list(self._temp_alerted):
            arm, mid = key
            m = next((m for m in self.motors.get(arm, {}).get('motors', []) if m['id'] == mid), None)
            if m and max(m['mos_c'], m['rotor_c']) < self.motor_temp_warn_c - 5:
                del self._temp_alerted[key]
                if self.agent is not None:
                    self.agent._say('daksha', f'{arm} arm motor {mid} cooled down to '
                                              f'{max(m["mos_c"], m["rotor_c"]):.0f} °C.')

    def wrist_snapshot(self, arm):
        """Fresh wrist frame {bgr, K, D, T_world_cam, T_world_mount, accuracy} or None; the camera pose comes
        from the arm's joints NOW (call it with the arm still)."""
        w = self._wrist.get(arm)
        if w is None or w['msg'] is None or w['K'] is None or w['T_mount_cam'] is None:
            return None
        if time.monotonic() - w['t'] > 1.0 or time.monotonic() - self.joint_t > 0.5:
            return None
        from aruco_hand_eye.ros_image import image_to_bgr
        T_wm = self.fk.transform('world', w['mount'], dict(self.joints))
        bgr = image_to_bgr(w['msg'])
        depth = None
        dm = w['depth_msg']
        if dm is not None and time.monotonic() - w['depth_t'] < 1.0 and dm.encoding in ('16UC1', 'mono16', '32FC1'):
            dt = np.uint16 if dm.encoding != '32FC1' else np.float32
            d = np.frombuffer(dm.data, dtype=dt).reshape(dm.height, -1)[:, :dm.width].astype(np.float32)
            if dt == np.uint16:
                d /= 1000.0                     # mm -> m
            d[d <= 0] = np.nan
            if d.shape[:2] == bgr.shape[:2]:
                depth = d
        return {'bgr': bgr, 'K': w['K'], 'D': w['D'], 'depth': depth,
                'T_world_cam': T_wm @ w['T_mount_cam'], 'T_world_mount': T_wm, 'accuracy': w.get('accuracy')}

    def camera_snapshot(self):
        """Fresh world frame {bgr, K, D, depth?} or None."""
        with self._cam_lock:
            c = dict(self._cam)
        now = time.monotonic()
        if c['bgr'] is None or c['K'] is None or now - c['t'] > 1.0:
            return None
        if c['depth'] is not None and (now - c['depth_t'] > 1.0 or c['depth'].shape[:2] != c['bgr'].shape[:2]):
            c['depth'] = None
        return c

    def _write_live_frame(self):
        """Feed the agent's live frame (it answers questions from stream/latest.jpg) and the UI."""
        if self.agent is None:
            return
        snap = self.camera_snapshot()
        if snap is None:
            return
        import cv2
        tmp = os.path.join(self.agent.stream_dir, 'latest.tmp.jpg')
        ok, jpg = cv2.imencode('.jpg', snap['bgr'], [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with open(tmp, 'wb') as f:
                f.write(jpg.tobytes())
            os.replace(tmp, os.path.join(self.agent.stream_dir, 'latest.jpg'))

    # ------------------------------------------------------------------ actions
    @property
    def execute(self):
        """aruco_reach's execute switch (False if it isn't running)."""
        return bool(self.motion.execute)

    def _apply_requested_execute(self):
        """pipeline_execute:=true arms aruco_reach once it is up."""
        if self._want_execute and self.motion.connected:
            self._want_execute = False
            try:
                self.log(self.motion.set_execute(True))
            except Exception as ex:
                self.log(f'could not enable execute on aruco_reach: {ex}')

    def submit(self, text):
        if self.agent is None:
            return {'ok': False, 'message': self.agent_error or 'the pipeline agent is still starting'}
        self.agent.submit(text)
        return {'ok': True, 'message': 'sent'}

    def stop(self):
        if self.agent is not None:
            self.agent.stop()
        return 'stopped; aruco_reach: ' + self.motion.stop()

    def emergency_stop(self):
        """The page's STOP button: stop the task and the arms, and switch execute OFF so nothing moves
        until the operator enables it again."""
        msg = self.stop()
        try:
            msg += ' | ' + self.motion.set_execute(False)
        except Exception as ex:
            msg += f' | could not switch execute off: {ex}'
        self.log(f'EMERGENCY STOP: {msg}')
        if hasattr(self, 'safety'):
            self.safety.event('warn', 'STOP pressed: task stopped, arms holding, execute is OFF.', 'estop')
        return msg

    def set_execute(self, on):
        """Toggles aruco_reach's execute switch (the one place real motion is armed)."""
        on = bool(on)
        if not on and self.agent is not None:
            self.agent.stop()
        try:
            msg = self.motion.set_execute(on)
        except RuntimeError as ex:
            msg = f'could not reach aruco_reach ({ex}). Start: ros2 launch aruco_hand_eye aruco_reach.launch.py'
        self.log(f'aruco_reach: {msg}')
        return msg

    def _set_table(self, z, source):
        self.table_z = float(z)
        s = self.cfg.get('scene', {})
        geo = self.R.apply_scene(table_z=self.table_z, grasp_dz=s.get('grasp_dz', 0.014),
                                 pregrasp_dz=s.get('pregrasp_dz'), release_dz=s.get('release_dz', 0.17),
                                 clear_pose=s.get('clear_pose'), handover_pose=s.get('handover_pose'),
                                 reach=s.get('reach'), regions=s.get('regions'),
                                 grasp_approach=s.get('grasp_approach'),
                                 grasp_tolerance_deg=s.get('grasp_tolerance_deg'))
        self.table_source = source
        self.log(f'table height {self.table_z:.3f} m ({source}); grasp TCP z {geo["grasp_tcp_z"]:.3f}')

    def measure_table(self):
        """Table height from an ArUco marker lying flat on the table, seen by the world camera."""
        try:
            if self.T_base_cam is None:
                raise ValueError(self.calib_problem)
            snap = self.camera_snapshot()
            if snap is None:
                raise ValueError('no recent world camera image')
            from aruco_hand_eye.world_camera import measure_table_z
            z, p = measure_table_z(snap['bgr'], snap['K'], snap['D'], self.T_world_base @ self.T_base_cam,
                                   self.cfg.get('table_marker'), float(self.cfg.get('table_marker_thickness_m', 0.0)))
            self._set_table(z, 'measured from ArUco')
            return True, f'table height {self.table_z:.3f} m (marker at x {p[0]:.3f}, y {p[1]:.3f})'
        except Exception as ex:
            return False, f'could not measure the table: {ex}'

    def log(self, msg):
        if self.agent is not None:
            self.agent.log(msg)
        elif self.with_agent:
            self._log_buf.append(msg)
        self.get_logger().info(f'[pipeline] {msg}')

    @staticmethod
    def _trig(res, *vals):
        if len(vals) == 2:
            res.success, res.message = bool(vals[0]), str(vals[1])
        else:
            res.success, res.message = True, str(vals[0])
        return res

    def state(self):
        snap = self.agent.snapshot() if self.agent is not None else {}
        now = time.monotonic()
        return {
            'ready': self.agent is not None, 'error': self.agent_error, 'model': self.model,
            'execute': self.execute, 'motion_server': self.motion.connected,
            'joints_ok': now - self.joint_t < 0.5, 'max_replans': self.max_replans,
            'grasp_choice': self.grasp_choice, 'grasp_style': self.cfg.get('grasp_style'),
            'motors': self.motor_state(),
            'safety': self.safety.state() if hasattr(self, 'safety') else None,
            'camera_ok': self.camera_snapshot() is not None,
            'depth': self._cam['depth'] is not None and now - self._cam['depth_t'] < 1.0,
            'table_z': self.table_z, 'table_source': getattr(self, 'table_source', None),
            'calibration': {**self.calib_info, 'usable': self.T_base_cam is not None,
                            'problem': self.calib_problem},
            'objects': self.cfg.get('objects'), 'containers': self.cfg.get('containers'),
            **{k: snap.get(k) for k in ('busy', 'current', 'queued', 'thinking', 'held', 'chat', 'logs',
                                        'interrupts', 'detections', 'grasp', 'results', 'focus', 'plan')},
            'grippers': self.gripper_feedback(snap.get('held') or {}),
        }

    # ------------------------------------------------------------------ detected objects (3-D boxes, goal, TF)
    @staticmethod
    def _slug(name):
        return ''.join(ch if ch.isalnum() else '_' for ch in str(name).lower()).strip('_') or 'object'

    def _goal_name(self, snap, names):
        """The detected object the current (or last) instruction is about."""
        texts = [snap.get('current') or '']
        for c in reversed(snap.get('chat') or []):
            who = c.get('who') if isinstance(c, dict) else c[0]
            if who == 'operator':
                texts.append(c.get('text', '') if isinstance(c, dict) else c[1])
                break
        for t in texts:
            hits = [n for n in names if n.lower() in str(t).lower()]
            if hits:
                return max(hits, key=len)         # "brown box" over "box"
        return None

    # links whose neighbourhood is "the hand" in the world image: wrist, gripper + its camera housing, fingers
    HAND_LINKS = ('link_7', 'link_8', 'gripper_mount', 'right_finger', 'left_finger', 'tcp')
    HAND_RADIUS_M = 0.045          # sphere sampled around each link origin (covers the camera housing)
    HAND_PAD_PX = 12               # + world calibration error (~1-2 cm)

    def hand_boxes(self, skip=()):
        """{arm: [x1, y1, x2, y2]}: world-camera pixel box around each hand (gripper, fingers, wrist
        camera housing), from the joints NOW + the world calibration. Arms in `skip` (e.g. holding the
        object) are left out. {} when joints / calibration / camera info are not available."""
        import cv2
        K, D = self._cam.get('K'), self._cam.get('D')
        if K is None or self.T_base_cam is None or time.monotonic() - self.joint_t > 0.5:
            return {}
        q = dict(self.joints)
        bgr = self._cam.get('bgr')
        H, W = (bgr.shape[:2] if bgr is not None else (2 * K[1, 2], 2 * K[0, 2]))
        T_cam_world = np.linalg.inv(self.T_world_base @ self.T_base_cam)
        rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
        r = self.HAND_RADIUS_M
        cube = np.array([[x, y, z] for x in (-r, 0, r) for y in (-r, 0, r) for z in (-r, 0, r)])
        out = {}
        for arm in ('left', 'right'):
            if arm in skip:
                continue
            pts = []
            for link in self.HAND_LINKS:
                try:
                    p = self.fk.transform('world', f'{arm}_{link}', q)[:3, 3]
                except (KeyError, ValueError):
                    continue
                pts.extend(p + cube)
            w = self._wrist.get(arm) or {}
            if w.get('mount') and w.get('T_mount_cam') is not None:      # the wrist camera itself
                try:
                    pts.extend((self.fk.transform('world', w['mount'], q) @ w['T_mount_cam'])[:3, 3] + cube)
                except (KeyError, ValueError):
                    pass
            if not pts:
                continue
            P = np.array(pts)
            Pc = (T_cam_world[:3, :3] @ P.T).T + T_cam_world[:3, 3]
            # only points in front of the camera and inside (a margin around) its field of view: the
            # distortion model sends points far outside it to absurd pixels
            front = Pc[:, 2] > 0.05
            P, Pc = P[front], Pc[front]
            uv0 = (K[:2, :2] @ (Pc[:, :2] / Pc[:, 2:3]).T).T + K[:2, 2]
            inside = (uv0[:, 0] > -0.2 * W) & (uv0[:, 0] < 1.2 * W) & (uv0[:, 1] > -0.2 * H) & (uv0[:, 1] < 1.2 * H)
            P = P[inside]
            if len(P) < 4:
                continue
            uv, _ = cv2.projectPoints(P, rvec, T_cam_world[:3, 3], K, D)
            uv = uv.reshape(-1, 2)
            pad = self.HAND_PAD_PX
            box = [max(0.0, uv[:, 0].min() - pad), max(0.0, uv[:, 1].min() - pad),
                   min(W, uv[:, 0].max() + pad), min(H, uv[:, 1].max() + pad)]
            if box[2] > box[0] and box[3] > box[1]:
                out[arm] = [float(v) for v in box]
        return out

    def robot_part(self, bbox, skip=(), frac=0.6):
        """The arm whose hand covers >= frac of detection `bbox` (world-camera pixels) in the image, else
        None. A vision model asked for e.g. "black cube" sometimes boxes the black wrist camera housing."""
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            hands = self.hand_boxes(skip)
        except Exception:
            return None
        area = max((x2 - x1) * (y2 - y1), 1.0)
        for arm, (hx1, hy1, hx2, hy2) in hands.items():
            iw, ih = min(x2, hx2) - max(x1, hx1), min(y2, hy2) - max(y1, hy1)
            if iw > 0 and ih > 0 and iw * ih / area >= frac:
                return arm
        return None

    def held_arms(self):
        """Arms the agent records as holding something (their hand may legitimately overlap an object)."""
        snap = self.agent.snapshot() if self.agent is not None else {}
        held = [a for a, o in (snap.get('held') or {}).items() if o]
        for a, g in (snap.get('grippers') or {}).items():
            if (g or {}).get('status') == 'holding' and a not in held:
                held.append(a)
        return held

    def _ray_to_plane(self, uv, z_world):
        from aruco_hand_eye.world_camera import ray_to_plane
        return ray_to_plane(uv, self._cam['K'], self._cam['D'], self.T_world_base @ self.T_base_cam, z_world)

    def objects_3d(self):
        """[{name, goal, top (world), box_centre, size [x, y, z] (m), yaw, bbox}] from the latest detections.
        Footprint width = the bbox's bottom edge back-projected onto the table; height = top - table."""
        snap = self.agent.snapshot() if self.agent is not None else {}
        dets = {k: v for k, v in (snap.get('detections') or {}).items() if v.get('centre')}
        goal = snap.get('focus') if snap.get('focus') in dets else self._goal_name(snap, list(dets))
        out = []
        for name, e in dets.items():
            c = np.array(e['centre'], float)
            tz = self.table_z if self.table_z is not None else c[2] - 0.05
            h = float(np.clip(c[2] - tz, 0.03, 0.5))
            w = 0.06
            bb = e.get('bbox')
            if bb and self._cam['K'] is not None and self.T_base_cam is not None:
                p1, p2 = self._ray_to_plane([bb[0], bb[3]], tz), self._ray_to_plane([bb[2], bb[3]], tz)
                if p1 is not None and p2 is not None:
                    w = float(np.clip(np.linalg.norm(p1[:2] - p2[:2]), 0.02, 0.5))
            out.append({'name': name, 'goal': name == goal, 'top': c.round(4).tolist(),
                        'box_centre': [round(c[0], 4), round(c[1], 4), round(tz + h / 2, 4)],
                        'size': [round(w, 3), round(w, 3), round(h, 3)], 'yaw': float(e.get('yaw', 0.0) or 0.0),
                        'bbox': bb, 'source': e.get('source')})
        # the live-tracked object (CSRT box, every frame): position from its footprint on the table
        tr = self.tracker.state() if self.tracker is not None else {}
        if tr.get('camera') == 'world' and tr.get('box') and tr.get('status') not in ('idle', 'detecting') \
                and self._cam['K'] is not None and self.T_base_cam is not None:
            name = self.tracked or tr.get('label') or tr.get('query') or 'tracked'
            x1, y1, x2, y2 = (float(v) for v in tr['box'])
            tz = self.table_z if self.table_z is not None else 0.8
            prev = next((o for o in out if o['name'] == name), None)
            h = prev['size'][2] if prev else 0.06
            p1, p2 = self._ray_to_plane([x1, y2], tz), self._ray_to_plane([x2, y2], tz)
            pc = self._ray_to_plane([(x1 + x2) / 2, y2], tz)
            if p1 is not None and p2 is not None and pc is not None:
                w = float(np.clip(np.linalg.norm(p1[:2] - p2[:2]), 0.02, 0.5))
                c = pc[:2] + (pc[:2] - self.T_world_base[:2, 3]) / max(np.linalg.norm(pc[:2] - self.T_world_base[:2, 3]), 1e-6) * w / 2
                o = {'name': name, 'goal': name == goal or prev is None and goal is None, 'tracked': True,
                     'top': [round(c[0], 4), round(c[1], 4), round(tz + h, 4)],
                     'box_centre': [round(c[0], 4), round(c[1], 4), round(tz + h / 2, 4)],
                     'size': [round(w, 3), round(w, 3), round(h, 3)], 'yaw': 0.0,
                     'bbox': [x1, y1, x2, y2], 'source': 'tracker'}
                if prev is not None:
                    o['goal'] = prev['goal']
                    out.remove(prev)
                out.append(o)
        return out

    def _publish_objects(self):
        try:
            objs = self.objects_3d() + self._wrist_objects()
        except Exception as ex:
            self.get_logger().warn(f'[pipeline] objects: {ex}', throttle_duration_sec=10.0)
            return
        self.objects_pub.publish(String(data=json.dumps({'frame': 'world', 'objects': objs})))
        if not objs:
            return
        from geometry_msgs.msg import TransformStamped
        stamp = self.get_clock().now().to_msg()
        T_base_world = np.linalg.inv(self.T_world_base)
        msgs = []
        for o in objs:
            p = T_base_world @ np.r_[o['top'], 1.0]
            yaw = o['yaw']
            extra = [f'wrist_goal_{o["arm"]}'] if o.get('wrist') else []
            for child in [f'object_{self._slug(o["name"])}' + ('_wrist' if o.get('wrist') else '')] + \
                    (['pipeline_goal'] if o['goal'] else []) + extra:
                t = TransformStamped()
                t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'base_link', child
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p[:3])
                t.transform.rotation.z, t.transform.rotation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
                msgs.append(t)
        self.tf_pub.sendTransform(msgs)

    def _wrist_objects(self):
        """The object as the wrist camera measures it during an approach (fresh for 2 s)."""
        w = self.wrist_obj
        if not w or time.monotonic() - w['t'] > 2.0:
            return []
        p = w['world']
        return [{'name': f"{w['name']} (wrist)", 'goal': False, 'wrist': True, 'arm': w['arm'],
                 'top': [round(v, 4) for v in p], 'box_centre': [round(p[0], 4), round(p[1], 4), round(p[2], 4)],
                 'size': [0.04, 0.04, 0.04], 'yaw': 0.0, 'bbox': None, 'source': 'wrist depth'}]

    def box_everywhere(self, name):
        """Put a live box on `name` in ALL three cameras: world + both wrist (gripper) cameras. Each camera
        has its own tracker (one vision-model detection, then CSRT every frame); a camera that cannot
        see it keeps looking slowly and shows 'looking for ...'. Non-blocking."""
        q = str(name).replace('_', ' ')
        trackers = [('world', self.tracker)] + [(f'{a}_wrist', t) for a, t in self.wrist_trackers.items()]
        for cam, tr in trackers:
            if tr is None:
                continue
            st = tr.state()
            if st.get('status') in ('tracking', 'detecting', 'searching') and (st.get('query') or '') == q \
                    and st.get('camera') == cam:
                continue                       # already boxing this object there
            try:
                tr.start(cam, query=q)
            except Exception as ex:
                self.get_logger().warn(f'[pipeline] box {q} in {cam}: {ex}', throttle_duration_sec=10.0)
        if self.tracker is not None:
            self.tracked = name

    def wrist_view_jpeg(self, arm):
        """This arm's wrist image with the wrist tracker's box (and the fingertip reference)."""
        import cv2
        from aruco_hand_eye.ros_image import image_to_bgr
        w = self._wrist.get(arm)
        if w is None or w['msg'] is None:
            return None
        img = image_to_bgr(w['msg']).copy()
        tr = self.wrist_trackers.get(arm)
        st = tr.state() if tr is not None else {}
        if st.get('camera') == f'{arm}_wrist' and not st.get('box') and st.get('status') not in (None, 'idle'):
            cv2.putText(img, f"looking for {st.get('label') or st.get('query')} ({st.get('status')})", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 255), 2)
        if st.get('camera') == f'{arm}_wrist' and st.get('box'):
            x1, y1, x2, y2 = (int(v) for v in st['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 200, 40), 3)
            tag = f"{st.get('label') or 'object'} ({st.get('status')})"
            ob = self.wrist_obj
            if ob and ob.get('arm') == arm and time.monotonic() - ob['t'] < 2.0:
                tag += ' ({:.3f}, {:.3f}, {:.3f})'.format(*ob['world'])
            cv2.putText(img, tag, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 200, 40), 2)
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes() if ok else None

    # finger joint per arm (width = 2 * |joint|), as aruco_reach's GRIPPER map
    GRIPPER_JOINTS = {'left': 'left_gripper_left_joint', 'right': 'right_gripper_right_joint'}

    def gripper_feedback(self, held):
        """Live: width between the fingers and whether a held object still looks held."""
        out = {}
        fresh = time.monotonic() - self.joint_t < 0.5
        for arm, j in self.GRIPPER_JOINTS.items():
            # joint at its travel (0.044) = closed, 0 = open (aruco_reach gripper_closed_at_travel)
            w = max(0.0, 2.0 * (0.044 - abs(self.joints[j]))) if fresh and j in self.joints else None
            obj = held.get(arm)
            if w is None:
                status = 'unknown'
            elif obj:
                status = 'holding' if w >= 0.012 else 'dropped?'   # closed on nothing = lost it
            elif 0.012 < w < 0.08:
                status, obj = 'holding', 'something'                # part-closed: stalled on an object
            else:
                status = 'empty'
            out[arm] = {'width_m': None if w is None else round(w, 4), 'object': obj, 'status': status}
        return out


def main():
    rclpy.init()
    node = PipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
