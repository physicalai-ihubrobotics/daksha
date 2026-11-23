#!/usr/bin/env python3
"""
Camera UI — a Flask operator console for mapping the robot's RealSense
cameras to their roles (World, Left gripper, Right gripper).

What it does
------------
  * Scans the USB bus for connected Intel RealSense devices and lists every
    serial number it finds (via pyrealsense2 — same library realsense2_camera
    uses under the hood).
  * Shows each detected camera *live* (color stream, MJPEG) so you can tell
    the physical cameras apart by just looking at them.
  * Lets you drag each live feed into one of three role slots and then saves
    that serial -> role mapping straight into
    camera_launch/config/cameras.yaml (the file start_all_cameras.launch.py
    reads), leaving every comment in that file untouched.

Why it exists
-------------
The D405 wrist cameras are identical and have been mislabelled before (see
the "swapped serials" comment in cameras.yaml). This tool removes the
guesswork: wiggle a camera, see which live tile moves, drag it into the right
slot, Save.

While the camera bringup is running
-----------------------------------
A RealSense device can only be opened by one process at a time, so while the
camera bringup (start_all_cameras.launch.py) has a camera open this tool
can't read it directly. It then shows that camera's picture from the bringup
instead: the JPEG frames on /<namespace>/<name>/color/image_raw/compressed,
matched to the camera by the running realsense node's own serial_no
parameter. Stop the bringup for the direct (lowest-latency) preview.

Run (after sourcing your ROS 2 workspace):
    pip install flask opencv-python pyrealsense2 pyyaml
    ros2 run camera_launch camera_ui      # open http://<hostname>:8120
"""

import os
import re
import json
import time
import glob
import signal
import socket
import logging
import threading

import numpy as np
import cv2
import yaml
from flask import Flask, Response, request, jsonify, render_template

import sys as _sys
# daksha_web_access.py is installed by the automation package into its lib dir.
from ament_index_python.packages import get_package_prefix
_AUTOMATION_DIR = os.path.join(get_package_prefix("automation"), "lib", "automation")
if _AUTOMATION_DIR not in _sys.path:
    _sys.path.insert(0, _AUTOMATION_DIR)
from daksha_web_access import setup_flask_gate

logging.getLogger("werkzeug").setLevel(logging.WARNING)

# pyrealsense2 is what actually talks to the cameras. Kept optional at import
# so the UI can still be opened (e.g. to inspect/edit assignments) on a box
# where it isn't installed — the scan just comes back empty with a clear note.
try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    RS_AVAILABLE = False
    RS_IMPORT_ERROR = str(_e)

# ROS is optional too. The node registers so it shows up in `ros2 node list`
# and logs through rclpy like the other operator consoles in this workspace,
# and it subscribes to the bringup's compressed color topics to preview
# cameras the bringup is holding (see RosPreview).
try:
    import rclpy
    from rclpy.node import Node
    ROS_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    ROS_AVAILABLE = False
    ROS_IMPORT_ERROR = str(_e)
    Node = object

try:
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from rcl_interfaces.srv import GetParameters
    ROS_PREVIEW = ROS_AVAILABLE
except Exception:  # pragma: no cover
    ROS_PREVIEW = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SELF_PORT = 8120                     # workspace convention: operator UIs live in 8100-8199
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_FPS = 30
JPEG_QUALITY = 70
ROS_FRAME_MAX_AGE = 2.0              # s; older bringup frames count as no picture
ROS_REFRESH_PERIOD = 5.0             # s; re-read topics and ask nodes their serial

# Default serial per role. These are the node's <role>_serial parameter
# defaults; override at launch without editing this file, e.g.:
#     ros2 run camera_launch camera_ui --ros-args \
#         -p world_serial:=_313522301897 -p left_serial:=_260322273428
# The leading underscore is the realsense2_camera string convention; the UI
# normalizes it away for display/compare (see norm_serial).
DEFAULT_SERIALS = {
    "world": "_313522301897",
    "left":  "_260322273428",
    "right": "_260322274109",
}

# The three role slots, in display order. The `key` is the top-level block
# name inside cameras.yaml whose `serial_no:` we rewrite on Save.
ROLES = [
    {"key": "world", "label": "World", "hint": "D455 overhead / scene camera"},
    {"key": "left",  "label": "Left gripper",  "hint": "D405 on the left wrist"},
    {"key": "right", "label": "Right gripper", "hint": "D405 on the right wrist"},
]
ROLE_KEYS = [r["key"] for r in ROLES]


def _share_yaml_path():
    """The installed share-dir cameras.yaml -- what the running launch reads.
    None if the package share dir can't be resolved (e.g. ROS not sourced)."""
    if not ROS_AVAILABLE:
        return None
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.normpath(os.path.join(
            get_package_share_directory('camera_launch'), 'config', 'cameras.yaml'))
        return p if os.path.exists(p) else None
    except Exception:
        return None


def _source_yaml_path():
    """The SOURCE-tree cameras.yaml (version controlled). Found either next to
    this module (running from src / symlink-install) or by searching the
    workspace's src/ tree -- the workspace root is the parent of the install
    prefix, which is where a copy-install's share dir lives."""
    local = os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'cameras.yaml'))
    if os.path.exists(local) and f'{os.sep}src{os.sep}' in local:
        return local
    share = _share_yaml_path()
    if share:
        parts = share.split(os.sep)
        if 'install' in parts:                       # <ws>/install/<pkg>/share/...
            ws = os.sep.join(parts[:parts.index('install')])
            hits = glob.glob(os.path.join(ws, 'src', '**', 'camera_launch',
                                          'config', 'cameras.yaml'), recursive=True)
            if hits:
                return os.path.normpath(hits[0])
    return local if os.path.exists(local) else None


def _cameras_yaml_path():
    """Primary cameras.yaml for reads/display: the source file if we can find
    it, otherwise the installed copy."""
    return _source_yaml_path() or _share_yaml_path() or os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'cameras.yaml'))


CAMERAS_YAML = _cameras_yaml_path()


def norm_serial(s):
    """Hardware reports a bare numeric serial ("313522301897"); cameras.yaml
    stores it with a leading underscore ("_313522301897") so realsense2_camera
    treats it as a string. Compare/display on the bare form."""
    if s is None:
        return None
    s = str(s)
    return s[1:] if s.startswith('_') else s


# --------------------------------------------------------------------------- #
# Live preview: one pipeline per device, each in its own grab thread, each
# holding only its most recent JPEG frame for the MJPEG endpoints to read.
# --------------------------------------------------------------------------- #
class CameraStreamer:
    def __init__(self, serial, ctx):
        self.serial = serial              # bare serial
        self._ctx = ctx
        self._lock = threading.Lock()
        self._jpeg = None                 # latest encoded frame
        self._status = "connecting"       # connecting | live | busy | error
        self._detail = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"rs-{serial}")
        self._thread.start()

    @property
    def status(self):
        with self._lock:
            return self._status, self._detail

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def _set_status(self, status, detail=""):
        with self._lock:
            self._status = status
            self._detail = detail

    def _start_pipeline(self):
        """Try color at the preview profile; fall back to whatever color
        format the device offers if that exact profile is rejected."""
        pipeline = rs.pipeline(self._ctx)
        cfg = rs.config()
        cfg.enable_device(self.serial)
        try:
            cfg.enable_stream(rs.stream.color, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                              rs.format.bgr8, PREVIEW_FPS)
            pipeline.start(cfg)
            return pipeline
        except Exception:
            # Retry letting the device pick its own default color profile.
            pipeline = rs.pipeline(self._ctx)
            cfg = rs.config()
            cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color)
            pipeline.start(cfg)
            return pipeline

    def _run(self):
        pipeline = None
        try:
            pipeline = self._start_pipeline()
        except Exception as e:
            msg = str(e)
            # "Device or resource busy" => the bringup (or another process)
            # already has the camera open.
            if "busy" in msg.lower() or "access denied" in msg.lower():
                self._set_status("busy", "Camera is open in another process "
                                         "(stop the camera bringup first).")
            else:
                self._set_status("error", msg)
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        try:
            while not self._stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(2000)
                except Exception:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                ok, buf = cv2.imencode('.jpg', img, encode_params)
                if not ok:
                    continue
                with self._lock:
                    self._jpeg = buf.tobytes()
                    self._status = "live"
                    self._detail = ""
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    @property
    def finished(self):
        """The grab thread has ended (the camera was busy or failed)."""
        return not self._thread.is_alive()

    def stop(self):
        # Join the grab thread so the pipeline is fully stopped before the
        # RealSense context is torn down -- otherwise librealsense can abort
        # on a thread still touching a destroyed context at process exit.
        self._stop.set()
        self._thread.join(timeout=3.0)


class DeviceManager:
    """Owns the shared RealSense context, the current scan result, and the
    per-serial streamers. Thread-safe for Flask's request threads."""

    def __init__(self, logger=None):
        self._lock = threading.Lock()
        self._ctx = rs.context() if RS_AVAILABLE else None
        self._devices = {}        # serial -> {serial,name,usb}
        self._streamers = {}      # serial -> CameraStreamer
        self._log = logger

    def _info(self, dev, field, default=""):
        try:
            return dev.get_info(field)
        except Exception:
            return default

    def scan(self, retry_busy=False):
        """Enumerate connected RealSense devices, (re)start streamers for new
        ones, drop streamers for devices that went away. With retry_busy (the
        Rescan button), also try again to open cameras that were busy, e.g.
        after the bringup stopped. Never automatically: a camera this tool
        holds is one a bringup started afterwards can't open."""
        if not RS_AVAILABLE:
            return []
        found = {}
        try:
            for dev in self._ctx.query_devices():
                serial = self._info(dev, rs.camera_info.serial_number)
                if not serial:
                    continue
                found[serial] = {
                    "serial": serial,
                    "name": self._info(dev, rs.camera_info.name, "RealSense"),
                    "usb": self._info(dev, rs.camera_info.usb_type_descriptor, "?"),
                }
        except Exception as e:
            if self._log:
                self._log(f"RealSense scan failed: {e}", "warn")
            return []

        with self._lock:
            self._devices = found
            # Start streaming any newly seen device.
            for serial in found:
                if serial not in self._streamers:
                    self._streamers[serial] = CameraStreamer(serial, self._ctx)
                elif retry_busy and self._streamers[serial].finished:
                    self._streamers[serial] = CameraStreamer(serial, self._ctx)
            # Tear down streamers whose device unplugged.
            for serial in list(self._streamers):
                if serial not in found:
                    self._streamers.pop(serial).stop()
        return list(found.values())

    def devices(self):
        with self._lock:
            out = []
            for serial, meta in self._devices.items():
                streamer = self._streamers.get(serial)
                status, detail = streamer.status if streamer else ("error", "")
                source = "device" if status == "live" else None
                if status == "busy":
                    _, node_name = ros_frame(serial)
                    if node_name:
                        status, source = "live", "ros"
                        detail = (f"Live view from the bringup's {node_name} node "
                                  "(the bringup has this camera open).")
                out.append({**meta, "status": status, "detail": detail, "source": source})
            return out

    def streamer(self, serial):
        with self._lock:
            return self._streamers.get(serial)

    def stop_all(self):
        with self._lock:
            for s in self._streamers.values():
                s.stop()
            self._streamers.clear()


# --------------------------------------------------------------------------- #
# cameras.yaml read / write. Reading is plain PyYAML; writing is a surgical
# line edit of just the `serial_no:` lines so every comment in that
# heavily-documented file survives untouched.
# --------------------------------------------------------------------------- #
def read_assignments():
    """role_key -> bare serial currently in cameras.yaml (or None)."""
    out = {k: None for k in ROLE_KEYS}
    try:
        with open(CAMERAS_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        for key, cam in (cfg.get('cameras') or {}).items():
            if key in out and cam and cam.get('serial_no'):
                out[key] = norm_serial(cam.get('serial_no'))
    except Exception:
        pass
    return out


_KEY_RE = re.compile(r'^(\s{2})([A-Za-z0-9_]+):\s*$')
_SERIAL_RE = re.compile(r'^(\s{4})serial_no:\s*.*$')


def yaml_targets():
    """Every cameras.yaml Save should write: the source file (version control)
    and the installed copy (used at runtime). Deduped by real path, so a
    symlink-install where install points back at src is written only once."""
    targets = []
    seen = set()
    for p in (_source_yaml_path(), _share_yaml_path()):
        if not p or not os.path.exists(p):
            continue
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            targets.append(p)
    return targets


def _rewrite_serials(path, mapping):
    """Rewrite only the matching `serial_no:` lines in one file, preserving
    indentation, comments, and everything else. Returns the roles changed."""
    with open(path) as f:
        lines = f.readlines()

    in_cameras = False
    cur_key = None
    changed = []
    out = []
    for line in lines:
        if re.match(r'^cameras:\s*$', line):
            in_cameras, cur_key = True, None
            out.append(line)
            continue
        if re.match(r'^\S', line):            # back to top level
            in_cameras, cur_key = False, None
            out.append(line)
            continue
        if in_cameras:
            m = _KEY_RE.match(line)
            if m:
                cur_key = m.group(2)
                out.append(line)
                continue
            if cur_key in mapping and mapping[cur_key] and _SERIAL_RE.match(line):
                indent = _SERIAL_RE.match(line).group(1)
                out.append(f'{indent}serial_no: "_{mapping[cur_key]}"\n')
                changed.append(cur_key)
                continue
        out.append(line)

    with open(path, 'w') as f:
        f.writelines(out)
    return changed


def write_assignments(mapping):
    """mapping: role_key -> bare serial (or None to leave that block alone).
    Writes every target in yaml_targets() (source + installed copy). Returns
    (changed_roles, written_paths)."""
    written = []
    changed = []
    for path in yaml_targets():
        changed = _rewrite_serials(path, mapping)
        written.append(path)
    return changed, written


# --------------------------------------------------------------------------- #
# Flask app
#
# Templates/static live next to this module (the symlink-install / run-in-place
# case); on a plain copy-install they're only under the package share dir, so
# resolve whichever exists.
# --------------------------------------------------------------------------- #
def _resource_dirs():
    here = os.path.dirname(os.path.realpath(__file__))
    tmpl = os.path.join(here, 'templates')
    static = os.path.join(here, 'static')
    if os.path.isdir(tmpl) and os.path.isdir(static):
        return tmpl, static
    if ROS_AVAILABLE:
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('camera_launch')
            return os.path.join(share, 'templates'), os.path.join(share, 'static')
        except Exception:
            pass
    return tmpl, static


_TMPL_DIR, _STATIC_DIR = _resource_dirs()
app = Flask(__name__, template_folder=_TMPL_DIR, static_folder=_STATIC_DIR)
setup_flask_gate(app, "camera_setup")

manager = None            # DeviceManager, set in main()
# Pending (unsaved) assignments edited in the UI: role_key -> bare serial|None.
# Seeded from cameras.yaml on startup.
assignments = {k: None for k in ROLE_KEYS}
assign_lock = threading.Lock()
_logger = None


def log(msg, level="info"):
    # Each severity gets its own call site: rclpy's logger keys on the caller's
    # source location and raises "Logger severity cannot be changed between
    # calls" if one line logs at more than one level. The try/except keeps a
    # logging hiccup from ever taking down a caller (e.g. the signal handler).
    try:
        if _logger is not None:
            if level == "warn":
                _logger.warning(msg)
            elif level == "error":
                _logger.error(msg)
            else:
                _logger.info(msg)
            return
    except Exception:
        pass
    print(f"[{level.upper()}] {msg}")


@app.route('/')
def index():
    return render_template('index.html',
                           roles=ROLES,
                           yaml_path=CAMERAS_YAML,
                           port=SELF_PORT)


@app.route('/api/scan')
def api_scan():
    if manager:
        # refresh device set / streamers; ?rescan=1 (the Rescan button) also
        # retries cameras that were busy
        manager.scan(retry_busy=request.args.get('rescan') == '1')
        devices = manager.devices()    # status-enriched view
    else:
        devices = []
    with assign_lock:
        current = dict(assignments)
    note = None
    if not RS_AVAILABLE:
        note = f"pyrealsense2 not available ({RS_IMPORT_ERROR}). Cannot scan hardware."
    elif not devices:
        note = ("No RealSense cameras detected. Check the USB cables, and make "
                "sure the camera bringup isn't already holding the devices.")
    return jsonify({
        "ok": True,
        "devices": devices,
        "assignments": current,
        "saved": read_assignments(),
        "yaml_path": CAMERAS_YAML,
        "rs_available": RS_AVAILABLE,
        "note": note,
    })


@app.route('/api/assign', methods=['POST'])
def api_assign():
    data = request.get_json(force=True, silent=True) or {}
    role = data.get('role')
    serial = data.get('serial')          # None / "" clears the slot
    serial = norm_serial(serial) if serial else None
    if role not in ROLE_KEYS:
        return jsonify({"ok": False, "error": f"unknown role '{role}'"}), 400
    with assign_lock:
        # A camera can only fill one slot — drop it from any other slot first.
        if serial:
            for k in ROLE_KEYS:
                if assignments[k] == serial and k != role:
                    assignments[k] = None
        assignments[role] = serial
        current = dict(assignments)
    return jsonify({"ok": True, "assignments": current})


@app.route('/api/save', methods=['POST'])
def api_save():
    with assign_lock:
        current = dict(assignments)
    assigned = {k: v for k, v in current.items() if v}
    if not assigned:
        return jsonify({"ok": False, "error": "Nothing assigned yet — drag a "
                                              "camera into at least one slot."}), 400
    # Distinctness guard: the same serial in two slots is always a mistake.
    dupes = [s for s in set(assigned.values())
             if list(assigned.values()).count(s) > 1]
    if dupes:
        return jsonify({"ok": False,
                        "error": f"Serial {dupes[0]} is assigned to two roles."}), 400
    try:
        changed, written = write_assignments(assigned)
    except Exception as e:
        log(f"Failed to write cameras.yaml: {e}", "error")
        return jsonify({"ok": False, "error": str(e)}), 500
    if not written:
        return jsonify({"ok": False, "error": "No cameras.yaml file found to write."}), 500
    serials = ", ".join(f"{k}=_{v}" for k, v in assigned.items())
    log(f"Saved camera assignments ({serials}) to: " + ", ".join(written))
    return jsonify({"ok": True, "changed": changed, "written": written,
                    "saved": read_assignments(), "yaml_path": CAMERAS_YAML})


def _placeholder_jpeg(text):
    img = np.full((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), 30, dtype=np.uint8)
    cv2.putText(img, text, (24, PREVIEW_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 170, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ok else b""


@app.route('/stream/<serial>')
def stream(serial):
    serial = norm_serial(serial)

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        idle = 0
        while True:
            streamer = manager.streamer(serial) if manager else None
            frame = streamer.latest_jpeg() if streamer else None
            if frame is None and streamer is not None:
                frame, _ = ros_frame(serial)    # busy: the bringup's picture
            if frame is None:
                if streamer is None:
                    frame = _placeholder_jpeg("camera not connected")
                else:
                    status, _ = streamer.status
                    frame = _placeholder_jpeg(
                        "device busy" if status == "busy" else "connecting...")
                idle += 1
            else:
                idle = 0
            if frame is not None and (frame is not last or idle):
                yield boundary + frame + b"\r\n"
                last = frame
            time.sleep(1.0 / PREVIEW_FPS)

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def camera_nodes():
    """Bringup camera nodes per cameras.yaml, named the way
    start_all_cameras.launch.py names them: [(node, compressed color topic)]."""
    out = []
    try:
        with open(CAMERAS_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        for key, cam in (cfg.get('cameras') or {}).items():
            if not cam or not cam.get('enabled', True):
                continue
            node = f"/{cam.get('namespace', key)}/{cam.get('name', 'camera')}"
            out.append((node, f"{node}/color/image_raw/compressed"))
    except Exception:
        pass
    return out


def ros_frame(serial):
    """(latest JPEG, node name) for the camera with this bare serial from the
    running bringup, or (None, None) if there's no fresh frame."""
    if _node is None:
        return None, None
    return _node.preview.frame(serial)


class RosPreview:
    """Keeps the newest bringup frame of every camera node, and the serial each
    node actually opened (its serial_no parameter, which can differ from
    cameras.yaml if the file changed after the bringup started)."""

    def __init__(self, node):
        self._node = node
        self._lock = threading.Lock()
        self._frames = {}         # node name -> (jpeg bytes, monotonic time)
        self._serials = {}        # node name -> bare serial it opened
        self._subs = {}           # node name -> subscription
        self._clients = {}        # node name -> GetParameters client
        self.refresh()
        node.create_timer(ROS_REFRESH_PERIOD, self.refresh)

    def refresh(self):
        for name, topic in camera_nodes():
            if name not in self._subs:
                self._subs[name] = self._node.create_subscription(
                    CompressedImage, topic,
                    lambda msg, name=name: self._on_frame(name, msg),
                    qos_profile_sensor_data)
            self._ask_serial(name)

    def _on_frame(self, name, msg):
        if 'jpeg' not in msg.format.lower() and 'jpg' not in msg.format.lower():
            return                    # the page shows JPEG only
        with self._lock:
            self._frames[name] = (bytes(msg.data), time.monotonic())

    def _ask_serial(self, name):
        client = self._clients.get(name)
        if client is None:
            client = self._clients[name] = self._node.create_client(
                GetParameters, f"{name}/get_parameters")
        if not client.service_is_ready():
            return                    # bringup not running; keep what we knew
        request = GetParameters.Request()
        request.names = ['serial_no']
        future = client.call_async(request)
        future.add_done_callback(lambda f, name=name: self._got_serial(name, f))

    def _got_serial(self, name, future):
        try:
            serial = norm_serial(future.result().values[0].string_value)
        except Exception:
            return
        if serial:
            with self._lock:
                self._serials[name] = serial

    def frame(self, serial):
        now = time.monotonic()
        with self._lock:
            for name, node_serial in self._serials.items():
                jpeg, stamp = self._frames.get(name, (None, 0))
                if node_serial == serial and jpeg and now - stamp < ROS_FRAME_MAX_AGE:
                    return jpeg, name
        return None, None


# --------------------------------------------------------------------------- #
# ROS 2 node: declares the per-role camera serial parameters (with defaults),
# provides logging, and previews cameras the bringup holds (RosPreview).
# --------------------------------------------------------------------------- #
class CameraUINode(Node):
    def __init__(self):
        super().__init__('camera_ui')
        # Runtime parameters — override at launch with `--ros-args -p name:=value`.
        for role in ROLE_KEYS:
            self.declare_parameter(f'{role}_serial', DEFAULT_SERIALS.get(role, ''))
        self.preview = RosPreview(self) if ROS_PREVIEW else _NoPreview()

    def serials(self):
        """role_key -> bare serial from the <role>_serial parameters."""
        return {role: norm_serial(self.get_parameter(f'{role}_serial').value) or None
                for role in ROLE_KEYS}


class _NoPreview:
    def frame(self, serial):
        return None, None


def seed_assignments():
    """Starting slots: a <role>_serial parameter set at launch, else what
    cameras.yaml holds, else the built-in default. (Seeding from the defaults
    alone showed stale serials after every restart, despite a Save.)"""
    saved = read_assignments()
    params = _node.serials() if _node is not None else {}
    seed = {}
    for role in ROLE_KEYS:
        default = norm_serial(DEFAULT_SERIALS.get(role)) or None
        explicit = params.get(role) if params.get(role) != default else None
        seed[role] = explicit or saved.get(role) or default
    return seed


_node = None
_cleanup_lock = threading.Lock()
_cleaned = False


def _cleanup():
    """Release the RealSense devices (and ROS) exactly once. This is the
    resource that gets left 'device busy' for the next run if we exit without
    stopping the pipelines, so it runs FIRST and from every exit path."""
    global _cleaned
    with _cleanup_lock:
        if _cleaned:
            return
        _cleaned = True
    try:
        if manager:
            manager.stop_all()          # stops each pipeline + joins its thread
    except Exception:
        pass
    if ROS_AVAILABLE and _node is not None:
        try:
            _node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def _on_signal(signum, _frame):
    log(f"Signal {signum} received — releasing cameras and exiting.", "warn")
    _cleanup()
    os._exit(0)


def _set_parent_death_signal():
    """Linux only: have the kernel send us SIGTERM if our parent dies. This
    covers the orphan case -- e.g. the automation console kills `ros2 run` but
    this child python survives still holding the cameras. With this, the child
    gets SIGTERM, _on_signal runs, and the devices are freed."""
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG,
                                                        int(signal.SIGTERM), 0, 0, 0)
    except Exception:
        pass


def main():
    global manager, _logger, assignments, _node

    # Orphan safety net: if our launcher dies, the kernel sends us SIGTERM.
    _set_parent_death_signal()

    if ROS_AVAILABLE:
        rclpy.init()
        _node = CameraUINode()
        _logger = _node.get_logger()
        threading.Thread(target=rclpy.spin, args=(_node,), daemon=True).start()
    else:
        print(f"[WARN] ROS 2 not available ({ROS_IMPORT_ERROR}). UI runs standalone.")

    # Install AFTER rclpy.init() so our handler overrides the one rclpy installs
    # -- we want to release the cameras ourselves before exiting. Covers Ctrl-C
    # and `kill` (SIGTERM, what the automation console sends).
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    manager = DeviceManager(logger=log)
    with assign_lock:
        assignments.update(seed_assignments())
    manager.scan()

    host = socket.gethostname().lower()
    log(f"Camera UI on http://{host}.local:{SELF_PORT}  (cameras.yaml: {CAMERAS_YAML})")
    if not RS_AVAILABLE:
        log(f"pyrealsense2 unavailable ({RS_IMPORT_ERROR}) — hardware scan disabled.", "warn")

    try:
        app.run(host="0.0.0.0", port=SELF_PORT, threaded=True)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
