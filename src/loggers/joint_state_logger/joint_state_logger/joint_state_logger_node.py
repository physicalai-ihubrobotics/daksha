#!/usr/bin/env python3
"""
joint_state_logger — a SINGLE node that records the arm's joint-state feedback
to Parquet files *and* serves its own web control panel (one process, one node).

Captures, per session, everything you'd want for offline analysis of the arms:

  * /joint_states  (sensor_msgs/JointState)      -> position, velocity, effort
  * <arm>/motor_status (hw_interface/MotorStatusArray)
                                                  -> mos_temp, rotor_temp (+ error)
  * /gesture_status_topic (std_msgs/String JSON) -> replay progress / loops

Storage is **Apache Parquet** (columnar, compressed) — no database server. Each
session is its own folder:

    <log_dir>/daksha_<YYYYmmdd_HHMMSS>/
        joint_states.parquet      # timestamp_ns, source_topic, joint_name,
                                  #   position, velocity, effort, stamp_iso, run_id
        motor_status.parquet      # stamp_iso, topic, arm_name, motor_id, error,
                                  #   error_name, mos_temp, rotor_temp, run_id
        gesture_status.parquet    # stamp_iso, active, name, percent, frame, total,
                                  #   output_topic, speed, message, loop_count, ...

The joint_states schema matches gesture_management's recordings, so a logged
session's joint file is replay-compatible. A Parquet file's footer is only
written when the session is finalised, so **Stop / New session closes and
finalises the current files** — rows are streamed to disk as row groups in the
meantime. The live web dashboard reads from an in-memory ring buffer (fast, and
independent of the not-yet-finalised file).

The web panel (served by this same process on :8140) lets the operator:
  * START / STOP saving and open a NEW session (timeline of saved folders),
    plus RENAME / DELETE a session and CHANGE the data directory;
  * choose WHICH streams to record;
  * trigger a gesture REPLAY (name + loops + speed) and watch live progress;
  * see TIMELINE GRAPHS of the current session.

Run (after sourcing the workspace):
    ros2 run joint_state_logger joint_state_logger      # logs + UI on :8140
"""

import os
import csv
import glob
import json
import time
import socket
import logging
import datetime
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from flask import Flask, jsonify, render_template, request

logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Parquet (pandas/pyarrow) is optional at import so the node still runs (CSV
# fallback) on a box where it isn't installed (`pip install pyarrow pandas`).
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    PARQUET_AVAILABLE = False
    PARQUET_IMPORT_ERROR = str(_e)

# Motor status (temperatures) is optional — if hw_interface isn't built we just
# log joint_states/gesture_status and skip the temperature topics.
try:
    from hw_interface.msg import MotorStatusArray
    MOTOR_STATUS_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    MOTOR_STATUS_AVAILABLE = False
    MOTOR_STATUS_IMPORT_ERROR = str(_e)

# gesture_management replay services — optional (only if that package is built).
try:
    from gesture_management.srv import PlayRecording
    from std_srvs.srv import Trigger
    GESTURE_SRV_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    GESTURE_SRV_AVAILABLE = False
    GESTURE_SRV_IMPORT_ERROR = str(_e)


SELF_PORT = 8140
COLLECTIONS = ["joint_states", "motor_status", "gesture_status"]
RECORD_STREAMS = COLLECTIONS
DEFAULT_OUTPUT_TOPIC = "/joint_cmd"

_node = None   # the single JointStateLogger instance (set in main)


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _ros_time_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# Explicit Parquet schemas (defined only when pyarrow is present).
# One unified schema so joint_states, motor_status and gesture_status all live
# in a SINGLE Parquet file per session. A `source` column tags each row; columns
# that don't apply to a row are null. (Filter source=="joint_states" to get the
# joint rows, which keep the gesture_management recording columns.)
_UNIFIED_FIELDS = [
    ("stamp_iso", "string"), ("source", "string"), ("run_id", "string"),
    ("timestamp_ns", "int64"),
    # joint_states
    ("source_topic", "string"), ("joint_name", "string"),
    ("position", "float64"), ("velocity", "float64"), ("effort", "float64"),
    # motor_status
    ("topic", "string"), ("arm_name", "string"), ("motor_id", "int64"),
    ("error", "int64"), ("error_name", "string"),
    ("mos_temp", "float64"), ("rotor_temp", "float64"),
    # gesture_status
    ("active", "bool"), ("name", "string"), ("percent", "float64"),
    ("frame", "int64"), ("total", "int64"), ("output_topic", "string"),
    ("speed", "float64"), ("message", "string"), ("loop_count", "int64"),
    ("repeat_mode", "string"), ("repeat_count", "int64"),
]
_UNIFIED_COLUMNS = [n for n, _ in _UNIFIED_FIELDS]

if PARQUET_AVAILABLE:
    _type_of = {"string": pa.string(), "int64": pa.int64(),
                "float64": pa.float64(), "bool": pa.bool_()}
    _UNIFIED_SCHEMA = pa.schema([(n, _type_of[t]) for n, t in _UNIFIED_FIELDS])


# --------------------------------------------------------------------------- #
# Sinks — one file per session, all three streams combined
# --------------------------------------------------------------------------- #
class ParquetSink:
    """A single Parquet FILE per session holding joint_states + motor_status +
    gesture_status rows (discriminated by the `source` column). Rows are buffered
    and streamed as row groups on flush; close() writes the footer."""

    def __init__(self, file_path, logger):
        self._log = logger
        self._path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
        self._buffer = []
        self._writer = None

    @property
    def path(self):
        return self._path

    def write_joint_state(self, doc):
        ts = int(float(doc.get("ros_time_sec") or 0.0) * 1e9)
        for j in doc["joints"]:
            self._buffer.append({
                "stamp_iso": doc["stamp_iso"], "source": "joint_states",
                "run_id": doc["run_id"], "timestamp_ns": ts,
                "source_topic": "/joint_states", "joint_name": j["name"],
                "position": _f(j["position"]), "velocity": _f(j["velocity"]),
                "effort": _f(j["effort"])})

    def write_motor_status(self, doc):
        for m in doc["motors"]:
            self._buffer.append({
                "stamp_iso": doc["stamp_iso"], "source": "motor_status",
                "run_id": doc["run_id"], "topic": doc["topic"],
                "arm_name": m["arm_name"], "motor_id": _i(m["id"]),
                "error": _i(m["error"]), "error_name": m["error_name"],
                "mos_temp": _f(m["mos_temp"]), "rotor_temp": _f(m["rotor_temp"])})

    def write_gesture_status(self, doc):
        r = doc.get("replay", {}) or {}
        self._buffer.append({
            "stamp_iso": doc["stamp_iso"], "source": "gesture_status",
            "run_id": doc["run_id"], "active": bool(doc.get("active")),
            "name": doc.get("name"), "percent": _f(doc.get("percent")),
            "frame": _i(doc.get("frame")), "total": _i(doc.get("total")),
            "output_topic": doc.get("output_topic"), "speed": _f(doc.get("speed")),
            "message": doc.get("message"), "loop_count": _i(r.get("loop_count")),
            "repeat_mode": r.get("repeat_mode"), "repeat_count": _i(r.get("repeat_count"))})

    def flush(self):
        if not self._buffer:
            return
        try:
            table = pa.Table.from_pylist(self._buffer, schema=_UNIFIED_SCHEMA)
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self._path, _UNIFIED_SCHEMA, compression="snappy")
            self._writer.write_table(table)
            self._buffer = []
        except Exception as e:
            self._log.warn(f"Parquet write failed: {e}")

    def describe(self):
        return f"Parquet {self._path} (single file, source-tagged rows)"

    def close(self):
        self.flush()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None


class CsvSink:
    """Fallback when pyarrow isn't installed: a single unified CSV per session
    (same `source`-tagged columns as the Parquet file)."""

    def __init__(self, file_path, logger):
        self._log = logger
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
        self._path = file_path
        self._file = open(file_path, "w", newline="")
        self._w = csv.DictWriter(self._file, fieldnames=_UNIFIED_COLUMNS,
                                 extrasaction="ignore")
        self._w.writeheader()

    @property
    def path(self):
        return self._path

    def write_joint_state(self, doc):
        ts = int(float(doc.get("ros_time_sec") or 0.0) * 1e9)
        for j in doc["joints"]:
            self._w.writerow({"stamp_iso": doc["stamp_iso"], "source": "joint_states",
                              "run_id": doc["run_id"], "timestamp_ns": ts,
                              "source_topic": "/joint_states", "joint_name": j["name"],
                              "position": j["position"], "velocity": j["velocity"],
                              "effort": j["effort"]})

    def write_motor_status(self, doc):
        for m in doc["motors"]:
            self._w.writerow({"stamp_iso": doc["stamp_iso"], "source": "motor_status",
                              "run_id": doc["run_id"], "topic": doc["topic"],
                              "arm_name": m["arm_name"], "motor_id": m["id"],
                              "error": m["error"], "error_name": m["error_name"],
                              "mos_temp": m["mos_temp"], "rotor_temp": m["rotor_temp"]})

    def write_gesture_status(self, doc):
        r = doc.get("replay", {}) or {}
        self._w.writerow({"stamp_iso": doc["stamp_iso"], "source": "gesture_status",
                          "run_id": doc["run_id"], "active": doc.get("active"),
                          "name": doc.get("name"), "percent": doc.get("percent"),
                          "frame": doc.get("frame"), "total": doc.get("total"),
                          "output_topic": doc.get("output_topic"), "speed": doc.get("speed"),
                          "message": doc.get("message"), "loop_count": r.get("loop_count"),
                          "repeat_mode": r.get("repeat_mode"), "repeat_count": r.get("repeat_count")})

    def flush(self):
        try:
            self._file.flush()
        except Exception:
            pass

    def describe(self):
        return f"CSV {self._path} (single file, source-tagged rows)"

    def close(self):
        self.flush()
        try:
            self._file.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Flask app (web control panel) — shares this one process/node. The dashboard
# reads the node's in-memory ring buffer (Parquet files aren't readable until a
# session is finalised).
# --------------------------------------------------------------------------- #
def _resource_dirs():
    here = os.path.dirname(os.path.realpath(__file__))
    tmpl, static = os.path.join(here, "templates"), os.path.join(here, "static")
    if os.path.isdir(tmpl) and os.path.isdir(static):
        return tmpl, static
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("joint_state_logger")
        return os.path.join(share, "templates"), os.path.join(share, "static")
    except Exception:
        return tmpl, static


_TMPL_DIR, _STATIC_DIR = _resource_dirs()
app = Flask(__name__, template_folder=_TMPL_DIR, static_folder=_STATIC_DIR)


def _load_store():
    """(data, error): recent in-memory docs {table: [docs]} for the dashboard."""
    if _node is None:
        return None, "node not ready"
    return _node.snapshot(), None


def _epoch_ms(doc):
    s = doc.get("stamp_iso")
    if s:
        try:
            return datetime.datetime.fromisoformat(s).timestamp() * 1000.0
        except Exception:
            pass
    return None


def _summarize_gesture(doc):
    if not doc:
        return None
    return {"active": doc.get("active"), "name": doc.get("name"),
            "percent": doc.get("percent"), "frame": doc.get("frame"),
            "total": doc.get("total"), "message": doc.get("message"),
            "stamp_iso": doc.get("stamp_iso")}


def _summarize_joint(doc):
    if not doc:
        return None
    joints = doc.get("joints", [])
    return {"stamp_iso": doc.get("stamp_iso"), "num_joints": len(joints),
            "first": joints[0] if joints else None}


def _summarize_motor(doc):
    if not doc:
        return None
    motors = doc.get("motors", [])
    hottest = None
    for m in motors:
        t = max(m.get("mos_temp", 0) or 0, m.get("rotor_temp", 0) or 0)
        if hottest is None or t > hottest[0]:
            hottest = (t, m)
    return {"stamp_iso": doc.get("stamp_iso"), "topic": doc.get("topic"),
            "num_motors": len(motors),
            "hottest": ({"arm_name": hottest[1].get("arm_name"),
                         "id": hottest[1].get("id"),
                         "mos_temp": hottest[1].get("mos_temp"),
                         "rotor_temp": hottest[1].get("rotor_temp")}
                        if hottest else None)}


@app.route("/")
def index():
    return render_template("index.html", port=SELF_PORT)


@app.route("/api/stats")
def api_stats():
    sess = _node.session_status() if _node else {}
    out = {"ok": True, "backend": sess.get("backend", "parquet"),
           "path": sess.get("db_path"), "basename": sess.get("basename"),
           "recording": sess.get("recording", False),
           "collections": COLLECTIONS, "csv_fallback_dir": sess.get("log_dir"),
           "connected": False, "size_bytes": sess.get("size_bytes"),
           "counts": {}, "latest": {}, "note": None}
    if _node is None:
        out["note"] = "node not ready"
        return jsonify(out)
    data = _node.snapshot()
    out["counts"] = _node.counts()
    out["connected"] = (sess.get("recording") or any(out["counts"].values()))
    g = data["gesture_status"][-1] if data.get("gesture_status") else None
    j = data["joint_states"][-1] if data.get("joint_states") else None
    m = data["motor_status"][-1] if data.get("motor_status") else None
    out["latest"] = {"gesture_status": _summarize_gesture(g),
                     "joint_states": _summarize_joint(j),
                     "motor_status": _summarize_motor(m)}
    if not PARQUET_AVAILABLE:
        out["note"] = (f"pyarrow not installed ({PARQUET_IMPORT_ERROR}); saving CSV "
                       "instead of Parquet. Install with: pip install pyarrow pandas")
    return jsonify(out)


# ----- session (start / stop / new / rename / delete / dir) + file list ----- #
@app.route("/api/session")
def api_session_get():
    return jsonify(_node.session_status() if _node else {})


@app.route("/api/session/start", methods=["POST"])
def api_session_start():
    return jsonify(_node.start_session())


@app.route("/api/session/stop", methods=["POST"])
def api_session_stop():
    return jsonify(_node.stop_session())


@app.route("/api/session/new", methods=["POST"])
def api_session_new():
    return jsonify(_node.new_session())


@app.route("/api/sessions")
def api_sessions():
    return jsonify({"ok": True, "sessions": _node.list_sessions() if _node else []})


@app.route("/api/session/delete", methods=["POST"])
def api_session_delete():
    body = request.get_json(silent=True) or {}
    return jsonify(_node.delete_session((body.get("name") or "").strip()))


@app.route("/api/session/rename", methods=["POST"])
def api_session_rename():
    body = request.get_json(silent=True) or {}
    return jsonify(_node.rename_session((body.get("old") or "").strip(),
                                        (body.get("new") or "").strip()))


@app.route("/api/session/dir", methods=["POST"])
def api_session_dir():
    body = request.get_json(silent=True) or {}
    return jsonify(_node.change_dir((body.get("dir") or "").strip()))


# ----- which data to record ----- #
@app.route("/api/record", methods=["GET"])
def api_record_get():
    flags, err = _node.get_record_flags()
    return jsonify({"ok": err is None, "note": err, "flags": flags})


@app.route("/api/record", methods=["POST"])
def api_record_set():
    body = request.get_json(silent=True) or {}
    wanted = {s: bool(body[s]) for s in RECORD_STREAMS if s in body}
    if not wanted:
        return jsonify({"ok": False, "note": "no known streams in request."})
    ok, err = _node.set_record_flags(wanted)
    flags, _ = _node.get_record_flags()
    return jsonify({"ok": ok, "note": err, "flags": flags})


# ----- replay control ----- #
@app.route("/api/replay", methods=["POST"])
def api_replay():
    if not GESTURE_SRV_AVAILABLE:
        return jsonify({"ok": False, "note": "gesture_management replay services unavailable."})
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "note": "recording name is required."})
    try:
        loops = int(body.get("loops", 1))
    except (TypeError, ValueError):
        loops = 1
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    try:
        interval = float(body.get("interval", 0.0))
    except (TypeError, ValueError):
        interval = 0.0
    output_topic = (body.get("output_topic") or DEFAULT_OUTPUT_TOPIC).strip()
    ok, msg = _node.trigger_replay(name, output_topic, speed, loops, interval)
    return jsonify({"ok": ok, "note": msg})


@app.route("/api/replay/stop", methods=["POST"])
def api_replay_stop():
    if not GESTURE_SRV_AVAILABLE:
        return jsonify({"ok": False, "note": "gesture_management replay services unavailable."})
    ok, msg = _node.stop_replay()
    return jsonify({"ok": ok, "note": msg})


@app.route("/api/replay/status")
def api_replay_status():
    return jsonify({"ok": True, "available": GESTURE_SRV_AVAILABLE,
                    "replay": _node.live_replay() if _node else None})


@app.route("/api/replays")
def api_replays():
    data, err = _load_store()
    if data is None:
        return jsonify({"ok": False, "note": err, "rows": []})
    docs = data.get("gesture_status", [])
    agg = {}
    cur_name, cur_active, cur_max_loop = None, False, 0

    def _flush():
        if cur_name is None:
            return
        a = agg.setdefault(cur_name, {"sessions": 0, "total_loops": 0,
                                      "last_iso": None, "last_mode": None, "last_count": None})
        a["sessions"] += 1
        a["total_loops"] += max(cur_max_loop, 1)

    for d in docs:
        active = bool(d.get("active"))
        name = d.get("name")
        r = d.get("replay", {}) or {}
        loop = int(r.get("loop_count") or 0)
        mode = r.get("repeat_mode")
        count = r.get("repeat_count")
        iso = d.get("stamp_iso")
        if active and (not cur_active or name != cur_name):
            _flush()
            cur_name, cur_active, cur_max_loop = name, True, loop
        elif active and name == cur_name:
            cur_max_loop = max(cur_max_loop, loop)
        elif not active and cur_active:
            _flush()
            cur_name, cur_active, cur_max_loop = None, False, 0
        if name is not None and iso:
            a = agg.setdefault(name, {"sessions": 0, "total_loops": 0,
                                      "last_iso": None, "last_mode": None, "last_count": None})
            a["last_iso"] = iso
            if active:
                a["last_mode"] = mode or a["last_mode"]
                a["last_count"] = count if count is not None else a["last_count"]
    if cur_active:
        _flush()
    rows = [{"name": n, **v} for n, v in agg.items() if n]
    rows.sort(key=lambda x: (x["last_iso"] or ""), reverse=True)
    return jsonify({"ok": True, "note": None, "rows": rows, "scanned": len(docs)})


# ----- timeline ----- #
@app.route("/api/timeline")
def api_timeline():
    metric = request.args.get("metric", "joint_pos")
    try:
        points = max(20, min(int(request.args.get("points", 240)), 1000))
    except (TypeError, ValueError):
        points = 240
    data, err = _load_store()
    if data is None:
        return jsonify({"ok": False, "note": err, "metric": metric,
                        "t": [], "iso": [], "series": [], "unit": ""})
    if metric in ("joint_pos", "joint_vel", "joint_eff"):
        return _timeline_joints(data, metric, points)
    if metric == "motor_temp":
        return _timeline_motor(data, points)
    if metric == "gesture":
        return _timeline_gesture(data, points)
    return jsonify({"ok": False, "note": f"unknown metric '{metric}'",
                    "metric": metric, "t": [], "iso": [], "series": [], "unit": ""})


def _timeline_joints(data, metric, points):
    field = {"joint_pos": "position", "joint_vel": "velocity", "joint_eff": "effort"}[metric]
    unit = {"joint_pos": "rad", "joint_vel": "rad/s", "joint_eff": "N·m"}[metric]
    docs = data.get("joint_states", [])[-points:]
    t, iso, series = [], [], {}
    for d in docs:
        t.append(_epoch_ms(d))
        iso.append(d.get("stamp_iso"))
        present = {j.get("name"): j for j in d.get("joints", [])}
        for n in (set(series.keys()) | set(present.keys())):
            series.setdefault(n, [None] * (len(t) - 1))
        for n in series:
            series[n].append(present.get(n, {}).get(field) if n in present else None)
    out = [{"label": n, "values": v} for n, v in sorted(series.items())]
    return jsonify({"ok": True, "note": None, "metric": metric, "unit": unit,
                    "t": t, "iso": iso, "series": out})


def _timeline_motor(data, points):
    docs = data.get("motor_status", [])[-points:]
    t, iso, series = [], [], {}
    for d in docs:
        t.append(_epoch_ms(d))
        iso.append(d.get("stamp_iso"))
        present = {}
        for m in d.get("motors", []):
            key = f"{m.get('arm_name', '?')}#{m.get('id')}"
            present[key] = max(m.get("mos_temp", 0) or 0, m.get("rotor_temp", 0) or 0)
        for k in (set(series.keys()) | set(present.keys())):
            series.setdefault(k, [None] * (len(t) - 1))
        for k in series:
            series[k].append(present.get(k))
    out = [{"label": k, "values": v} for k, v in sorted(series.items())]
    return jsonify({"ok": True, "note": None, "metric": "motor_temp", "unit": "°C",
                    "t": t, "iso": iso, "series": out})


def _timeline_gesture(data, points):
    docs = data.get("gesture_status", [])[-points:]
    t, iso, pct, loops = [], [], [], []
    for d in docs:
        t.append(_epoch_ms(d))
        iso.append(d.get("stamp_iso"))
        pct.append(d.get("percent"))
        loops.append((d.get("replay", {}) or {}).get("loop_count"))
    return jsonify({"ok": True, "note": None, "metric": "gesture", "unit": "%",
                    "t": t, "iso": iso,
                    "series": [{"label": "percent", "values": pct},
                               {"label": "loop_count", "values": loops, "axis": "right"}]})


# --------------------------------------------------------------------------- #
# The single node
# --------------------------------------------------------------------------- #
class JointStateLogger(Node):

    def __init__(self):
        super().__init__("joint_state_logger")
        global SELF_PORT, DEFAULT_OUTPUT_TOPIC

        _log_dir = os.path.expanduser("~/daksha/logs")
        self.declare_parameter("log_dir", _log_dir)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("motor_status_topics",
                               ["/LeftArmSystem/motor_status", "/RightArmSystem/motor_status"])
        self.declare_parameter("gesture_status_topic", "/gesture_status_topic")
        self.declare_parameter("port", SELF_PORT)
        self.declare_parameter("autostart", True)
        self.declare_parameter("flush_period_s", 2.0)
        self.declare_parameter("joint_states_max_hz", 20.0)
        self.declare_parameter("recent_buffer", 3000)   # in-memory docs/table for dashboard
        self.declare_parameter("log_joint_states", True)
        self.declare_parameter("log_motor_status", True)
        self.declare_parameter("log_gesture_status", True)
        self.declare_parameter("default_output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("play_recording_service", "play_recording")
        self.declare_parameter("stop_replay_service", "stop_replay")

        p = lambda n: self.get_parameter(n).value
        self._log_dir = os.path.expanduser(p("log_dir"))
        self._port = int(p("port"))
        self._recent_max = max(100, int(p("recent_buffer")))

        SELF_PORT = self._port
        DEFAULT_OUTPUT_TOPIC = p("default_output_topic")

        self._enabled = {
            "joint_states": bool(p("log_joint_states")),
            "motor_status": bool(p("log_motor_status")),
            "gesture_status": bool(p("log_gesture_status")),
        }
        self.add_on_set_parameters_callback(self._on_set_params)

        hz = float(p("joint_states_max_hz"))
        self._js_min_dt = (1.0 / hz) if hz and hz > 0 else 0.0
        self._js_last_write = 0.0

        # session + dashboard state (guarded by _sink_lock)
        self._sink_lock = threading.Lock()
        self._sink = None
        self._db_path = None     # current session directory
        self._run_id = None
        self._recording = False
        self._recent = {t: deque(maxlen=self._recent_max) for t in COLLECTIONS}
        self._counts = {t: 0 for t in COLLECTIONS}

        self._live_lock = threading.Lock()
        self._live_replay = None

        # Subscriptions
        self.create_subscription(
            JointState, p("joint_states_topic"), self._joint_state_cb, qos_profile_sensor_data)
        if MOTOR_STATUS_AVAILABLE:
            for topic in p("motor_status_topics"):
                self.create_subscription(
                    MotorStatusArray, topic, self._make_motor_status_cb(topic),
                    qos_profile_sensor_data)
        else:
            self.get_logger().warn(
                f"hw_interface not available ({MOTOR_STATUS_IMPORT_ERROR}); "
                "temperatures (motor_status) will NOT be logged.")
        self._gesture_topic = p("gesture_status_topic")
        self.create_subscription(String, self._gesture_topic, self._gesture_status_cb, 10)
        self._last_gesture_key = None

        self._call_lock = threading.Lock()
        self._play_cli = self._stop_cli = None
        if GESTURE_SRV_AVAILABLE:
            self._play_cli = self.create_client(PlayRecording, p("play_recording_service"))
            self._stop_cli = self.create_client(Trigger, p("stop_replay_service"))

        self._flush_timer = self.create_timer(
            max(0.2, float(p("flush_period_s"))), self._periodic_flush)

        self.get_logger().info("joint_state_logger (single node: logging + web UI) started")
        self.get_logger().info(f"  storage:       {'Parquet' if PARQUET_AVAILABLE else 'CSV (pyarrow missing)'}")
        self.get_logger().info(f"  web UI:        http://<host>:{self._port}")
        self.get_logger().info(f"  joint_states:  {p('joint_states_topic')} (throttle {hz} Hz)")
        if bool(p("autostart")):
            self.start_session()
        else:
            self.get_logger().info("  autostart off — press Start in the UI to record.")

    # ----- parameter callback ----- #
    def _on_set_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for prm in params:
            key = {"log_joint_states": "joint_states",
                   "log_motor_status": "motor_status",
                   "log_gesture_status": "gesture_status"}.get(prm.name)
            if key is not None:
                self._enabled[key] = bool(prm.value)
        return SetParametersResult(successful=True)

    # ----- sink / session management ----- #
    def _make_sink(self, run_id):
        if PARQUET_AVAILABLE:
            path = os.path.join(self._log_dir, f"daksha_{run_id}.parquet")
            try:
                sink = ParquetSink(path, self.get_logger())
                self.get_logger().info(f"Recording to {path} (single Parquet file)")
                return sink
            except Exception as e:
                self.get_logger().warn(f"Parquet open failed ({e}); falling back to CSV.")
        else:
            self.get_logger().warn(
                f"pyarrow not installed ({PARQUET_IMPORT_ERROR}); CSV fallback. "
                "Install with: pip install pyarrow pandas")
        return CsvSink(os.path.join(self._log_dir, f"daksha_{run_id}.csv"),
                       self.get_logger())

    def _reset_recent(self):
        self._recent = {t: deque(maxlen=self._recent_max) for t in COLLECTIONS}
        self._counts = {t: 0 for t in COLLECTIONS}

    def _unique_run_id(self):
        base = _stamp()
        rid, n = base, 1
        while (os.path.exists(os.path.join(self._log_dir, f"daksha_{rid}.parquet"))
               or os.path.exists(os.path.join(self._log_dir, f"daksha_{rid}.csv"))):
            rid = f"{base}_{n}"
            n += 1
        return rid

    def start_session(self):
        with self._sink_lock:
            if self._sink is None:
                self._run_id = self._unique_run_id()
                self._sink = self._make_sink(self._run_id)
                self._db_path = self._sink.path
                self._reset_recent()
            self._recording = True
        self.get_logger().info(f"Recording STARTED ({self._basename()})")
        return self.session_status()

    def stop_session(self):
        with self._sink_lock:
            self._recording = False
            if self._sink is not None:
                self._sink.close()      # finalises the Parquet footers
                self._sink = None
        self.get_logger().info(f"Recording STOPPED ({self._basename()})")
        return self.session_status()

    def new_session(self):
        with self._sink_lock:
            if self._sink is not None:
                self._sink.close()
                self._sink = None
            self._run_id = self._unique_run_id()
            self._sink = self._make_sink(self._run_id)
            self._db_path = self._sink.path
            self._reset_recent()
            self._recording = True
        self.get_logger().info(f"NEW session ({self._basename()})")
        return self.session_status()

    def _basename(self):
        return os.path.basename(self._db_path) if self._db_path else "—"

    def _path_size(self, path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def session_status(self):
        size = self._path_size(self._db_path) if self._db_path else None
        return {"ok": True, "recording": self._recording,
                "db_path": self._db_path, "basename": self._basename(),
                "run_id": self._run_id, "log_dir": self._log_dir,
                "size_bytes": size,
                "backend": "parquet" if PARQUET_AVAILABLE else "csv"}

    def list_sessions(self):
        # Each session is a single .parquet (or .csv) file in log_dir.
        try:
            files = [f for f in glob.glob(os.path.join(self._log_dir, "*"))
                     if os.path.isfile(f) and f.endswith((".parquet", ".csv"))]
            # include the current session even before its first flush writes it
            if self._db_path and self._db_path not in files:
                files.append(self._db_path)
        except Exception:
            return []
        out = []
        for f in sorted(set(files))[-40:]:
            try:
                mt = os.path.getmtime(f) if os.path.exists(f) else 0
                out.append({"name": os.path.basename(f), "bytes": self._path_size(f),
                            "mtime": datetime.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
                            "current": (f == self._db_path)})
            except OSError:
                continue
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def _in_dir(self, name):
        if not name:
            return None
        base = os.path.basename(name)
        if base != name or ".." in base or "\0" in base:
            return None
        return os.path.join(self._log_dir, base)

    def delete_session(self, name):
        path = self._in_dir(name)
        if path is None or not os.path.isfile(path):
            return {"ok": False, "note": "invalid or missing session file"}
        with self._sink_lock:
            if path == self._db_path and self._recording:
                return {"ok": False,
                        "note": "can't delete the active recording — Stop or New session first"}
            try:
                os.remove(path)
            except OSError as e:
                return {"ok": False, "note": str(e)}
            if path == self._db_path:
                self._db_path = None
        self.get_logger().info(f"deleted session {os.path.basename(path)}")
        return {"ok": True, "note": f"deleted {os.path.basename(path)}"}

    def rename_session(self, old, new_name):
        src = self._in_dir(old)
        if src is None or not os.path.isfile(src):
            return {"ok": False, "note": "source session file not found"}
        newbase = os.path.basename(new_name or "")
        if not newbase or newbase != new_name or ".." in newbase or "\0" in newbase:
            return {"ok": False, "note": "invalid new name"}
        # keep the original extension if the user didn't give one
        if not newbase.endswith((".parquet", ".csv")):
            newbase += os.path.splitext(src)[1]
        dst = os.path.join(self._log_dir, newbase)
        if os.path.exists(dst):
            return {"ok": False, "note": "a session with that name already exists"}
        with self._sink_lock:
            if src == self._db_path and self._recording:
                return {"ok": False,
                        "note": "can't rename the active recording — Stop or New session first"}
            try:
                os.rename(src, dst)
            except OSError as e:
                return {"ok": False, "note": str(e)}
            if src == self._db_path:
                self._db_path = dst
        self.get_logger().info(f"renamed session -> {newbase}")
        return {"ok": True, "note": f"renamed to {newbase}"}

    def change_dir(self, new_dir):
        if not new_dir:
            return {"ok": False, "note": "empty path"}
        path = os.path.abspath(os.path.expanduser(new_dir))
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            return {"ok": False, "note": str(e)}
        with self._sink_lock:
            self._log_dir = path
            was_recording = self._recording
            if self._sink is not None:
                self._sink.close()
                self._sink = None
            if was_recording:
                self._run_id = self._unique_run_id()
                self._sink = self._make_sink(self._run_id)
                self._db_path = self._sink.path
                self._reset_recent()
                self._recording = True
            else:
                self._db_path = None
        self.get_logger().info(f"data directory changed -> {path}")
        return {"ok": True, "note": f"data dir → {path}", "log_dir": path}

    def _periodic_flush(self):
        with self._sink_lock:
            if self._sink is not None:
                self._sink.flush()

    # ----- dashboard data (in-memory) ----- #
    def snapshot(self):
        with self._sink_lock:
            return {t: list(self._recent[t]) for t in COLLECTIONS}

    def counts(self):
        with self._sink_lock:
            return dict(self._counts)

    # ----- record toggles ----- #
    def get_record_flags(self):
        return dict(self._enabled), None

    def set_record_flags(self, wanted):
        for s, v in wanted.items():
            if s in self._enabled:
                self._enabled[s] = bool(v)
        return True, None

    # ----- live replay + replay services ----- #
    def live_replay(self):
        with self._live_lock:
            return dict(self._live_replay) if self._live_replay else None

    def _call(self, client, req, timeout=6.0):
        if client is None:
            return None, "service client unavailable"
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=1.5):
            return None, f"service '{client.srv_name}' not available"
        with self._call_lock:
            future = client.call_async(req)
            deadline = time.time() + timeout
            while not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if not future.done():
                return None, "service call timed out"
            try:
                return future.result(), None
            except Exception as e:
                return None, str(e)

    def trigger_replay(self, name, output_topic, speed, loops, interval):
        if not GESTURE_SRV_AVAILABLE:
            return False, "gesture_management services not built."
        req = PlayRecording.Request()
        req.recording_name = name
        req.output_topic = output_topic
        req.replay_speed = float(speed)
        if loops <= 0:
            req.repeat_mode, req.repeat_count = "infinite", 0
        elif loops == 1:
            req.repeat_mode, req.repeat_count = "once", 1
        else:
            req.repeat_mode, req.repeat_count = "count", int(loops)
        req.interval_s = float(interval)
        resp, err = self._call(self._play_cli, req)
        if err:
            return False, err
        return bool(resp.success), resp.message

    def stop_replay(self):
        if not GESTURE_SRV_AVAILABLE:
            return False, "gesture_management services not built."
        resp, err = self._call(self._stop_cli, Trigger.Request())
        if err:
            return False, err
        return bool(resp.success), resp.message

    # ----- callbacks ----- #
    def _write(self, kind, doc):
        with self._sink_lock:
            if not self._recording or self._sink is None or not self._enabled[kind]:
                return
            {"joint_states": self._sink.write_joint_state,
             "motor_status": self._sink.write_motor_status,
             "gesture_status": self._sink.write_gesture_status}[kind](doc)
            self._recent[kind].append(doc)
            self._counts[kind] += 1

    def _joint_state_cb(self, msg):
        if not self._recording or not self._enabled["joint_states"]:
            return
        if self._js_min_dt:
            now = time.monotonic()
            if now - self._js_last_write < self._js_min_dt:
                return
            self._js_last_write = now
        joints = []
        pos, vel, eff = list(msg.position), list(msg.velocity), list(msg.effort)
        for i, name in enumerate(msg.name):
            joints.append({"name": name,
                           "position": pos[i] if i < len(pos) else None,
                           "velocity": vel[i] if i < len(vel) else None,
                           "effort": eff[i] if i < len(eff) else None})
        self._write("joint_states", {
            "run_id": self._run_id, "source": "joint_states",
            "stamp_iso": _now_iso(), "ros_time_sec": _ros_time_sec(msg), "joints": joints})

    def _make_motor_status_cb(self, topic):
        def cb(msg):
            if not self._recording or not self._enabled["motor_status"]:
                return
            motors = [{"arm_name": m.arm_name, "id": int(m.id), "error": int(m.error),
                       "error_name": m.error_name, "mos_temp": float(m.mos_temp),
                       "rotor_temp": float(m.rotor_temp)} for m in msg.motors]
            self._write("motor_status", {
                "run_id": self._run_id, "source": "motor_status", "topic": topic,
                "stamp_iso": _now_iso(), "motors": motors})
        return cb

    def _gesture_status_cb(self, msg):
        try:
            status = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        r = status.get("replay", {}) if isinstance(status, dict) else {}
        with self._live_lock:
            self._live_replay = r
        if not self._recording or not self._enabled["gesture_status"]:
            return
        key = (r.get("active"), r.get("name"), r.get("frame"), r.get("percent"), r.get("message"))
        if key == self._last_gesture_key:
            return
        self._last_gesture_key = key
        self._write("gesture_status", {
            "run_id": self._run_id, "source": "gesture_status", "stamp_iso": _now_iso(),
            "active": bool(r.get("active", False)), "name": r.get("name"),
            "percent": r.get("percent"), "frame": r.get("frame"), "total": r.get("total"),
            "output_topic": r.get("output_topic"), "speed": r.get("speed"),
            "message": r.get("message"), "replay": r})

    def destroy_node(self):
        with self._sink_lock:
            if self._sink is not None:
                self._sink.close()
                self._sink = None
        super().destroy_node()


def main(args=None):
    global _node
    rclpy.init(args=args)
    node = JointStateLogger()
    _node = node

    host = socket.gethostname().lower()
    node.get_logger().info(f"web UI → http://{host}.local:{node._port}")
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=node._port, threaded=True,
                               use_reloader=False),
        daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
