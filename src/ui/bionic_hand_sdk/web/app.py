#!/usr/bin/env python3
"""Web UI for both bionic hands, built on the bionic_hand SDK.

    python3 web/app.py                  # this laptop only: http://127.0.0.1:5000
    python3 web/app.py --host 0.0.0.0   # also reachable from phones on the same Wi-Fi

Use the port "sim://" in the page to try everything without hardware.
"""
import argparse
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bionic_hand import BionicHands, list_ports, find_bionic_hand_port  # noqa: E402
from bionic_hand.config import FINGER_NAMES  # noqa: E402
from bionic_hand.protocol import DEFAULT_BAUD, DEFAULT_PORT  # noqa: E402
from bionic_hand.simulator import SIM_PORT  # noqa: E402

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
hands = BionicHands(DEFAULT_PORT, DEFAULT_BAUD)
state = {"enabled": False}
hand_scan = {"left": False, "right": False, "left_ids": [], "right_ids": [], "scanned": False}
log = deque(maxlen=60)


def _scan_hands():
    """Ping all 16 motors and update hand_scan with which hands responded."""
    if not hands.connected:
        hand_scan.update(left=False, right=False, left_ids=[], right_ids=[], scanned=False)
        return
    try:
        ping = hands.ping_all()
        left_ok = [sid for sid in range(1, 9) if ping.get(sid)]
        right_ok = [sid for sid in range(9, 17) if ping.get(sid)]
        hand_scan.update(left=len(left_ok) > 0, right=len(right_ok) > 0,
                         left_ids=left_ok, right_ids=right_ok, scanned=True)
    except Exception:
        hand_scan.update(left=False, right=False, left_ids=[], right_ids=[], scanned=False)


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def add_log(message, level="info"):
    log.appendleft({"t": time.strftime("%H:%M:%S"), "msg": message, "level": level})


class Runner:
    """Runs one movement at a time in a background thread so STOP stays responsive."""

    def __init__(self):
        self.thread = None
        self.label = None
        self.lock = threading.Lock()

    @property
    def busy(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, label, fn):
        with self.lock:
            if self.busy:
                raise ApiError(f"Busy: {self.label} is still running. Wait or press STOP.", 409)
            self.label = label
            self.thread = threading.Thread(target=self._run, args=(label, fn), daemon=True)
            self.thread.start()

    def _run(self, label, fn):
        add_log(f"Started: {label}")
        try:
            done = fn()
            if done is False:
                add_log(f"Stopped: {label}", "warn")
            else:
                add_log(f"Done: {label}", "ok")
        except Exception as exc:
            add_log(f"Error in {label}: {exc}", "error")


runner = Runner()


# -- helpers ------------------------------------------------------------------
def body():
    return request.get_json(silent=True) or {}


def num(data, key, default, lo, hi, cast=float):
    try:
        value = cast(data.get(key, default))
    except (TypeError, ValueError):
        raise ApiError(f"{key} must be a number.")
    return max(lo, min(hi, value))


def motion_args(data):
    motion = data.get("motion") or None
    if motion in ("auto", ""):
        motion = None
    return {
        "speed": num(data, "speed", 500, 1, 1000, int),
        "duration": num(data, "duration", 2.0, 0.0, 20.0),
        "motion": motion,
    }


def hand_arg(data, allow_both=True):
    hand = data.get("hand", "both")
    valid = ("left", "right", "both") if allow_both else ("left", "right")
    if hand not in valid:
        raise ApiError(f"hand must be one of {valid}.")
    return hand


def require_motion():
    if not hands.connected:
        raise ApiError("Not connected. Connect to the serial port first.")
    if not state["enabled"]:
        raise ApiError("Movement is locked. Switch on 'Enable movement' first.")


def selected_hands(hand):
    return [h for n, h in hands.hands.items() if hand in (n, "both")]


def resolve_motion(hand, motion):
    """Pick the motion style for one hand.

    smooth/staged interpolate from the current position, so if it is not known
    yet, try reading it from the motors; if that fails, use direct for this move.
    """
    motion = motion or hand.config.motion
    if motion == "direct" or hand.pose_known:
        return motion
    try:
        hand.read_positions()
    except Exception:
        pass
    if hand.pose_known:
        add_log(f"{hand.name.title()} hand: read current motor positions.")
        return motion
    add_log(f"{hand.name.title()} hand: position unknown, using direct motion for this move.", "warn")
    return "direct"


def run_hands(selected, parallel, call, motion):
    """Job body: ``call(hand, motion)`` for each hand, in order or simultaneously."""
    def one(h):
        return call(h, resolve_motion(h, motion))
    if parallel and len(selected) > 1:
        return BionicHands._parallel([(one, (h,), {}) for h in selected])
    return all(one(h) for h in selected)


def closure(hand, finger):
    """Current commanded closure of a finger, 0 = open, 1 = closed."""
    cfg = hand.config
    values = []
    for sid in cfg.fingers[finger]:
        span = cfg.close_pose[sid] - cfg.open_pose[sid]
        if span:
            values.append((hand.pose[sid] - cfg.open_pose[sid]) / span)
    return round(max(0.0, min(1.0, sum(values) / len(values))), 3) if values else 0.0


@app.errorhandler(ApiError)
def api_error(err):
    return jsonify(ok=False, error=str(err)), err.status


@app.errorhandler(Exception)
def unexpected_error(err):
    if isinstance(err, HTTPException):
        return err
    add_log(f"Error: {err}", "error")
    return jsonify(ok=False, error=str(err)), 500


# -- pages / info ---------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/info")
def info():
    return jsonify(ok=True, gestures=BionicHands.gestures(), fingers=list(FINGER_NAMES),
                   waves=list(BionicHands.WAVE_SEQUENCES), default_port=DEFAULT_PORT,
                   default_baud=DEFAULT_BAUD)


@app.get("/api/ports")
def ports():
    return jsonify(ok=True, ports=list_ports() + [SIM_PORT])


@app.get("/api/detect")
def detect():
    """Auto-detect bionic hand by USB vendor/product ID."""
    port = find_bionic_hand_port()
    return jsonify(ok=True, detected=port is not None, port=port,
                   hand_scan=dict(hand_scan))


@app.post("/api/scan")
def scan():
    """Re-scan: ping all motors to detect left/right hands."""
    if not hands.connected:
        raise ApiError("Connect first.")
    if runner.busy:
        raise ApiError("Wait for the current movement to finish.", 409)
    _scan_hands()
    return jsonify(ok=True, hand_scan=dict(hand_scan))


@app.get("/api/status")
def status():
    out = {}
    for name, hand in hands.hands.items():
        out[name] = {
            "pose_known": hand.pose_known,
            "moving": hand.is_moving,
            "ids": hand.ids,
            "fingers": {f: list(ids) for f, ids in hand.fingers.items()},
            "pose": {str(sid): round(hand.pose[sid], 1) for sid in hand.ids},
            "closure": {f: closure(hand, f) for f in hand.fingers},
            "motion": hand.config.motion,
        }
    detected_port = find_bionic_hand_port()
    return jsonify(ok=True, connected=hands.connected, port=hands.bus.port, baud=hands.bus.baud,
                   enabled=state["enabled"], busy=runner.busy, job=runner.label if runner.busy else None,
                   detected=detected_port is not None, detected_port=detected_port,
                   hand_scan=dict(hand_scan),
                   hands=out, log=list(log))


# -- connection / safety ----------------------------------------------------------
@app.post("/api/connect")
def connect():
    data = body()
    port = data.get("port") or DEFAULT_PORT
    baud = int(num(data, "baud", DEFAULT_BAUD, 9600, 3_000_000, int))
    if runner.busy:
        raise ApiError("Wait for the current movement to finish.", 409)
    hands.connect(port, baud)
    actual_port = hands.bus.port
    add_log(f"Connected to {actual_port} at {baud} baud.", "ok")
    _scan_hands()
    if hand_scan["left"] and hand_scan["right"]:
        add_log(f"Left hand ({len(hand_scan['left_ids'])}/8) and Right hand ({len(hand_scan['right_ids'])}/8) detected.", "ok")
    elif hand_scan["left"]:
        add_log(f"Left hand detected ({len(hand_scan['left_ids'])}/8 motors). Right hand not responding.", "warn")
    elif hand_scan["right"]:
        add_log(f"Right hand detected ({len(hand_scan['right_ids'])}/8 motors). Left hand not responding.", "warn")
    else:
        add_log("No hand motors responded to ping. Check power and wiring.", "error")
    return jsonify(ok=True)


@app.post("/api/disconnect")
def disconnect():
    state["enabled"] = False
    hands.disconnect()
    hand_scan.update(left=False, right=False, left_ids=[], right_ids=[], scanned=False)
    add_log("Disconnected. Movement locked.")
    return jsonify(ok=True)


@app.post("/api/enable")
def enable():
    on = bool(body().get("enabled"))
    if on and not hands.connected:
        raise ApiError("Connect first.")
    state["enabled"] = on
    add_log("Movement ENABLED." if on else "Movement locked.", "warn" if on else "info")
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    hands.stop()
    add_log("STOP requested (software stop - motor power stays on).", "warn")
    return jsonify(ok=True)


# -- movement -------------------------------------------------------------------
@app.post("/api/gesture")
def gesture():
    data = body()
    require_motion()
    hand, name = hand_arg(data), str(data.get("name", ""))
    if name.upper() not in BionicHands.gestures():
        raise ApiError(f"Unknown gesture {name!r}.")
    kwargs, parallel = motion_args(data), bool(data.get("parallel"))
    motion = kwargs.pop("motion")
    runner.start(f"{name} ({hand})", lambda: run_hands(
        selected_hands(hand), parallel, lambda h, m: h.gesture(name, motion=m, **kwargs), motion))
    return jsonify(ok=True)


@app.post("/api/fingers")
def fingers():
    data = body()
    require_motion()
    hand = hand_arg(data)
    targets = {}
    for finger in FINGER_NAMES:
        key = finger.lower()
        if data.get(key) is not None:
            targets[key] = num(data, key, 0.0, 0.0, 1.0)
    if not targets:
        raise ApiError("Select at least one finger.")
    kwargs = motion_args(data)
    motion = kwargs.pop("motion")
    desc = ", ".join(f"{k} {round(v * 100)}%" for k, v in targets.items())
    runner.start(f"fingers {desc} ({hand})", lambda: run_hands(
        selected_hands(hand), False, lambda h, m: h.set_fingers(**targets, motion=m, **kwargs), motion))
    return jsonify(ok=True)


@app.post("/api/wave")
def wave():
    data = body()
    require_motion()
    sequence = data.get("sequence", "both")
    if sequence not in BionicHands.WAVE_SEQUENCES:
        raise ApiError(f"sequence must be one of {BionicHands.WAVE_SEQUENCES}.")
    kwargs = motion_args(data)
    motion = kwargs.pop("motion") or "direct"
    kwargs["cycles"] = num(data, "cycles", 2, 1, 10, int)
    order = {"left": ["left"], "right": ["right"], "both": ["left", "right"],
             "left_to_right": ["left", "right"], "right_to_left": ["right", "left"]}[sequence]
    runner.start(f"wave {sequence}", lambda: run_hands(
        [hands.hands[n] for n in order], sequence == "both",
        lambda h, m: h.wave(motion=m, **kwargs), motion))
    return jsonify(ok=True)


@app.post("/api/motor")
def motor():
    data = body()
    require_motion()
    sid = int(num(data, "id", 0, 1, 16, int))
    angle = num(data, "angle", 150.0, 0.0, 300.0)
    speed = num(data, "speed", 300, 1, 1000, int)
    hand = hands.left if sid in hands.left.ids else hands.right
    runner.start(f"motor {sid} -> {angle:g} deg", lambda: hand.set_motor(sid, angle, speed))
    return jsonify(ok=True)


# -- diagnostics ----------------------------------------------------------------
@app.post("/api/ping")
def ping():
    if not hands.connected:
        raise ApiError("Connect first.")
    if runner.busy:
        raise ApiError("Wait for the current movement to finish.", 409)
    result = hands.ping_all()
    answered = [sid for sid, ok in result.items() if ok]
    missing = [sid for sid, ok in result.items() if not ok]
    if not answered:
        add_log("Ping: no motors responded. Check wiring and power.", "error")
    elif not missing:
        add_log("All 16 motors answered.", "ok")
    else:
        add_log(f"Ping: motors {answered} OK, no answer from {missing}.", "warn")
    return jsonify(ok=True, result={str(k): v for k, v in result.items()})


@app.post("/api/read_positions")
def read_positions():
    if not hands.connected:
        raise ApiError("Connect first.")
    if runner.busy:
        raise ApiError("Wait for the current movement to finish.", 409)
    hand = hand_arg(body())
    positions = hands.read_positions(hand)
    expected = set()
    for h in selected_hands(hand):
        expected.update(h.ids)
    missing = sorted(expected - set(positions))
    if not positions:
        raise ApiError("No motors responded. Check wiring and power.")
    if missing:
        add_log(f"Read positions ({hand}): motors {missing} did not respond.", "warn")
    else:
        add_log(f"Read present positions ({hand}).", "ok")
    return jsonify(ok=True, positions={str(k): round(v, 1) for k, v in positions.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the LAN")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"Bionic Hand Web UI:  http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            print(f"On your Wi-Fi:       http://{s.getsockname()[0]}:{args.port}")
            s.close()
        except OSError:
            pass
        print("WARNING: no login. Use only on a trusted private network.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
