"""Real-time bounding box on a chosen object.

The vision LLM (Qwen3-VL) finds the object -- by name, or at a point the
operator clicked -- and returns a bounding box. Qwen3-VL takes ~1 s per look,
far too slow to redraw a box every frame, so an OpenCV CSRT tracker follows
the box on every camera frame (~50 fps on Thor at 640x480) and the LLM
re-checks every `redetect_s` seconds, or immediately when the tracker loses
the object.

Qwen3-VL reports boxes as [x1, y1, x2, y2] on a 0-1000 grid relative to the
image, whatever the image's real size.
"""

import json
import re
import threading
import time

import cv2
import numpy as np

from .camera import data_uri
from .llm_client import LLMError, extract_json

FIND_PROMPT = (
    'Locate every "{query}" in the image. Reply with a JSON list only: '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short name>"}}, ...] using '
    'coordinates from 0 to 1000 relative to the image width and height. If '
    'there is none, reply [].')

# Asked for a box straight from click coordinates, Qwen3-VL anchors the box at
# the click instead of fitting the object. So a click is two steps: name the
# object under a drawn marker, then locate it by name (accurate) and keep the
# instance under the click.
NAME_PROMPT = (
    'What is the single object directly under the red crosshair marker? Reply '
    'with ONE JSON object only: {"label": "<short name, 1-3 words>"}. If there '
    'is no distinct object there (only floor, wall or empty space), reply '
    '{"label": null}.')

def _decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _to_pixels(b, width, height):
    if not (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)):
        return None
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    box = (max(0, round(x1 / 1000 * width)), max(0, round(y1 / 1000 * height)),
           min(width, round(x2 / 1000 * width)), min(height, round(y2 / 1000 * height)))
    return box if box[2] - box[0] >= 4 and box[3] - box[1] >= 4 else None


def parse_boxes(reply, width, height):
    """[(pixel box, label), ...] from a Qwen3-VL grounding reply: a JSON list of
    {"bbox_2d", "label"} objects, or a single such object. Raises ValueError if
    the reply has no JSON at all."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    items = None
    a, b = text.find("["), text.rfind("]")
    brace = text.find("{")
    # A top-level list only if "[" comes before the first "{" -- otherwise the
    # "[" is the bbox_2d coordinates inside a single object.
    if a != -1 and b > a and (brace == -1 or a < brace):
        try:
            items = json.loads(text[a:b + 1])
        except ValueError:
            items = None
    if not isinstance(items, list):
        items = [extract_json(text)]
    out = []
    for it in items:
        if isinstance(it, dict):
            box = _to_pixels(it.get("bbox_2d"), width, height)
            if box:
                out.append((box, str(it.get("label") or "").strip()))
    return out


def parse_box(reply, width, height):
    """First (pixel box, label) of a grounding reply, or (None, label)."""
    found = parse_boxes(reply, width, height)
    if found:
        return found[0]
    try:
        return None, str(extract_json(reply).get("label") or "").strip()
    except (ValueError, AttributeError):
        return None, ""


def pick_box(found, ref):
    """The detection to keep: the one containing point `ref` (x, y pixels), else
    the one whose centre is nearest it; the first one when there is no ref."""
    if not found:
        return None, ""
    if ref is None:
        return found[0]
    rx, ry = ref
    inside = [f for f in found if f[0][0] <= rx <= f[0][2] and f[0][1] <= ry <= f[0][3]]
    pool = inside or found
    # among boxes containing the point prefer the smallest (most specific)
    key = (lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1])) if inside else \
          (lambda f: ((f[0][0] + f[0][2]) / 2 - rx) ** 2 + ((f[0][1] + f[0][3]) / 2 - ry) ** 2)
    return min(pool, key=key)


class ObjectTracker:
    """One tracked object at a time. All public methods are thread-safe."""

    def __init__(self, cameras, llm, log, publish=None, redetect_s=5.0):
        self.cameras, self.llm, self.log = cameras, llm, log
        self.publish = publish or (lambda state: None)
        self.redetect_s = redetect_s
        self._lock = threading.Lock()
        self._gen = 0                  # bumps on every start/stop; stale threads exit
        self._state = {"status": "idle"}
        self._tracker = None
        self._detecting = False
        self._last_error = None
        # reject(box) -> reason or None: boxes that are not an object (e.g. the robot's own hand).
        # Applied to the FIRST detection only: later re-checks keep following the object even when
        # the hand comes close to it during a pick.
        self.reject = None

    # ------------------------------------------------------------------ API
    def state(self):
        with self._lock:
            return {k: v for k, v in self._state.items() if not k.startswith("_")}

    def start(self, camera, query=None, point=None):
        """Box `query` (object name) or the object at `point` ([x, y] in 0..1 of the
        image) in `camera`. Returns immediately; detection runs in the background."""
        if camera not in self.cameras.cameras:
            raise KeyError(f"unknown camera '{camera}'")
        if not self.llm:
            raise LLMError("LLM disabled; cannot find objects")
        if not query and not point:
            raise ValueError("give an object name or a point")
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._tracker = None
            self._state = {"status": "detecting", "camera": camera,
                           "query": query or None, "point": point,
                           "label": query or "", "box": None, "size": None,
                           "fps": 0.0, "detected_ago_s": None, "message": "",
                           "_detected_at": None}
        threading.Thread(target=self._run, args=(gen,), daemon=True,
                         name="ros_agent_tracker").start()
        return self.state()

    def stop(self):
        with self._lock:
            self._gen += 1
            self._tracker = None
            self._state = {"status": "idle"}
        self.publish(self.state())

    def wait_detected(self, timeout=30.0):
        """Block until the first detection finished; returns the state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.state()
            if s["status"] != "detecting":
                return s
            time.sleep(0.1)
        return self.state()

    def overlay(self, camera):
        """Small dict for the live view of `camera` (None if not tracking it)."""
        s = self.state()
        if s.get("camera") != camera or s["status"] == "idle":
            return None
        return {k: s.get(k) for k in ("status", "label", "box", "size", "fps",
                                      "detected_ago_s", "message")}

    # ------------------------------------------------------------- internals
    def _current(self, gen):
        return self._gen == gen

    def _update(self, gen, **kw):
        with self._lock:
            if not self._current(gen):
                return False
            self._state.update(kw)
            at = self._state.get("_detected_at")
            self._state["detected_ago_s"] = None if at is None else round(time.monotonic() - at, 1)
            snap = {k: v for k, v in self._state.items() if not k.startswith("_")}
        self.publish(snap)
        return True

    def _detect(self, gen, frame):
        """Ask the LLM for the box in `frame` (BGR). Runs in its own thread."""
        try:
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self._lock:
                query, point = self._state.get("query"), self._state.get("point")
                prev = self._state.get("box")
            t0 = time.monotonic()
            label = query
            if not query:
                # Step 1 (clicks only): name the object under a drawn marker.
                px, py = round(point[0] * w), round(point[1] * h)
                marked = frame.copy()
                r = max(10, w // 40)
                cv2.circle(marked, (px, py), r, (0, 0, 255), 3)
                cv2.line(marked, (px - 2 * r, py), (px + 2 * r, py), (0, 0, 255), 2)
                cv2.line(marked, (px, py - 2 * r), (px, py + 2 * r), (0, 0, 255), 2)
                ok, mbuf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 85])
                named = extract_json(self.llm.chat([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri(mbuf.tobytes())}},
                    {"type": "text", "text": NAME_PROMPT}]}], max_tokens=48))
                label = str(named.get("label") or "").strip()
                if not label or label.lower() in ("null", "none"):
                    self._update(gen, status="not_found", box=None,
                                 message="no distinct object at that spot — click on the object itself")
                    with self._lock:
                        if self._current(gen):
                            self._gen += 1        # nothing to keep looking for
                    return
            # Step 2: locate by name on the clean frame, keep the right instance.
            reply = self.llm.chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri(buf.tobytes())}},
                {"type": "text", "text": FIND_PROMPT.format(query=label)}]}], max_tokens=256)
            if prev:
                ref = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
            elif point:
                ref = (point[0] * w, point[1] * h)
            else:
                ref = None
            found = parse_boxes(reply, w, h)
            if self.reject is not None and not prev and found:
                kept = []
                for f in found:
                    why = self.reject(f[0])
                    if why:
                        self.log(f"ignored box {[round(v) for v in f[0]]} for '{query or label}': {why}")
                    else:
                        kept.append(f)
                found = kept
            box, found_label = pick_box(found, ref)
            label = found_label or label
            if self._last_error:
                self.log("detection working again")
                self._last_error = None
                self._update(gen, message="")
            self.log(f"detect {query or point} -> {box} '{label}' of {len(found)} "
                     f"({time.monotonic() - t0:.1f}s)")
            if box is None:
                with self._lock:
                    was_tracking = self._state.get("box") is not None
                self._update(gen, status="lost" if was_tracking else "not_found", box=None,
                             message=f"'{query or label or 'object'}' is not visible")
                with self._lock:
                    if self._current(gen):
                        self._tracker = None
                return
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
            with self._lock:
                if not self._current(gen):
                    return
                self._tracker = tracker
                self._state["_detected_at"] = time.monotonic()
                # A clicked point is only valid for the frame it was clicked on;
                # re-checks look the object up by the name the model gave it.
                if not self._state.get("query") and label:
                    self._state["query"] = label
            self._update(gen, status="tracking", box=list(box), size=[w, h],
                         label=label or query or "object", message="")
        except Exception as e:  # LLM down, bad reply, ...
            # Log a failure once, not on every retry (every redetect_s while it lasts).
            err = str(e).split(" If the model server")[0]
            if err != self._last_error:
                self.log(f"detect failed (will keep retrying quietly): {err}")
                self._last_error = err
            with self._lock:
                had_box = self._state.get("box") is not None
            if isinstance(e, LLMError) and "unreachable" in str(e):
                msg = ("AI offline — box kept by the tracker only, not re-checked"
                       if had_box else "AI offline — can't look for the object until it's back")
            else:
                msg = "the AI gave an unusable answer; trying again"
            self._update(gen, status="tracking" if had_box else "error", message=msg)
        finally:
            with self._lock:
                self._detecting = False

    def _kick_detect(self, gen, frame):
        with self._lock:
            if self._detecting or not self._current(gen):
                return
            self._detecting = True
        threading.Thread(target=self._detect, args=(gen, frame.copy()), daemon=True).start()

    def _run(self, gen):
        with self._lock:
            camera = self._state["camera"]
        n, frames, t_fps = 0, 0, time.monotonic()
        last_detect = 0.0
        while self._current(gen):
            jpeg, n = self.cameras.wait_frame(camera, n, timeout=2.0)
            if not self._current(gen):
                return
            if jpeg is None:
                self._update(gen, message=f"no frames from camera '{camera}'")
                continue
            frame = _decode(jpeg)
            if frame is None:
                continue
            with self._lock:
                tracker = self._tracker
                detected_at = self._state.get("_detected_at")
            now = time.monotonic()
            if tracker is None:
                # Nothing to track yet (first look, or lost): ask the LLM, at most
                # once per redetect_s while the object stays out of view.
                if now - last_detect >= (0 if detected_at is None and last_detect == 0
                                         else self.redetect_s):
                    last_detect = now
                    self._kick_detect(gen, frame)
                continue
            ok, (x, y, bw, bh) = tracker.update(frame)
            frames += 1
            if now - t_fps >= 1.0:
                fps, frames, t_fps = frames / (now - t_fps), 0, now
            else:
                fps = None
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, int(x)), max(0, int(y))          # CSRT can drift past the edge
            x2, y2 = min(fw, int(x + bw)), min(fh, int(y + bh))
            if ok and x2 - x1 > 2 and y2 - y1 > 2:
                kw = {"box": [x1, y1, x2, y2], "status": "tracking"}
                if fps is not None:
                    kw["fps"] = round(fps, 1)
                self._update(gen, **kw)
                if detected_at is not None and now - detected_at >= self.redetect_s \
                        and now - last_detect >= self.redetect_s:
                    last_detect = now
                    self._kick_detect(gen, frame)       # periodic re-check vs. drift
            else:
                with self._lock:
                    if self._current(gen):
                        self._tracker = None
                self._update(gen, status="searching", box=None,
                             message="lost track; looking again")
                last_detect = now
                self._kick_detect(gen, frame)

