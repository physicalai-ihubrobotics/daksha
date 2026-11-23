"""Chat page for ros_agent (stdlib http.server, behind the shared Daksha gate).

    GET  /                       chat page
    GET  /api/info               robot name, DOF, LLM status, cameras
    GET  /api/llm                model server state: ready | loading | down | disabled
    GET  /api/models             {"current", "models": [{"name", "vision", ...}]} from Ollama
    POST /api/model              {"model": name} -> switch the LLM (next question uses it)
    GET  /api/cameras            camera status
    GET  /api/camera/<id>.jpg    latest frame (as published, not re-encoded); header
                                 X-Track = live bounding box for this camera, if any
    GET  /api/track              tracked-object state
    POST /api/track              {"camera", "object": name} or {"camera", "point": [x, y]}
                                 (point in 0..1 of the image) -> start boxing it
    POST /api/track/stop         clear the box
    POST /api/ask                {"question", "history"?, "attach_cameras"?: [ids],
                                  "think"?: bool} -> {"ok", "answer", "trace"}
    POST /api/stop               stop the running question

  Real-robot VLM pipeline (pipeline_node.py), same page:
    GET  /api/pipeline/state           JSON state (execute, table, calibration, chat, logs)
    GET  /api/pipeline/live.jpg        world camera frame the pipeline sees
    GET  /api/pipeline/vlm.jpg         last VLM detections drawn on the frame
    POST /api/pipeline/command         {"text"} instruction or question
    POST /api/pipeline/stop            interrupt now, drop queued commands
    POST /api/pipeline/execute         {"on": bool} arm / disarm real motion
    POST /api/pipeline/measure_table   table height from an ArUco marker on the table
    POST /api/pipeline/allow_calibration  use an over-the-limit world calibration (session)
"""

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ament_index_python.packages import get_package_share_directory

log = logging.getLogger("ros_agent.web")

APP_NAME = "ros_agent"          # key in automation/config/web_access.yaml
_MAX_BODY = 1 << 20


def _gate(handler_class):
    """Wrap with the Daksha browser-access gate if automation is installed."""
    try:
        from ament_index_python.packages import get_package_prefix
        auto_dir = os.path.join(get_package_prefix("automation"), "lib", "automation")
        if auto_dir not in sys.path:
            sys.path.insert(0, auto_dir)
        from daksha_web_access import guard_http_handler
    except Exception as e:
        log.warning("Daksha gate unavailable (%s); chat page is open to any browser", e)
        return handler_class
    return guard_http_handler(handler_class, APP_NAME)


def make_server(agent, host, port):
    page_path = os.path.join(get_package_share_directory("ros_agent"), "web", "chat.html")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the node's console quiet
            pass

        def _send(self, status, body, ctype="application/json", extra=()):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, default=str).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                with open(page_path, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self._send(200, agent.web_info())
            if path == "/api/llm":
                return self._send(200, agent.llm_status())
            if path == "/api/models":
                try:
                    return self._send(200, agent.list_models())
                except Exception as e:
                    return self._send(503, {"error": f"cannot list Ollama models: {e}"})
            if path == "/api/cameras":
                return self._send(200, {"cameras": agent.cameras.status()})
            if path.startswith("/api/camera/") and path.endswith(".jpg"):
                cam = path[len("/api/camera/"):-len(".jpg")]
                data, _, age = agent.cameras.latest(cam)
                if data is None:
                    return self._send(404, {"error": f"no frame for camera '{cam}'"})
                extra = [("X-Frame-Age", f"{age:.3f}")]
                track = agent.tracker.overlay(cam)
                if track:   # box drawn on exactly this frame's response
                    extra.append(("X-Track", json.dumps(track)))
                return self._send(200, data, "image/jpeg", extra)
            if path == "/api/track":
                return self._send(200, agent.tracker.state())
            if path.startswith("/api/pipeline/"):
                return self._pipeline_get(path[len("/api/pipeline/"):])
            self._send(404, {"error": "not found"})

        # ── real-robot pipeline ────────────────────────────────────────────
        def _pipe(self):
            p = getattr(agent, "pipeline", None)
            if p is None:
                self._send(503, {"ok": False, "message": getattr(agent, "pipeline_error", None)
                                 or "pipeline disabled (pipeline:=false)"})
            return p

        def _pipeline_get(self, what):
            p = self._pipe()
            if p is None:
                return
            if what == "state":
                return self._send(200, p.state())
            if what == "live.jpg":
                # straight from the camera, so the view works while the VLM is still loading
                snap = p.camera_snapshot()
                if snap is None:
                    return self._send(404, {"error": "no world camera image"})
                import cv2
                img = snap["bgr"].copy()
                try:   # detected objects: box + name (+ 3-D size); the prompt's goal in green
                    for o in p.objects_3d():
                        if not o.get("bbox"):
                            continue
                        x1, y1, x2, y2 = (int(round(v)) for v in o["bbox"])
                        col = (90, 200, 40) if o["goal"] else (40, 190, 240)
                        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if o["goal"] else 2)
                        s = o["size"]
                        tag = (("GOAL: " if o["goal"] else "") + o["name"] +
                               f"  {s[0] * 100:.0f}x{s[2] * 100:.0f} cm")
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ty = max(th + 6, y1)
                        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
                        cv2.putText(img, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                except Exception:
                    pass
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return self._send(200, jpg.tobytes(), "image/jpeg") if ok else self._send(500, {"error": "encode"})
            if what.startswith("wrist/") and what.endswith(".jpg"):
                data = p.wrist_view_jpeg(what[len("wrist/"):-len(".jpg")])
                return self._send(200, data, "image/jpeg") if data else self._send(404, {"error": "no wrist image"})
            if what == "vlm.jpg":
                if p.agent is None:
                    return self._send(404, {"error": "pipeline starting"})
                f = os.path.join(p.agent.stream_dir, "vlm.jpg")
                try:
                    with open(f, "rb") as fh:
                        return self._send(200, fh.read(), "image/jpeg")
                except OSError:
                    return self._send(404, {"error": "no image yet"})
            return self._send(404, {"error": "not found"})

        def _pipeline_post(self, what):
            p = self._pipe()
            if p is None:
                return
            try:
                req = self._json_body()
                if what == "command":
                    text = str(req.get("text") or "").strip()[:500]
                    if not text:
                        raise ValueError("need 'text'")
                    r = p.submit(text)
                    return self._send(200 if r["ok"] else 503, r)
                if what == "stop":
                    return self._send(200, {"ok": True, "message": p.stop()})
                if what == "emergency_stop":
                    return self._send(200, {"ok": True, "message": p.emergency_stop()})
                if what == "execute":
                    return self._send(200, {"ok": True, "message": p.set_execute(bool(req.get("on")))})
                if what == "measure_table":
                    ok, msg = p.measure_table()
                    return self._send(200 if ok else 400, {"ok": ok, "message": msg})
                if what == "grasp_style":
                    return self._send(200, {"ok": True, "message": p.set_grasp_style(req.get("style"))})
                if what == "max_replans":
                    return self._send(200, {"ok": True, "message": p.set_max_replans(req.get("n"))})
                if what == "allow_calibration":
                    return self._send(200, {"ok": True, "message": p.allow_inaccurate_calibration()})
                return self._send(404, {"error": "not found"})
            except ValueError as e:
                return self._send(400, {"ok": False, "message": str(e)})

        def _json_body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > _MAX_BODY:
                raise ValueError("request too large")
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise ValueError("body must be a JSON object")
            return req

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/pipeline/"):
                return self._pipeline_post(path[len("/api/pipeline/"):])
            if path == "/api/model":
                try:
                    name = str(self._json_body().get("model") or "").strip()
                    if not name:
                        raise ValueError("need 'model'")
                    return self._send(200, agent.set_model(name))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                except Exception as e:
                    return self._send(503, {"error": str(e)})
            if path == "/api/stop":
                self._json_body()
                return self._send(200, agent.stop_question())
            if path == "/api/track/stop":
                self._json_body()
                agent.tracker.stop()
                return self._send(200, agent.tracker.state())
            if path == "/api/track":
                try:
                    req = self._json_body()
                    point = req.get("point")
                    if point is not None and not (
                            isinstance(point, list) and len(point) == 2
                            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in point)):
                        raise ValueError("point must be [x, y] in 0..1")
                    query = str(req.get("object") or "").strip()[:200] or None
                    return self._send(200, agent.tracker.start(
                        str(req.get("camera") or "world"), query=query, point=point))
                except (ValueError, KeyError) as e:
                    return self._send(400, {"status": "error", "message": str(e)})
                except Exception as e:
                    return self._send(503, {"status": "error", "message": str(e)})
            if path != "/api/ask":
                return self._send(404, {"error": "not found"})
            try:
                req = self._json_body()
                if not str(req.get("question", "")).strip():
                    raise ValueError("need a non-empty 'question'")
            except ValueError as e:
                return self._send(400, {"ok": False, "answer": f"bad request: {e}"})
            self._send(200, agent.process_request({
                "question": str(req["question"]),
                "history": req.get("history"),
                "attach_cameras": req.get("attach_cameras") or [],
                "think": bool(req.get("think")),
            }, trace_images=True))

    server = ThreadingHTTPServer((host, port), _gate(Handler))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="ros_agent_web").start()
    return server
