"""ros_agent node.

Interfaces (all under the node namespace, default /ros_agent):
  ~/describe_robot   std_srvs/Trigger   structural summary (JSON in message)
  ~/check_collision  std_srvs/Trigger   live collision status (JSON in message)
  ~/request          std_msgs/String    JSON {"id", "question", "history"?, "attach_cameras"?}
                                        or {"id", "tool", "args"}
  ~/response         std_msgs/String    JSON {"id", "ok", "answer" | "result", "trace"}

Chat page: http://<robot>:<web_port>/ (web_port 0 disables it).
Cameras come from automation's config/camera_topics.yaml; the agent can look
through them with the look_at_camera tool (needs a vision LLM, e.g. Qwen3-VL).
"""

import json
import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml

from ament_index_python.packages import get_package_share_directory

from .camera import CameraFeed, default_camera_config, load_cameras
from .llm_client import ChatClient, LLMError, run_agent
from .robot_model import RobotModel, resolve_path
from .robot_state import LiveState, load_joint_directions
from .safety import validate_tool_call
from .tools import Tool, ToolBox
from .topics import TopicReader
from .tracker import ObjectTracker


class RosAgent(Node):
    def __init__(self):
        super().__init__("ros_agent")
        robot_id = self.declare_parameter("robot_id", "gen2").value
        cfg_path = self.declare_parameter("robots_config", "").value or os.path.join(
            get_package_share_directory("ros_agent"), "config", "robots.yaml")
        base_url = self.declare_parameter("llm_base_url", "http://localhost:11434/v1").value
        llm_model = self.declare_parameter("llm_model", "qwen3.5:9b").value
        api_key = self.declare_parameter("llm_api_key", "ollama").value
        timeout = self.declare_parameter("llm_timeout", 120.0).value
        self.max_steps = self.declare_parameter("llm_max_steps", 8).value
        max_age = self.declare_parameter("max_state_age", 0.5).value
        cam_cfg = self.declare_parameter("camera_topics_config", "").value or \
            default_camera_config()
        cam_max_age = self.declare_parameter("camera_max_age", 2.0).value
        web_host = self.declare_parameter("web_host", "0.0.0.0").value
        web_port = self.declare_parameter("web_port", 8150).value
        self.robot_id, self.llm_model = robot_id, llm_model
        self.llm_base = llm_model           # the name picked in the UI; llm_model = its rax-*:64k twin
        self.llm_base_url = base_url
        self.pipeline = None

        with open(cfg_path) as f:
            robots = (yaml.safe_load(f) or {}).get("robots", {})
        if robot_id not in robots:
            raise RuntimeError(f"robot '{robot_id}' not in {cfg_path}; known: {list(robots)}")
        cfg = robots[robot_id]

        self.model = RobotModel(resolve_path(cfg["urdf"]),
                                resolve_path(cfg.get("srdf")),
                                cfg.get("groups"), cfg.get("end_effectors"))
        directions = load_joint_directions(resolve_path(cfg.get("joint_direction_file")))
        self.live = LiveState(self, cfg, directions)
        cameras = load_cameras(cam_cfg)
        if not cameras:
            self.get_logger().warn(f"[ros_agent] no cameras loaded from '{cam_cfg}'")
        self.cameras = CameraFeed(self, cameras)
        self.tools = ToolBox(self.model, self.live, cfg, max_age, self.cameras, cam_max_age)
        # Any topic in the ROS graph: list it, read its values, watch it.
        self.topics = TopicReader(self)
        for t in self.topics.tools(Tool):
            self.tools.registry[t.name] = t
        self.llm = ChatClient(base_url, llm_model, api_key, timeout) if base_url else None
        # One question at a time against the LLM, whichever interface asked.
        self._llm_lock = threading.Lock()
        self._cancel = threading.Event()   # set by Stop: abandon the running question

        # Real-time bounding box on a chosen object (LLM finds it, CSRT follows it).
        redetect_s = self.declare_parameter("track_redetect_s", 5.0).value
        self.track_pub = self.create_publisher(String, "~/tracked_object", 10)
        self.tracker = ObjectTracker(
            self.cameras, self.llm, log=lambda s: self.get_logger().info(f"[ros_agent] track: {s}"),
            publish=lambda st: self.track_pub.publish(String(data=json.dumps(st))),
            redetect_s=redetect_s)
        if self.cameras.cameras and self.llm:
            self.tools.registry["box_object"] = Tool(
                "box_object", "Draw a live bounding box around an object in a camera view "
                "(shown on the operator's chat page and published on ~/tracked_object), and "
                "keep it on the object as it moves. Use when asked to box, mark, highlight, "
                "track or point out an object. Returns where it was found.",
                self._box_object,
                {"object": ("string", True, "what to box, e.g. 'red cup'; 'none' clears it"),
                 "camera": ("string", False, "camera id (default 'world')")})

        self.create_service(Trigger, "~/describe_robot", self._srv("describe_robot"))
        self.create_service(Trigger, "~/check_collision", self._srv("check_collision"))
        self.pub = self.create_publisher(String, "~/response", 10)
        self.create_subscription(String, "~/request", self._on_request, 10)

        self._jobs = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

        d = self.model.describe()
        self.get_logger().info(
            f"[ros_agent] {robot_id}: '{d['name']}' {d['dof']} DOF, groups "
            f"{list(d['groups'])}, cameras {list(self.cameras.cameras)}, "
            f"LLM {'at ' + base_url if self.llm else 'disabled'}")

        self.web = None
        if web_port:
            threading.Thread(target=self._start_web, args=(web_host, int(web_port)),
                             daemon=True).start()

    def _start_web(self, host, port):
        """Bind the chat page, retrying while the port is held (e.g. by an old
        agent that is still shutting down)."""
        from .web import make_server
        warned = False
        while rclpy.ok():
            try:
                self.web = make_server(self, host, port)
                self.get_logger().info(f"[ros_agent] chat page {self.chat_url(host, port)}")
                return
            except OSError as e:
                if not warned:
                    self.get_logger().warn(f"[ros_agent] chat page port {port} busy ({e}); "
                                           "retrying every 5 s")
                    warned = True
                time.sleep(5.0)

    @staticmethod
    def chat_url(host, port):
        """Link a browser on the LAN can open: <hostname>.local (mDNS) when bound to
        all interfaces, e.g. http://s2.local:8150/."""
        if host in ("0.0.0.0", "::", ""):
            import socket
            name = socket.gethostname().split(".")[0]
            host = f"{name}.local"
        return f"http://{host}:{port}/"

    def web_info(self):
        d = self.model.describe()
        return {"robot_id": self.robot_id, "name": d["name"], "dof": d["dof"],
                "groups": list(d["groups"]), "llm": self.llm_base if self.llm else None,
                "llm_status": self.llm_status(),
                "cameras": self.cameras.status(),
                "tools": [t["name"] for t in self.tools.specs()]}

    def list_models(self):
        """{"current", "models": [...]} -- models available in Ollama."""
        if not self.llm:
            return {"current": None, "models": []}
        # the rax-*:64k twins are an implementation detail: the picker shows the base models
        models = [m for m in self.llm.list_models() if not m["name"].startswith("rax-")]
        return {"current": self.llm_base, "models": models}

    def stop_question(self):
        """Stop the running question at its next step (the model call already
        in flight finishes in the background)."""
        self._cancel.set()
        return {"ok": True}

    def set_model(self, name):
        """Switch the LLM at runtime (the next question uses it)."""
        if not self.llm:
            raise ValueError("no LLM configured (llm_base_url is empty)")
        names = [m["name"] for m in self.llm.list_models()]
        if name not in names:
            raise ValueError(f"'{name}' is not in Ollama; have: {', '.join(names)}")
        # ONE model for the whole pipeline: Ask, the box tracker, the VLM detector and the RAX
        # agent all use the same Ollama model (its rax-*:64k twin), so only one copy is loaded
        from .rax_agent import ensure_agent_model
        effective, note = ensure_agent_model(self.llm_base_url, name)
        self.llm.set_model(effective)
        self.llm_model, self.llm_base = effective, name
        if self.pipeline is not None:
            self.pipeline.set_model(effective)
        self.get_logger().info(f"[ros_agent] model for the whole pipeline: {name} ({effective})"
                               + (f"; {note}" if note else ""))
        return {**self.llm_status(), **({"warning": note} if note else {})}

    def _box_object(self, a):
        obj = a["object"].strip()
        if obj.lower() in ("", "none", "clear", "nothing"):
            self.tracker.stop()
            return {"cleared": True}
        cam = a.get("camera") or "world"
        self.tracker.start(cam, query=obj)
        s = self.tracker.wait_detected(timeout=30.0)
        out = {"camera": cam, "object": obj, "status": s["status"], "label": s.get("label")}
        if s.get("box"):
            x1, y1, x2, y2 = s["box"]
            w, h = s["size"]
            out.update(box_px=s["box"], image_size=s["size"],
                       center_in_image={"x": round((x1 + x2) / 2 / w, 2),
                                        "y": round((y1 + y2) / 2 / h, 2)},
                       note="box is now shown live on the chat page and follows the object")
        elif s.get("message"):
            out["message"] = s["message"]
        return out

    def llm_status(self):
        """{"state": ready|down|disabled, "model", "url", "error"?}.

        Ready means the Ollama server answers and has llm_model pulled; Ollama
        loads the model into memory on the first question."""
        if not self.llm:
            return {"state": "disabled", "model": None}
        base = self.llm.url[:-len("/chat/completions")]
        out = {"model": self.llm_model, "url": base}
        if self.llm.ready():
            return {**out, "state": "ready"}
        if self.llm.last_models is not None:   # server up, model not pulled
            return {**out, "state": "down",
                    "error": f"model '{self.llm_model}' not in Ollama; run "
                             f"`ollama pull {self.llm_model}` (have: "
                             f"{', '.join(self.llm.last_models) or 'none'})"}
        return {**out, "state": "down", "error": f"Ollama not reachable at {base}"}

    def _srv(self, tool):
        def cb(_req, res):
            try:
                res.message = json.dumps(self.tools.call(tool, {}), default=str)
                res.success = True
            except Exception as e:
                res.success, res.message = False, str(e)
            return res
        return cb

    def _on_request(self, msg):
        try:
            req = json.loads(msg.data)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
        except ValueError as e:
            self._reply({"id": None, "ok": False, "answer": f"bad request: {e}"})
            return
        self._jobs.put(req)

    def _reply(self, obj):
        self.pub.publish(String(data=json.dumps(obj, default=str)))

    def process_request(self, req, trace_images=False):
        """Process one request dict; returns the reply dict (without "id")."""
        try:
            if "tool" in req:
                args = req.get("args", {}) or {}
                ok, err = validate_tool_call(self.tools.registry, req["tool"], args)
                if not ok:
                    raise ValueError(err)
                result = self.tools.call(req["tool"], args)
                if isinstance(result, dict) and "_image" in result:
                    result["image_bytes"] = len(result.pop("_image"))
                return {"ok": True, "result": result}
            if "question" in req:
                if not self.llm:
                    raise LLMError("LLM disabled (llm_base_url is empty); use a tool request")
                images = []
                for cam in req.get("attach_cameras") or []:
                    jpeg = self.cameras.snapshot(cam)[0] if cam in self.cameras.cameras else None
                    if jpeg is not None:
                        images.append((cam, jpeg))
                with self._llm_lock:
                    self._cancel.clear()   # a Stop meant for the previous question
                    answer, trace = run_agent(
                        self.llm, self.tools, self.model, req["question"], self.max_steps,
                        lambda n, a: validate_tool_call(self.tools.registry, n, a),
                        log=lambda s: self.get_logger().info(f"[ros_agent] {s}"),
                        history=req.get("history"), images=images,
                        trace_images=trace_images, topics=self.topics.published(),
                        think=bool(req.get("think")), cancel=self._cancel)
                return {"ok": True, "answer": answer, "trace": trace}
            if req.get("list_tools"):
                return {"ok": True, "result": self.tools.specs()}
            raise ValueError("request needs 'question', 'tool' or 'list_tools'")
        except Exception as e:
            return {"ok": False, "answer": str(e)}

    def _worker(self):
        while rclpy.ok():
            req = self._jobs.get()
            self._reply({"id": req.get("id"), **self.process_request(req)})


def main():
    """One process, one web page: the read-only chat agent and (unless pipeline:=false)
    the real-robot VLM pick-and-place pipeline node, spun on one executor."""
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = RosAgent()
    pipeline = None
    if node.declare_parameter("pipeline", True).value:
        try:
            from .pipeline_node import PipelineNode
            overrides = {"execute": node.declare_parameter("pipeline_execute", False).value}
            # one model: pipeline_model (if given) or the chat's llm_model, as its rax-*:64k twin
            model = node.declare_parameter("pipeline_model", "").value or node.llm_base
            node.llm_base = model
            try:
                from .rax_agent import ensure_agent_model
                overrides["model"] = ensure_agent_model(node.llm_base_url, model)[0]
            except Exception as e:
                node.get_logger().warn(f"[ros_agent] model twin for {model}: {e}")
                overrides["model"] = model
            pipeline = PipelineNode(overrides)
            node.pipeline = pipeline          # the chat page serves its /api/pipeline/* routes

            # one chat: questions typed to the robot are answered by this Ask agent (ROS topics,
            # joints, cameras, read-only tools) with the world camera frame attached
            def ask(question):
                r = node.process_request({"question": question, "attach_cameras": ["world"]})
                return r.get("answer") if r.get("ok") else f"Could not answer: {r.get('answer') or r.get('error') or r}"
            pipeline.ask_fn = ask
            pipeline.tracker = node.tracker       # live box tracking feeds the pipeline's objects

            def not_robot(box):
                arm = pipeline.robot_part(box, skip=pipeline.held_arms())
                return f'that is the {arm} gripper / wrist camera, not an object' if arm else None
            node.tracker.reject = not_robot
            # a second tracker for the wrist camera of the picking arm (the world one keeps its object)
            # one tracker per wrist (gripper) camera: the object gets a live box in all three views.
            # A slower re-check than the world one keeps the shared vision model free for the agent.
            wrist_redetect = node.declare_parameter("wrist_track_redetect_s", 10.0).value
            pipeline.wrist_trackers = {arm: ObjectTracker(
                node.cameras, node.llm, log=lambda s, arm=arm: node.get_logger().info(f"[ros_agent] {arm} wrist track: {s}"),
                redetect_s=wrist_redetect) for arm in ("right", "left")} if node.llm else {}

            # RAX as the task agent: every prompt -> Claude Code on the UI-picked Ollama model ->
            # RAX robot tools (in process, on this pipeline) -> aruco_reach
            use_rax = node.declare_parameter("rax_agent", True).value

            def wire():
                while pipeline.agent is None and pipeline.agent_error is None and rclpy.ok():
                    time.sleep(0.5)
                if pipeline.agent is None:
                    return
                try:
                    node.set_model(node.llm_base)          # one model everywhere (creates the 64k twin once)
                except Exception as e:
                    pipeline.log(f"model setup: {e}")
                if not use_rax:
                    return
                try:
                    from .rax_agent import RaxAgent
                    rax = RaxAgent(pipeline, node, node.llm_base_url)
                    rax.start()
                    node.rax = rax
                    pipeline.agent.rax_run, pipeline.agent.rax_stop = rax.run, rax.stop
                except Exception as e:
                    pipeline.log(f"RAX agent not available ({type(e).__name__}: {e}); using the local planner")
            threading.Thread(target=wire, daemon=True, name="pipeline_wire").start()
        except Exception as e:
            node.get_logger().error(f"[ros_agent] pipeline not started: {type(e).__name__}: {e}")
            node.pipeline_error = f"{type(e).__name__}: {e}"
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    if pipeline is not None:
        executor.add_node(pipeline)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, "rax", None) is not None:
            node.rax.close()
        if pipeline is not None:
            pipeline.stop()               # never leave a motion running
        node.tracker.stop()
        if node.web:
            node.web.shutdown()
        executor.shutdown(timeout_sec=2.0)
        if pipeline is not None:
            pipeline.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
