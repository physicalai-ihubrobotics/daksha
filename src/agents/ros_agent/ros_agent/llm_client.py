"""Minimal OpenAI-compatible chat client (Ollama's /v1 endpoint) over urllib,
plus the JSON tool-use loop. No third-party dependencies."""

import json
import re
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You are a ROS 2 robot intelligence agent for the robot below.
You can ONLY observe the robot through the tools listed. You cannot move it.

Robot: {robot}
Groups: {groups}
End effectors: {ees}

Tools:
{tools}

Rules:
- Never invent joint, link, group or frame names; use names from the tools.
- Ground every claim about the current state in a tool result. If data is
  unavailable or stale, say so instead of guessing.
- If asked to move, plan or execute motion, explain that this agent is
  read-only and describe what you observed instead.
- Requests to "box", "put a box on", "mark", "highlight", "track" or "point
  out" an object mean drawing a bounding box on the camera view: call
  box_object. It only annotates the video for the operator; it moves nothing,
  so it is always allowed.
- For data the specific tools do not cover (sensors, gripper, battery, modes,
  controller or node status, camera calibration, diagnostics, any topic the
  user names), pick the topic from the list below (or list_topics), then
  read_topic for its values or monitor_topic to see whether it changes and how
  fast. When several topics relate to the question (e.g. x_status, x_info and
  x_health), read EVERY one of them before answering. Quote
  the actual values you read and base the answer on them; if they contradict
  each other or look like placeholders (e.g. a status saying "waiting" or
  "disconnected"), say so instead of picking one.
{topics}
- Earlier turns of the conversation may be out of date. For anything about the
  current state or scene, check again with a tool (or the attached image)
  instead of repeating an earlier answer.
{vision}

Reply with exactly ONE JSON object and nothing else, either
  {{"tool": "<name>", "args": {{...}}}}   to call a tool, or
  {{"answer": "<final answer for the user>"}}"""


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url, model, api_key="EMPTY", timeout=60.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout

    last_models = None   # model ids from the last /models answer; None = unreachable

    def ready(self, max_age=5.0):
        """True if the server answers /models AND lists self.model (Ollama answers
        /models even when the model is not pulled). Cached for `max_age` s so
        status polls from the chat page stay cheap."""
        now = time.monotonic()
        cached = getattr(self, "_ready_cache", None)
        if cached and now - cached[1] < max_age:
            return cached[0]
        try:
            req = urllib.request.Request(self.url[:-len("/chat/completions")] + "/models",
                                         headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=1.0) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            self.last_models = ids
            # "qwen3-vl" means "qwen3-vl:latest" to Ollama
            names = {self.model, self.model + ":latest"}
            ok = bool(names & set(ids))
        except Exception:
            self.last_models = None
            ok = False
        self._ready_cache = (ok, now)
        return ok

    def list_models(self):
        """[{"name", "vision", "thinking", "size_gb"}] from Ollama's native API."""
        root = self.url[:-len("/chat/completions")]
        root = root[:-3] if root.endswith("/v1") else root

        def get(path, payload=None):
            req = urllib.request.Request(root + path, data=payload and json.dumps(payload).encode())
            with urllib.request.urlopen(req, timeout=5.0) as r:
                return json.loads(r.read())

        out = []
        for m in get("/api/tags").get("models", []):
            try:
                caps = get("/api/show", {"model": m["name"]}).get("capabilities") or []
            except Exception:
                caps = []
            out.append({"name": m["name"], "vision": "vision" in caps,
                        "thinking": "thinking" in caps,
                        "size_gb": round(m.get("size", 0) / 1e9, 1)})
        return sorted(out, key=lambda m: m["name"])

    def set_model(self, name):
        self.model = name
        self._ready_cache = None

    def chat(self, messages, max_tokens=1024, think=False):
        return self.chat_full(messages, max_tokens, think)[0]

    def chat_full(self, messages, max_tokens=1024, think=False):
        """(content, reasoning). think=False asks thinking models to answer
        directly (Ollama honours it for models with thinking levels); think=True
        lets them reason first and adds room for the reasoning tokens."""
        req = {"model": self.model, "messages": messages, "temperature": 0.1,
               "max_tokens": max_tokens + (4096 if think else 0)}
        if not think:
            req["reasoning_effort"] = "none"
        body = json.dumps(req).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:  # server is up but rejected the request
            detail = e.read().decode(errors="replace")[:400]
            try:
                detail = json.loads(detail).get("error", {}).get("message", detail)
            except (ValueError, AttributeError):
                pass
            raise LLMError(f"LLM server error (HTTP {e.code}): {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LLMError(f"LLM endpoint {self.url} unreachable ({e}). If the model "
                           "server was just started it may still be loading; try again "
                           "in a minute.") from e
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response: {str(data)[:200]}") from e
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        if not content.strip() and reasoning:
            raise LLMError(
                f"model '{self.model}' spent its whole reply thinking and gave no answer. " +
                ("Turn Thinking off, or ask a simpler question." if think else
                 "Its thinking cannot be turned off; pick another model, e.g. qwen3.5:9b."))
        return content, reasoning


def extract_json(text):
    """First JSON object in a model reply (tolerates <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in reply")


VISION_RULES = """- You can SEE through the robot's cameras. For any question about the
  scene, objects, people, what the arms are holding or where things are, look
  first (look_at_camera, default "world") unless a current image is already
  attached. Describe only what is visible; say when something is unclear,
  occluded or out of frame. Camera images are not metric: do not state
  distances or sizes from an image as if measured."""

_MAX_HISTORY = 20


def _image_part(jpeg):
    from .camera import data_uri
    return {"type": "image_url", "image_url": {"url": data_uri(jpeg)}}


def _drop_old_images(messages):
    """Keep only the newest image in the conversation (bounds context/latency)."""
    for m in messages:
        if isinstance(m["content"], list):
            text = " ".join(p["text"] for p in m["content"] if p["type"] == "text")
            m["content"] = text + " [image omitted: superseded by a newer image]"


def _clean_history(history):
    out = []
    for m in (history or [])[-_MAX_HISTORY:]:
        if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str) and m["content"].strip()):
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _topics_block(topics, limit=90):
    """Compact list of published topics for the system prompt."""
    if not topics:
        return ""
    names = sorted(topics)
    more = f" (+{len(names) - limit} more: use list_topics)" if len(names) > limit else ""
    return "  Topics with a publisher right now: " + ", ".join(names[:limit]) + more


def run_agent(client, toolbox, model, question, max_steps, validate, log=print,
              history=None, images=None, trace_images=False, topics=None,
              think=False, cancel=None):
    """Tool-use loop. Returns (answer, trace).

    think   -- let the model reason before each reply; the reasoning goes into
               the trace as {"tool": "thinking", "thinking": text} steps.
    cancel  -- threading.Event; when set, stop before the next model call.
    history -- earlier [{"role": "user"|"assistant", "content": str}] turns.
    images  -- [(label, jpeg_bytes)] attached to the question (e.g. the current
               world-camera frame).
    trace_images -- put the frames the model saw into the trace as data URIs.
    topics  -- names of the topics currently published (shown to the model).
    """
    from .camera import data_uri
    has_vision = "look_at_camera" in toolbox.registry or bool(images)
    system = SYSTEM_PROMPT.format(
        robot=f"{model.name} ({model.describe()['dof']} DOF)",
        groups=json.dumps({g: d["joints"] for g, d in model.groups.items()}),
        ees=json.dumps(model.end_effectors),
        tools=json.dumps(toolbox.specs(), indent=1),
        vision=VISION_RULES if has_vision else "",
        topics=_topics_block(topics))
    messages = [{"role": "system", "content": system}] + _clean_history(history)
    trace = []
    if images:
        labels = ", ".join(label for label, _ in images)
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"{question}\n\n(Attached: current frame from {labels}.)"},
            *[_image_part(jpeg) for _, jpeg in images]]})
        for label, jpeg in images:
            step = {"tool": "attached_image", "args": {"camera": label}, "result": {}}
            if trace_images:
                step["image"] = data_uri(jpeg)
            trace.append(step)
    else:
        messages.append({"role": "user", "content": question})
    for _ in range(max_steps):
        if cancel is not None and cancel.is_set():
            return "Stopped.", trace
        reply, reasoning = client.chat_full(messages, think=think)
        if cancel is not None and cancel.is_set():
            return "Stopped.", trace
        if think and reasoning.strip():
            trace.append({"tool": "thinking", "args": {}, "result": {},
                          "thinking": reasoning.strip()})
        messages.append({"role": "assistant", "content": reply})
        try:
            obj = extract_json(reply)
        except ValueError:
            messages.append({"role": "user", "content":
                             "Invalid reply. Respond with one JSON object only."})
            continue
        if "answer" in obj:
            return str(obj["answer"]), trace
        name, args = obj.get("tool"), obj.get("args", {}) or {}
        ok, err = validate(name, args)
        if ok:
            try:
                result = toolbox.call(name, args)
            except Exception as e:  # tool errors go back to the model, not the user
                result = {"error": str(e)}
        else:
            result = {"error": f"rejected: {err}"}
        image = result.pop("_image", None) if isinstance(result, dict) else None
        log(f"tool {name}({args}) -> {str(result)[:160]}")
        step = {"tool": name, "args": args, "result": result}
        text = f"TOOL_RESULT {name}: {json.dumps(result, default=str)}"
        if image is None:
            messages.append({"role": "user", "content": text})
        else:
            _drop_old_images(messages)
            messages.append({"role": "user", "content": [
                {"type": "text", "text": text + "\nThe camera image is attached."},
                _image_part(image)]})
            if trace_images:
                step["image"] = data_uri(image)
        trace.append(step)
    return "I could not finish within the tool-call budget.", trace
