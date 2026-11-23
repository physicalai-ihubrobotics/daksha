"""Generic, read-only access to every ROS topic for the agent.

list_topics   -- every topic with its type and publisher/subscriber counts
read_topic    -- wait for the next message on any topic and return its values
monitor_topic -- watch a topic for a few seconds: rate, and min/max/mean/last of
                 every numeric field (is it moving? how noisy? how fast?)

The message type is resolved at runtime from the graph, so no topic list has to
be maintained. Subscriptions are temporary (created per call, destroyed after)
and use a QoS matched to the publishers, so latched topics like
/robot_description are received too. Nothing is ever published.

Messages are converted to plain JSON. Large arrays (images, point clouds, laser
scans) are summarised -- length, first values, min/max/mean -- and the result is
shrunk to a character budget so it fits the model's context.
"""

import array
import json
import math
import threading
import time

from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

MAX_CHARS = 3500          # per tool result, the LLM context is only 8k tokens
LIST_LIMIT = 120          # topics listed without a filter


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _r(x, nd=5):
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else str(x)
    return x


def _is_msg(x):
    return hasattr(x, "get_fields_and_field_types")


def _seq(x):
    """Sequence-like field values (list, tuple, array.array, numpy, bytes)."""
    if isinstance(x, (list, tuple, array.array, bytes, bytearray)):
        return True
    return type(x).__name__ == "ndarray"


def to_plain(msg, preview=8):
    """Message -> JSON-able dict. Arrays longer than 2*preview are summarised."""
    if _is_msg(msg):
        return {f: to_plain(getattr(msg, f), preview) for f in msg.get_fields_and_field_types()}
    if _seq(msg):
        items = list(msg)
        n = len(items)
        if n <= 2 * preview:
            return [to_plain(v, preview) for v in items]
        out = {"_array_length": n,
               "first": [to_plain(v, preview) for v in items[:preview]]}
        nums = [float(v) for v in items if _numeric(v) or type(v).__module__ == "numpy"]
        if len(nums) == n:
            finite = [v for v in nums if math.isfinite(v)]
            if finite:
                out.update(min=_r(min(finite)), max=_r(max(finite)),
                           mean=_r(sum(finite) / len(finite)))
            if len(finite) < n:
                out["non_finite"] = n - len(finite)
        return out
    if type(msg).__module__ == "numpy":
        msg = msg.item()
    if isinstance(msg, float):
        return _r(msg, 6)
    if isinstance(msg, str) and len(msg) > 600:
        return msg[:600] + f"... ({len(msg)} chars)"
    return msg


def select_field(data, path):
    """'pose.position.x' or 'position[3]' or 'name' from a to_plain() dict."""
    cur = data
    for part in path.replace("]", "").replace("[", ".").split("."):
        if not part:
            continue
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"no field '{part}' in path '{path}'; fields here: {list(cur)}")
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise KeyError(f"bad index '{part}' in path '{path}' (list of {len(cur)})")
        else:
            raise KeyError(f"'{path}': cannot go into a {type(cur).__name__}")
    return cur


def fit(data, rebuild, budget=MAX_CHARS):
    """Shrink a result until its JSON fits `budget`: smaller array previews first,
    then hard truncation. `rebuild(preview)` regenerates the data."""
    for preview in (8, 4, 2, 1):
        text = json.dumps(data, default=str)
        if len(text) <= budget:
            return data, False
        data = rebuild(preview)
    text = json.dumps(data, default=str)
    if len(text) <= budget:
        return data, False
    return {"truncated_json": text[:budget] + " ...",
            "hint": "message is large; use the 'field' argument to read one part"}, True


def _flatten(data, prefix="", out=None):
    """{'a.b': 1.0, 'name[2]': 'j3', ...} for every scalar number/string, skipping
    summarised big arrays and header timestamps (they always change)."""
    out = {} if out is None else out
    if isinstance(data, dict):
        if "_array_length" in data:
            return out
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if path == "header.stamp" or path.endswith(".header.stamp"):
                continue
            _flatten(v, path, out)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            _flatten(v, f"{prefix}[{i}]", out)
    elif _numeric(data) and math.isfinite(data):
        out[prefix or "value"] = float(data)
    elif isinstance(data, str):
        out[prefix or "value"] = data
    return out


class TopicReader:
    def __init__(self, node):
        self.node = node

    # ------------------------------------------------------------- discovery
    def _types(self):
        return dict(self.node.get_topic_names_and_types())

    def published(self):
        """Names of topics that have at least one publisher (skips ROS internals)."""
        skip = ("/parameter_events", "/rosout")
        return [t for t in self._types() if t not in skip and "/_action/" not in t
                and self.node.count_publishers(t) > 0]

    def list_topics(self, a):
        flt = (a.get("filter") or "").lower()
        rows = []
        for name, types in sorted(self._types().items()):
            if flt and flt not in name.lower() and not any(flt in t.lower() for t in types):
                continue
            rows.append([name, types[0] if len(types) == 1 else types,
                         self.node.count_publishers(name)])
        total = len(rows)
        out = {"count": total,
               "columns": ["topic", "type", "publishers"],
               "topics": rows[:LIST_LIMIT]}
        if total > LIST_LIMIT:
            out["note"] = (f"showing {LIST_LIMIT} of {total}; pass 'filter' "
                           "(part of a topic name or type) to narrow it")
        if not total:
            out["note"] = "no topic matches" if flt else "no topics visible (ROS graph empty?)"
        return out

    def _resolve(self, topic):
        if not topic.startswith("/"):
            topic = "/" + topic
        types = self._types()
        if topic not in types:
            import difflib
            key = topic.strip("/").split("/")[-1]
            near = [t for t in types if key and key in t][:8] or \
                difflib.get_close_matches(topic, list(types), n=5, cutoff=0.5)
            raise KeyError(f"topic '{topic}' not found" +
                           (f"; similar: {near}" if near else "; use list_topics"))
        from rosidl_runtime_py.utilities import get_message
        type_name = types[topic][0]
        try:
            return topic, type_name, get_message(type_name)
        except (AttributeError, ModuleNotFoundError, ValueError) as e:
            raise RuntimeError(f"message type {type_name} is not installed here: {e}")

    def _qos(self, topic):
        """Best effort (matches any publisher); transient local only when every
        publisher latches, so latched topics deliver their last message."""
        pubs = self.node.get_publishers_info_by_topic(topic)
        latched = bool(pubs) and all(
            p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in pubs)
        reliable = bool(pubs) and all(
            p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in pubs)
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE if (latched and reliable)
            else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if latched
            else DurabilityPolicy.VOLATILE), len(pubs)

    def _collect(self, topic, cls, seconds, first_only):
        """Messages received within `seconds` (stops at the first if first_only)."""
        qos, npub = self._qos(topic)
        got, done = [], threading.Event()

        def cb(msg):
            got.append((time.monotonic(), msg))
            if first_only:
                done.set()

        sub = self.node.create_subscription(cls, topic, cb, qos)
        try:
            done.wait(seconds)
        finally:
            self.node.destroy_subscription(sub)
        return got, npub

    # --------------------------------------------------------------- reading
    def read_topic(self, a):
        topic, type_name, cls = self._resolve(a["topic"])
        timeout = min(max(float(a.get("timeout", 3.0)), 0.2), 10.0)
        got, npub = self._collect(topic, cls, timeout, first_only=True)
        base = {"topic": topic, "type": type_name, "publishers": npub}
        if not got:
            return {**base, "received": False,
                    "reason": (f"no message within {timeout:g}s" +
                               ("" if npub else " -- the topic has no publisher"))}
        msg = got[0][1]
        field = a.get("field")

        def build(preview):
            data = to_plain(msg, preview)
            return select_field(data, field) if field else data

        data, _ = fit(build(8), build)
        out = {**base, "received": True}
        if field:
            out["field"] = field
        out["message"] = data
        stamp = _stamp_age(msg, self.node)
        if stamp is not None:
            out["header_age_s"] = stamp
        if not field and type_name.startswith("std_msgs/"):
            related = self._related(topic)
            if related:
                out["related_topics"] = related
        return out

    def _related(self, topic, limit=4, wait=0.6):
        """Current values of sibling std_msgs topics (/battery_status ->
        /battery_info, /battery_health) so the answer can use all of them."""
        ns, _, leaf = topic.rpartition("/")
        stem = leaf.split("_")[0]
        types = self._types()
        out = {}
        for t in sorted(types):
            tns, _, tleaf = t.rpartition("/")
            if t == topic or tns != ns or tleaf.split("_")[0] != stem \
                    or not types[t][0].startswith("std_msgs/") \
                    or self.node.count_publishers(t) == 0:
                continue
            try:
                from rosidl_runtime_py.utilities import get_message
                got, _ = self._collect(t, get_message(types[t][0]), wait, first_only=True)
            except Exception:
                continue
            if got:
                val = to_plain(got[-1][1], 4)
                val = val.get("data", val) if isinstance(val, dict) and len(val) == 1 else val
                if len(json.dumps(val, default=str)) <= 300:
                    out[t] = val
            if len(out) >= limit:
                break
        return out

    def monitor_topic(self, a):
        topic, type_name, cls = self._resolve(a["topic"])
        seconds = min(max(float(a.get("seconds", 2.0)), 0.5), 10.0)
        got, npub = self._collect(topic, cls, seconds, first_only=False)
        out = {"topic": topic, "type": type_name, "publishers": npub,
               "window_s": seconds, "messages": len(got)}
        if not got:
            out["reason"] = "no messages" + ("" if npub else " -- the topic has no publisher")
            return out
        if len(got) > 1:
            span = got[-1][0] - got[0][0]
            out["rate_hz"] = round((len(got) - 1) / span, 2) if span > 0 else None
        field = a.get("field")
        series, texts = {}, {}
        for _, msg in got:
            data = to_plain(msg, 8)
            if field:
                data = select_field(data, field)
                if not isinstance(data, (dict, list)):
                    data = {field: data}
            for k, v in _flatten(data).items():
                if isinstance(v, str):
                    seen = texts.setdefault(k, [])
                    if v not in seen:
                        seen.append(v)
                else:
                    series.setdefault(k, []).append(v)
        if texts:   # strings: the distinct values seen, in order (last = current)
            out["text_values"] = {k: (v if len(v) <= 6 else v[:2] + ["..."] + v[-3:])
                                  for k, v in list(texts.items())[:20]}
        stats = {}
        for k, vals in series.items():
            lo, hi = min(vals), max(vals)
            stats[k] = {"last": _r(vals[-1]), "min": _r(lo), "max": _r(hi),
                        "mean": _r(sum(vals) / len(vals)), "change": _r(hi - lo)}
        changing = [k for k, s in stats.items() if s["change"] not in (0, 0.0)]
        out["changing_fields"] = changing[:40]
        out["constant_fields"] = len(stats) - len(changing)
        # keep the per-field stats only for what changed (plus a few constants)
        keep = {k: stats[k] for k in changing}
        for k in stats:
            if len(keep) >= len(changing) + 10:
                break
            keep.setdefault(k, {"value": stats[k]["last"]})
        out["fields"] = keep
        text = json.dumps(out, default=str)
        if len(text) > MAX_CHARS:
            out["fields"] = {k: keep[k] for k in list(keep)[:25]}
            out["note"] = "many numeric fields; pass 'field' to focus on one part"
        return out

    def tools(self, Tool):
        return [
            Tool("list_topics", "List the ROS topics that exist right now, with message type "
                 "and number of publishers. Use it to find where some data is published.",
                 self.list_topics,
                 {"filter": ("string", False, "only topics whose name or type contains this, "
                             "e.g. 'joint', 'camera', 'sensor_msgs/Imu'")}),
            Tool("read_topic", "Read the latest message on ANY ROS topic and return its "
                 "values (large arrays are summarised). Use for data no other tool covers: "
                 "sensors, status, diagnostics, controller state, gripper, battery...",
                 self.read_topic,
                 {"topic": ("string", True, "topic name from list_topics"),
                  "field": ("string", False, "only this part, e.g. 'pose.position' or "
                            "'position[2]'"),
                  "timeout": ("number", False, "seconds to wait for a message (default 3, "
                              "max 10)")}),
            Tool("monitor_topic", "Watch a topic for a few seconds: message rate and "
                 "min/max/mean/last of every numeric field, plus which fields changed. Use "
                 "to tell whether something is moving, publishing, noisy or stuck.",
                 self.monitor_topic,
                 {"topic": ("string", True, "topic name from list_topics"),
                  "field": ("string", False, "only this part of the message"),
                  "seconds": ("number", False, "how long to watch (default 2, max 10)")}),
        ]


def _stamp_age(msg, node):
    h = getattr(msg, "header", None)
    st = getattr(h, "stamp", None)
    if st is None or not hasattr(st, "sec") or (st.sec == 0 and st.nanosec == 0):
        return None
    now = node.get_clock().now().nanoseconds * 1e-9
    return round(now - (st.sec + st.nanosec * 1e-9), 3)
