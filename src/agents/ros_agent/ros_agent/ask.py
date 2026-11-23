"""CLI client for ros_agent.

    ros2 run ros_agent ask "is the right arm close to colliding?"
    ros2 run ros_agent ask --tool check_collision
    ros2 run ros_agent ask --tool get_joint_states group=left_arm
    ros2 run ros_agent ask --list-tools
"""

import argparse
import json
import sys
import time
import uuid

import rclpy
from std_msgs.msg import String


def _parse_arg(kv):
    key, _, val = kv.partition("=")
    try:
        return key, json.loads(val)
    except json.JSONDecodeError:
        return key, val


def main():
    ap = argparse.ArgumentParser(description="Ask the ros_agent node.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--tool")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--ns", default="/ros_agent")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("args", nargs="*", help="tool args as key=value")
    a, extra = ap.parse_known_args()

    rid = uuid.uuid4().hex[:8]
    if a.list_tools:
        req = {"id": rid, "list_tools": True}
    elif a.tool:
        tool_args = a.args + ([a.question] if a.question else []) + extra
        req = {"id": rid, "tool": a.tool, "args": dict(_parse_arg(x) for x in tool_args)}
    elif a.question:
        req = {"id": rid, "question": a.question}
    else:
        ap.error("give a question, --tool NAME, or --list-tools")

    rclpy.init()
    node = rclpy.create_node(f"ros_agent_ask_{rid}")
    result = {}
    node.create_subscription(String, a.ns + "/response",
                             lambda m: result.update(json.loads(m.data))
                             if json.loads(m.data).get("id") == rid else None, 10)
    pub = node.create_publisher(String, a.ns + "/request", 10)

    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print(f"ros_agent not found on {a.ns}/request", file=sys.stderr)
        sys.exit(1)

    pub.publish(String(data=json.dumps(req)))
    deadline = time.time() + a.timeout
    while not result and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if not result:
        print("timed out waiting for ros_agent", file=sys.stderr)
        sys.exit(1)
    if "answer" in result and result.get("ok"):
        print(result["answer"])
        for step in result.get("trace", []):
            print(f"  · {step['tool']}({step['args']})", file=sys.stderr)
    elif result.get("ok"):
        print(json.dumps(result["result"], indent=2, default=str))
    else:
        print(f"error: {result.get('answer')}", file=sys.stderr)
        sys.exit(2)
