#!/usr/bin/env python3
"""Web UI for the leader_combined node (port 8105).

Same console as leader_controller_ui, but runs gen2_leader_combined.launch.py
once and switches normal <-> mirror through the node's ~/set_mirror service,
so changing mode never kills the leader launch.
"""

import logging
import os
import signal
import socket
import subprocess
import threading

import rclpy
from flask import Flask, jsonify, render_template_string, request
from std_msgs.msg import String
from std_srvs.srv import SetBool

# Imported first: it puts automation/ on sys.path for daksha_web_access.
from gen2_leader import leader_controller_ui as base
from daksha_web_access import setup_flask_gate
from gen2_leader.leader_combined import MODE_QOS, MODES

PORT = 8105
LEADER_NODE = "/leader_combined"
DIR_KEYS = ("right_direction", "left_direction")


def _patched_html():
    """leader_controller_ui's page with the launch card swapped for a single
    launch + a service-driven Normal/Mirror toggle."""
    html = base.HTML
    replacements = [
        ("<title>Daksha Leader Control</title>",
         "<title>Daksha Leader Combined</title>"),
        ("<h1>Daksha Leader Controller<small>Launch Control Console</small></h1>",
         "<h1>Daksha Leader Combined<small>Normal / Mirror via Service</small></h1>"),
        ('''      <div class="eyebrow">Launch Controls</div>
      <div class="btns">
        <button class="b-primary" onclick="run('/normal')"><span class="dot"></span> Start Normal</button>
        <button class="b-play" onclick="run('/mirror')"><span class="dot"></span> Start Mirror</button>
        <button onclick="run('/restart')"><span class="dot"></span> Restart</button>
        <button class="b-rec" onclick="run('/kill')"><span class="dot"></span> Kill Launch</button>
      </div>
      <div class="readout">
        <div><div class="k">Status</div><div class="v" id="statusVal">—</div></div>
        <div><div class="k">Mode</div><div class="v" id="modeVal">None</div></div>
        <div><div class="k">Port</div><div class="v">8102</div></div>
      </div>''',
         '''      <div class="eyebrow">Launch Controls</div>
      <div class="btns">
        <button class="b-primary" onclick="run('/start')"><span class="dot"></span> Start Leader</button>
        <button onclick="run('/restart')"><span class="dot"></span> Restart</button>
        <button class="b-rec" onclick="run('/kill')"><span class="dot"></span> Kill Launch</button>
      </div>
      <div class="eyebrow" style="margin-top:18px">Teleop Mode</div>
      <div class="toggle">
        <button id="btnLeaderNormal" onclick="setLeaderMode('normal')">Normal</button>
        <button id="btnLeaderMirror" onclick="setLeaderMode('mirror')">Mirror</button>
      </div>
      <div class="readout">
        <div><div class="k">Status</div><div class="v" id="statusVal">—</div></div>
        <div><div class="k">Mode</div><div class="v" id="modeVal">None</div></div>
        <div><div class="k">Port</div><div class="v">8105</div></div>
      </div>'''),
        # Changes apply live to the one running node; no relaunch needed.
        ('''      dirConfig[data.mode] = data.config;
      switchDirTab(data.mode);
''', '''      dirConfig.normal = data.config.normal;
      dirConfig.mirror = data.config.mirror;
'''),
        ('''        modeVal.textContent = data.mode == null ? "None" : data.mode;
''', '''        modeVal.textContent = data.mode == null ? "None" : data.mode;
        document.getElementById("btnLeaderNormal").className = data.mode === "normal" ? "active" : "";
        document.getElementById("btnLeaderMirror").className = data.mode === "mirror" ? "active" : "";
'''),
        ('''function setRobotMode(mode)''', '''function setLeaderMode(mode)
{
    fetch("/leader_mode/" + mode)
    .then(r=>r.json())
    .then(data=>{
        if(!data.ok) ihubToast("Mode switch failed: " + (data.error || "unknown"), true);
        updateStatus();
    });
}

function setRobotMode(mode)'''),
    ]
    for old, new in replacements:
        if html.count(old) != 1:
            raise RuntimeError(
                f"leader_combined_ui: base HTML changed, cannot patch: {old[:60]!r}")
        html = html.replace(old, new)
    return html


HTML = _patched_html()

app = Flask(__name__)
setup_flask_gate(app, "leader_combined")

node_instance = None


class LeaderCombinedUINode(base.Gen2LeaderNode):
    """Gen2LeaderNode (robot mode, home, speed limits, param helpers) plus a
    single combined launch and the leader_combined mode service/topic."""

    def __init__(self):
        super().__init__(node_name='gen2_leader_combined_ui')
        self._leader_mode = None
        self.create_subscription(
            String, f"{LEADER_NODE}/mode", self._leader_mode_cb, MODE_QOS)
        self._set_mirror_client = self.create_client(
            SetBool, f"{LEADER_NODE}/set_mirror")

    def _leader_mode_cb(self, msg):
        self._leader_mode = msg.data

    def get_leader_mode(self):
        return self._leader_mode if self.is_running() else None

    def set_leader_mode(self, mode):
        if mode not in MODES:
            return False, f"unknown mode '{mode}'"
        try:
            response = self._call_param_service(
                self._set_mirror_client,
                SetBool.Request(data=(mode == "mirror")),
                timeout_s=3.0)
        except Exception as e:
            return False, str(e)
        if not response.success:
            return False, response.message
        self._leader_mode = response.message
        return True, response.message

    def start_launch(self, mode="combined"):
        self.get_logger().info('Starting combined leader launch...')
        self.kill_process()
        cmd = f"""
        source /opt/ros/jazzy/setup.bash
        source {base._WORKSPACE_INSTALL}/setup.bash
        ros2 launch gen2_leader gen2_leader_combined.launch.py
        """
        try:
            self.process = subprocess.Popen(
                ["bash", "-c", cmd], preexec_fn=os.setsid)
            self.last_mode = "combined"
            self.get_logger().info(
                f'Combined launch started (PID: {self.process.pid})')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to start combined launch: {e}')
            return False

    def kill_process(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None
        self.last_mode = None
        self._leader_mode = None


# ============================================================================
# Flask Routes
# ============================================================================

@app.route("/")
def index():
    return render_template_string(
        HTML, mirror_right=base.MIRROR_DEFAULT_RIGHT, mirror_left=base.MIRROR_DEFAULT_LEFT)


@app.route("/start")
def start():
    if node_instance:
        node_instance.start_launch()
    return jsonify(ok=True)


@app.route("/restart")
def restart():
    if node_instance:
        node_instance.start_launch()
    return jsonify(ok=True)


@app.route("/kill")
def kill():
    if node_instance:
        node_instance.kill_process()
    return jsonify(ok=True)


@app.route("/leader_mode/<mode>")
def leader_mode(mode):
    if not node_instance or not node_instance.is_running():
        return jsonify(ok=False, error="Leader launch not running")
    ok, msg = node_instance.set_leader_mode(mode)
    return jsonify(ok=ok, mode=msg if ok else None, error=None if ok else msg)


@app.route("/status")
def status():
    if not node_instance:
        return jsonify(running=False, mode=None, robot_mode="unknown")
    return jsonify(
        running=node_instance.is_running(),
        mode=node_instance.get_leader_mode(),
        robot_mode=node_instance.get_robot_mode(),
    )


@app.route("/set_joint_direction", methods=["POST"])
def set_joint_direction_live():
    """Push both modes' direction settings to the running leader_combined."""
    if not node_instance or not node_instance.is_running():
        return jsonify(ok=False, error="Leader launch not running")

    config = (request.get_json(force=True) or {}).get("config", {})
    errors = []
    for mode in MODES:
        section = config.get(mode, {})
        for key in DIR_KEYS:
            arr = section.get(key)
            if arr and len(arr) == 7:
                if not node_instance.set_integer_array_param(
                        LEADER_NODE, f"{mode}.{key}", arr):
                    errors.append(f"Failed to set {mode}.{key}")
        for key in base.GRIPPER_KEYS:
            val = section.get(key)
            if val in (1, -1):
                if not node_instance.set_integer_param(
                        LEADER_NODE, f"{mode}.{key}", val):
                    errors.append(f"Failed to set {mode}.{key}")

    if errors:
        return jsonify(ok=False, error="; ".join(errors))
    return jsonify(ok=True, mode="leader_combined")


@app.route("/joint_direction_live")
def get_joint_direction_live():
    """Read both modes' direction settings from the running leader_combined."""
    if not node_instance or not node_instance.is_running():
        return jsonify(ok=False, error="Leader launch not running")

    config = {}
    for mode in MODES:
        config[mode] = {}
        for key in DIR_KEYS:
            arr = node_instance.get_integer_array_param(LEADER_NODE, f"{mode}.{key}")
            if arr is None:
                return jsonify(ok=False, error=f"Failed to read {mode}.{key}")
            config[mode][key] = arr
        for key in base.GRIPPER_KEYS:
            val = node_instance.get_integer_param(LEADER_NODE, f"{mode}.{key}")
            if val is None:
                return jsonify(ok=False, error=f"Failed to read {mode}.{key}")
            config[mode][key] = val
    return jsonify(ok=True, mode="leader_combined", config=config)


# Unchanged behaviour from leader_controller_ui: robot mode, home, speed
# limits, leader portal check and joint_direction.yaml load/save.
for rule, endpoint, view, methods in [
    ("/home", "go_home", base.go_home, ["GET"]),
    ("/mode/<mode>", "set_mode", base.set_mode, ["GET"]),
    ("/velocity_status", "velocity_status", base.velocity_status, ["GET"]),
    ("/velocity_set/<value>", "velocity_set", base.velocity_set, ["GET"]),
    ("/acceleration_status", "acceleration_status", base.acceleration_status, ["GET"]),
    ("/acceleration_set/<value>", "acceleration_set", base.acceleration_set, ["GET"]),
    ("/check_leader", "check_leader", base.check_leader, ["GET"]),
    ("/joint_direction_config", "get_joint_direction_config",
     base.get_joint_direction_config, ["GET"]),
    ("/joint_direction_config", "save_joint_direction_config",
     base.save_joint_direction_config, ["POST"]),
]:
    app.add_url_rule(rule, endpoint, view, methods=methods)


# ============================================================================
# Main
# ============================================================================

def run_flask():
    print(f"Web server starting on http://0.0.0.0:{PORT}")
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)


def main():
    global node_instance

    rclpy.init()

    node = LeaderCombinedUINode()
    node_instance = node
    # The reused base routes (home, speed limits, robot mode) read the
    # node through base.node_instance.
    base.node_instance = node

    print("=" * 60)
    print(" Daksha Leader Combined Web Controller (ROS2 Node)")
    print("=" * 60)
    print(f"  http://{socket.gethostname().lower()}.local:{PORT}")
    print(f"  or http://<ROBOT_IP>:{PORT}")
    print("=" * 60)

    threading.Thread(target=run_flask, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.kill_process()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
