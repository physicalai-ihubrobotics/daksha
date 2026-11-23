"""RAX as the robot task agent of ros_agent's single pipeline.

    operator prompt (web chat / /pipeline/command)
        -> RaxAgent.run(): Claude Code (`claude -p`) on the LOCAL Ollama model picked in the UI
           (Anthropic-compatible /v1/messages; no cloud model)
        -> RAX robot tools over MCP, served IN THIS PROCESS by the RAX frontend + DakshaRosBackend
           on ros_agent's own PipelineNode (same camera, calibration, table, STOP)
        -> aruco_reach actions -> /joint_cmd
        <- the agent ends with daksha_report(complete, summary) -> the task log

Tools the agent gets (MCP server "robot"): RAX's daksha_* robot tools (status, observe,
pixel_to_position, move_eef, move_joints, gripper, home, measure_table) plus, from here:
  daksha_find      VLM bounding box + 3-D position of a named object (boxed on the camera
                   and in the IK view, goal TF object_<name> / pipeline_goal)
  daksha_track     live bounding-box tracking (VLM finds, OpenCV CSRT follows every frame)
  daksha_pick_up   the pipeline's pick skill (pre-grasp, tilted approach, gripper check)
  daksha_put_down  put what a hand holds on the table (region / near an object / xy) or in a container
  daksha_ros_info  read-only ROS tools of the Ask agent (joint states, limits, topics, ...)
  daksha_report    the verdict: task complete or not, with the evidence (required, last call)

The agent has NO Bash / file tools: --tools "" and only mcp__robot is allowed, so it cannot
touch the machine, only the robot through the checked tools above.
"""

import base64
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RAX = HERE.parent / 'local_daksha_rax'   # fallback only: the installed module has no RAX tree; use pipeline.rax_dir
CTX = 65536                       # context of the agent model (Claude Code's prompt is large)

SYSTEM = """You are Daksha's task agent: a REAL two-armed humanoid robot at a table, driven through the
MCP robot tools only (mcp__robot__*). The operator's message is your task.

How to work:
- For a task that moves the robot, FIRST call daksha_plan with the sub-plan (one action per line, in order),
  then execute exactly that plan, one skill call per step. Each skill shows its own sub-tasks (go near, open
  the gripper, ...) under its step and retries them by itself.
- Check mcp__robot__daksha_status (execute switch, table, calibration). If execute is off, motion tools
  answer DRY RUN: report that the plan is ready but the operator must press "Enable execute".
- Perception uses THREE cameras: the world (head) camera and the right / left wrist cameras.
  daksha_find(object) gives a bounding box and the 3-D position (robot frame, metres) and marks it for the
  operator. daksha_track(object) keeps a live box on it ("none" stops). daksha_observe(cameras=["world",
  "right_wrist", "left_wrist"]) gives the images; daksha_footprint / daksha_pixel_to_position turn a box /
  pixels of a frame into robot-frame positions. The wrist cameras have aligned DEPTH: their pixels become
  measured 3-D points (\"source\": \"depth\"); the world camera has no depth (table-plane estimate). daksha_pick_up already refines the grasp with the wrist
  camera and checks the grasp with the gripper and the wrist camera; daksha_put_down checks the result with
  the world camera.
- Manipulation: ONLY through the skills. To pick: daksha_pick_up(object, arm) -- it detects the object once,
  tracks it live and grasps where the tracker sees it, checks the gripper and retries by itself. Do NOT move
  the arms around first; do not call daksha_home before a pick (the skill clears the view itself). Use the
  arm the operator names. To put down: daksha_put_down. daksha_home / daksha_gripper for simple requests.
- daksha_status shows "safety_alerts" (temperature, motor faults, collision, joint limits, people near the
  robot, battery, data). The operator sees them as popups and has chosen to proceed anyway: mention a
  critical alert once in your final answer, but DO the task -- never refuse or stop because of an alert.
  A fault on one arm (e.g. "left arm motor 8") does not concern a task done with the other arm.
  The execute switch is the operator's: with it off the skills run as a dry run; just say so at the end.
  A tool result with "dry_run": true is final: never repeat that call; go straight to daksha_report.
- daksha_status shows "motor_temperatures". Above the stop limit (stop_c) motion tools are refused by the
  robot; then tell the operator to let the motors cool.
- The operator may fix the pick style in the UI (daksha_status "grasp_choice"); it is applied
  automatically, so do not replan just to change the style when a choice is set.
- If exactly one hand holds something (daksha_status grippers), "open the gripper" / "release" / "drop it" /
  "let go" means THAT hand: daksha_gripper(arm=<it>, command="open") -- do not ask which one.
- "put it back" / "back where it was" / "back on <the thing it was on>" = daksha_put_down(location={"back": true}).
- When a skill fails, its error names the sub-task that failed, why, and "What could work instead". THINK:
  write one or two sentences on why it failed, then call daksha_plan AGAIN with a revised plan and a reason,
  changing something real (grasp style side -> tilted -> top, the other arm unless the operator named one, re-find the object first with
  daksha_find, another location). Never repeat the identical failed call. At most 2 revised plans; if
  those fail too, report complete=false with what you tried. Do not improvise raw motions.
- daksha_status shows "grippers": a hand with status "holding" HAS an object. "put down / place / release
  it / set it down" = daksha_put_down with that arm. NEVER call daksha_pick_up for a hand that is holding
  something (it would open the gripper and drop it).
- If the operator NAMES a hand ("with the right hand") always use that hand; never switch to the other
  one on your own. If the named hand is holding something, do not pick: report complete=false and ask
  whether to put the held object down first.
- Questions about the robot itself (joints, limits, topics, battery): daksha_ros_info.
- Keep it short: few tool calls, no long explanations. Never invent results: base every claim on tool output.
- A refused or failed tool call is information: adapt (another arm, re-find the object) at most twice.
- VERIFY before you finish: e.g. after a pick, the gripper result (stalled_before_target, width) or
  daksha_find again; after a placement, daksha_find the object at its new place.
- ALWAYS finish with exactly one daksha_report(complete=true|false, summary, evidence). Answers to plain
  questions are reports too (complete=true, the answer in summary)."""


def _tool(name, description, props, required=()):
    return {'name': name, 'description': description,
            'inputSchema': {'type': 'object', 'properties': props, 'required': list(required),
                            'additionalProperties': False}}


class RaxAgent:
    """One task at a time; STOP kills the agent process and interrupts the robot."""

    def __init__(self, pipeline, ros_node, ollama_url='http://localhost:11434', claude=None):
        self.p, self.n = pipeline, ros_node
        self.url = ollama_url.rstrip('/').removesuffix('/v1')
        self.claude = claude or shutil.which('claude') or str(pathlib.Path.home() / '.local/bin/claude')
        # the RAX sources (harness/mcp_bridge.py, services/) live in src/, NOT next to the installed module
        self.rax = pathlib.Path(getattr(pipeline, 'rax_dir', None) or RAX)
        self.root = self.rax / 'runtime' / 'ui_agent'
        self.ws = self.root / 'agent-workspace'
        # AF_UNIX paths are limited to 107 bytes: the workspace path is too deep, so the socket lives in /tmp
        import tempfile
        self.sock = pathlib.Path(tempfile.gettempdir()) / f'daksha_rax_{os.getuid()}.sock'
        self._proc = None
        self._lock = threading.Lock()
        self._report = None
        self._service = None
        self.error = None

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        """The effective Ollama model (same for agent, VLM, tracker and Ask)."""
        return getattr(self.n, 'llm_model', None)

    # ------------------------------------------------------------------ MCP robot service (in process)
    def start(self):
        import sys
        if str(self.rax) not in sys.path:
            sys.path.insert(0, str(self.rax))
        from services.controller.config import Configuration
        from services.controller.frontend import Frontend
        from services.controller.session import Session
        from services.daksha_ros.backend import DakshaRosBackend
        self.root.mkdir(parents=True, exist_ok=True)
        self.ws.mkdir(exist_ok=True)
        (self.ws / 'CLAUDE.md').write_text(SYSTEM + '\n')
        config = Configuration(root=self.root, robot='daksha_ros', task='fruit_to_bowl')
        session = Session(config)
        real = self.p.agent.robot if self.p.agent is not None else None
        hide = () if self.p.cfg.get('rax_raw_motion', False) else ('daksha_move_joints', 'daksha_move_eef',
                                                                   'daksha_move_eef_both')
        if real is not None:
            real.frames_dir = str(self.ws / 'frames')       # frames on disk like the simulator (frames/<id>/...)
        backend = DakshaRosBackend(config, node=self.p, real=real, extra_tools=self._extra_tools(), hide_tools=hide)
        self._wrap_plan(backend)
        self.frontend = Frontend(session, backend, mode='formal')   # robot tools + status, no episodes
        self.sock.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(self.sock))
        os.chmod(self.sock, 0o600)
        listener.listen(16)
        self._stop_service = threading.Event()
        self._service = threading.Thread(target=self.frontend.serve_socket, args=(listener, self._stop_service),
                                         daemon=True, name='rax_mcp')
        self._service.start()
        home = self.root / 'claude'
        home.mkdir(exist_ok=True)
        py = sys.executable
        (home / 'mcp.json').write_text(json.dumps({'mcpServers': {'robot': {
            'type': 'stdio', 'command': py, 'args': [str(self.rax / 'harness' / 'mcp_bridge.py'), str(self.sock)]}}}))
        (home / 'settings.json').write_text(json.dumps({
            'permissions': {'defaultMode': 'dontAsk', 'allow': ['mcp__robot'],
                            'deny': ['Bash', 'Edit', 'Write', 'Read', 'NotebookEdit', 'WebFetch', 'WebSearch']},
            'includeCoAuthoredBy': False,
            'env': {'MCP_TOOL_TIMEOUT': '900000', 'MCP_TIMEOUT': '60000', 'DISABLE_AUTOUPDATER': '1',
                    'ENABLE_TOOL_SEARCH': 'false'}}))
        self.p.log(f'RAX agent ready: robot tools on {self.sock}, model from the UI picker')

    PLAN_TOOLS = {'daksha_pick_up': 'pick up', 'daksha_put_down': 'put down', 'daksha_home': 'go home',
                  'daksha_gripper': 'gripper', 'daksha_find': 'find', 'daksha_track': 'track',
                  'daksha_measure_table': 'measure the table', 'daksha_move_eef': 'move the hand',
                  'daksha_move_joints': 'move the arm', 'daksha_move_eef_both': 'move both hands'}

    def _wrap_plan(self, backend):
        """Every ACTION tool call is one step of the visible plan: it starts the next planned step
        (the skill then lists its sub-tasks under it) and ends it done / failed."""
        orig = backend.call

        def call(name, args):
            a = self.p.agent
            if name not in self.PLAN_TOOLS or a is None:
                return orig(name, args)
            what = ' '.join(str(args.get(k)) for k in ('object', 'command', 'arm') if args.get(k))
            a.plan_step_start(f'{self.PLAN_TOOLS[name]} {what}'.strip())
            from local_pipeline.robot import DryRun
            try:
                out = orig(name, args)
            except DryRun as exc:
                # execute is off: the step stopped at its first motion. Not an error to work around: tell
                # the model it is final, so it does not repeat the identical call
                a.plan_step_end(False, exc)
                if a._task is not None:
                    a._task['dry_run'] = True
                return {'dry_run': True, 'stopped_at': str(exc),
                        'note': 'DRY RUN (execute is off): perception and planning ran, nothing moved. This is '
                                'final: do NOT call this tool again. Finish now with daksha_report(complete=false) '
                                'and tell the operator to enable execute.'}, []
            except Exception as exc:
                a.plan_step_end(False, exc)
                raise
            a.plan_step_end(True)
            return out
        backend.call = call

    def _do_plan(self, args):
        steps = args.get('steps') or []
        if not steps:
            raise ValueError('give the plan as a list of steps')
        self.p.agent.plan_set(steps, reason=args.get('reason'))
        return {'plan': steps, 'note': 'shown to the operator; now execute it step by step'}, []

    def close(self):
        self.stop()
        if self._service is not None:
            self._stop_service.set()
            self._service.join(timeout=2)
            try:
                self.frontend.close()
            except Exception:
                pass
        self.sock.unlink(missing_ok=True)

    # ------------------------------------------------------------------ running a task
    @property
    def busy(self):
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                self._stopped = True             # an operator STOP, not a failure
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def run(self, text):
        """Blocking: one task from start to the report (call from a worker thread)."""
        a = self.p.agent
        model = self.model
        if not model:
            raise RuntimeError('no model selected in the UI')
        self._report = None
        self._stopped = False
        env = {**os.environ,
               'ANTHROPIC_BASE_URL': self.url, 'ANTHROPIC_AUTH_TOKEN': 'ollama', 'ANTHROPIC_API_KEY': '',
               'ANTHROPIC_MODEL': model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
               'ANTHROPIC_DEFAULT_SONNET_MODEL': model, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': model,
               'ANTHROPIC_SMALL_FAST_MODEL': model, 'CLAUDE_CODE_SUBAGENT_MODEL': model,
               'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_TELEMETRY': '1',
               'RAX_MCP_SOCKET': str(self.sock)}
        env.pop('CLAUDECODE', None)
        home = self.root / 'claude'
        cmd = [self.claude, '-p', '--output-format', 'stream-json', '--verbose',
               '--settings', str(home / 'settings.json'), '--setting-sources', 'project',
               '--mcp-config', str(home / 'mcp.json'), '--strict-mcp-config',
               '--permission-mode', 'dontAsk', '--tools', '', '--allowedTools', 'mcp__robot',
               '--model', model, '--append-system-prompt', SYSTEM, '--', text]
        a.log(f'RAX agent ({model}): "{text}"')
        with self._lock:
            self._proc = subprocess.Popen(cmd, cwd=self.ws, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1)
        transcript = (self.root / 'transcript.jsonl').open('a')
        final = ''
        try:
            for line in self._proc.stdout:
                transcript.write(line)
                try:
                    ev = json.loads(line)
                except ValueError:
                    if line.strip():
                        a.log('agent: ' + line.strip()[:300])
                    continue
                if ev.get('type') == 'system' and ev.get('subtype') == 'init':
                    bad = [s.get('name') for s in ev.get('mcp_servers') or [] if s.get('status') != 'connected']
                    if bad:
                        self.stop()
                        raise RuntimeError(f'the robot tools (MCP {bad}) did not connect, so the agent has no tools; '
                                           f'see {self.root / "transcript.jsonl"}')
                    a.log(f'agent connected: {len(ev.get("tools") or [])} tools, model {ev.get("model")}')
                if ev.get('type') == 'assistant':
                    for b in ev.get('message', {}).get('content', []):
                        if b.get('type') == 'text' and b.get('text', '').strip():
                            a._say('daksha', b['text'].strip())
                        elif b.get('type') == 'tool_use':
                            name = b.get('name', '').removeprefix('mcp__robot__')
                            args = json.dumps(b.get('input') or {})[:160]
                            a._say('step', f'▸ {name} {args}')
                            a.log(f'agent tool: {name} {args}')
                elif ev.get('type') == 'user':
                    for b in ev.get('message', {}).get('content', []):
                        if b.get('type') == 'tool_result' and b.get('is_error') and not self._stopped:
                            txt = b.get('content')
                            txt = txt if isinstance(txt, str) else json.dumps(txt)[:400]
                            a._say('step', f'✗ {txt[:300]}')
                elif ev.get('type') == 'result':
                    final = ev.get('result') or ''
                    if ev.get('is_error') or ev.get('subtype') not in (None, 'success'):
                        a.log(f'agent result: {ev.get("subtype")} {final[:300]}')
            self._proc.wait()
        finally:
            transcript.close()
            rc = self._proc.returncode
            with self._lock:
                self._proc = None
            # A robot-tool call may still be running in the in-process MCP server (e.g. a pick skill)
            # after the agent process ended or was killed by STOP. Wait for it HERE, while the robot
            # interrupt is still set (the worker clears it only after this returns), so the skill
            # aborts at its next check instead of carrying on after STOP.
            if getattr(self, 'frontend', None) is not None:
                if not self.frontend._dispatch.acquire(timeout=120):
                    a.log('a robot-tool call is still running after 120 s')
                else:
                    self.frontend._dispatch.release()
        if self._stopped:
            raise InterruptedError('stopped by the operator')
        if self._report is None:
            if rc not in (0, None) and rc < 0:
                raise InterruptedError('the agent was stopped')
            raise ValueError('the agent ended without daksha_report: completion not verified'
                             + (f' (last words: {final[:200]})' if final else ''))
        return self._report

    # ------------------------------------------------------------------ tools from this pipeline
    def _skills(self):
        a = self.p.agent
        if a is None or a.skills is None:
            raise ValueError('the pipeline is still starting (VLM loading)')
        return a, a.skills

    def _find(self, args):
        a, sk = self._skills()
        obj = str(args['object']).strip()
        if obj not in a.objects and obj not in a.containers:
            a.objects.append(obj)
            sk.all_labels = a.objects + a.containers
        camera = args.get('camera') or 'world'
        found, wrist_img = sk.perceive_camera([obj], camera)
        e = found.get(obj)
        if not e or not e.get('centre'):
            raise ValueError(f'"{obj}" is not visible to the {camera} camera')
        if camera != 'world':
            import io as _io
            from PIL import Image as _Image
            buf = _io.BytesIO()
            _Image.fromarray(wrist_img).save(buf, format='JPEG', quality=85)
            return {'object': obj, 'camera': camera, 'top_position_m': [round(v, 4) for v in e['centre']],
                    'bbox_px': e.get('bbox'), 'from_depth': e.get('depth', False),
                    'note': 'wrist-camera position: measured with its aligned depth; it relies on the wrist '
                            'hand-eye calibration'}, [{'type': 'image', 'mimeType': 'image/jpeg',
                                                       'data': base64.b64encode(buf.getvalue()).decode()}]
        a._set(focus=obj)
        self.p.box_everywhere(obj)                       # live boxes in all three cameras
        o = next((o for o in self.p.objects_3d() if o['name'] == obj), {})
        out = {'object': obj, 'camera': 'world', 'top_position_m': [round(v, 4) for v in e['centre']],
               'bbox_px': e.get('bbox'), 'from_depth': bool(self.p._cam.get('depth') is not None),
               'size_m_estimate': o.get('size'), 'source': e.get('source'),
               'note': 'boxed on the operator camera view and in the IK view (TF object_<name>, pipeline_goal)'}
        return out, self._image(a)

    def _image(self, a):
        f = pathlib.Path(a.stream_dir) / 'vlm.jpg'
        try:
            return [{'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(f.read_bytes()).decode()}]
        except OSError:
            return []

    def _track(self, args):
        obj = str(args['object']).strip()
        tr = getattr(self.n, 'tracker', None)
        if tr is None:
            raise ValueError('tracking is not available (no LLM / camera)')
        if obj.lower() in ('', 'none', 'stop', 'clear'):
            tr.stop()
            self.p.tracked = None
            return {'tracking': None}, []
        tr.start('world', query=obj)
        s = tr.wait_detected(timeout=60.0)
        if not s.get('box'):
            tr.stop()
            raise ValueError(f'"{obj}" not found to track: {s.get("message") or s.get("status")}')
        self.p.tracked = obj
        self.p.box_everywhere(obj)                       # and in both gripper cameras
        if self.p.agent is not None:
            self.p.agent._set(focus=obj)
        return {'tracking': obj, 'status': s['status'], 'bbox_px': s['box'],
                'note': 'a live box follows it on the camera view and in the IK view'}, []

    def _pick(self, args):
        a, sk = self._skills()
        obj = str(args['object']).strip()
        if obj not in a.objects:
            a.objects.append(obj)
            sk.all_labels = a.objects + a.containers
        arm = args.get('arm') if args.get('arm') in ('left', 'right') else None
        named = self._named_arm(a)
        if named and arm is None:
            arm = named
        elif named and arm != named:
            held = (sk.held or {}).get(named)
            raise ValueError(f'the operator asked for the {named} hand, not the {arm}. '
                             + (f'The {named} hand is holding the {held}: do NOT switch hands; report '
                                f'complete=false and ask the operator whether to put the {held} down first.'
                                if held else f'Call daksha_pick_up again with arm="{named}".'))
        # the operator's choice in the UI (side / top / tilted) wins over the style the model asked for
        choice = getattr(self.p, 'grasp_choice', 'auto')
        style = choice if choice in ('side', 'top', 'tilted') else args.get('style')
        if choice != 'auto' and args.get('style') and args.get('style') != choice:
            a.log(f'pick style {choice} (operator choice) instead of {args.get("style")}')
        held = self._pick_tracked(a, sk, obj, arm, style=style)
        g = sk.last_grasp or {}
        a._task_step('pick_up', obj, 'done', f'holding it in the {held} hand')
        return {'picked': True, 'arm': held, 'gripper_width_m': g.get('width_m'),
                'stalled_before_target': g.get('stalled'), 'evidence': g.get('why')}, []

    @staticmethod
    def _named_arm(a):
        """'left' / 'right' when the operator's task names exactly one hand ("with the right hand",
        "left arm", "right gripper"; not "right hand side"), else None."""
        task = ((a._task or {}).get('task') or '').lower()
        sides = {s for s in ('left', 'right') if re.search(rf'\b{s}[\s-]+(hand|arm|gripper)\b(?![\s-]*side)', task)}
        return sides.pop() if len(sides) == 1 else None

    def _tracked_entry(self, obj):
        """The live tracker's view of obj as a pick target ({'centre': top}), or None."""
        for o in self.p.objects_3d():
            if o.get('tracked') and o['name'] == obj:
                return {'centre': list(o['top']), 'bbox': o.get('bbox'), 'source': 'tracker'}
        return None

    def _pick_tracked(self, a, sk, obj, arm, attempts=1, style=None):
        """Detect ONCE (vision model), then follow the box with the live CSRT tracker: every attempt
        takes the position from the tracker (re-read with the arms out of view), so a box that moved
        is followed without another vision-model call. Falls back to a fresh detection if the
        tracker lost it."""
        tr = getattr(self.n, 'tracker', None)
        if tr is None:
            return sk.pick_up(obj, arm=arm, style=style)
        from local_pipeline.robot import StepFailed
        last = None
        for attempt in range(1, attempts + 1):
            sk.check()
            sk.clear_view_real()                         # arms out of the world camera's view
            st = tr.state()
            if not (st.get('status') == 'tracking' and self.p.tracked == obj):
                a.log(f'tracking {obj}: one vision-model detection, then the live tracker follows it')
                self._track({'object': obj})             # raises if not found
            time.sleep(0.5)                              # a few tracker frames with the arms clear
            e = self._tracked_entry(obj)
            if e is None:
                raise StepFailed(f'the tracker lost the {obj}')
            a.log(f'pick attempt {attempt}: tracked {obj} at ({e["centre"][0]:.3f}, {e["centre"][1]:.3f})')
            try:
                return sk.pick(obj, e, arm=arm, style=style, locate=lambda: self._tracked_entry(obj))
            except StepFailed as exc:
                last = exc
                a.log(f'pick attempt {attempt} failed: {exc}')
                sk.recover()
        raise StepFailed(f'could not pick up the {obj}: {last}')

    def _put(self, args):
        a, sk = self._skills()
        arm = args.get('arm') if args.get('arm') in ('left', 'right') else None
        # "put it back (on ...)": back where it was picked, whatever location the model chose
        task = ((a._task or {}).get('task') or '').lower()
        if ' back' in f' {task}' and not args.get('container'):
            held = sk.holding_arm(arm)
            if held and (sk.grasp_pose.get(held) or {}).get('pick_tcp'):
                if args.get('location') != {'back': True}:
                    a.log(f'"{task}": putting it back where it was picked (not {args.get("location")})')
                args = {**args, 'location': {'back': True}}
        if args.get('container'):
            sk.put_in(args['container'], arm=arm)
            what = f'into the {args["container"]}'
        else:
            loc = args.get('location') or {'region': 'center'}
            sk.place_at(loc, arm=arm)
            what = f'on the table at {json.dumps(loc)}'
        a._task_step('put_down', '', 'done', what)
        return {'placed': True, 'where': what, 'held_now': dict(sk.held)}, []

    def _ros_info(self, args):
        name = args['tool']
        if name not in self._ros_tools():
            raise ValueError(f'unknown tool {name}; use one of {self._ros_tools()}')
        r = self.n.tools.call(name, args.get('args') or {})
        if isinstance(r, dict):
            r = {k: v for k, v in r.items() if k != '_image'}
        return r, []

    def _ros_tools(self):
        return sorted(k for k in self.n.tools.registry if k not in ('box_object',))

    def _do_report(self, args):
        self._report = {'complete': bool(args['complete']), 'summary': str(args.get('summary', '')),
                        'evidence': str(args.get('evidence', ''))}
        return {'recorded': True}, []

    def _extra_tools(self):
        ros_tools = self._ros_tools()
        loc = {'type': 'object', 'description': 'where: {"back": true} = exactly where it was picked from (e.g. back '
               'on the box it stood on); or on the table {"region": "left|right|front|back|center|front-left|..."}, '
               '{"near": "<object>", "side": "left|right|front|behind"}, {"xy": [x, y]}'}
        return {
            'daksha_find': (_tool('daksha_find', 'Find a named object with the vision model in one camera: '
                                  '"world" (default; boxes it for the operator on the camera + IK view, TF '
                                  'object_<name>) or "right_wrist" / "left_wrist" (close range; 3-D from the wrist '
                                  "camera's aligned DEPTH). Returns the bounding box, 3-D top position (robot "
                                  'frame, m), whether depth was used, and the annotated image.',
                                  {'object': {'type': 'string'},
                                   'camera': {'type': 'string', 'enum': ['world', 'right_wrist', 'left_wrist']}},
                                  ['object']), self._find),
            'daksha_track': (_tool('daksha_track', 'Live bounding-box tracking of an object on the world camera '
                                   '(vision model finds it, a CSRT tracker follows it every frame, re-checked every '
                                   'few s). object "none" stops tracking.', {'object': {'type': 'string'}},
                                   ['object']), self._track),
            'daksha_pick_up': (_tool('daksha_pick_up', 'Pick an object up and keep holding it, lifted: finds it, '
                                     'opens, approaches from above (tilted), closes and checks the gripper actually '
                                     'holds something. Fails with the reason (e.g. empty grasp).',
                                     {'object': {'type': 'string'}, 'arm': {'type': 'string', 'enum': ['left', 'right']},
                                      'style': {'type': 'string', 'enum': ['side', 'top', 'tilted'],
                                                'description': 'side = from the side (default), top = from above, '
                                                               'tilted = 30 deg forward'}},
                                     ['object']), self._pick),
            'daksha_put_down': (_tool('daksha_put_down', 'Put what a hand holds down on the table at a location, '
                                      'or into a container.', {'location': loc, 'container': {'type': 'string'},
                                                               'arm': {'type': 'string', 'enum': ['left', 'right']}}),
                                self._put),
            'daksha_ros_info': (_tool('daksha_ros_info', 'Read-only robot information from ROS: ' + ', '.join(ros_tools),
                                      {'tool': {'type': 'string', 'enum': ros_tools},
                                       'args': {'type': 'object', 'description': 'arguments of that tool'}},
                                      ['tool']), self._ros_info),
            'daksha_plan': (_tool('daksha_plan', 'REQUIRED FIRST CALL for a robot task: the sub-plan, one short line '
                                  'per action in order (each = one skill call), e.g. ["pick up the brown box with '
                                  'the right hand", "put it down on the table"]. Shown to the operator, then '
                                  'execute it. After a failure call it AGAIN with the revised plan and `reason` '
                                  '(why the plan changes).',
                                  {'steps': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 12},
                                   'reason': {'type': 'string', 'description': 'for a revised plan: what failed and '
                                                                               'what is different now'}}, ['steps']),
                            self._do_plan),
            'daksha_report': (_tool('daksha_report', 'REQUIRED last call: was the task completed? complete=true '
                                    'only if verified by tool results. summary = one sentence for the operator; '
                                    'evidence = the tool results that prove it.',
                                    {'complete': {'type': 'boolean'}, 'summary': {'type': 'string'},
                                     'evidence': {'type': 'string'}}, ['complete', 'summary']), self._do_report),
        }


def ensure_agent_model(url, base, ctx=CTX):
    """rax-<base>:<ctx>k: the base model with a context big enough for the agent (same weights,
    created once). Everything (Ask, VLM, tracker, agent) uses this one so Ollama keeps ONE copy
    loaded. Returns (model, error): error when base cannot call tools (then base is used as is)."""
    url = url.rstrip('/').removesuffix('/v1')

    def api(path, body=None, timeout=30):
        req = urllib.request.Request(url + path, method='POST' if body is not None else 'GET',
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()
    if base.startswith('rax-'):
        return base, None
    info = json.loads(api('/api/show', {'model': base}))
    if 'tools' not in (info.get('capabilities') or []):
        return base, f'{base} cannot call tools, so the RAX agent cannot use it'
    name = f'rax-{base.replace(":", "-").replace("/", "-")}:{ctx // 1024}k'
    names = [m['name'] for m in json.loads(api('/api/tags'))['models']]
    if name not in names:
        api('/api/create', {'model': name, 'from': base, 'parameters': {'num_ctx': int(ctx)}, 'stream': False},
            timeout=600)
    return name, None
