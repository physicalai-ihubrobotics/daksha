"""Prepare and launch a Daksha RAX session with Claude Code (adapted from RAX real/rainbow
harness/claude_cli/launcher.py, Apache-2.0).

    python -m harness.launcher prepare --name s1 --episodes 24
    python -m harness.launcher launch  --name s1            # interactive Claude Code TUI
    python -m harness.launcher launch  --name s1 --non-interactive
    python -m harness.launcher service --name s1            # only the robot service (manual use)

    # the agent on a LOCAL Ollama model instead of Claude (Claude Code talks to Ollama's
    # Anthropic-compatible /v1/messages endpoint; harness, sandbox, MCP tools unchanged):
    python -m harness.launcher launch --name s1 --llm ollama                       # qwen3.5:35b, 64k context
    python -m harness.launcher launch --name s1 --llm ollama --ollama-model 9b --ollama-ctx 32768

Session layout (runtime/sessions/<name>/):
    configuration.json, state.json, mcp.jsonl, frontend.log   operator files (hidden from the agent)
    robot.sock                                                the robot service socket
    stream/latest.jpg                                         operator viewer frames
    claude/settings.json, claude/mcp.json                     Claude Code settings and MCP config
    agent-workspace/                                          the agent's cwd and only writable place

The agent runs with your normal Claude Code login, ``--permission-mode dontAsk`` and an explicit
allow list; its Bash runs in Claude Code's bubblewrap sandbox: writes only in the workspace, no
network, and the simulator source, session ledger and your credentials are unreadable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.controller.config import Configuration  # noqa: E402

SESSIONS = PROJECT/'runtime'/'sessions'
MCP_SERVER = 'robot'
DEFAULT_CLAUDE = ('/home/ihub/.vscode-server/extensions/anthropic.claude-code-2.1.322/resources/'
                  'native-binary/claude')
DENIED_TOOLS = ('WebSearch', 'WebFetch', 'Workflow', 'CronCreate', 'CronDelete', 'CronList', 'ScheduleWakeup',
                'RemoteTrigger', 'SendMessage', 'ListAgents', 'EnterWorktree', 'ExitWorktree', 'PushNotification',
                'Artifact', 'ArtifactComments', 'ArtifactData', 'SendToProject', 'SendToProjectContext')
SUBAGENT_TOOLS = ('Agent', 'Task')
HOME = Path.home()
# Unreadable to the agent: the harness and simulator source, the real robot's code, credentials.
PROTECTED = [PROJECT/'services', PROJECT/'harness', PROJECT/'scripts', PROJECT/'local_tests', PROJECT/'docs',
             PROJECT/'assets', PROJECT/'scratch', PROJECT/'agent_skills', PROJECT/'auto_research_agent', PROJECT/'examples',
             HOME/'uni', HOME/'.ssh', HOME/'.aws', HOME/'.config/gcloud', HOME/'.netrc', HOME/'.git-credentials',
             HOME/'.claude', HOME/'.claude.json', HOME/'Desktop/rax/rax']
PRIVATE_FILES = ('configuration.json', 'state.json', 'mcp.jsonl', 'frontend.log', 'state.json.lock', 'formal')
UID = os.getuid()
HIDDEN_SOCKETS = (Path('/run/docker.sock'), Path('/var/run/docker.sock'), Path('/run/containerd'),
                  Path('/run/dbus'), Path(f'/run/user/{UID}'), Path('/tmp/.X11-unix'))


def claude_binary(path=None):
    for candidate in (path, os.environ.get('CLAUDE_BINARY'), shutil.which('claude'), DEFAULT_CLAUDE):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError('Claude Code not found: pass --claude /path/to/claude or set CLAUDE_BINARY')


def agent_python():
    """Python interpreter shared with the Daksha ROS workspace."""
    interpreter = Path(os.environ.get('DAKSHA_PYTHON', '/home/ihub/daksha_venv/bin/python'))
    if not interpreter.is_file():
        raise FileNotFoundError(f'Daksha Python not found: {interpreter} (set DAKSHA_PYTHON)')
    return interpreter


def session_root(name):
    return SESSIONS/name


def generate_contract(config):
    """MCP_SESSION.md: the live tool list with schemas, written from the frontend's contract."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from services.backends import make_backend
    from services.controller.frontend import Frontend

    class _NoSession:
        def __init__(self, cfg):
            self.config = cfg
            self.state = {'current': None, 'episodes_started': 0, 'history': []}
    backend = make_backend(config)
    try:
        tools = Frontend(_NoSession(config), backend).tools()
    finally:
        backend.close()
    lines = ['# MCP session contract', '',
             f'Task `{config.task}`, robot `{config.robot}`, observation profile `{config.observation_profile}`, '
             f'exploration budget {config.episodes} episodes. Call these as MCP tools (server `robot`) or from '
             'Python with `ctx.call(name, **arguments)`.', '']
    for t in tools:
        lines += [f'## `{t["name"]}`', '', t['description'], '']
        props = t['inputSchema'].get('properties', {})
        if props:
            req = set(t['inputSchema'].get('required', []))
            lines += ['| argument | type | required | notes |', '|---|---|---|---|']
            for k, v in props.items():
                typ = v.get('type', '')
                if 'enum' in v:
                    typ += ' ' + '/'.join(map(str, v['enum']))
                note = v.get('description', '')
                rng = [f'{b} {v[b]}' for b in ('minimum', 'maximum', 'minItems', 'maxItems') if b in v]
                lines.append(f'| `{k}` | {typ} | {"yes" if k in req else ""} | {note} {"(" + ", ".join(rng) + ")" if rng else ""} |')
            lines.append('')
    return '\n'.join(lines) + '\n'


def populate_workspace(config):
    from services.daksha_sim import sorting, tasks, tray_transport
    task_text = {'fruit_toy_sorting': sorting.TASK_TEXT,
                 'tray_transport': tray_transport.TASK_TEXT}.get(config.task, tasks.TASK_TEXT)
    ws = config.workspace
    for name in ('code', 'memory', 'output', 'frames', '.pylib', 'experiments'):
        (ws/name).mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT/'auto_research_agent'/'AGENTS.md', ws/'AGENTS.md')
    (ws/'CLAUDE.md').write_text('@AGENTS.md\n')
    (ws/'TASK.md').write_text(task_text)
    shutil.copytree(PROJECT/'auto_research_agent'/'api', ws/'api', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PROJECT/'agent_skills', ws/'.claude'/'skills')
    model = ws/'robot-model'
    shutil.copytree(PROJECT/'assets'/'daksha', model, ignore=shutil.ignore_patterns('_tmp.urdf'))
    for p in model.rglob('*'):
        p.chmod(0o444 if p.is_file() else 0o555)
    model.chmod(0o555)
    for name, text in (('PROGRESS.md', '# Progress\n\n(nothing yet)\n'), ('TIPS.md', '# Tips\n\n| Task phase | Preconditions | What worked | Visible effect | Evidence |\n|---|---|---|---|---|\n'),
                       ('RUNS.md', '# Controller runs\n\n| episode | code version | outcome | failing step | notes |\n|---|---|---|---|---|\n'),
                       ('INDEX.md', '# Memory index\n\n| Skill | Tags | Confirmation | Last confirmed |\n|---|---|---|---|\n')):
        (ws/'memory'/name).write_text(text)
    (ws/'MCP_SESSION.md').write_text(generate_contract(config))


def settings(config, socket):
    ws = config.workspace
    root = config.root
    allow = [f'Read(/{ws}/**)', f'Edit(/{ws}/**)', 'Bash', 'Glob', 'Grep', 'TodoWrite', f'mcp__{MCP_SERVER}']
    if config.subagents:
        allow.append('Agent')
    protected = [*PROTECTED, *(root/f for f in PRIVATE_FILES), *(p for p in SESSIONS.glob('*') if p != root)]
    deny = list(DENIED_TOOLS) + [f'Edit(/{ws}/robot-model/**)', f'Edit(/{ws}/api/**)', f'Edit(/{ws}/.claude/**)']
    for path in protected:
        for tool in ('Read', 'Edit'):
            deny += [f'{tool}(/{path})', f'{tool}(/{path}/**)']
    timeout_ms = (config.session_seconds + 300)*1000
    from harness.sandbox import bwrap_level
    isolated = bwrap_level() == 'full'
    return {
        'permissions': {'defaultMode': 'dontAsk', 'allow': allow, 'deny': deny},
        'sandbox': {
            'enabled': isolated, 'failIfUnavailable': isolated, 'autoAllowBashIfSandboxed': True,
            'allowUnsandboxedCommands': False,
            'filesystem': {'allowWrite': [str(ws)], 'denyRead': [str(p) for p in (*protected, *HIDDEN_SOCKETS)]},
            'network': {'allowAllUnixSockets': True, 'allowLocalBinding': False, 'allowedDomains': []},
        },
        'includeCoAuthoredBy': False,
        'env': {
            'RAX_MCP_SOCKET': str(socket), 'MUJOCO_GL': 'egl',
            'PYTHONPATH': f'{ws}:{ws}/.pylib', 'PYTHONDONTWRITEBYTECODE': '1',
            'PATH': f'{agent_python().parent}:/usr/local/bin:/usr/bin:/bin',
            'MCP_TOOL_TIMEOUT': str(timeout_ms), 'MCP_TIMEOUT': '60000',
            'BASH_DEFAULT_TIMEOUT_MS': '900000', 'BASH_MAX_TIMEOUT_MS': str(config.session_seconds*1000),
            'DISABLE_AUTOUPDATER': '1', 'ENABLE_TOOL_SEARCH': 'false'}}


def refresh(args):
    """Re-copy the task text, AGENTS.md, skills, api/ and the live tool list into an existing session's
    workspace (after the backend or docs changed). memory/, code/, frames/, output/ and the episode
    budget are kept."""
    root = session_root(args.name)
    if not (root/'configuration.json').exists():
        raise SystemExit(f'No session {args.name!r}')
    config = Configuration.load(root/'configuration.json')
    from services.daksha_sim import sorting, tasks, tray_transport
    ws = config.workspace
    task_text = {'fruit_toy_sorting': sorting.TASK_TEXT,
                 'tray_transport': tray_transport.TASK_TEXT}.get(config.task, tasks.TASK_TEXT)
    (ws/'TASK.md').write_text(task_text)
    shutil.copy(PROJECT/'auto_research_agent'/'AGENTS.md', ws/'AGENTS.md')
    for src, dst in ((PROJECT/'auto_research_agent'/'api', ws/'api'), (PROJECT/'agent_skills', ws/'.claude'/'skills')):
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
    (ws/'MCP_SESSION.md').write_text(generate_contract(config))
    print(f'Refreshed {ws}: TASK.md, AGENTS.md, api/, .claude/skills/, MCP_SESSION.md (memory/ and code/ kept)')


def prepare(args):
    root = session_root(args.name)
    if root.exists():
        raise SystemExit(f'{root} exists: choose another --name (or launch/resume it)')
    backend_options = {}
    if args.backend_options:
        backend_options = json.loads(args.backend_options.read_text())
    config = Configuration(root=root, task=args.task, robot=args.robot, episodes=args.episodes, formal_episodes=args.formal_episodes,
                           backend_options=backend_options,
                           observation_profile=args.profile, agent_model=args.model, agent_effort=args.effort,
                           session_seconds=int(args.hours*3600), subagents=args.subagents)
    root.mkdir(parents=True, mode=0o700)
    (root/'configuration.json').write_text(json.dumps(config.to_dict(), indent=2))
    populate_workspace(config)
    socket = root/'robot.sock'
    home = root/'claude'
    home.mkdir()
    (home/'settings.json').write_text(json.dumps(settings(config, socket), indent=2))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {MCP_SERVER: {
        'type': 'stdio', 'command': str(agent_python()),
        'args': [str(PROJECT/'harness'/'mcp_bridge.py'), str(socket)]}}}, indent=2))
    from harness.sandbox import FIX, bwrap_level
    if bwrap_level() != 'full':
        print('WARNING: bubblewrap is blocked on this host, so the agent\'s Bash runs WITHOUT the OS sandbox. '
              'Permission rules still confine its Read/Edit tools to the workspace, but shell commands could read '
              'other files. Re-run prepare after enabling it to get full isolation.\n' + FIX)
    print(f'Prepared session {args.name}: {root}')
    print(f'  workspace: {config.workspace}')
    print(f'  launch:    python -m harness.launcher launch --name {args.name}')
    return config


def start_service(config, *, started_at):
    socket = config.root/'robot.sock'
    log = (config.root/'frontend.log').open('a')
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'PYTHONPATH': str(PROJECT)}
    process = subprocess.Popen([str(agent_python()), '-m', 'services.controller.frontend',
                                '--config', str(config.root/'configuration.json'), '--socket', str(socket),
                                '--started-at', str(started_at)], cwd=PROJECT, stdout=log, stderr=log, env=env,
                               start_new_session=True)
    deadline = time.time() + 120
    while not socket.exists():
        if process.poll() is not None:
            raise SystemExit(f'Robot service exited ({process.returncode}); see {config.root/"frontend.log"}')
        if time.time() > deadline:
            process.terminate()
            raise SystemExit('Robot service did not start within 120 s')
        time.sleep(0.2)
    return process


def stop(process, grace=10):
    if process and process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(grace)
        except subprocess.TimeoutExpired:
            process.kill()


def has_conversation(config):
    """Whether Claude Code has a saved conversation for this workspace (~/.claude/projects/<key>/)."""
    import re
    key = re.sub(r'[^A-Za-z0-9]', '-', str(config.workspace))
    folder = HOME/'.claude'/'projects'/key
    return folder.is_dir() and any(folder.glob('*.jsonl'))


OLLAMA_ALIASES = {'35b': 'qwen3.5:35b', '9b': 'qwen3.5:9b', '8b': 'qwen3-vl:8b'}


def ollama_agent_model(base, ctx, url):
    """A local Ollama model with a context window big enough for Claude Code + the RAX
    contract and skills. Ollama's server default (OLLAMA_CONTEXT_LENGTH) is often 4-16k,
    which silently truncates the agent's instructions, so a derived model with num_ctx=ctx
    is created once (same weights, no download): rax-<base>-<ctx/1024>k."""
    import urllib.request
    base = OLLAMA_ALIASES.get(base, base)

    def api(path, body=None, timeout=30):
        req = urllib.request.Request(url.rstrip('/') + path, method='POST' if body is not None else 'GET',
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()

    try:
        names = [m['name'] for m in json.loads(api('/api/tags'))['models']]
    except Exception as exc:
        raise SystemExit(f'Ollama not reachable at {url} ({exc}); start it with `ollama serve`') from None
    if base not in names and f'{base}:latest' not in names:
        raise SystemExit(f'Ollama has no model {base} (has: {", ".join(names)}); `ollama pull {base}`')
    info = json.loads(api('/api/show', {'model': base}))
    if 'tools' not in (info.get('capabilities') or []):
        raise SystemExit(f'{base} does not support tool calling, which the RAX agent needs')
    name = f'rax-{base.replace(":", "-").replace("/", "-")}:{ctx // 1024}k'
    if name not in names:
        print(f'Creating Ollama model {name} (from {base}, num_ctx {ctx}; same weights, no download)...', flush=True)
        api('/api/create', {'model': name, 'from': base, 'parameters': {'num_ctx': int(ctx)}, 'stream': False},
            timeout=600)
    return name


def ollama_env(model, url):
    """Point Claude Code at Ollama's Anthropic-compatible API for every model slot."""
    return {'ANTHROPIC_BASE_URL': url.rstrip('/'), 'ANTHROPIC_AUTH_TOKEN': 'ollama', 'ANTHROPIC_API_KEY': '',
            'ANTHROPIC_MODEL': model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': model, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': model,
            'ANTHROPIC_SMALL_FAST_MODEL': model, 'CLAUDE_CODE_SUBAGENT_MODEL': model,
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_TELEMETRY': '1'}


def claude_arguments(config, *, prompt, interactive, resume):
    home = config.root/'claude'
    denied = DENIED_TOOLS + (() if config.subagents else SUBAGENT_TOOLS)
    args = ['--settings', str(home/'settings.json'), '--setting-sources', 'project',
            '--mcp-config', str(home/'mcp.json'), '--strict-mcp-config',
            '--permission-mode', 'dontAsk', '--effort', config.agent_effort, '--disallowedTools', *denied]
    if config.agent_model:
        args += ['--model', config.agent_model]
    if resume:
        args += ['--continue']
    if not interactive:
        args = ['-p', *args, '--output-format', 'stream-json', '--verbose']
    return [*args, '--', prompt]


def launch(args):
    root = session_root(args.name)
    if not (root/'configuration.json').exists():
        raise SystemExit(f'No session {args.name}: run prepare first')
    config = Configuration.load(root/'configuration.json')
    # Rewrite the Claude Code settings from the current code and host (sandbox availability can change).
    (root/'claude'/'settings.json').write_text(json.dumps(settings(config, root/'robot.sock'), indent=2))
    from harness.sandbox import FIX, bwrap_level
    if bwrap_level() != 'full':
        if not args.allow_unsandboxed:
            raise SystemExit('bubblewrap is blocked on this host, so the agent\'s Bash would run without the OS '
                             'sandbox (it could read files outside its workspace, including the simulator source).\n'
                             + FIX + '\nOr pass --allow-unsandboxed to accept permission-rule-only confinement.')
        print('WARNING: running without the OS sandbox for Bash (bubblewrap is blocked); see README "Full isolation".')
    llm_env = {}
    if args.llm == 'ollama':                 # before the robot service starts: a failure here leaks nothing
        import dataclasses
        url = args.ollama_url if '://' in args.ollama_url else 'http://' + args.ollama_url
        model = ollama_agent_model(args.ollama_model, args.ollama_ctx, url)
        llm_env = ollama_env(model, url)
        config = dataclasses.replace(config, agent_model=model)   # also passed as --model
        print(f'Agent LLM: local Ollama {model} at {url} (not Claude)', flush=True)
    started = time.time()
    service = start_service(config, started_at=started)
    print(f'Robot service up ({root/"robot.sock"}). Viewer: python scripts/viewer.py --name {args.name}')
    if args.resume and not has_conversation(config):
        print('No earlier conversation in this workspace: starting fresh instead of resuming.')
        args.resume = False
    prompt_file = 'RESUME_PROMPT.md' if args.resume else 'START_PROMPT.md'
    prompt = args.prompt or (PROJECT/'auto_research_agent'/prompt_file).read_text().strip()
    env = {**os.environ, 'RAX_MCP_SOCKET': str(root/'robot.sock')}
    env.pop('CLAUDECODE', None)
    env.update(llm_env)
    command = [str(claude_binary(args.claude)), *claude_arguments(config, prompt=prompt,
                                                                  interactive=not args.non_interactive,
                                                                  resume=args.resume)]
    try:
        if args.non_interactive:
            with (root/'agent-transcript.jsonl').open('a') as transcript:
                agent = subprocess.Popen(command, cwd=config.workspace, env=env, stdout=subprocess.PIPE, text=True)
                for line in agent.stdout:
                    transcript.write(line)
                    transcript.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('type') == 'assistant':
                        for block in event.get('message', {}).get('content', []):
                            if block.get('type') == 'text':
                                print(block['text'], flush=True)
                            elif block.get('type') == 'tool_use':
                                print(f'  -> {block["name"]} {json.dumps(block.get("input"))[:200]}', flush=True)
                    elif event.get('type') == 'result':
                        print('RESULT:', event.get('subtype'), event.get('result', '')[:2000], flush=True)
                agent.wait()
        else:
            subprocess.run(command, cwd=config.workspace, env=env)
    finally:
        stop(service)
        print(f'Session {args.name} stopped. Evaluate the controller with: '
              f'python scripts/evaluate_controller.py --name {args.name} --project <project>')


def service(args):
    config = Configuration.load(session_root(args.name)/'configuration.json')
    process = start_service(config, started_at=time.time())
    print(f'Robot service running on {config.root/"robot.sock"} (Ctrl-C stops it)')
    try:
        process.wait()
    except KeyboardInterrupt:
        stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--name', required=True)
    p.add_argument('--task', default='fruit_to_bowl', choices=['fruit_to_bowl', 'fruit_toy_sorting', 'tray_transport'])
    p.add_argument('--episodes', type=int, default=24)
    p.add_argument('--formal-episodes', type=int, default=10)
    p.add_argument('--profile', choices=['rgbd', 'rgb-only'], default='rgbd')
    p.add_argument('--backend-options', type=Path,
                   help='JSON file with backend options, such as calibrated camera specs')
    p.add_argument('--model', default=None, help='Claude model alias or id (default: your Claude Code default)')
    p.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--hours', type=float, default=6.0)
    p.add_argument('--subagents', action='store_true')
    p.add_argument('--robot', default='daksha_sim', choices=['daksha_sim', 'daksha_sim_ollama', 'daksha_ros'],
                   help='daksha_sim_ollama adds the vlm_query tool backed by local Ollama qwen3-vl:8b; '
                        'daksha_ros = the REAL robot over ROS 2 (source the workspace, run aruco_reach)')
    p.set_defaults(func=prepare)
    p = sub.add_parser('launch')
    p.add_argument('--name', required=True)
    p.add_argument('--non-interactive', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prompt')
    p.add_argument('--claude')
    p.add_argument('--allow-unsandboxed', action='store_true',
                   help='Launch even when bubblewrap is blocked (permission rules only; see README)')
    p.add_argument('--llm', choices=['claude', 'ollama'], default='claude',
                   help='ollama = run the agent on a local Ollama model (Anthropic-compatible API)')
    p.add_argument('--ollama-model', default='35b',
                   help='35b = qwen3.5:35b (default), 9b, 8b = qwen3-vl:8b, or any tool-capable Ollama model')
    p.add_argument('--ollama-ctx', type=int, default=65536, help='context window for the agent model (tokens)')
    p.add_argument('--ollama-url', default=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
    p.set_defaults(func=launch)
    p = sub.add_parser('refresh', help='re-copy task text, skills and tool list into an existing session')
    p.add_argument('--name', required=True)
    p.set_defaults(func=refresh)
    p = sub.add_parser('service')
    p.add_argument('--name', required=True)
    p.set_defaults(func=service)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
