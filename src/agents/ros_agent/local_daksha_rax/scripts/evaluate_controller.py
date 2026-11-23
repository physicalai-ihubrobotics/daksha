"""Formal evaluation (RAX code mode): freeze code/<project>/ and run its main(ctx) unattended on
held-out layouts (seeds 0..N-1, never used in exploration), each in a fresh bubblewrap sandbox.

    MUJOCO_GL=egl PYTHONPATH=. /home/ihub/daksha_venv/bin/python scripts/evaluate_controller.py --name s1 --project fruit_bowl

The controller sees only its frozen project, the runtime API, its own new frames and a writable
output/; it has no network, and the repository and your home directory are hidden. The robot
service runs in formal mode: status and the daksha_* tools only. Results are written to
runtime/sessions/<name>/formal/results.json.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

PROJECT = pathlib.Path(__file__).resolve().parents[1]
DAKSHA_PYTHON = pathlib.Path(os.environ.get('DAKSHA_PYTHON', '/home/ihub/daksha_venv/bin/python'))
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MUJOCO_GL', 'egl')
from services.controller.config import Configuration  # noqa: E402
from services.controller.frontend import serve  # noqa: E402

WALL_LIMIT_S = 900


def freeze(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    digest = hashlib.sha256()
    for p in sorted(dst.rglob('*')):
        if p.is_file():
            digest.update(str(p.relative_to(dst)).encode() + b'\0' + p.read_bytes())
    return digest.hexdigest()[:16]


def python_base():
    out = subprocess.run([str(DAKSHA_PYTHON), '-c', 'import sys; print(sys.base_prefix)'],
                         capture_output=True, text=True, check=True)
    return pathlib.Path(out.stdout.strip())


def sandbox_command(ws, project, socket, level):
    venv = DAKSHA_PYTHON.parent.parent
    run = [str(DAKSHA_PYTHON), '-m', 'api.runtime', 'run', f'code/{project}/controller.py']
    if level == 'none':
        return run
    base = python_base()
    unshare = ['--unshare-all'] if level == 'full' else ['--unshare-user', '--unshare-pid', '--unshare-ipc']
    return ['bwrap', '--ro-bind', '/', '/', '--tmpfs', '/home', '--dir', '/home/ihub',
            '--ro-bind', str(venv), str(venv), '--tmpfs', '/tmp', '--tmpfs', '/run',
            '--dev', '/dev', '--proc', '/proc',
            '--ro-bind', str(base), str(base),
            '--ro-bind', str(ws), str(ws), '--bind', str(ws/'output'), str(ws/'output'),
            '--ro-bind', str(ws/'frames'), str(ws/'frames'),
            '--bind', str(socket), str(socket),
            *unshare, '--die-with-parent', '--new-session', '--chdir', str(ws),
            '--setenv', 'RAX_MCP_SOCKET', str(socket), '--setenv', 'PYTHONPATH', str(ws),
            '--setenv', 'HOME', str(ws/'output'), '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', *run]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='session name (runtime/sessions/<name>)')
    ap.add_argument('--project', required=True, help='code/<project>/ in the agent workspace')
    ap.add_argument('--episodes', type=int, help='held-out layouts (default: the session configuration)')
    ap.add_argument('--first-seed', type=int, default=0)
    args = ap.parse_args()
    session = PROJECT/'runtime'/'sessions'/args.name
    config = Configuration.load(session/'configuration.json')
    src = config.workspace/'code'/args.project
    if not (src/'controller.py').exists():
        raise SystemExit(f'{src}/controller.py not found')
    formal_root = session/'formal'
    formal = Configuration(**{**config.to_dict(), 'root': formal_root})
    ws = formal.workspace
    if ws.exists():
        shutil.rmtree(ws)
    for d in ('code', 'output', 'frames'):
        (ws/d).mkdir(parents=True)
    shutil.copytree(PROJECT/'auto_research_agent'/'api', ws/'api', ignore=shutil.ignore_patterns('__pycache__'))
    bundle = freeze(src, ws/'code'/args.project)
    socket = formal_root/'formal.sock'
    holder, stop = {}, threading.Event()
    thread = threading.Thread(target=serve, args=(formal, socket), daemon=True,
                              kwargs=dict(mode='formal', ready=lambda f: holder.setdefault('f', f), stop=stop,
                                          state_name='formal-state.json', audit_name='formal-mcp.jsonl'))
    thread.start()
    while 'f' not in holder:
        if not thread.is_alive():
            raise SystemExit('Formal robot service failed to start')
        time.sleep(0.1)
    frontend = holder['f']
    from harness.sandbox import FIX, bwrap_level
    level = bwrap_level()
    if level != 'full':
        print(f'WARNING: sandbox level "{level}": the controller runs '
              + ('without network isolation' if level == 'mount' else 'as a plain subprocess (no OS isolation)')
              + '.\n' + FIX)
    env = {'PATH': '/usr/bin:/bin', 'RAX_MCP_SOCKET': str(socket), 'PYTHONPATH': str(ws), 'HOME': str(ws/'output'),
           'PYTHONDONTWRITEBYTECODE': '1', 'LANG': 'C.UTF-8'}
    n = args.episodes or config.formal_episodes
    results = []
    print(f'Evaluating code/{args.project} (bundle {bundle}) on {n} held-out layouts')
    for seed in range(args.first_seed, args.first_seed + n):
        with frontend._dispatch:
            frontend.backend.reset_episode(seed)
        for d in ('output', 'frames'):
            shutil.rmtree(ws/d)
            (ws/d).mkdir()
        log = formal_root/f'run_{seed:03d}.log'
        t0 = time.time()
        with log.open('w') as out:
            try:
                proc = subprocess.run(sandbox_command(ws, args.project, socket, level), stdout=out,
                                      stderr=subprocess.STDOUT, timeout=WALL_LIMIT_S, cwd=ws, env=env)
                code, timeout = proc.returncode, False
            except subprocess.TimeoutExpired:
                code, timeout = None, True
        with frontend._dispatch:
            frontend.backend._settle(['left', 'right'], 1.0)
            state = frontend.backend.task_state()
            frontend.backend.end_episode()
        result = {'seed': seed, 'task_complete': state['task_complete'], 'inside': state['inside'],
                  'exit_code': code, 'timeout': timeout, 'wall_s': round(time.time() - t0, 1),
                  'sim_s': round(float(frontend.backend.d.time), 1), 'log': str(log.relative_to(session))}
        results.append(result)
        print(f"  seed {seed:3d}: {'SUCCESS' if result['task_complete'] else 'fail   '} inside={state['inside']} "
              f"exit={code} {'TIMEOUT ' if timeout else ''}{result['wall_s']}s wall")
    stop.set()
    thread.join(10)
    rate = sum(r['task_complete'] for r in results)/len(results)
    item_count = len(results[0]['inside']) if results else 0
    items_mean = (sum(sum(r['inside'].values()) for r in results)/len(results)) if results else 0.0
    summary = {'project': args.project, 'bundle': bundle, 'sandbox': level, 'episodes': len(results),
               'success_rate': rate, 'items_in_correct_bowls_mean': items_mean, 'item_count': item_count,
               'results': results}
    (formal_root/'results.json').write_text(json.dumps(summary, indent=1))
    print(f'Success rate {rate:.0%} ({sum(r["task_complete"] for r in results)}/{len(results)}); '
          f'mean items in correct bowls {items_mean:.2f}/{item_count}; results in {formal_root/"results.json"}')


if __name__ == '__main__':
    main()
