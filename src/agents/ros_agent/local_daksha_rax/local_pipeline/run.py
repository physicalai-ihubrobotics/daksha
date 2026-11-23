"""Run the local interactive pipeline: Daksha fruit scene + local VLM (Ollama) + browser UI + terminal.

    cd /home/ihub/robot/src/agents/ros_agent/local_daksha_rax
    MUJOCO_GL=egl PYTHONPATH=$PWD /home/ihub/daksha_venv/bin/python -m local_pipeline.run            # open http://localhost:8092

Type instructions or questions at the `you>` prompt (or in the browser). Ctrl-C interrupts
the running command; Ctrl-C again while idle (or `quit`) exits.
"""
import argparse
import pathlib
import signal
import sys
import threading
import time

from .agent import LocalAgent
from .ollama_vlm import OllamaVLM
from .web import serve

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='local', help='run folder under runtime/local_pipeline/')
    ap.add_argument('--task', default='fruit_to_bowl', choices=['fruit_to_bowl', 'fruit_toy_sorting'])
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--model', default='35b',
                    help='35b = qwen3.5:35b (default), 9b = qwen3.5:9b, 8b = qwen3-vl:8b, or any Ollama model name')
    ap.add_argument('--ollama', default='http://localhost:11434')
    ap.add_argument('--ik', choices=['sim', 'human'], default='sim',
                    help='human = human-posture IK (local_pipeline/human_ik.py, placo + kinematics/ package)')
    ap.add_argument('--realtime', type=float, default=1.0, help='sim speed vs wall clock (0 = unpaced)')
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to reach the UI from another machine')
    ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--no-repl', action='store_true', help='browser only (for running in the background)')
    ap.add_argument('--do', action='append', default=[], help='command(s) to run at start, e.g. --do look')
    args = ap.parse_args()

    vlm = OllamaVLM(host=args.ollama, model=args.model)
    ok, names = vlm.available()
    args.model = vlm.model
    if not ok:
        sys.exit(f'Ollama at {args.ollama} does not serve {args.model} (has: {names or "nothing / not running"}). '
                 f'Start it with `ollama serve` and `ollama pull {args.model}`.')
    print(f'Loading {args.model} on Ollama…', flush=True)
    vlm.warm_up()

    root = PROJECT/'runtime'/'local_pipeline'/args.name
    agent = LocalAgent(root, task=args.task, seed=args.seed, realtime=args.realtime, vlm=vlm, ik=args.ik).start()
    server = serve(agent, args.host, args.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'UI: http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}   '
          f'(Esc or the STOP button interrupts)', flush=True)
    for c in args.do:
        agent.submit(c)

    def on_sigint(*_):
        if agent.state['busy'] or agent.q.qsize():
            print('\n[interrupt]', flush=True)
            agent.stop()
        else:
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_sigint)

    try:
        if args.no_repl or not sys.stdin.isatty():
            while True:
                time.sleep(3600)
        print('Commands: look · put the apple in the bowl · put all the fruit in the bowl · what do you see? · '
              'stop · home · reset [seed] · quit', flush=True)
        while True:
            try:
                line = input('you> ')
            except EOFError:
                break
            if line.strip() in ('quit', 'exit'):
                break
            agent.submit(line)
    except KeyboardInterrupt:
        pass
    print('shutting down', flush=True)
    agent.close()
    server.shutdown()


if __name__ == '__main__':
    main()
