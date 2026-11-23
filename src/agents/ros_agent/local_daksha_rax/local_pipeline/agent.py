"""The interactive local agent: any operator wording -> local VLM understanding -> robot skills, interruptible.

Every message (typed in the browser or the terminal) goes to the local VLM together with the live head-camera
image, a scene memory (where each object was last seen, what each hand holds, what is running) and the last
few chat turns. The model decides what the message is:
- a question or chat: answered from the image and memory, the robot keeps doing what it does;
- a command, in any phrasing ("grab the red one and drop it in the white dish", "put the banana next to the
  orange", "hand me the apple", "move your left hand up a bit", "then tidy up"): a list of actions.
"stop/halt/wait..." is caught instantly without the model.

Threads
- worker: the only thread that touches the simulator (plans run here, one at a time).
- router threads: one model call per message (serialized), never touching the robot.
- callers (web UI, terminal): `submit(text)` and `stop()` only set events and queue work.

Interrupt rules
- stop stops the arm at the next control tick and drops queued plans; the arm holds where it stopped.
- A new command interrupts the running one first (barge-in), unless the operator says to do it after
  ("then ...", "after that ...", "also ..."): then it is queued.
- Questions never interrupt.
"""
from __future__ import annotations

import collections
import io
import json
import os
import pathlib
import queue
import re
import threading
import copy
import time
import types

import numpy as np
from PIL import Image, ImageDraw

from .ollama_vlm import OllamaVLM, VLMError, parse_json
from .robot import DryRun, Interrupted, InterruptibleBackend, Skills, StepFailed

FRUITS = ('orange', 'apple', 'banana')
STOP_WORDS = re.compile(r'^\s*(stop|halt|wait|freeze|pause|cancel|abort|interrupt|ruko|niruthu|nillu)\b', re.I)
SYNONYMS = {  # last-resort name mapping when the model returns a description instead of a name
    'red': 'apple', 'red fruit': 'apple', 'yellow': 'banana', 'yellow fruit': 'banana', 'curved': 'banana',
    'orange fruit': 'orange', 'citrus': 'orange', 'dish': 'bowl', 'white bowl': 'bowl', 'container': 'bowl',
    'basket': 'bowl', 'plate': 'bowl',
}
ACTIONS = {'pick_place', 'pick_up', 'put_in', 'place_at', 'move_object', 'hand_over', 'hover', 'move_arm',
           'gripper', 'home', 'look', 'find', 'reset', 'wait'}

PLANNER_SYSTEM = """You are the language brain of Daksha, a two-armed humanoid robot standing at a table (a simulation).
The image is Daksha's head camera right now. Understand ANY way the operator phrases a message: any wording,
synonyms, colours and shapes, pronouns ("it", "that one", "the other one"), typos, short or long sentences,
polite or blunt, English mixed with other languages. Turn it into robot actions or an answer.

Scene memory:
SCENE

Objects: OBJECTS. Containers: CONTAINERS. Arms: "left", "right" (the robot's own left/right; omit "arm" to let it choose).
Robot frame: x forward (away from the robot), y to the robot's left, z up, metres.

Actions (JSON objects, executed in order):
 {"do":"pick_place","object":O,"target":C}           pick O and put it into container C
 {"do":"pick_up","object":O,"arm":A}                 pick O and keep holding it (lifted)
 {"do":"put_in","target":C,"arm":A}                  put what a hand holds into C
 {"do":"place_at","location":L,"arm":A}              put what a hand holds down on the table at L
 {"do":"move_object","object":O,"location":L}        pick O and put it down on the table at L
 {"do":"hand_over","object":O}                       pick O (unless already held) and hold it out to the person
 {"do":"hover","object":O,"arm":A}                   point at O: hand above it
 {"do":"move_arm","arm":A,"dx":m,"dy":m,"dz":m}      move a hand relative to where it is ("up a bit" = dz 0.05, "up 10 cm" = dz 0.10)
 {"do":"gripper","arm":A or "both","command":"open" or "close"}   open = release / drop / let go
 {"do":"find","object":O}                           locate O and mark it: bounding box on the camera, 3-D box and
                                                     goal frame in the IK view, position and size in the reply
 {"do":"home"}  {"do":"look"}  {"do":"reset"}  {"do":"wait","seconds":s}
L = {"region":"left"|"right"|"front"|"back"|"center"|"front-left"|"front-right"|"back-left"|"back-right"}
 or {"near":O,"side":"left"|"right"|"front"|"behind"}  or {"xy":[x,y]}.

Rules:
- Use only the object and container names above. Map descriptions to names (red fruit -> apple, yellow or curved
  -> banana, the round orange one -> orange, white dish / container / basket -> bowl).
- "pick / grab / take / lift X" with no destination, or with "hold it" / "keep it" / "don't place it" = pick_up
  (keep holding). Use pick_place only when the operator names where it goes (in the bowl, into the dish ...).
- "it" / "that" = the object mentioned last in the conversation, or the one a hand holds.
- "everything", "all the fruit", "clean up", "tidy the table" = one pick_place per object not already in the bowl.
- "bounding box / box / mark / highlight / find / locate / show me / where is X" = {"do":"find","object":X}
  ("type":"command"). Never say you cannot draw boxes: find draws them.
- Questions, greetings, chat: "type":"question", answer from the image and the memory in "reply", no actions.
- Anything the robot should do: "type":"command". "when":"after" only if the operator says to do it after the
  current task ("then", "next", "after that", "also", "when you are done"); otherwise "when":"now".
- Plan ONLY what the newest operator message asks. Earlier requests in the conversation are already done or
  running; never repeat their actions ("after that go home" -> just [{"do":"home"}] with "when":"after").
- If it is impossible or unclear, ask one short clarifying question in "reply" with no actions.
Reply with JSON only:
{"type":"command" or "question","when":"now" or "after","reply":"<one short sentence to the operator>","actions":[...]}"""

REAL_ROBOT_RULES = """This is a real robot. The supplied image is the current full world-camera frame.
Use open-vocabulary vision: identify the object the operator names directly from the live image, even if it has
not appeared in the known-object list or a previous scan. Never reject a visible object just because it is absent
from that list. For "look" / "what do you see", scan the camera and name the visible movable objects. If the
object is ambiguous or not visible, ask the operator to clarify before planning an action. Do not invent objects.
All motion must still go through the existing robot skills and their safety checks."""


class LocalAgent:
    def __init__(self, root, task='fruit_to_bowl', seed=1000, realtime=1.0, vlm=None, ik='sim', backend=None):
        # backend: None = the MuJoCo sim; or a callable(config, realtime=...) returning a robot with the
        # same daksha_* tool interface (e.g. the ROS 2 real-robot backend in ros_agent).
        self._backend_factory = backend
        self.ask_fn = None   # optional: question -> answer text (ros_agent's Ask agent: ROS topics, cameras)
        self.rax_run = None  # optional: task text -> report dict (ros_agent's RAX agent); then EVERY prompt goes to it
        self.rax_stop = None
        self.max_replans = 2   # revised plans per task (daksha_plan refuses more)
        self.is_real = False
        self.root = pathlib.Path(root)
        self.stream_dir = self.root/'stream'
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        self.task, self.seed = task, seed
        self.vlm = vlm or OllamaVLM()
        self._cfg = types.SimpleNamespace(root=self.root, workspace=self.root/'workspace', backend_options={},
                                          observation_profile='rgbd', task=task)
        self._realtime = realtime
        self.ik_mode = ik  # 'sim' = the simulator's IK, 'human' = local_pipeline/human_ik.py (placo)
        self._ready = threading.Event()
        self._init_error = None
        self.robot = self.skills = None  # built on the worker thread: it owns the simulator and its GL contexts
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.logs = collections.deque(maxlen=300)
        self.chat = collections.deque(maxlen=100)
        self.results = collections.deque(maxlen=100)   # task log: was each task completed? (see _finish_task)
        self._task = None
        self.state = {'busy': False, 'current': None, 'queued': 0, 'detections': {}, 'task_state': None,
                      'vlm': {}, 'seed': seed, 'task': task, 'model': self.vlm.model, 'interrupts': 0, 'ik': ik,
                      'thinking': False, 'held': {}, 'grasp': None, 'focus': None, 'plan': None}
        self._route_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name='robot-worker', daemon=True)
        self._closed = False

    # ------------------------------------------------------------------ public API (any thread)
    def start(self):
        self._worker.start()
        self._ready.wait()
        if self._init_error:
            raise self._init_error
        return self

    def submit(self, text):
        text = (text or '').strip()
        if not text:
            return
        self._say('operator', text)
        if STOP_WORDS.match(text) and len(text.split()) <= 3:
            self.stop()
            return
        if self.rax_run is not None:
            # single pipeline: the RAX agent plans with tools; a new task while one runs replaces it
            if self.state['busy'] or not self.q.empty():
                self.log(f'new task "{text}": stopping "{self.state["current"]}"')
                self.stop(say=False)
            self.q.put(('rax', {'text': text}, text))
            self._set(queued=self.q.qsize())
            return
        threading.Thread(target=self._route, args=(text,), daemon=True).start()

    # ------------------------------------------------------------------ the visible sub-plan
    # plan = {'task', 'cur', 'steps': [{'label', 'status', 'detail', 'subs': [{'label', 'status', 'tries', 'detail'}]}]}
    # status: pending | running | done | failed (subs also: retry)
    def plan_set(self, steps, task=None, reason=None):
        """A (new) plan. A plan made while one exists is a REVISION: the old one moves to history
        (with its outcome) and `reason` says why the plan changed."""
        steps = [str(s).strip() for s in steps if str(s).strip()][:12]
        with self.lock:
            old = self.state.get('plan')
            if old and old.get('steps') and old.get('rev', 1) >= 1 + int(getattr(self, 'max_replans', 2)):
                raise ValueError(f'replan limit reached ({getattr(self, "max_replans", 2)} revised plans): stop and call daksha_report '
                                 'with complete=false and what you tried')
            history = list(old.get('history', [])) if old else []
            if old and old.get('steps'):
                history.append({'rev': old.get('rev', 1), 'reason': old.get('reason'),
                                'steps': [{'label': s['label'], 'status': s['status'], 'detail': s.get('detail', '')}
                                          for s in old['steps']]})
            rev = (old.get('rev', 1) + 1) if old and old.get('steps') else 1
            self.state['plan'] = {'task': task or (self._task or {}).get('task', ''), 'cur': -1, 'rev': rev,
                                  'reason': reason if rev > 1 else None, 'history': history[-3:],
                                  'steps': [{'label': s, 'status': 'pending', 'detail': '', 'subs': []} for s in steps]}
        head = (f'Plan {rev} (replanned: {reason})' if rev > 1 and reason else f'Plan {rev}' if rev > 1 else 'Plan')
        self._say('plan', head + ':\n' + '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(steps)))

    def plan_step_start(self, label):
        """A skill starts: the next pending plan step (or a new one if the agent did not plan it)."""
        with self.lock:
            p = self.state.get('plan')
            if p is None:
                p = self.state['plan'] = {'task': (self._task or {}).get('task', ''), 'cur': -1, 'steps': []}
            nxt = next((i for i, s in enumerate(p['steps']) if s['status'] == 'pending'), None)
            if nxt is None:
                p['steps'].append({'label': label, 'status': 'pending', 'detail': '', 'subs': []})
                nxt = len(p['steps']) - 1
            p['cur'] = nxt
            p['steps'][nxt]['status'] = 'running'
            p['steps'][nxt]['subs'] = []

    def plan_step_end(self, ok, detail=''):
        with self.lock:
            p = self.state.get('plan')
            if p and 0 <= p['cur'] < len(p['steps']):
                s = p['steps'][p['cur']]
                s['status'], s['detail'] = ('done' if ok else 'failed'), str(detail)[:200]

    def plan_subs(self, labels):
        """A skill announces its sub-tasks (all shown as pending at once)."""
        with self.lock:
            p = self.state.get('plan')
            if p and 0 <= p['cur'] < len(p['steps']):
                p['steps'][p['cur']]['subs'] = [{'label': l, 'status': 'pending', 'tries': 0, 'detail': ''}
                                                for l in labels]

    def plan_sub(self, label, status, detail='', attempt=1):
        with self.lock:
            p = self.state.get('plan')
            if not p or not (0 <= p['cur'] < len(p['steps'])):
                return
            subs = p['steps'][p['cur']]['subs']
            sub = next((s for s in subs if s['label'] == label and s['status'] not in ('done', 'failed')), None) \
                or next((s for s in subs if s['label'] == label), None)
            if sub is None:
                sub = {'label': label, 'status': 'pending', 'tries': 0, 'detail': ''}
                subs.append(sub)
            sub['status'], sub['tries'] = status, max(sub['tries'], attempt)
            if detail:
                sub['detail'] = str(detail)[:160]

    # ------------------------------------------------------------------ task log
    def _task_step(self, do, what, status, detail=''):
        if self._task is not None:
            self._task['steps'].append({'do': do, 'object': what, 'status': status, 'detail': detail})

    def _finish_task(self, status, reason):
        """One entry per task: completed only if it ran to the end and no step failed."""
        t = self._task or {'task': '?', 'start': time.time(), 'steps': []}
        # a step that failed but succeeded later (a retry of the whole pick) is recovered, not a failure
        bad = [s for i, s in enumerate(t['steps']) if s['status'] != 'done'
               and not any(u['do'] == s['do'] and u['status'] == 'done' for u in t['steps'][i + 1:])]
        if status == 'completed' and bad:
            status, reason = 'failed', '; '.join(f"{s['do']} {s['object'] or ''}: {s['detail']}".strip() for s in bad)
        if status == 'completed' and t.get('note'):
            reason = t['note']
        elif status == 'completed' and not t['steps']:
            reason = 'nothing to do'
        entry = {'task': t['task'], 'status': status, 'reason': reason, 'steps': t['steps'],
                 'seconds': round(time.time() - t['start'], 1), 'time': time.time(),
                 'grasp': g if (g := self.state.get('grasp')) and g.get('time', 0) >= t['start'] else None,
                 # the task's own plan (with its sub-tasks and earlier revisions), shown inside the task log entry
                 'plan': copy.deepcopy(pl) if (pl := self.state.get('plan')) and pl.get('steps') else None}
        with self.lock:
            self.results.append(entry)
        self.log(f'TASK {status.upper()}: "{t["task"]}" ({entry["seconds"]} s)' + (f': {reason}' if reason else ''))
        self._task = None

    def _on_skill_plan(self, kind, **kw):
        if kind == 'subs':
            self.plan_subs(kw['labels'])
        elif kind == 'sub':
            self.plan_sub(kw['label'], kw['status'], kw.get('detail', ''), kw.get('attempt', 1))

    def _ask(self, text, fallback=None):
        """A question for the Ask agent (read-only tools); its answer goes into this chat.
        Falls back to the planner's own short reply when no Ask agent is available."""
        self._set(thinking=True)
        try:
            ans = self.ask_fn(text)
        except Exception as exc:
            ans = f'Could not answer: {exc}'
        finally:
            self._set(thinking=False)
        self._say('daksha', str(ans or fallback or 'I have no answer for that.').strip())

    def _route(self, text):
        """Understand one message (model call, serialized) and dispatch it. Never touches the robot."""
        with self._route_lock:
            self._set(thinking=True)
            try:
                plan = self._understand(text)
            finally:
                self._set(thinking=False)
            actions = plan.get('actions') or []
            if plan.get('type') == 'question' and not actions and self.ask_fn is not None:
                # same chat, Ask's tools: answered in the background so commands are never blocked
                threading.Thread(target=self._ask, args=(text, plan.get('reply')), daemon=True).start()
                return
            if plan.get('reply'):
                self._say('daksha', plan['reply'])
            if not actions:
                return
            busy = self.state['busy'] or not self.q.empty()
            if busy and plan.get('when') == 'after':
                self.log(f'queued after the current task: "{text}"')
            elif busy:
                self.log(f'barge-in: interrupting "{self.state["current"]}" for "{text}"')
                self.stop(say=False)
            self.q.put(('plan', {'plan': plan, 'text': text}, text))
            self._set(queued=self.q.qsize())

    def stop(self, say=True):
        dropped = 0
        while True:
            try:
                self.q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if self.state['busy']:
            self.robot.interrupt.set()
            if self.rax_stop is not None:
                self.rax_stop()             # kills the agent process; run() then waits for the in-flight tool
            self._set(interrupts=self.state['interrupts'] + 1)
        if say:
            self._say('daksha', 'Stopping.' + (f' Dropped {dropped} queued command(s).' if dropped else ''))
        self._set(queued=0)

    def snapshot(self):
        with self.lock:
            plan = None if self.state.get('plan') is None else json.loads(json.dumps(self.state['plan']))
            return {**self.state, 'plan': plan, 'logs': list(self.logs)[-80:], 'chat': list(self.chat)[-40:],
                    'results': list(self.results)[-50:]}

    def close(self):
        self._closed = True
        self.robot.interrupt.set()
        self.q.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=10)  # the worker owns the GL renderers; let it release them

    # ------------------------------------------------------------------ worker
    def _run(self):
        try:
            self.robot = (self._backend_factory or InterruptibleBackend)(self._cfg, realtime=self._realtime)
            self.is_real = bool(getattr(self.robot, 'is_real', False))
            self.objects = list(getattr(self.robot.tasks, 'OBJECTS', FRUITS))
            self.containers = list(getattr(self.robot.tasks, 'BOWL_NAMES', ('bowl',)))
            hik = None
            if self.ik_mode == 'human':
                from .human_ik import DualHumanIK
                hik = DualHumanIK()
                self.log('human-posture IK ready (placo, kinematics/ package)')
            self.skills = Skills(self.robot, self.vlm, log=self.log, on_detections=self._draw, human_ik=hik)
            self.skills.all_labels = self.objects + self.containers
            self.skills.containers = list(self.containers)

            def on_grasp(g):
                self._set(grasp=g, held=dict(self.skills.held))
                self._say('daksha', (f'Picked the {g["object"]} with the {g["arm"]} hand '
                                     f'({g["width_m"] * 100:.1f} cm between the fingers).') if g['picked'] else
                          f'Did NOT pick the {g["object"]}: {g["why"]}.')
            self.skills.on_grasp = on_grasp
            self.skills.on_step = lambda label, status, detail: self._task_step(label, '', status, detail)
            self.skills.on_plan = self._on_skill_plan
            if not self.is_real:
                self._reset(self.seed)  # the first layout exists before any command is accepted
            self._update_task_state()
        except Exception as exc:
            self._init_error = exc
            return
        finally:
            self._ready.set()
        while not self._closed:
            item = self.q.get()
            if item is None:
                break
            kind, args, label = item
            self.robot.interrupt.clear()
            self._task = {'task': label, 'start': time.time(), 'steps': []}
            # a new task starts clean: the last task's boxes may be of an object that has moved since, and
            # its grasp result is old news unless that object is still in the hand
            g = self.state.get('grasp')
            held = self.state.get('held') or {}
            keep = g if g and g.get('picked') and held.get(g.get('arm')) == g.get('object') else None
            self._set(plan=None, detections={}, grasp=keep)
            status, reason = 'completed', ''
            self._set(busy=True, current=label, queued=self.q.qsize())
            try:
                if kind == 'reset':
                    if self.is_real:
                        self._say('daksha', 'reset is for the simulator; the real table is not changed.')
                    else:
                        self._reset(args.get('seed'))
                elif kind == 'rax':
                    rep = self.rax_run(args['text'])
                    note = rep['summary'] + (f' ({rep["evidence"]})' if rep.get('evidence') else '')
                    if not rep['complete'] and self._task.get('dry_run'):
                        raise DryRun(rep['summary'] or 'execute is off')
                    if not rep['complete']:
                        raise StepFailed(note or 'the agent reports the task is not complete')
                    self._task['note'] = note
                    self._say('daksha', '✓ ' + (rep['summary'] or 'done'))
                elif kind == 'plan':
                    self._execute(args['plan'], args['text'])
                else:  # a raw instruction (e.g. from code): understand it here
                    self._execute(self._understand(args['text']), args['text'])
            except (Interrupted, InterruptedError) as exc:
                status, reason = 'interrupted', str(exc)
                self.log(f'"{label}" {exc}')
                self._say('daksha', f'Interrupted "{label}". Holding position.')
            except DryRun as exc:
                status, reason = 'dry run', f'nothing moved: {exc}'
                self.log(f'"{label}" dry run: {exc}')
                self._say('daksha', f'Dry run, not moving: {exc}')
            except (StepFailed, ValueError, VLMError) as exc:
                status, reason = 'failed', str(exc)
                self.log(f'"{label}" failed: {exc}')
                self._say('daksha', f'Could not finish "{label}": {exc}')
            except Exception as exc:  # keep the worker alive for the next command
                status, reason = 'error', f'{type(exc).__name__}: {exc}'
                self.log(f'"{label}" error {type(exc).__name__}: {exc}')
                self._say('daksha', f'Could not finish "{label}": {exc}')
            finally:
                self._finish_task(status, reason)
                self.robot.interrupt.clear()
                self._update_task_state()
                self._set(busy=False, current=None, queued=self.q.qsize(), held=dict(self.skills.held))
        self.robot.close()

    def _reset(self, seed=None):
        self.seed = self.seed + 1 if seed is None else int(seed)
        self.robot.reset_episode(self.seed)
        if self.skills is not None and self.skills.hik is not None:
            try:
                self.skills.home()  # from the sim's HOME to the human ready posture
            except (StepFailed, ValueError, Interrupted) as exc:
                self.log(f'human home after reset: {exc}')
        self._set(seed=self.seed, detections={})
        try:
            (self.stream_dir/'vlm.jpg').unlink()
        except OSError:
            pass
        self.log(f'new layout, seed {self.seed}')

    def _execute(self, plan, text):
        actions = [x for x in (plan.get('actions') or []) if isinstance(x, dict)]
        self.log('plan: ' + json.dumps(actions))
        placed = {}  # obj -> target, for the verification sweep
        sk = self.skills
        said_arm = re.search(r'\b(left|right)\b', text, re.I) is not None
        for a in actions:
            sk.check()
            do = a.get('do')
            rec = None  # (status, detail) for the task log; None = done
            obj, target, arm = self._name(a.get('object')), self._name(a.get('target')), a.get('arm')
            arm = arm if arm in ('left', 'right') else None
            # The model often names an arm on its own; for grasping, honour it only if the operator said it.
            if do in ('pick_place', 'pick_up', 'move_object', 'hand_over', 'hover') and not said_arm:
                arm = None
            self._set(current=f'{text} → {do} {obj or target or ""}'.strip(), held=dict(sk.held))
            if do in ('pick_place', 'pick_up', 'move_object', 'hover') and obj not in self.objects:
                if self.is_real and obj:
                    self.objects.append(obj)
                    sk.all_labels = self.objects + self.containers
                else:
                    self._say('daksha', f'I do not know which object "{a.get("object")}" is.')
                    continue
            if do in ('pick_place', 'put_in') and target not in self.containers:
                target = self.containers[0]
            if do == 'pick_place':
                ok = sk.pick_place(obj, target)
                placed[obj] = target
                rec = ('done', f'placed in the {target}') if ok else ('failed', 'failed after retries')
                self._say('daksha', f'{obj} → {target}: ' + ('placed' if ok else 'failed after retries'))
            elif do == 'pick_up':
                held_arm = sk.pick_up(obj, arm=arm)
                self._say('daksha', f'Holding the {obj} with my {held_arm} hand.')
                g = sk.last_grasp or {}
                rec = ('done', f'holding it in the {held_arm} hand' + (f' (fingers {g["width_m"] * 100:.1f} cm, gripping)' if g.get('picked') else ''))
            elif do == 'put_in':
                what = sk.held.get(sk.holding_arm(arm) or '', None)
                sk.put_in(target, arm=arm)
                if what:
                    placed[what] = target
            elif do == 'place_at':
                sk.place_at(a.get('location'), arm=arm)
            elif do == 'move_object':
                held_arm = sk.holding_arm(None, obj) or sk.pick_up(obj, arm=arm)
                sk.place_at(a.get('location'), arm=held_arm)
                self._say('daksha', f'Moved the {obj}.')
            elif do == 'hand_over':
                sk.hand_over(obj if obj in self.objects else None, arm=arm)
                self._say('daksha', 'Here you go. Say "release" or "let go" when you have it.')
            elif do == 'hover':
                sk.hover(obj, arm=arm)
            elif do == 'move_arm':
                which = arm or sk.holding_arm() or 'left'
                sk.move_arm(which, *(float(a.get(k, 0) or 0) for k in ('dx', 'dy', 'dz')))
            elif do == 'gripper':
                sk.gripper(a.get('arm') if a.get('arm') in ('left', 'right') else 'both',
                           'close' if a.get('command') == 'close' else 'open')
            elif do == 'home':
                sk.home()
            elif do == 'look':
                if self.is_real:
                    found = sk.perceive_all()
                    for label in found:
                        if label not in self.objects and label not in self.containers:
                            self.objects.append(label)
                    sk.all_labels = self.objects + self.containers
                else:
                    found = sk.perceive(self.objects + self.containers)
                names = ', '.join(f'{k} ({v["source"]})' for k, v in found.items())
                self._say('daksha', 'I see: ' + names if names else 'I could not identify any movable objects in the live camera.')
                rec = ('done', f'{len(found)} object(s) seen')
            elif do == 'find':
                if not obj:
                    self._say('daksha', 'Which object should I find?')
                    continue
                if obj not in self.objects and obj not in self.containers:
                    self.objects.append(obj)
                    sk.all_labels = self.objects + self.containers
                found = sk.perceive([obj])
                e = found.get(obj)
                if not e or not e.get('centre'):
                    self._say('daksha', f'I could not find the {obj} in the camera image.')
                    rec = ('failed', 'not found in the camera image')
                else:
                    self._set(focus=obj)
                    rec = ('done', 'found at ({:.3f}, {:.3f}, {:.3f})'.format(*e['centre']))
                    x, y, z = e['centre']
                    self._say('daksha', f'Found the {obj}: boxed on the camera and in the IK view '
                                        f'(goal frame object_{re.sub(r"[^a-z0-9]+", "_", obj.lower()).strip("_")}), '
                                        f'top at x {x:.3f}, y {y:.3f}, z {z:.3f} m.')
            elif do == 'reset':
                self._reset(a.get('seed'))
            elif do == 'wait':
                end = time.time() + min(30.0, float(a.get('seconds', 2) or 2))
                while time.time() < end:
                    sk.check()
                    time.sleep(0.1)
            else:
                self.log(f'unknown action {a}')
                rec = ('failed', 'unknown action')
            self._task_step(do, obj or target, *(rec or ('done', '')))
        self._set(held=dict(sk.held))
        if len(placed) > 1:  # multi-object jobs only (a later placement can knock an earlier one out)
            self._verify(placed)

    def _name(self, value):
        """An object/container name from what the model returned (exact, plural, or a description)."""
        if not value:
            return None
        v = str(value).lower().strip().removeprefix('the ').replace(' ', '_')
        names = self.objects + self.containers
        if v in names:
            return v
        if v.rstrip('s') in names:
            return v.rstrip('s')
        v2 = v.replace('_', ' ')
        if v2 in SYNONYMS:
            return SYNONYMS[v2]
        return next((n for n in names if n in v or n.replace('_', ' ') in v2), v)

    def _scene_summary(self):
        from .robot import inside
        sk = self.skills
        lines = []
        mem = dict(sk.memory) if sk else {}
        if self.is_real:
            positioned = {name: entry for name, entry in mem.items() if entry and 'centre' in entry}
            if positioned:
                lines.append('- camera-seen objects: ' + ', '.join(positioned))
            else:
                lines.append('- no objects have been localized yet; inspect the live camera')
            held = {a: o for a, o in (sk.held.items() if sk else []) if o}
            lines.append('- hands: ' + (', '.join(f'{a} hand holds the {o}' for a, o in held.items()) or 'both empty'))
            lines.append('- robot is ' + (f'busy with: {self.state["current"]}' if self.state['busy'] else 'idle'))
            return '\n'.join(lines)
        bowl = mem.get(self.containers[0])
        for name in self.objects + self.containers:
            e = mem.get(name)
            if e is None:
                lines.append(f'- {name}: not seen yet')
                continue
            x, y = e['centre'][:2]
            side = 'left side' if y > 0.08 else 'right side' if y < -0.08 else 'middle'
            extra = ''
            if name in self.objects and bowl is not None:
                extra = ', in the bowl' if inside(e, bowl) else ', on the table'
            lines.append(f'- {name}: last seen at x={x:.2f}, y={y:.2f} ({side}{extra})')
        held = {a: o for a, o in (sk.held.items() if sk else []) if o}
        lines.append('- hands: ' + (', '.join(f'{a} hand holds the {o}' for a, o in held.items()) or 'both empty'))
        lines.append('- robot is ' + (f'busy with: {self.state["current"]}' if self.state['busy'] else 'idle'))
        with self.lock:
            recent = list(self.chat)[-7:-1]
        if recent:
            lines.append('- recent conversation: ' + ' | '.join(f'{m["who"]}: {m["text"][:80]}' for m in recent))
        return '\n'.join(lines)

    def _understand(self, text):
        """One model call: message + live head-camera image + scene memory -> {"type","when","reply","actions"}."""
        exact = self._fast_path(text)
        if exact is not None:
            return exact
        try:
            frame = Image.open(self.stream_dir/'latest.jpg').convert('RGB')
            if self.is_real:
                head = frame
            else:
                w, h = frame.size
                head = frame.crop((w//2, 0, w, h))  # simulator stream = [overview | head camera]
        except OSError:
            head = None
        system = (PLANNER_SYSTEM.replace('OBJECTS', ', '.join(self.objects))
                  .replace('CONTAINERS', ', '.join(self.containers)).replace('SCENE', self._scene_summary()))
        if self.is_real:
            system = system.replace('(a simulation)', '(a real robot)')
            system = system.replace(
                f'Objects: {", ".join(self.objects)}. Containers: {", ".join(self.containers)}.',
                'Use movable objects and containers visible in the current live camera; known labels are hints only.')
            system = system.replace(
                '- Use only the object and container names above. Map descriptions to names (red fruit -> apple, yellow or curved\n'
                '  -> banana, the round orange one -> orange, white dish / container / basket -> bowl).',
                '- Use the named object from the current live image, even if it is not in the known-object list. '
                'Use the live image for object identity; do not map arbitrary objects to fruit names.')
            system += '\n\n' + REAL_ROBOT_RULES
        try:
            plan = self.vlm.ask_json(head, f'Operator: {text}', system=system, prefill='{"type": "', max_tokens=600)
            self._set(vlm=self.vlm.last)
            if isinstance(plan, dict) and isinstance(plan.get('actions', []), list):
                bad = [x for x in plan.get('actions', []) if not isinstance(x, dict) or x.get('do') not in ACTIONS]
                if bad:
                    self.log(f'dropping unknown actions from the model: {bad}')
                    plan['actions'] = [x for x in plan['actions'] if x not in bad]
                self.log(f'understood ({self.vlm.last.get("seconds", "?")} s): {plan.get("type")} {plan.get("when", "")} '
                         f'{json.dumps(plan.get("actions", []))}')
                return plan
            self.log(f'model reply was not a plan: {plan}')
        except VLMError as exc:
            self.log(f'model gave no answer ({exc}); trying the keyword fallback')
        rule = self._rule_plan(text)
        if rule is not None:
            return rule
        return {'type': 'question', 'reply': 'Sorry, I did not get that. Could you say it another way?', 'actions': []}

    def _fast_path(self, text):
        """Exact one-word commands skip the model (instant)."""
        t = text.lower().strip(' .!')
        table = {'home': [{'do': 'home'}], 'go home': [{'do': 'home'}], 'look': [{'do': 'look'}],
                 'scan': [{'do': 'look'}], 'reset': [{'do': 'reset'}], 'release': [{'do': 'gripper', 'arm': 'both', 'command': 'open'}],
                 'let go': [{'do': 'gripper', 'arm': 'both', 'command': 'open'}], 'open': [{'do': 'gripper', 'arm': 'both', 'command': 'open'}]}
        m = re.fullmatch(r'reset (\d+)', t)
        if m:
            return {'type': 'command', 'when': 'now', 'actions': [{'do': 'reset', 'seed': int(m.group(1))}]}
        if t in table:
            return {'type': 'command', 'when': 'now', 'actions': table[t]}
        return None

    def _verify(self, placed, rounds=1):
        """Look again after placing: a later placement can knock an earlier object out. Retry what is outside."""
        from .robot import inside
        for r in range(rounds + 1):
            self._set(current='verify: looking at the table')
            targets = sorted(set(placed.values()))
            scene = self.skills.perceive(list(placed) + targets, clear=False)  # a look in place, no arm motion
            outside = [o for o, t in placed.items() if o in scene and t in scene and not inside(scene[o], scene[t])]
            unseen = [o for o in placed if o not in scene]
            if not outside:
                self._say('daksha', f'Checked with {self.vlm.model}: ' + ', '.join(f'{o} in the {t}' for o, t in placed.items()
                                                                          if o not in unseen)
                          + (f' ({", ".join(unseen)} not visible)' if unseen else '') + '.')
                return
            if r == rounds:
                self._say('daksha', f'Still outside after a retry: {", ".join(outside)}.')
                return
            self._say('daksha', f'Verification: {", ".join(outside)} not in place, retrying.')
            for o in outside:
                self._set(current=f'retry {o}')
                self.skills.pick_place(o, placed[o])

    def _rule_plan(self, text):
        """Keyword fallback, used only when the model gives no answer (e.g. Ollama down)."""
        t = text.lower()
        if re.fullmatch(r'\s*(go )?home\s*', t):
            return {'actions': [{'do': 'home'}]}
        if re.fullmatch(r'\s*(reset|new layout|shuffle)( \d+)?\s*', t):
            m = re.search(r'\d+', t)
            return {'actions': [{'do': 'reset', **({'seed': int(m.group())} if m else {})}]}
        if re.fullmatch(r'\s*(look|detect|scan)\s*', t):
            return {'actions': [{'do': 'look'}]}
        if re.fullmatch(r'\s*open( grippers?)?\s*', t):
            return {'actions': [{'do': 'open'}]}
        if re.search(r'\b(all|every|everything|clean|clear)\b', t) and re.search(r'\b(fruits?|objects?|toys?|table|everything)\b', t):
            target = next((c for c in self.containers if c.replace('_', ' ') in t), self.containers[0])
            return {'reply': f'Putting everything into the {target}, one object at a time.',
                    'actions': [{'do': 'pick_place', 'object': o, 'target': target} for o in self.objects]}
        named = [o for o in self.objects if re.search(rf'\b{o.replace("_", "[ _]")}\b', t)]
        hold = re.search(r"\b(hold|keep|carry)\b|don'?t (put|place|drop)|do not (put|place|drop)|without (putting|placing)", t)
        destination = re.search(r'\b(in|into|inside|bowl|dish|basket|container)\b', t)
        if named and re.search(r'\b(pick|grab|lift|take)\b', t) and (hold or not destination):
            return {'type': 'command', 'when': 'now', 'reply': f'Picking up the {named[0]} and holding it.',
                    'actions': [{'do': 'pick_up', 'object': named[0]}]}
        if named and re.search(r'\b(put|place|pick|move|grab|drop)\b', t):
            target = next((c for c in self.containers if c.replace('_', ' ') in t), self.containers[0])
            return {'reply': f'Moving {", ".join(named)} to the {target}.',
                    'actions': [{'do': 'pick_place', 'object': o, 'target': target} for o in named]}
        return None

    def _update_task_state(self):
        try:
            s = self.robot.task_state()
            self._set(task_state={'task_complete': s.get('task_complete'), 'inside': s.get('inside')})
        except Exception:
            pass

    # ------------------------------------------------------------------ overlay + bookkeeping
    def _draw(self, rgb, dets, found, frame_id):
        img = Image.fromarray(rgb.astype(np.uint8)).convert('RGB')
        dr = ImageDraw.Draw(img)
        palette = {'orange': (255, 150, 0), 'apple': (230, 40, 40), 'banana': (250, 220, 0), 'bowl': (80, 200, 255)}
        for label, e in found.items():
            if 'bbox' not in e:
                continue
            col = palette.get(label, (120, 255, 120))
            x1, y1, x2, y2 = e['bbox']
            dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
            if e.get('point'):
                u, v = e['point']
                dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill=col)
            c = e.get('centre')
            tag = (f'{label} ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})' if c else label)
            tag += '' if e['source'] == 'vlm' else ' [fallback]'
            tw = dr.textlength(tag)
            dr.rectangle([x1, y1 - 14, x1 + tw + 6, y1], fill=(0, 0, 0))
            dr.text((x1 + 3, y1 - 13), tag, fill=col)
        dr.rectangle([0, 0, 640, 16], fill=(0, 0, 0))
        dr.text((4, 2), f'{self.vlm.model}  frame {frame_id}  {self.vlm.last.get("seconds", "?")} s', fill=(220, 220, 220))
        tmp = self.stream_dir/'vlm.tmp.jpg'
        img.save(tmp, quality=88)
        os.replace(tmp, self.stream_dir/'vlm.jpg')
        self._set(detections={k: {kk: vv for kk, vv in v.items() if kk in ('centre', 'source', 'yaw', 'bbox', 'point')}
                              for k, v in found.items()}, vlm=self.vlm.last)

    def log(self, msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with self.lock:
            self.logs.append(line)

    def _say(self, who, text):
        with self.lock:
            self.chat.append({'who': who, 'text': text, 't': time.strftime('%H:%M:%S')})
        if who != 'operator':
            print(f'  daksha> {text}', flush=True)

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)
