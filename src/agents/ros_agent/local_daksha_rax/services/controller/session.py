"""Durable session owner: one robot owner per session, the episode ledger and the MCP audit log.

state.json and mcp.jsonl live in the session directory, outside the agent's workspace, and
the agent's sandbox cannot read them (they hold seeds and per-episode results).
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time

from services.json_io import write_private_json

EXPLORATION_SEED_BASE = 1000  # exploration episode i plays seed 1000 + i; formal uses 0..N-1


class Session:
    def __init__(self, config, *, state_name='state.json', audit_name='mcp.jsonl'):
        self.config = config
        root = config.root
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._owner = (root/(state_name + '.lock')).open('a')
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner.close()
            raise RuntimeError('Session already has a robot owner')
        self._lock = threading.RLock()
        self.state_path = root/state_name
        self.audit_path = root/audit_name
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {'schema': 1, 'created_at': time.time(), 'episodes_started': 0,
                          'current': None, 'history': []}
            self.save()

    def save(self):
        with self._lock:
            write_private_json(self.state_path, self.state)

    def audit(self, event):
        raw = (json.dumps({'time': time.time(), **event}, allow_nan=False, default=str) + '\n').encode()
        with self._lock, self.audit_path.open('ab') as stream:
            stream.write(raw)

    # ------------------------------------------------------------------ episodes
    def episodes_left(self):
        return self.config.episodes - self.state['episodes_started']

    def begin_episode(self):
        with self._lock:
            if self.episodes_left() <= 0:
                raise PermissionError(f'The exploration budget of {self.config.episodes} episodes is used up')
            index = self.state['episodes_started']
            self.state['episodes_started'] += 1
            self.state['current'] = {'index': index, 'seed': EXPLORATION_SEED_BASE + index,
                                     'started_at': time.time(), 'evaluations': []}
            self.save()  # charged before the simulator resets, so a failed reset stays charged
            return self.state['current']

    def record_evaluation(self, complete):
        with self._lock:
            if self.state['current'] is not None:
                self.state['current']['evaluations'].append({'time': time.time(), 'task_complete': bool(complete)})
                self.save()

    def end_episode(self, final):
        with self._lock:
            current = self.state['current']
            if current is None:
                return None
            current.update(ended_at=time.time(), final=final)
            self.state['history'].append(current)
            self.state['current'] = None
            self.save()
            return current

    def close(self):
        self._owner.close()
