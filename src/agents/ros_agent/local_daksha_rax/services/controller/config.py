"""Operator-owned session configuration. No values are accepted from agent code."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

PROFILES = ('rgbd', 'rgb-only')
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
TASKS = ('fruit_to_bowl', 'fruit_toy_sorting', 'tray_transport')


@dataclass(frozen=True)
class Configuration:
    root: Path                         # session directory (runtime/sessions/<name>)
    task: str = 'fruit_to_bowl'
    robot: str = 'daksha_sim'          # services.backends registry key
    backend_options: dict = field(default_factory=dict)
    observation_profile: str = 'rgbd'
    episodes: int = 24                 # exploration episode budget
    formal_episodes: int = 10          # held-out layouts for scripts/evaluate_controller.py
    agent_model: str | None = None     # Claude Code model alias or id; None: the CLI default
    agent_effort: str = 'high'
    session_seconds: int = 6*3600
    subagents: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).resolve())
        if not re.fullmatch(r'[A-Za-z0-9_-]+', self.task):
            raise ValueError('Invalid task id')
        if self.task not in TASKS:
            raise ValueError(f'task must be one of {TASKS}')
        from services.backends import BACKENDS
        if self.robot not in BACKENDS:
            raise ValueError(f"Unknown robot; choose one of {', '.join(sorted(BACKENDS))}")
        if not isinstance(self.backend_options, dict):
            raise ValueError('backend_options must be an object')
        if self.observation_profile not in PROFILES:
            raise ValueError(f'observation_profile must be one of {PROFILES}')
        if type(self.episodes) is not int or not 1 <= self.episodes <= 1000:
            raise ValueError('episodes must be 1..1000')
        if type(self.formal_episodes) is not int or not 1 <= self.formal_episodes <= 200:
            raise ValueError('formal_episodes must be 1..200')
        if self.agent_effort not in EFFORTS:
            raise ValueError(f'agent_effort must be one of {EFFORTS}')

    @property
    def workspace(self):
        return self.root/'agent-workspace'

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=str))
