"""Calibrated endpoints, finger mapping and gesture definitions for both hands.

Values are taken from the final working GUIs in ``akashy/*/final/``.
"""
import json
from dataclasses import dataclass, field

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring")

# Finger states in order (thumb, index, middle, ring). True = OPEN, False = CLOSED.
GESTURES = {
    "OPEN HAND":                (True,  True,  True,  True),
    "CLOSE / FIST":             (False, False, False, False),
    "THUMBS UP":                (True,  False, False, False),
    "POINT":                    (False, True,  False, False),
    "PEACE / V":                (False, True,  True,  False),
    "VICTORY / V (THUMB OPEN)": (True,  True,  True,  False),
    "THREE FINGERS":            (False, True,  True,  True),
    "MIDDLE FINGER CLOSED":     (True,  True,  False, True),
    "INDEX CLOSED":             (True,  False, True,  True),
    "INDEX + RING":             (False, True,  False, True),
    "MIDDLE + RING":            (False, False, True,  True),
}


@dataclass
class HandConfig:
    """Motor IDs, finger mapping and OPEN/CLOSE angles (degrees) for one hand."""
    name: str
    fingers: dict                 # finger name -> (joint1 id, joint2 id)
    open_pose: dict               # motor id -> angle
    close_pose: dict              # motor id -> angle
    motion: str = "direct"        # default motion style: direct | smooth | staged
    ids: list = field(init=False)

    def __post_init__(self):
        self.ids = sorted(sid for pair in self.fingers.values() for sid in pair)

    def to_dict(self):
        return {
            "fingers": {f: list(ids) for f, ids in self.fingers.items()},
            "open": {str(k): v for k, v in self.open_pose.items()},
            "close": {str(k): v for k, v in self.close_pose.items()},
            "motion": self.motion,
        }

    def update_from_dict(self, data):
        if "fingers" in data:
            self.fingers = {f: tuple(ids) for f, ids in data["fingers"].items()}
            self.ids = sorted(sid for pair in self.fingers.values() for sid in pair)
        self.open_pose.update({int(k): float(v) for k, v in data.get("open", {}).items()})
        self.close_pose.update({int(k): float(v) for k, v in data.get("close", {}).items()})
        self.motion = data.get("motion", self.motion)


def left_config():
    return HandConfig(
        name="left",
        fingers={"Thumb": (1, 2), "Index": (3, 4), "Middle": (5, 6), "Ring": (7, 8)},
        open_pose={1: 300.0, 2: 0.0, 3: 300.0, 4: 0.0, 5: 300.0, 6: 0.0, 7: 300.0, 8: 0.0},
        close_pose={1: 114.0, 2: 195.0, 3: 133.0, 4: 182.0, 5: 142.0, 6: 191.0, 7: 128.0, 8: 182.0},
        motion="staged",
    )


def right_config():
    # Physical mapping on the right hand runs Ring, Middle, Index, Thumb.
    return HandConfig(
        name="right",
        fingers={"Thumb": (15, 16), "Index": (13, 14), "Middle": (11, 12), "Ring": (9, 10)},
        open_pose={9: 300.0, 10: 0.0, 11: 300.0, 12: 0.0, 13: 300.0, 14: 0.0, 15: 300.0, 16: 0.0},
        close_pose={9: 147.0, 10: 170.0, 11: 138.0, 12: 160.0, 13: 123.0, 14: 130.0, 15: 138.0, 16: 190.0},
        motion="smooth",
    )


def save_configs(path, configs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({c.name: c.to_dict() for c in configs}, f, indent=2)


def load_configs(path, configs):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in configs:
        if c.name in data:
            c.update_from_dict(data[c.name])
