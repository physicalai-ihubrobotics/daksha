#!/usr/bin/env python3
"""Demo: run a few gestures on both hands.

Before running: close every other controller using the serial port and put
BOTH hands physically at their calibrated OPEN pose.
"""
import sys
from pathlib import Path

# Allow running from a source checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bionic_hand import BionicHands

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"

with BionicHands(port) as hands:
    print("Motors responding:", hands.ping_all())
    hands.confirm_open()                       # software: both hands are at OPEN

    try:
        hands.left.gesture("point", speed=300, duration=2)
        hands.right.gesture("peace / v", speed=300, duration=2)
        hands.gesture("thumbs up", parallel=True)
        hands.right.set_fingers(index=0.5, middle="close")
        hands.wave("left_to_right", cycles=2, speed=500, duration=3)
        hands.open()
    except KeyboardInterrupt:
        hands.stop()
        print("Stopped - check the hands and confirm OPEN before moving again.")
