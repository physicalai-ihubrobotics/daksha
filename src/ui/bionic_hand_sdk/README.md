# bionic_hand — Python SDK

Control the SCS0009 bionic hands from Python code instead of a GUI.
Protocol, calibration and motion logic come from the final working scripts in
`akashy/*/final/`.

- One shared serial adapter (`/dev/ttyACM0`, 1,000,000 baud)
- Left hand: motor IDs 1–8, Right hand: motor IDs 9–16
- Each finger (Thumb, Index, Middle, Ring) = 2 motors

## Install

```bash
cd bionic_hand_sdk
pip install -e .        # or: pip install pyserial  and run from this folder
```

## Quick start

```python
from bionic_hand import BionicHands

with BionicHands("/dev/ttyACM0") as hands:   # connects, disconnects on exit
    hands.confirm_open()                     # both hands are physically OPEN

    hands.gesture("thumbs up")               # both hands, one after the other
    hands.gesture("point", parallel=True)    # both hands at the same time
    hands.left.gesture("peace / v", speed=300, duration=2)
    hands.right.close()
    hands.right.open()

    hands.left.set_fingers(index="open", middle=0.5, ring=True)  # 0.0 open .. 1.0 closed
    hands.wave("left_to_right", cycles=2)
```

## Web UI

```bash
python3 web/app.py                   # open http://127.0.0.1:5000 on this laptop
python3 web/app.py --host 0.0.0.0    # also from a phone on the same Wi-Fi
```

Connect, switch on **Enable movement**, then use per-finger sliders, gestures,
waves or the single-motor test. When a hand's position is unknown, smooth and
staged moves first read the motor positions; if that fails, that move is sent
as direct (only the selected fingers are commanded). **STOP** (or the
Esc key) stops the running sequence. Pick the port `sim://` to try the UI
without hardware. There is no login: use `--host 0.0.0.0` only on a trusted
private network.

## Main classes

| Class | What it is |
|---|---|
| `BionicHands(port, baud, calibration_file=None)` | Both hands on one bus. Attributes `.left`, `.right`, `.bus` |
| `Hand` | One hand. Gestures, fingers, single motors, wave, read positions |
| `SCSBus` | Raw FEETECH SCS bus: `set_angle`, `set_speed`, `read_angle`, `ping`, `write_u16`, `read_register` |
| `HandConfig` | Motor IDs, finger mapping, OPEN/CLOSE angles, default motion style |

### `BionicHands`
| Method | Description |
|---|---|
| `connect()` / `disconnect()` | Open/close the serial port (or use `with`) |
| `confirm_open(hand="both")` | Tell the SDK the hand(s) are physically at OPEN |
| `gesture(name, hand="both", parallel=False, speed=, duration=, motion=)` | Run a named gesture |
| `open(hand)` / `close(hand)` | Open hand / fist |
| `wave(sequence, cycles=2, speed=500, duration=3.0)` | `left`, `right`, `left_to_right`, `right_to_left`, `both` |
| `read_positions(hand)` | Read present angles from the motors |
| `ping_all()` | `{id: True/False}` for all 16 motors |
| `stop()` | Stop running sequences (software only) |
| `save_calibration(path)` / `load_calibration(path)` | JSON calibration file |
| `BionicHands.gestures()` / `BionicHands.list_ports()` | Helpers |

### `Hand` (`hands.left`, `hands.right`, or `hands["left"]`)
| Method | Description |
|---|---|
| `gesture(name, ...)`, `open()`, `close()` | Named gestures |
| `set_fingers(thumb=, index=, middle=, ring=, ...)` | `True/"open"`, `False/"close"` or `0.0–1.0` |
| `move_to({id: angle}, speed, duration, motion, max_step)` | Any custom pose |
| `set_motor(id, angle, speed)` | Test one motor (marks pose unknown) |
| `wave(cycles, speed, duration)` | Close/open repeatedly |
| `read_positions()`, `confirm_open()`, `stop()` | Pose tracking / stop |

Every movement **blocks** until done and returns `True`, or `False` if `stop()`
was called from another thread.

### Motion styles (`motion=`)
| Style | From | Behaviour |
|---|---|---|
| `"direct"` | combined GUI | Send each goal once; servo moves at `speed`; wait `duration` |
| `"smooth"` | right-hand GUI (right default) | Interpolate in `max_step`° steps over `duration` |
| `"staged"` | left-hand GUI (left default) | Finger by finger with easing — fewer finger collisions |

`smooth` and `staged` need a known start pose: call `confirm_open()` (hand is
physically OPEN) or `read_positions()` first, otherwise `PoseUnknownError`.
After a stop or error the pose becomes unknown again.

### Gestures
`OPEN HAND`, `CLOSE / FIST`, `THUMBS UP`, `POINT`, `PEACE / V`,
`VICTORY / V (THUMB OPEN)`, `THREE FINGERS`, `MIDDLE FINGER CLOSED`,
`INDEX CLOSED`, `INDEX + RING`, `MIDDLE + RING` (case-insensitive).

### Running in the background
```python
import threading
t = threading.Thread(target=hands.wave, kwargs={"sequence": "both", "cycles": 5})
t.start()
...
hands.stop()
```

### Custom calibration
```python
hands.left.config.close_pose[3] = 140.0     # change one endpoint
hands.save_calibration("calibration.json")
hands = BionicHands(calibration_file="calibration.json")
```

## Safety
- Only one program may use the serial port at a time — close the GUIs / web app.
- `stop()` is **not** an emergency stop; it does not cut motor power.
- Pose tracking is the last *commanded* target, not measured feedback.
- `read_positions()` uses register 56, which the old reader GUIs assumed and
  is not yet confirmed for your firmware.
- Test new poses at low speed with the fingers clear.
