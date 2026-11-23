"""High-level ``Hand`` and ``BionicHands`` classes."""
import threading
import time

from .config import FINGER_NAMES, GESTURES, left_config, right_config, load_configs, save_configs
from .protocol import ANGLE_MAX, DEFAULT_BAUD, DEFAULT_PORT, SCSBus, list_ports

MOTION_STYLES = ("direct", "smooth", "staged")


class PoseUnknownError(RuntimeError):
    """Raised when interpolated motion is requested but the start pose is unknown."""


class BusyError(RuntimeError):
    """Raised when a hand is asked to move while it is already moving."""


def _finger_amount(value):
    """Convert a finger command to a closure fraction: 0.0 = OPEN, 1.0 = CLOSED."""
    if isinstance(value, bool):
        return 0.0 if value else 1.0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("open", "o"):
            return 0.0
        if v in ("close", "closed", "c"):
            return 1.0
        raise ValueError(f"Unknown finger state {value!r}; use 'open', 'close' or 0.0-1.0.")
    amount = float(value)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Finger closure must be between 0.0 (open) and 1.0 (closed).")
    return amount


class Hand:
    """One bionic hand: 4 fingers x 2 SCS0009 motors on a shared bus.

    Every movement method blocks until finished and returns ``True`` on
    completion or ``False`` if :meth:`stop` was called from another thread.
    ``stop()`` only ends the software sequence; it does not cut motor power.
    """

    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        self.pose = dict(config.open_pose)   # last commanded targets (not measured)
        self.pose_known = False
        self._stop = threading.Event()
        self._motion_lock = threading.Lock()

    def __repr__(self):
        return f"<Hand {self.name} ids={self.ids} pose_known={self.pose_known}>"

    @property
    def name(self):
        return self.config.name

    @property
    def ids(self):
        return self.config.ids

    @property
    def fingers(self):
        return self.config.fingers

    @property
    def is_moving(self):
        return self._motion_lock.locked()

    # -- pose tracking ----------------------------------------------------
    def confirm_open(self):
        """Tell the SDK the hand is physically at its calibrated OPEN pose.

        This only updates software tracking; it does not move or read motors.
        """
        self.pose = dict(self.config.open_pose)
        self.pose_known = True

    def read_positions(self, update_pose=True):
        """Read present angles from every motor. Returns ``{id: angle}``."""
        angles = {}
        for sid in self.ids:
            try:
                angles[sid] = self.bus.read_angle(sid)
            except Exception:
                pass
            time.sleep(0.003)
        if update_pose:
            if angles:
                self.pose.update(angles)
            self.pose_known = len(angles) == len(self.ids)
        return angles

    # -- target builders --------------------------------------------------
    def finger_target(self, finger, amount):
        """Angles for one finger at ``amount`` closure (0 = open, 1 = closed)."""
        finger = finger.capitalize()
        if finger not in self.fingers:
            raise ValueError(f"Unknown finger {finger!r}; choose from {list(self.fingers)}.")
        a = _finger_amount(amount)
        o, c = self.config.open_pose, self.config.close_pose
        return {sid: o[sid] + (c[sid] - o[sid]) * a for sid in self.fingers[finger]}

    def gesture_target(self, name):
        key = name.upper()
        if key not in GESTURES:
            raise ValueError(f"Unknown gesture {name!r}; choose from {list(GESTURES)}.")
        targets = {}
        for finger, is_open in zip(FINGER_NAMES, GESTURES[key]):
            targets.update(self.finger_target(finger, is_open))
        return targets

    # -- movement ---------------------------------------------------------
    def move_to(self, targets, speed=500, duration=2.0, motion=None, max_step=2.0):
        """Move to ``{motor_id: angle}``. Motors not listed keep their last target.

        motion:
          ``"direct"`` - send each goal once, let the servo travel (combined GUI).
          ``"smooth"`` - linear interpolation in ``max_step``-degree steps (right GUI).
          ``"staged"`` - finger by finger with ease-in/out (left GUI, fewer collisions).
        Defaults to the hand's configured style. smooth/staged need a known pose:
        call :meth:`confirm_open` or :meth:`read_positions` first.
        """
        motion = motion or self.config.motion
        if motion not in MOTION_STYLES:
            raise ValueError(f"motion must be one of {MOTION_STYLES}.")
        unknown = set(targets) - set(self.ids)
        if unknown:
            raise ValueError(f"Motor IDs {sorted(unknown)} do not belong to the {self.name} hand.")
        for sid, angle in targets.items():
            if not 0.0 <= float(angle) <= ANGLE_MAX:
                raise ValueError(f"Motor ID {sid}: angle {angle} outside 0-{ANGLE_MAX:g} degrees.")
        if motion != "direct" and not self.pose_known:
            raise PoseUnknownError(
                f"{self.name} hand pose unknown. Put it at OPEN and call confirm_open(), "
                "or call read_positions(), before using smooth/staged motion.")
        speed = max(1, min(1000, int(speed)))
        duration = max(0.0, float(duration))
        full = dict(self.pose)
        full.update({sid: float(a) for sid, a in targets.items()})

        if not self._motion_lock.acquire(blocking=False):
            raise BusyError(f"The {self.name} hand is already moving.")
        try:
            self._stop.clear()
            for sid in self.ids:
                if self._stop.is_set():
                    break
                self.bus.set_speed(sid, speed)
            if not self._stop.is_set():
                getattr(self, f"_move_{motion}")(dict(self.pose), full, duration, max_step, set(targets))
            if self._stop.is_set():
                self._hold_current_position()
                return False
            self.pose = full
            if motion == "direct" and set(targets) >= set(self.ids):
                self.pose_known = True  # every motor was sent an absolute goal
            return True
        except Exception:
            self.pose_known = False
            raise
        finally:
            self._motion_lock.release()

    def _hold_current_position(self):
        """Read actual motor positions and command servos to hold there."""
        angles = self.read_positions(update_pose=True)
        for sid, angle in angles.items():
            try:
                self.bus.set_angle(sid, angle)
            except Exception:
                pass

    def _wait(self, seconds):
        return not self._stop.wait(seconds)

    def _move_direct(self, start, target, duration, max_step, requested):
        # Only command the requested motors, so unselected fingers never move.
        for sid in (sid for sid in self.ids if sid in requested):
            if self._stop.is_set():
                return
            self.bus.set_angle(sid, target[sid])
        self._wait(duration)  # time for servos to travel; not feedback

    def _max_steps(self, duration, motors):
        """Most steps that fit in ``duration`` given the bus time per write."""
        per_step = max(1e-3, motors * getattr(self.bus, "write_delay", 0.006) * 1.5)
        return max(2, int(duration / per_step))

    def _step_through(self, ids, start, target, duration, steps, ease=False):
        """Write ``steps`` interpolated poses, each on its own deadline within ``duration``."""
        t0 = time.monotonic()
        for step in range(1, steps + 1):
            if self._stop.is_set():
                return False
            t = step / steps
            ratio = t * t * (3.0 - 2.0 * t) if ease else t
            for sid in ids:
                self.bus.set_angle(sid, start[sid] + (target[sid] - start[sid]) * ratio)
            if not self._wait(max(0.0, t0 + duration * t - time.monotonic())):
                return False
        return True

    def _move_smooth(self, start, target, duration, max_step, requested):
        moving = [sid for sid in self.ids if target[sid] != start[sid]] or self.ids
        largest = max(abs(target[sid] - start[sid]) for sid in self.ids)
        steps = max(2, min(600, int(max(1.0, largest / max(0.5, max_step)))))
        steps = min(steps, self._max_steps(duration, len(moving)))
        self._step_through(moving, start, target, duration, steps)

    def _move_staged(self, start, target, duration, max_step, requested):
        distances = {f: max(abs(target[sid] - start[sid]) for sid in self.fingers[f])
                     for f in FINGER_NAMES if f in self.fingers}
        order = [f for f, d in distances.items() if d > 0]
        if not order:
            return
        total = sum(distances.values()) or 1.0
        min_slice = min(0.6, duration / len(order))
        remaining = max(0.0, duration - min_slice * len(order))
        for finger in order:
            if self._stop.is_set():
                return
            finger_time = max(0.15, min_slice + remaining * distances[finger] / total)
            ids = self.fingers[finger]
            steps = max(4, min(30, int(finger_time * 8), self._max_steps(finger_time, len(ids))))
            if not self._step_through(ids, start, target, finger_time, steps, ease=True):
                return

    # -- convenience ------------------------------------------------------
    def gesture(self, name, **kwargs):
        """Run a named gesture, e.g. ``hand.gesture("thumbs up")``."""
        return self.move_to(self.gesture_target(name), **kwargs)

    def open(self, **kwargs):
        return self.gesture("OPEN HAND", **kwargs)

    def close(self, **kwargs):
        return self.gesture("CLOSE / FIST", **kwargs)

    def set_fingers(self, thumb=None, index=None, middle=None, ring=None, **kwargs):
        """Set any fingers: ``True``/``"open"``, ``False``/``"close"`` or 0.0-1.0 closure.

        Fingers left as ``None`` keep their current target.
        """
        targets = {}
        for finger, value in (("Thumb", thumb), ("Index", index), ("Middle", middle), ("Ring", ring)):
            if value is not None:
                targets.update(self.finger_target(finger, value))
        return self.move_to(targets, **kwargs)

    def set_motor(self, servo_id, angle, speed=500):
        """Send one motor to ``angle`` directly (diagnostic). Marks the pose unknown."""
        if servo_id not in self.ids:
            raise ValueError(f"Motor ID {servo_id} is not on the {self.name} hand.")
        if not 0.0 <= float(angle) <= ANGLE_MAX:
            raise ValueError(f"Angle must be 0-{ANGLE_MAX:g} degrees.")
        self.bus.set_speed(servo_id, speed)
        time.sleep(0.03)
        self.bus.set_angle(servo_id, angle)
        self.pose[servo_id] = float(angle)
        self.pose_known = False

    def wave(self, cycles=2, speed=500, duration=3.0, motion="direct"):
        """Close and reopen the whole hand ``cycles`` times (no wrist rotation)."""
        close_t, open_t = self.gesture_target("CLOSE / FIST"), self.gesture_target("OPEN HAND")
        for _ in range(max(1, int(cycles))):
            if not self.move_to(close_t, speed=speed, duration=duration / 2, motion=motion):
                return False
            if not self.move_to(open_t, speed=speed, duration=duration / 2, motion=motion):
                return False
        return True

    def stop(self):
        """Stop the running sequence (software only, not an emergency stop)."""
        self._stop.set()


class BionicHands:
    """Controller for both hands on one shared serial adapter.

    Example::

        from bionic_hand import BionicHands

        with BionicHands("/dev/ttyACM0") as hands:
            hands.confirm_open()               # both hands are physically OPEN
            hands.gesture("thumbs up")         # both hands
            hands.left.gesture("point")        # one hand
            hands.right.set_fingers(index=0.5) # half-close right index
            hands.wave("left_to_right", cycles=2)
    """

    WAVE_SEQUENCES = ("left", "right", "left_to_right", "right_to_left", "both")

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, calibration_file=None):
        self.bus = SCSBus(port, baud)
        self.left = Hand(self.bus, left_config())
        self.right = Hand(self.bus, right_config())
        if calibration_file:
            self.load_calibration(calibration_file)

    def __repr__(self):
        state = "connected" if self.connected else "disconnected"
        return f"<BionicHands {self.bus.port} {state}>"

    def __getitem__(self, name):
        return self._select(name)[0]

    @property
    def hands(self):
        return {"left": self.left, "right": self.right}

    @staticmethod
    def list_ports():
        return list_ports()

    @staticmethod
    def gestures():
        return list(GESTURES)

    # -- connection -------------------------------------------------------
    @property
    def connected(self):
        return self.bus.connected

    def connect(self, port=None, baud=None):
        self.bus.connect(port, baud)
        return self

    def disconnect(self):
        self.stop()
        self.bus.disconnect()
        for hand in self.hands.values():
            hand.pose_known = False

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # -- helpers ----------------------------------------------------------
    def _select(self, hand):
        if hand in ("both", None):
            return [self.left, self.right]
        if hand in ("left", "right"):
            return [self.hands[hand]]
        raise ValueError("hand must be 'left', 'right' or 'both'.")

    @staticmethod
    def _parallel(calls):
        """Run ``(fn, args, kwargs)`` calls in threads; re-raise the first error."""
        results, errors = [None] * len(calls), []

        def run(i, fn, args, kwargs):
            try:
                results[i] = fn(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i, *call), daemon=True) for i, call in enumerate(calls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return all(results)

    # -- commands ---------------------------------------------------------
    def confirm_open(self, hand="both"):
        for h in self._select(hand):
            h.confirm_open()

    def read_positions(self, hand="both"):
        positions = {}
        for h in self._select(hand):
            positions.update(h.read_positions())
        return positions

    def ping_all(self):
        """Return ``{motor_id: True/False}`` for all 16 motors."""
        result = {}
        for h in self.hands.values():
            for sid in h.ids:
                try:
                    result[sid] = self.bus.ping(sid)
                except Exception:
                    result[sid] = False
                time.sleep(0.003)
        return result

    def gesture(self, name, hand="both", parallel=False, **kwargs):
        """Run a gesture on one or both hands (sequentially unless ``parallel``)."""
        selected = self._select(hand)
        if parallel and len(selected) > 1:
            return self._parallel([(h.gesture, (name,), kwargs) for h in selected])
        return all(h.gesture(name, **kwargs) for h in selected)

    def open(self, hand="both", **kwargs):
        return self.gesture("OPEN HAND", hand, **kwargs)

    def close(self, hand="both", **kwargs):
        return self.gesture("CLOSE / FIST", hand, **kwargs)

    def wave(self, sequence="both", **kwargs):
        """Wave sequences: left, right, left_to_right, right_to_left, both (simultaneous)."""
        if sequence not in self.WAVE_SEQUENCES:
            raise ValueError(f"sequence must be one of {self.WAVE_SEQUENCES}.")
        if sequence == "both":
            return self._parallel([(h.wave, (), kwargs) for h in (self.left, self.right)])
        order = {"left": [self.left], "right": [self.right],
                 "left_to_right": [self.left, self.right],
                 "right_to_left": [self.right, self.left]}[sequence]
        return all(h.wave(**kwargs) for h in order)

    def stop(self):
        for h in self.hands.values():
            h.stop()

    # -- calibration ------------------------------------------------------
    def save_calibration(self, path):
        save_configs(path, [self.left.config, self.right.config])

    def load_calibration(self, path):
        load_configs(path, [self.left.config, self.right.config])
        for h in self.hands.values():
            h.pose = dict(h.config.open_pose)
            h.pose_known = False
