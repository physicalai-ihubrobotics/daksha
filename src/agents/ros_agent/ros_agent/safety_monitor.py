"""Safety monitor for the real Daksha: reads ROS topics / pipeline state once a second and raises alerts.

Each check returns alerts {key, level: 'warn' | 'critical', msg}. The pipeline node shows them on
the page (banner + popup + voice), posts each new alert to the chat once a minute (and an all-clear
when it goes away), gives them to the agent in daksha_status, refuses motion while a CRITICAL alert
is active, and stops a running motion when a person is seen while the robot executes (human_stop).

  temperature   motor >= motor_temp_warn_c (warn) / >= motor_temp_stop_c (critical)
  motor fault   a motor whose status is not "Enabled"                       (critical)
  collision     self-collision in the live pose (URDF collision pairs)      (critical)
                TCPs of the two arms closer than arms_min_dist_m             (warn)
                a TCP near / below the table top                             (warn / critical)
                aruco_reach stopped an arm: "not following ... colliding?"   (critical)
  joint limits  a joint within joint_limit_margin_deg of its limit           (warn)
  human         a person in the world camera (OpenCV HOG people detector)    (critical)
  battery       /battery_info % below battery_warn_pct / battery_stop_pct    (warn / critical)
  data          /joint_states or the world camera stale; /diagnostics ERROR  (critical / warn)
  gripper       a held object seems dropped                                  (warn)
"""

import json
import time

import numpy as np


class SafetyMonitor:
    def __init__(self, node, cfg):
        self.n = node
        c = cfg.get('safety', {}) or {}
        self.c = {
            'arms_min_dist_m': float(c.get('arms_min_dist_m', 0.10)),
            'table_margin_m': float(c.get('table_margin_m', 0.02)),
            'joint_limit_margin_deg': float(c.get('joint_limit_margin_deg', 3.0)),
            'battery_warn_pct': float(c.get('battery_warn_pct', 20.0)),
            'battery_stop_pct': float(c.get('battery_stop_pct', 10.0)),
            'human_detect': bool(c.get('human_detect', True)),
            'human_stop': bool(c.get('human_stop', True)),
            'human_min_height_frac': float(c.get('human_min_height_frac', 0.25)),
            'alert_repeat_s': float(c.get('alert_repeat_s', 60.0)),
            # operator choice: alerts are only shown (popups), they never block or stop motion
            # (the motor over-temperature stop at motor_temp_stop_c and the STOP button still apply)
            'ignore_warnings': bool(c.get('ignore_warnings', False)),
        }
        self._t0 = time.monotonic()
        self.alerts = []                 # current alerts
        self.events, self._seq = [], 0   # popups for the page
        self._last_said = {}             # key -> time an alert was posted to the chat
        self.battery_pct = None
        self.diag = {}                   # name -> (level, message, t)
        self.reach_msgs = {}             # arm -> last aruco_reach message seen
        self._reach_seen = set()
        self._hog = None
        self._people = {'t': 0.0, 'boxes': []}
        self._collision = None

    # ------------------------------------------------------------------ inputs (ROS callbacks)
    def on_battery(self, msg):
        self.battery_pct = float(msg.data)

    def on_diagnostics(self, msg):
        now = time.monotonic()
        for s in msg.status:
            lvl = s.level if isinstance(s.level, int) else (s.level[0] if s.level else 0)
            self.diag[s.name] = (int(lvl), s.message.strip(), now)

    def on_reach_status(self, msg):
        try:
            st = json.loads(msg.data)
        except ValueError:
            return
        for arm, a in (st.get('arms') or {}).items():
            m = a.get('last_msg') or ''
            if m and m != self.reach_msgs.get(arm, ('', 0))[0]:      # only a NEW message counts
                # the message aruco_reach already shows when the monitor starts is history, not an event
                self.reach_msgs[arm] = (m, time.monotonic() if arm in self._reach_seen else 0.0)
            self._reach_seen.add(arm)

    # ------------------------------------------------------------------ checks
    def check(self):
        out = []
        for fn in (self._temperature, self._motor_faults, self._collision_checks, self._joint_limits,
                   self._human, self._battery, self._data, self._gripper):
            try:
                out += fn()
            except Exception as ex:                       # one broken check never hides the others
                out.append({'key': f'monitor:{fn.__name__}', 'level': 'warn',
                            'msg': f'safety check {fn.__name__} failed: {ex}'})
        self.alerts = out
        return out

    @property
    def critical(self):
        return [a for a in self.alerts if a['level'] == 'critical']

    def _temperature(self):
        ms = self.n.motor_state() if hasattr(self.n, 'motor_state') else {}
        out = []
        for h in ms.get('hot', []):
            crit = h['temp_c'] >= ms['stop_c']
            out.append({'key': f"temp:{h['arm']}:{h['id']}", 'level': 'critical' if crit else 'warn',
                        'msg': f"{h['arm']} arm motor {h['id']} at {h['temp_c']:.0f} °C (limit {ms['warn_c']:.0f} °C)"
                               + ('; motion stopped until it cools' if crit else '; let it cool before long tasks')})
        return out

    def _motor_faults(self):
        out = []
        now = time.monotonic()
        for arm, d in getattr(self.n, 'motors', {}).items():
            if now - d['t'] > 3.0:
                continue
            for m in d['motors']:
                if m.get('error') and m['error'] != 'Enabled':
                    out.append({'key': f"fault:{arm}:{m['id']}", 'level': 'critical',
                                'msg': f"{arm} arm motor {m['id']} reports {m['error']}"})
        return out

    def _collision_checks(self):
        out = []
        n = self.n
        if time.monotonic() - n.joint_t > 0.5:
            return out
        joints = dict(n.joints)
        # self-collision in the live pose
        if self._collision is None:
            try:
                from aruco_hand_eye.aruco_reach import CollisionChecker
                self._collision = CollisionChecker(n.urdf_path)
            except Exception:
                self._collision = False
        if self._collision:
            hit = self._collision.check(joints)
            if hit:
                out.append({'key': 'collision:self', 'level': 'critical', 'msg': f'self-collision now: {hit}'})
        # the two hands close to each other / a hand at the table
        tcp = {}
        for arm in ('left', 'right'):
            try:
                tcp[arm] = np.array(n.fk.transform('world', f'{arm}_tcp', joints)[:3, 3])
            except Exception:
                pass
        if len(tcp) == 2:
            d = float(np.linalg.norm(tcp['left'] - tcp['right']))
            if d < self.c['arms_min_dist_m']:
                out.append({'key': 'collision:arms', 'level': 'warn',
                            'msg': f'the two hands are {d * 100:.0f} cm apart (collision risk)'})
        tz = n.table_z
        if tz is not None:
            for arm, p in tcp.items():
                if p[2] < tz + self.c['table_margin_m'] and 0.15 < p[0] < 1.0 and abs(p[1]) < 0.8:
                    below = p[2] < tz - 0.005
                    out.append({'key': f'collision:table:{arm}', 'level': 'critical' if below else 'warn',
                                'msg': f'{arm} hand {"below" if below else "at"} the table top '
                                       f'({(p[2] - tz) * 100:+.1f} cm)'})
        # aruco_reach stopped an arm because it did not follow (blocked / contact)
        now = time.monotonic()
        for arm, (m, t) in self.reach_msgs.items():
            if 'STOPPED' in m and now - t < 30.0:
                out.append({'key': f'collision:blocked:{arm}', 'level': 'critical',
                            'msg': f'{arm} arm was stopped: {m[:140]}'})
        return out

    def _joint_limits(self):
        out = []
        n = self.n
        if time.monotonic() - n.joint_t > 0.5 or self._collision in (None, False):
            return out
        margin = np.radians(self.c['joint_limit_margin_deg'])
        robot = self._collision.robot
        for name, v in n.joints.items():
            if not name.startswith(('left_joint_', 'right_joint_')):
                continue
            try:
                lo, hi = robot.get_joint_limits(name)
            except Exception:
                continue
            if v < lo + margin or v > hi - margin:
                out.append({'key': f'limit:{name}', 'level': 'info',     # the home pose sits at j1's limit
                            'msg': f'{name} at {np.degrees(v):.0f}° is within {self.c["joint_limit_margin_deg"]:.0f}° '
                                   f'of its limit [{np.degrees(lo):.0f}°, {np.degrees(hi):.0f}°]'})
        return out

    def _human(self):
        if not self.c['human_detect']:
            return []
        now = time.monotonic()
        if now - self._people['t'] >= 1.0:              # ~1 Hz: cheap enough on the CPU
            self._people['t'] = now
            snap = self.n.camera_snapshot()
            self._people['boxes'] = self._detect_people(snap['bgr']) if snap is not None else []
        if not self._people['boxes']:
            return []
        return [{'key': 'human', 'level': 'critical',
                 'msg': f'person near the robot ({len(self._people["boxes"])} in the world camera)'
                        + (': motion stopped' if self.c['human_stop'] else '')}]

    def _detect_people(self, bgr):
        import cv2
        if self._hog is None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        h = bgr.shape[0]
        small = cv2.resize(bgr, (bgr.shape[1] // 2, h // 2))
        rects, weights = self._hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes = []
        for (x, y, w, hh), wt in zip(rects, np.ravel(weights) if len(rects) else []):
            if float(wt) > 0.5 and hh * 2 >= self.c['human_min_height_frac'] * h:
                boxes.append([int(x * 2), int(y * 2), int((x + w) * 2), int((y + hh) * 2)])
        return boxes

    def _battery(self):
        b = self.battery_pct
        if b is None:
            return []
        if b < self.c['battery_stop_pct']:
            return [{'key': 'battery', 'level': 'critical', 'msg': f'battery at {b:.0f} %: charge now'}]
        if b < self.c['battery_warn_pct']:
            return [{'key': 'battery', 'level': 'warn', 'msg': f'battery low: {b:.0f} %'}]
        return []

    def _data(self):
        out = []
        n = self.n
        now = time.monotonic()
        if now - self._t0 < 5.0:                        # startup: topics are still connecting
            return out
        if now - n.joint_t > 1.0:
            out.append({'key': 'data:joints', 'level': 'critical', 'msg': 'no /joint_states for over 1 s'})
        if n.camera_snapshot() is None:
            out.append({'key': 'data:camera', 'level': 'warn', 'msg': 'no fresh world camera image'})
        for name, (lvl, m, t) in self.diag.items():
            if lvl >= 2 and now - t < 10.0:                 # ERROR (2): timing jitter is routine -> info; STALE (3) -> warn
                out.append({'key': f'diag:{name}', 'level': 'warn' if lvl >= 3 else 'info', 'msg': f'{name}: {m[:120]}'})
        return out

    def _gripper(self):
        out = []
        held = (self.n.agent.snapshot().get('held') if self.n.agent is not None else {}) or {}
        for arm, g in self.n.gripper_feedback(held).items():
            if g['status'] == 'dropped?':
                out.append({'key': f'gripper:{arm}', 'level': 'warn',
                            'msg': f'{arm} gripper closed on nothing: the {g["object"]} may have dropped'})
        return out

    # ------------------------------------------------------------------ announcing
    def to_announce(self):
        """New / repeating alerts to post to the chat, and alerts that cleared since."""
        now = time.monotonic()
        new = [a for a in self.alerts if a['level'] != 'info'
               and now - self._last_said.get(a['key'], -1e9) >= self.c['alert_repeat_s']]
        for a in new:
            self._last_said[a['key']] = now
        active = {a['key'] for a in self.alerts if a['level'] != 'info'}
        cleared = [k for k in list(self._last_said) if k not in active]
        for k in cleared:
            del self._last_said[k]
        return new, cleared

    def event(self, level, msg, key=''):
        """A popup for the page (newest last; the page shows each id once)."""
        self._seq += 1
        self.events.append({'id': self._seq, 'level': level, 'msg': msg, 'key': key, 't': time.time()})
        del self.events[:-20]

    def state(self):
        return {'alerts': self.alerts, 'critical': bool(self.critical), 'events': list(self.events),
                'human_boxes': self._people['boxes'], 'battery_pct': self.battery_pct}
