#!/usr/bin/env python3
"""drax_can_watchdog - independent safety watchdog for the Drax CANopen drives.

Runs outside ROS (stdlib only) so it keeps working when ros2_control_node crashes,
hangs, is killed, or its hardware component drops into ERROR.

1. Stall stop: daksha_hardware talks SDO to every drive each control cycle (~150 req/s
   per node at 50 Hz, also while torque is disabled). When those requests have been
   flowing and then stop for --stall seconds, the watchdog sends zero speed to every
   node (and optionally disables them). It re-arms once traffic resumes.
2. CAN link: keeps the interface UP with the right bitrate and a non-zero restart-ms,
   so a bus-off recovers automatically. Needs root (run it as a systemd service).

Usage:  sudo drax_can_watchdog.py --iface can0 --bitrate 500000 --nodes 1 2
"""

import argparse
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import time

CAN_FMT = '=IB3x8s'
SDO_TX = 0x600  # host -> drive
SDO_RX = 0x580  # drive -> host

log = logging.getLogger('drax_can_watchdog')


def sdo_write(node, index, sub, value, size):
    cmd = {1: 0x2F, 2: 0x2B, 4: 0x23}[size]
    data = bytes([cmd, index & 0xFF, index >> 8, sub]) + value.to_bytes(4, 'little', signed=True)
    return struct.pack(CAN_FMT, SDO_TX + node, 8, data)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=5)


class Watchdog:

    def __init__(self, args):
        self.a = args
        self.sock = None
        self.last_traffic = None     # monotonic time of last host SDO request seen
        self.armed = False           # traffic seen since the last stop
        # Listen first, so a busy link is recognised before any link repair is considered.
        self.next_link_check = time.monotonic() + 1.5
        self.deferred_warned = False

    # ---------------------------------------------------------------- CAN link
    def link_info(self):
        r = run(['ip', '-details', '-json', 'link', 'show', self.a.iface])
        if r.returncode != 0:
            return None
        info = json.loads(r.stdout)[0]
        data = info.get('linkinfo', {}).get('info_data', {})
        return {
            'up': 'UP' in info.get('flags', []),
            'state': data.get('state', '?'),
            'bitrate': data.get('bittiming', {}).get('bitrate'),
            'restart_ms': data.get('restart_ms', 0),
        }

    def check_link(self):
        li = self.link_info()
        if li is None:
            log.error('CAN interface %s not found (driver/cable?)', self.a.iface)
            return False
        bad = []
        if not li['up']:
            bad.append('DOWN')
        if li['state'] == 'BUS-OFF':
            bad.append('BUS-OFF')
        if self.a.bitrate and li['bitrate'] not in (None, self.a.bitrate):
            bad.append('bitrate %s' % li['bitrate'])
        if self.a.restart_ms and li['restart_ms'] == 0:
            bad.append('restart-ms 0')
        if not bad:
            return True
        broken = li['state'] == 'BUS-OFF' or not li['up']
        if not broken and self.armed:
            # Link works and the controller is using it: never bounce it under a
            # running robot just to fix settings. Applied once the traffic stops.
            if not self.deferred_warned:
                log.warning('CAN %s: %s (deferred until controller traffic stops)',
                            self.a.iface, ', '.join(bad))
                self.deferred_warned = True
            return True
        if self.a.no_link_fix:
            log.warning('CAN %s: %s (link fix disabled)', self.a.iface, ', '.join(bad))
            return li['up']
        log.warning('CAN %s: %s -> reconfiguring', self.a.iface, ', '.join(bad))
        cfg = ['type', 'can']
        if self.a.bitrate:
            cfg += ['bitrate', str(self.a.bitrate)]
        if self.a.restart_ms:
            cfg += ['restart-ms', str(self.a.restart_ms)]
        for cmd in (['ip', 'link', 'set', self.a.iface, 'down'],
                    ['ip', 'link', 'set', self.a.iface] + cfg,
                    ['ip', 'link', 'set', self.a.iface, 'up']):
            r = run(cmd)
            if r.returncode != 0:
                log.error('%s failed: %s', ' '.join(cmd), r.stderr.strip())
                return False
        self.close_socket()
        self.deferred_warned = False
        log.info('CAN %s is up again', self.a.iface)
        return True

    # ------------------------------------------------------------------ socket
    def open_socket(self):
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((self.a.iface,))
        s.settimeout(0.05)
        self.sock = s

    def close_socket(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    # ------------------------------------------------------------------ stop
    def transact(self, frame, node, index, sub, timeout=0.05):
        """Send one SDO request, wait for the matching response. True if acknowledged."""
        self.sock.send(frame)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                cid, _, d = struct.unpack(CAN_FMT, self.sock.recv(16))
            except socket.timeout:
                continue
            if (cid & 0x7FF) == SDO_RX + node and d[1] | d[2] << 8 == index and d[3] == sub:
                return d[0] == 0x60
        return False

    def stop_drives(self, reason):
        log.warning('%s -> sending zero speed to nodes %s', reason, self.a.nodes)
        steps = [(0x2FF0, 0x09, 0, 2), (0x60FF, 0x00, 0, 4)]
        if self.a.disable:
            steps.append((0x6040, 0x00, 0x06, 2))
        for node in self.a.nodes:
            results = []
            for index, sub, value, size in steps:
                ok = any(self.transact(sdo_write(node, index, sub, value, size), node, index, sub)
                         for _ in range(3))
                results.append('%04X:%02X=%s' % (index, sub, 'ok' if ok else 'NO ACK'))
            level = logging.INFO if all(r.endswith('ok') for r in results) else logging.ERROR
            log.log(level, 'node %d: %s', node, ', '.join(results))

    # ------------------------------------------------------------------ loop
    def loop(self):
        requests = {SDO_TX + n for n in self.a.nodes}
        while True:
            now = time.monotonic()
            if now >= self.next_link_check:
                self.next_link_check = now + self.a.link_check_period
                if not self.check_link():
                    self.close_socket()
                    time.sleep(1.0)
                    continue
            if self.sock is None:
                try:
                    self.open_socket()
                except OSError as e:
                    log.error('cannot open CAN socket on %s: %s', self.a.iface, e)
                    time.sleep(1.0)
                    continue
            try:
                cid, _, _ = struct.unpack(CAN_FMT, self.sock.recv(16))
                if (cid & 0x7FF) in requests:
                    if not self.armed:
                        log.info('controller traffic detected, watchdog armed')
                    self.armed = True
                    self.last_traffic = time.monotonic()
            except socket.timeout:
                pass
            except OSError as e:  # ENETDOWN etc.
                log.error('CAN socket error: %s', e)
                self.close_socket()
                self.next_link_check = 0.0
                continue

            if self.armed and time.monotonic() - self.last_traffic > self.a.stall:
                self.armed = False
                self.stop_drives('no controller CAN traffic for %.2fs' % self.a.stall)
                if self.a.on_stall:
                    log.info('running --on-stall: %s', self.a.on_stall)
                    subprocess.Popen(self.a.on_stall, shell=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--iface', default='can0')
    p.add_argument('--bitrate', type=int, default=500000, help='0 = do not manage bitrate')
    p.add_argument('--restart-ms', type=int, default=100, help='bus-off auto restart, 0 = leave')
    p.add_argument('--nodes', type=int, nargs='+', default=[1, 2])
    p.add_argument('--stall', type=float, default=0.5, help='seconds without controller SDO traffic')
    p.add_argument('--disable', action='store_true',
                   help='also disable the drives (0x6040=0x06, wheels free-wheel) after zero speed')
    p.add_argument('--no-link-fix', action='store_true', help='only report CAN link problems')
    p.add_argument('--link-check-period', type=float, default=1.0)
    p.add_argument('--on-stall', default='', help='shell command to run after a stall stop')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if os.geteuid() != 0 and not args.no_link_fix:
        log.warning('not root: CAN link repair will fail (use --no-link-fix or run as root)')
    log.info('watching %s nodes=%s stall=%.2fs bitrate=%s restart-ms=%s disable=%s',
             args.iface, args.nodes, args.stall, args.bitrate, args.restart_ms, args.disable)
    try:
        Watchdog(args).loop()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
