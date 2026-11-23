"""FEETECH SCS/SCSCL packet helpers and the shared serial bus for SCS0009 servos."""
import threading
import time

import serial
import serial.tools.list_ports

DEFAULT_PORT = "auto"
DEFAULT_BAUD = 1_000_000

HEADER = b"\xff\xff"
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03

REG_GOAL_POSITION = 42
REG_GOAL_SPEED = 46
REG_PRESENT_POSITION = 56  # Assumed by the existing reader GUIs; confirm for your firmware.

POSITION_MAX = 1023
ANGLE_MAX = 300.0
SPEED_MAX = 1000


class BusError(RuntimeError):
    """Raised for serial-bus or servo communication failures."""


def checksum(data):
    return (~sum(data)) & 0xFF


def make_packet(servo_id, instruction, params=()):
    body = [servo_id, len(params) + 2, instruction] + list(params)
    return HEADER + bytes(body + [checksum(body)])


def angle_to_raw(angle):
    return max(0, min(POSITION_MAX, round(float(angle) / ANGLE_MAX * POSITION_MAX)))


def raw_to_angle(raw):
    return float(raw) / POSITION_MAX * ANGLE_MAX


BIONIC_HAND_VID_PID = (0x1A86, 0x55D3)  # QinHeng Electronics USB Single Serial


def list_ports():
    """Return the device names of all serial ports on this machine."""
    return [p.device for p in serial.tools.list_ports.comports()]


def find_bionic_hand_port():
    """Auto-detect the bionic hand serial port by USB vendor/product ID.

    Returns the device path (e.g. '/dev/ttyACM1') or None.
    """
    vid, pid = BIONIC_HAND_VID_PID
    for p in serial.tools.list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    return None


class SCSBus:
    """Thread-safe connection to one shared SCS servo bus.

    All 16 motors (left IDs 1-8, right IDs 9-16) sit on the same bus, so both
    hands must share a single ``SCSBus`` instance.
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, write_delay=0.006):
        self.port = port
        self.baud = baud
        self.write_delay = write_delay
        self.ser = None
        self.lock = threading.Lock()

    # -- connection -------------------------------------------------------
    @property
    def connected(self):
        return bool(self.ser and self.ser.is_open)

    def connect(self, port=None, baud=None):
        self.disconnect()
        self.port = port or self.port
        self.baud = baud or self.baud
        if self.port == "auto":
            detected = find_bionic_hand_port()
            if detected:
                self.port = detected
            else:
                raise BusError("Bionic hand not detected on USB. Check connection.")
        if str(self.port).startswith("sim://"):
            from .simulator import SimulatedSerial
            self.ser = SimulatedSerial()
            return self
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baud,
                                     timeout=0.05, write_timeout=0.25)
        except serial.SerialException as exc:
            raise BusError(f"Cannot open {self.port}: {exc}") from exc
        time.sleep(0.15)
        self.ser.reset_input_buffer()
        return self

    def disconnect(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    def _require(self):
        if not self.connected:
            raise BusError("Not connected. Call connect() first.")

    # -- low level --------------------------------------------------------
    def write_u16(self, servo_id, address, value):
        self._require()
        value = max(0, min(65535, int(value)))
        packet = make_packet(servo_id, INST_WRITE, [address, *value.to_bytes(2, "big")])
        with self.lock:
            # Clear stale input, write one packet, flush, let the bus settle.
            self.ser.reset_input_buffer()
            self.ser.write(packet)
            self.ser.flush()
            time.sleep(self.write_delay)

    def _read_status(self, servo_id, timeout):
        """Read one status packet (FF FF ID LEN ERR PARAMS.. CHK) for ``servo_id``."""
        end = time.monotonic() + timeout
        buf = bytearray()
        while time.monotonic() < end:
            b = self.ser.read(1)
            if not b:
                continue
            buf.extend(b)
            while len(buf) >= 2 and buf[:2] != HEADER:
                del buf[0]
            if len(buf) >= 4:
                length = buf[3]
                total = length + 4
                if length >= 2 and len(buf) >= total:
                    candidate = bytes(buf[:total])
                    if checksum(list(candidate[2:-1])) == candidate[-1] and candidate[2] == servo_id:
                        return candidate
                    del buf[0]
        return None

    def read_register(self, servo_id, address, size=2, timeout=0.15):
        self._require()
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(make_packet(servo_id, INST_READ, [address, size]))
            self.ser.flush()
            response = self._read_status(servo_id, timeout)
        if response is None:
            raise BusError(f"No valid response from motor ID {servo_id}.")
        if response[4] != 0:
            raise BusError(f"Motor ID {servo_id} returned status error {response[4]}.")
        params = response[5:-1]
        if len(params) < size:
            raise BusError(f"Short read response from motor ID {servo_id}.")
        return bytes(params[:size])

    def ping(self, servo_id, timeout=0.1):
        """Return True if the servo answers a PING."""
        self._require()
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(make_packet(servo_id, INST_PING))
            self.ser.flush()
            return self._read_status(servo_id, timeout) is not None

    # -- servo helpers ----------------------------------------------------
    def set_speed(self, servo_id, speed):
        self.write_u16(servo_id, REG_GOAL_SPEED, max(1, min(SPEED_MAX, int(speed))))

    def set_angle(self, servo_id, angle):
        self.write_u16(servo_id, REG_GOAL_POSITION, angle_to_raw(angle))

    def read_angle(self, servo_id):
        raw = int.from_bytes(self.read_register(servo_id, REG_PRESENT_POSITION, 2), "big")
        if raw > POSITION_MAX:
            raise BusError(f"Motor ID {servo_id} returned unexpected raw position {raw}.")
        return raw_to_angle(raw)
