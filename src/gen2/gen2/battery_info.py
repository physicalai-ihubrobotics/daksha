#!/usr/bin/env python3
"""
Smart BMS -> ROS 2 node (Waveshare USB-CAN-A, serial)

Self-contained: the serial/CAN/BMS protocol is included in this one file,
nothing is imported from bms_serial.py.

Reads the BMS through the Waveshare USB-CAN-A adapter and publishes:

    battery_info    std_msgs/Float32   battery percentage (SOC, %)
    battery_health  std_msgs/String    full human-readable pack state

If the adapter is unplugged or the BMS stops answering, the node stays alive
and publishes 50.0 on battery_info and a "disconnected: ..." message on
battery_health, retrying the connection in the background.

Protocol: BMS CAN, 250 kbit/s, 29-bit extended IDs
    Host -> BMS : 0x18 <cmd> 01 40   e.g. 0x18900140
    BMS  -> Host: 0x18 <cmd> 40 01   e.g. 0x18904001

Waveshare variable-length serial frame:
    0xAA | info | ID (2 bytes std / 4 bytes ext, LSB first) | data | 0x55
    info = 0xC0 | (extended << 5) | (remote << 4) | DLC

Usage:
    ./bms_ros2_node.py
    ./bms_ros2_node.py --ros-args -p port:=/dev/ttyUSB0 -p bitrate:=250000
"""

import time

import rclpy
import serial
from rclpy.node import Node
from serial.tools import list_ports
from std_msgs.msg import Float32, String

# --------------------------------------------------------------------------
# BMS / Waveshare protocol constants
# --------------------------------------------------------------------------

BMS_ADDRESS = 0x01
HOST_ADDRESS = 0x40

CELL_COUNT = 8

# Standard BMS read command IDs
COMMANDS = [
    0x90,  # Pack voltage/current/SOC
    0x91,  # Maximum/minimum cell voltage
    0x92,  # Maximum/minimum temperature
    0x93,  # Charge/discharge MOS status
    0x94,  # BMS information
    0x95,  # Individual cell voltages
    0x96,  # Temperature sensors
    0x98,  # Fault status
]

# Waveshare adapter speed codes
CAN_SPEEDS = {
    1000000: 0x01, 800000: 0x02, 500000: 0x03, 400000: 0x04,
    250000: 0x05, 200000: 0x06, 125000: 0x07, 100000: 0x08,
    50000: 0x09, 20000: 0x0A, 10000: 0x0B, 5000: 0x0C,
}
FRAME_EXTENDED = 0x02
MODE_NORMAL = 0x00

# Waveshare USB-CAN-A adapter uses a CH340 serial chip.
ADAPTER_VID = 0x1A86
ADAPTER_PID = 0x7523

# Node timing / behaviour
REQUEST_INTERVAL = 0.15  # seconds between polling each command
POLL_PERIOD = 0.02       # how often the poll timer fires
PUBLISH_PERIOD = 0.1     # how often we publish the topics (10 Hz)
STALE_TIMEOUT = 3.0      # no fresh BMS data within this -> treated as disconnected
RECONNECT_PERIOD = 2.0   # how often to retry opening the adapter

# Published when the adapter is missing or the BMS is not answering.
FALLBACK_PCT = 50.0


# --------------------------------------------------------------------------
# Waveshare USB-CAN-A serial protocol
# --------------------------------------------------------------------------

class WaveshareCan:
    def __init__(self, port, baud, bitrate):
        if bitrate not in CAN_SPEEDS:
            raise ValueError(f"unsupported bitrate {bitrate}")
        self.ser = serial.Serial(
            port,
            baudrate=baud,
            timeout=0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO,
        )
        self.buf = bytearray()
        self.configure(bitrate)

    def configure(self, bitrate):
        """Send the 20-byte settings frame: CAN speed, frame type, mode."""
        cmd = bytearray(20)
        cmd[0] = 0xAA
        cmd[1] = 0x55
        cmd[2] = 0x12
        cmd[3] = CAN_SPEEDS[bitrate]
        cmd[4] = FRAME_EXTENDED
        # bytes 5..12: filter ID / mask ID = 0 (accept everything)
        cmd[13] = MODE_NORMAL
        cmd[14] = 0x01
        cmd[19] = sum(cmd[2:19]) & 0xFF
        self.ser.reset_input_buffer()
        self.ser.write(bytes(cmd))
        self.ser.flush()
        time.sleep(0.1)  # let the adapter apply the settings
        self.ser.reset_input_buffer()

    def send(self, can_id, data, extended=True):
        data = bytes(data)
        info = 0xC0 | (0x20 if extended else 0) | len(data)
        id_len = 4 if extended else 2
        frame = bytes([0xAA, info]) + can_id.to_bytes(id_len, "little") \
            + data + bytes([0x55])
        self.ser.write(frame)

    def recv(self):
        """Return a list of (can_id, extended, data) for all complete frames."""
        waiting = self.ser.in_waiting
        if waiting:
            self.buf += self.ser.read(waiting)

        frames = []
        while True:
            start = self.buf.find(0xAA)
            if start < 0:
                self.buf.clear()
                break
            del self.buf[:start]
            if len(self.buf) < 2:
                break

            info = self.buf[1]
            if (info & 0xC0) != 0xC0 or (info & 0x0F) > 8:
                del self.buf[0]  # not a data frame header, resync
                continue

            extended = bool(info & 0x20)
            dlc = info & 0x0F
            id_len = 4 if extended else 2
            total = 2 + id_len + dlc + 1
            if len(self.buf) < total:
                break

            if self.buf[total - 1] != 0x55:
                del self.buf[0]
                continue

            can_id = int.from_bytes(self.buf[2:2 + id_len], "little")
            data = bytes(self.buf[2 + id_len:2 + id_len + dlc])
            del self.buf[:total]

            if not info & 0x10:  # skip remote frames
                frames.append((can_id, extended, data))
        return frames

    def close(self):
        self.ser.close()


def find_port(vid=ADAPTER_VID, pid=ADAPTER_PID):
    """Return the tty of the first serial device matching vid:pid, or None."""
    for p in list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    return None


# --------------------------------------------------------------------------
# BMS protocol decoding
# --------------------------------------------------------------------------

def request_id(cmd):
    return 0x18000000 | (cmd << 16) | (BMS_ADDRESS << 8) | HOST_ADDRESS


def response_cmd(arbitration_id):
    """Return the command of a BMS -> host frame, or None for other frames."""
    if (arbitration_id >> 24) != 0x18:
        return None
    if (arbitration_id & 0xFFFF) != ((HOST_ADDRESS << 8) | BMS_ADDRESS):
        return None
    return (arbitration_id >> 16) & 0xFF


def u16(data, index):
    return (data[index] << 8) | data[index + 1]


class BMS:
    def __init__(self):
        self.voltage = None
        self.current = None
        self.soc = None
        self.cells = {}
        self.temperatures = {}
        self.max_cell_v = None
        self.max_cell_no = None
        self.min_cell_v = None
        self.min_cell_no = None
        self.max_temp = None
        self.min_temp = None
        self.max_temp_sensor = None
        self.min_temp_sensor = None
        self.charge_mos = None
        self.discharge_mos = None
        self.state = None
        self.remaining_mah = None
        self.cell_count = None
        self.sensor_count = None
        self.cycles = None
        self.fault_bytes = None
        self.last_update = {}

    def decode(self, cmd, data):
        if len(data) != 8:
            return

        self.last_update[cmd] = time.monotonic()

        if cmd == 0x90:
            self.voltage = round(u16(data, 0) * 0.1, 1)
            # Offset 30000; positive means charging, negative discharging.
            self.current = round((u16(data, 4) - 30000) * 0.1, 1)
            self.soc = round(u16(data, 6) * 0.1, 1)

        elif cmd == 0x91:
            self.max_cell_v = u16(data, 0) / 1000.0
            self.max_cell_no = data[2]
            self.min_cell_v = u16(data, 3) / 1000.0
            self.min_cell_no = data[5]

        elif cmd == 0x92:
            self.max_temp = data[0] - 40
            self.max_temp_sensor = data[1]
            self.min_temp = data[2] - 40
            self.min_temp_sensor = data[3]

        elif cmd == 0x93:
            self.state = {0: "idle", 1: "charging", 2: "discharging"}.get(
                data[0], data[0]
            )
            self.charge_mos = data[1]
            self.discharge_mos = data[2]
            self.remaining_mah = (
                (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
            )

        elif cmd == 0x94:
            self.cell_count = data[0]
            self.sensor_count = data[1]
            self.cycles = u16(data, 5)

        elif cmd == 0x95:
            # Byte 0 = frame number (from 1); bytes 1..6 = 3 cell voltages.
            frame = data[0]
            if frame == 0 or frame == 0xFF:
                return

            for offset in range(3):
                index = (frame - 1) * 3 + offset + 1
                if index <= CELL_COUNT:
                    millivolts = u16(data, 1 + offset * 2)
                    if 0 < millivolts < 6000:
                        self.cells[index] = millivolts / 1000.0

        elif cmd == 0x96:
            # Byte 0 = frame number (from 1); bytes 1..7 = 7 temperatures.
            frame = data[0]
            if frame == 0 or frame == 0xFF:
                return

            limit = self.sensor_count or 16
            for offset in range(7):
                index = (frame - 1) * 7 + offset + 1
                raw = data[1 + offset]
                if index <= limit and raw != 0xFF:
                    self.temperatures[index] = raw - 40

        elif cmd == 0x98:
            self.fault_bytes = bytes(data)


# --------------------------------------------------------------------------
# ROS 2 node
# --------------------------------------------------------------------------

class BatteryInfo(Node):

    def __init__(self):
        super().__init__("battery_info")

        # Parameters (overridable with --ros-args -p name:=value).
        # port "auto" -> locate the adapter by its USB VID:PID (1a86:7523);
        # set an explicit path (e.g. /dev/ttyUSB0) to override.
        self.declare_parameter("port", "auto")
        self.declare_parameter("baud", 2000000)
        self.declare_parameter("bitrate", 250000)
        self.declare_parameter("deduplicate", False)
        self.declare_parameter("dummy_percentage", -1.0)

        self._port_param = self.get_parameter("port").value
        self._baud = int(self.get_parameter("baud").value)
        self._bitrate = int(self.get_parameter("bitrate").value)

        # Publishers
        self.pct_pub = self.create_publisher(Float32, "battery_info", 10)
        self.health_pub = self.create_publisher(String, "battery_health", 10)
        self.status_pub = self.create_publisher(String, "battery_status", 10)

        # BMS decoder + adapter. The adapter may be absent at startup; the
        # node stays alive, publishes fallback values and keeps retrying.
        self.bms = BMS()
        self.adapter = None
        self._last_reconnect = 0.0
        self._last_pct = None
        self._last_health = None
        self._connect()

        self.get_logger().info(
            "BMS node started; publishing battery_info and battery_health"
        )

        # Timers: fast polling of the CAN bus, slow publishing of topics.
        self._command_index = 0
        self._next_request = 0.0
        self.create_timer(POLL_PERIOD, self._poll)
        self.create_timer(PUBLISH_PERIOD, self._publish)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _resolve_port(self):
        """Resolve the configured port, auto-detecting by USB id if asked."""
        if self._port_param and self._port_param != "auto":
            return self._port_param
        return find_port()

    def _connect(self):
        """Try to (re)open the adapter. Returns True on success."""
        if self.adapter is not None:
            return True

        port = self._resolve_port()
        if port is None:
            self.get_logger().warn(
                f"Adapter {ADAPTER_VID:04x}:{ADAPTER_PID:04x} not found; "
                "publishing fallback values.",
                throttle_duration_sec=10.0,
            )
            return False

        try:
            self.adapter = WaveshareCan(port, self._baud, self._bitrate)
        except Exception as exc:  # serial.SerialException / ValueError
            self.get_logger().warn(
                f"Cannot open adapter on {port}: {exc}; will retry.",
                throttle_duration_sec=10.0,
            )
            self.adapter = None
            return False

        self.get_logger().info(
            f"Adapter connected on {port} (CAN {self._bitrate} bit/s)"
        )
        return True

    def _drop_adapter(self):
        """Close and forget the adapter so the next poll reconnects."""
        if self.adapter is not None:
            try:
                self.adapter.close()
            except Exception:
                pass
        self.adapter = None

    # ------------------------------------------------------------------
    # CAN polling
    # ------------------------------------------------------------------
    def _poll(self):
        now = time.monotonic()

        # If the adapter is gone, retry opening it every RECONNECT_PERIOD.
        if self.adapter is None:
            if now - self._last_reconnect >= RECONNECT_PERIOD:
                self._last_reconnect = now
                self._connect()
            return

        # Send one request at a time, spaced by REQUEST_INTERVAL.
        if now >= self._next_request:
            cmd = COMMANDS[self._command_index]
            try:
                self.adapter.send(request_id(cmd), bytes(8), extended=True)
            except Exception as exc:
                self.get_logger().warn(f"Serial write error: {exc}; dropping adapter")
                self._drop_adapter()
                return
            self._command_index = (self._command_index + 1) % len(COMMANDS)
            self._next_request = now + REQUEST_INTERVAL

        # Drain whatever replies have arrived.
        try:
            frames = self.adapter.recv()
        except Exception as exc:
            self.get_logger().warn(f"Serial read error: {exc}; dropping adapter")
            self._drop_adapter()
            return

        for can_id, extended, data in frames:
            if not extended:
                continue
            cmd = response_cmd(can_id)
            if cmd is not None:
                self.bms.decode(cmd, data)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish(self):
        bms = self.bms
        pct = Float32()
        health = String()

        dummy = self.get_parameter("dummy_percentage").value
        if dummy >= 0.0:
            pct.data = float(dummy)
            health.data = f"dummy: {dummy:.1f}%"
        else:
            connected = self.adapter is not None
            fresh = bms.last_update and (
                time.monotonic() - max(bms.last_update.values()) < STALE_TIMEOUT
            )

            if connected and fresh and bms.soc is not None:
                pct.data = float(bms.soc)
                health.data = self._describe(bms)
            else:
                pct.data = FALLBACK_PCT
                if not connected:
                    reason = "adapter not connected"
                elif not bms.last_update:
                    reason = "no data received from BMS yet"
                else:
                    reason = "BMS stopped responding (stale data)"
                health.data = f"disconnected: {reason}"

        if self.get_parameter("deduplicate").value:
            if pct.data == self._last_pct and health.data == self._last_health:
                return

        self._last_pct = pct.data
        self._last_health = health.data
        self.pct_pub.publish(pct)
        self.health_pub.publish(health)

        status_msg = String()
        if dummy >= 0.0:
            status_msg.data = "DUMMY"
        elif bms.current is None:
            status_msg.data = "WAITING FOR DATA"
        elif bms.current > 0.1:
            status_msg.data = "CHARGING"
        elif bms.current < -0.1:
            status_msg.data = "DISCHARGING"
        else:
            status_msg.data = "IDLE"
        self.status_pub.publish(status_msg)

    @staticmethod
    def _describe(bms):
        """Build a full human-readable description of the pack state."""
        lines = []
        lines.append(f"state={bms.state}")
        lines.append(f"soc={_fmt(bms.soc, '%')}")
        lines.append(f"voltage={_fmt(bms.voltage, 'V')}")
        lines.append(f"current={_fmt(bms.current, 'A')}")
        lines.append(f"remaining={_fmt(bms.remaining_mah, 'mAh')}")

        # Per-cell voltages
        cells = []
        for i in range(1, CELL_COUNT + 1):
            v = bms.cells.get(i)
            cells.append(f"c{i}={v if v is not None else '-'}")
        lines.append("cells[" + " ".join(cells) + "]")

        # Temperatures
        if bms.temperatures:
            temps = " ".join(
                f"t{i}={v}C" for i, v in sorted(bms.temperatures.items())
            )
        else:
            temps = "none"
        lines.append("temps[" + temps + "]")

        lines.append(
            f"cell_v_max={_fmt(bms.max_cell_v, 'V')}(#{bms.max_cell_no}) "
            f"min={_fmt(bms.min_cell_v, 'V')}(#{bms.min_cell_no})"
        )
        lines.append(
            f"temp_max={_fmt(bms.max_temp, 'C')}(#{bms.max_temp_sensor}) "
            f"min={_fmt(bms.min_temp, 'C')}(#{bms.min_temp_sensor})"
        )
        lines.append(f"charge_mos={bms.charge_mos} discharge_mos={bms.discharge_mos}")
        lines.append(f"cells={bms.cell_count} sensors={bms.sensor_count}")
        lines.append(f"cycles={bms.cycles}")

        if bms.fault_bytes is not None:
            faulty = any(bms.fault_bytes)
            lines.append(
                f"faults={'YES ' + bms.fault_bytes.hex(' ').upper() if faulty else 'none'}"
            )

        # Overall one-word verdict.
        lines.append(f"health={_verdict(bms)}")

        return "; ".join(lines)


def _fmt(value, unit):
    return "N/A" if value is None else f"{value}{unit}"


def _verdict(bms):
    """Rough overall health/condition verdict from the available data."""
    # Any active fault bit -> fault.
    if bms.fault_bytes is not None and any(bms.fault_bytes):
        return "FAULT"

    # Cell voltage spread (imbalance) and absolute limits.
    if bms.max_cell_v is not None and bms.min_cell_v is not None:
        spread = bms.max_cell_v - bms.min_cell_v
        if bms.max_cell_v > 3.65 or bms.min_cell_v < 2.8:
            return "CRITICAL"
        if spread > 0.1:
            return "IMBALANCED"

    # Temperature limits.
    if bms.max_temp is not None and bms.max_temp > 55:
        return "HOT"
    if bms.min_temp is not None and bms.min_temp < -10:
        return "COLD"

    # Nothing decoded yet.
    if bms.soc is None and bms.voltage is None:
        return "NO_DATA"

    return "OK"


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BatteryInfo()
    except Exception:
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._drop_adapter()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
