#!/usr/bin/env python3
"""
Smart BMS reader over a Waveshare USB-CAN-A adapter (serial, no SocketCAN)

Everything happens in this one script: it configures the adapter's CAN
bitrate over the serial port, sends the BMS request frames and decodes the
replies. No can0, no vcan, no sudo (only read/write access to the tty -
add yourself to the 'dialout' group if needed).

Protocol: BMS CAN, 250 kbit/s, 29-bit extended IDs
    Host -> BMS : 0x18 <cmd> 01 40   e.g. 0x18900140
    BMS  -> Host: 0x18 <cmd> 40 01   e.g. 0x18904001

Waveshare variable-length serial frame:
    0xAA | info | ID (2 bytes std / 4 bytes ext, LSB first) | data | 0x55
    info = 0xC0 | (extended << 5) | (remote << 4) | DLC

Usage:
    ./bms_serial.py [--port /dev/ttyUSB0] [--baud 2000000] [--bitrate 250000]
"""

import argparse
import sys
import time

import serial

BMS_ADDRESS = 0x01
HOST_ADDRESS = 0x40

CELL_COUNT = 8
REQUEST_INTERVAL = 0.15
PRINT_INTERVAL = 1.0

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


# --------------------------------------------------------------------------
# BMS protocol
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

    def show(self):
        print("\n" + "=" * 45)
        print("       SMART BMS STATUS")
        print("=" * 45)

        print("Pack voltage :", self.voltage, "V")
        print("Current      :", self.current, "A")
        print("SOC          :", self.soc, "%")
        print("Remaining    :", self.remaining_mah, "mAh")

        print("\nCell voltages:")
        for i in range(1, CELL_COUNT + 1):
            value = self.cells.get(i)
            print(f"  Cell {i}: {value if value is not None else 'Waiting'} V")

        print("\nTemperatures:")
        if self.temperatures:
            for i, value in sorted(self.temperatures.items()):
                print(f"  Sensor {i}: {value} °C")
        else:
            print("  Waiting for sensor data")

        print("\nMaximum cell voltage:", self.max_cell_v, "V", f"(cell {self.max_cell_no})")
        print("Minimum cell voltage:", self.min_cell_v, "V", f"(cell {self.min_cell_no})")
        print("Maximum temperature:", self.max_temp, "°C", f"(sensor {self.max_temp_sensor})")
        print("Minimum temperature:", self.min_temp, "°C", f"(sensor {self.min_temp_sensor})")
        print("Charge MOS status:", self.charge_mos)
        print("Discharge MOS status:", self.discharge_mos)
        print("Operating state:", self.state)
        print("Cell / sensor count:", self.cell_count, "/", self.sensor_count)
        print("Cycle count:", self.cycles)

        if self.fault_bytes is not None:
            print("Fault bytes:", self.fault_bytes.hex(" ").upper())

        print("=" * 45)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=2000000,
                        help="serial baud rate of the adapter")
    parser.add_argument("--bitrate", type=int, default=250000,
                        help="CAN bus bitrate")
    args = parser.parse_args()

    try:
        adapter = WaveshareCan(args.port, args.baud, args.bitrate)
    except (serial.SerialException, ValueError) as exc:
        sys.exit(f"Cannot open adapter: {exc}")

    bms = BMS()

    print("BMS serial reader started")
    print("Port:", args.port, "| CAN bitrate:", args.bitrate)
    print("Press Ctrl+C to exit")

    next_request = 0
    next_print = time.monotonic()
    command_index = 0

    try:
        while True:
            now = time.monotonic()

            # Poll one command at a time.
            if now >= next_request:
                cmd = COMMANDS[command_index]
                try:
                    adapter.send(request_id(cmd), bytes(8), extended=True)
                except serial.SerialException as exc:
                    print("Serial write error:", exc)

                command_index = (command_index + 1) % len(COMMANDS)
                next_request = now + REQUEST_INTERVAL

            for can_id, extended, data in adapter.recv():
                if extended:
                    cmd = response_cmd(can_id)
                    if cmd is not None:
                        bms.decode(cmd, data)

            if now >= next_print:
                bms.show()
                next_print = now + PRINT_INTERVAL

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        adapter.close()


if __name__ == "__main__":
    main()
