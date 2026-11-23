#!/usr/bin/env python3
"""Self-contained ROS 2 keyboard teleop node for two CANopen motors with separate motor control topic."""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int64MultiArray, String

# Custom message for motor commands
from geometry_msgs.msg import Twist  # We'll still import this but not use it for cmd_vel

try:
    import can
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("python-can is required. Install with: sudo apt install python3-can") from exc


class MotorError(RuntimeError):
    pass


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def rpm_to_dec(rpm: int, feedback_resolution: int = 4096) -> int:
    return int(round((rpm * 512 * feedback_resolution) / 1875))


def read_key(input_stream, timeout_sec: float) -> str | None:
    readable, _, _ = select.select([input_stream], [], [], timeout_sec)
    if not readable:
        return None
    ch = input_stream.read(1)
    return ch if ch else None


@dataclass
class SocketCanBus:
    interface: str = "can0"
    debug: bool = False

    def __post_init__(self) -> None:
        self.bus = can.interface.Bus(interface="socketcan", channel=self.interface, receive_own_messages=False)

    def send(self, can_id: int, data: bytes) -> None:
        msg = can.Message(arbitration_id=can_id, is_extended_id=False, data=data)
        self.bus.send(msg, timeout=0.2)
        if self.debug:
            print(f"[can tx] 0x{can_id:03X} {data.hex(' ')}")

    def recv(self, timeout: float) -> can.Message | None:
        msg = self.bus.recv(timeout=timeout)
        if msg is not None and self.debug:
            print(f"[can rx] 0x{msg.arbitration_id:03X} {bytes(msg.data).hex(' ')}")
        return msg

    def send_nmt_start(self, node_id: int = 0) -> None:
        self.send(0x000, bytes([0x01, node_id & 0x7F]))

    def close(self) -> None:
        self.bus.shutdown()


@dataclass
class CanOpenNode:
    io: SocketCanBus
    node_id: int
    timeout: float = 0.35
    retries: int = 3
    debug: bool = False

    @property
    def tx_id(self) -> int:
        return 0x600 + self.node_id

    @property
    def rx_id(self) -> int:
        return 0x580 + self.node_id

    def _request(self, req: bytes) -> bytes:
        if len(req) != 8:
            raise ValueError("SDO request must be 8 bytes")

        for _ in range(self.retries + 1):
            if self.debug:
                print(f"[tx n{self.node_id}] 0x{self.tx_id:03X} {req.hex(' ')}")
            self.io.send(self.tx_id, req)
            end = time.monotonic() + self.timeout
            while time.monotonic() < end:
                msg = self.io.recv(max(0.01, end - time.monotonic()))
                if msg is None:
                    continue
                if msg.is_error_frame or msg.is_remote_frame or msg.is_extended_id:
                    continue
                if msg.arbitration_id != self.rx_id:
                    continue
                data = bytes(msg.data)
                if len(data) < 8:
                    continue
                if self.debug:
                    print(f"[rx n{self.node_id}] 0x{self.rx_id:03X} {data[:8].hex(' ')}")
                return data[:8]

        raise TimeoutError(f"No SDO reply from node {self.node_id}")

    def write(self, index: int, sub: int, value: int, size: int, signed: bool = True) -> None:
        if size not in (1, 2, 4):
            raise ValueError("size must be 1, 2, or 4")
        cs = {1: 0x2F, 2: 0x2B, 4: 0x23}[size]
        raw = int(value).to_bytes(size, "little", signed=signed).ljust(4, b"\x00")
        resp = self._request(bytes([cs, index & 0xFF, (index >> 8) & 0xFF, sub, *raw]))
        if resp[0] == 0x80:
            code = int.from_bytes(resp[4:8], "little")
            raise MotorError(f"SDO abort 0x{index:04X}:{sub:02X}, code=0x{code:08X}")
        ok = resp[0] == 0x60 and resp[1] == (index & 0xFF) and resp[2] == ((index >> 8) & 0xFF) and resp[3] == sub
        if not ok:
            raise MotorError(f"Unexpected write response node {self.node_id}: {resp.hex(' ')}")

    def read_u16(self, index: int, sub: int) -> int:
        resp = self._request(bytes([0x40, index & 0xFF, (index >> 8) & 0xFF, sub, 0, 0, 0, 0]))
        if resp[0] == 0x80:
            code = int.from_bytes(resp[4:8], "little")
            raise MotorError(f"SDO abort 0x{index:04X}:{sub:02X}, code=0x{code:08X}")
        if resp[0] != 0x4B:
            raise MotorError(f"Unexpected read_u16 response node {self.node_id}: {resp.hex(' ')}")
        return int.from_bytes(resp[4:6], "little", signed=False)

    def read_i32(self, index: int, sub: int) -> int:
        resp = self._request(bytes([0x40, index & 0xFF, (index >> 8) & 0xFF, sub, 0, 0, 0, 0]))
        if resp[0] == 0x80:
            code = int.from_bytes(resp[4:8], "little")
            raise MotorError(f"SDO abort 0x{index:04X}:{sub:02X}, code=0x{code:08X}")
        if resp[0] not in (0x43, 0x42):
            raise MotorError(f"Unexpected read_i32 response node {self.node_id}: {resp.hex(' ')}")
        return int.from_bytes(resp[4:8], "little", signed=True)


@dataclass
class Motor:
    node: CanOpenNode

    def init_speed_mode(self, mode: int = -3, fix_din3: bool = True) -> None:
        self.node.write(0x6060, 0x00, mode, size=1, signed=True)
        self.node.write(0x6040, 0x00, 0x86, size=2, signed=True)
        self.node.write(0x605A, 0x11, 0, size=1, signed=False)
        if fix_din3:
            pol = self.node.read_u16(0x2010, 0x01)
            sim = self.node.read_u16(0x2010, 0x02)
            self.node.write(0x2010, 0x01, pol & ~(1 << 2), size=2, signed=False)
            self.node.write(0x2010, 0x02, sim & ~(1 << 2), size=2, signed=False)
        self.node.write(0x6040, 0x00, 0x06, size=2, signed=True)
        time.sleep(0.02)
        self.node.write(0x6040, 0x00, 0x07, size=2, signed=True)
        time.sleep(0.02)
        self.node.write(0x6040, 0x00, 0x0F, size=2, signed=True)
        self.set_rpm(0, precise=True)

    def set_rpm(self, rpm: int, precise: bool = True) -> None:
        self.node.write(0x2FF0, 0x09, rpm, size=2, signed=True)
        if precise:
            self.node.write(0x60FF, 0x00, rpm_to_dec(rpm), size=4, signed=True)

    def read_encoder(self, index: int = 0x6063, sub: int = 0x00) -> int:
        return self.node.read_i32(index, sub)

    def stop(self) -> None:
        self.set_rpm(0, precise=True)

    def disable(self) -> None:
        self.node.write(0x6040, 0x00, 0x06, size=2, signed=True)


@dataclass
class TwoMotorController:
    left: Motor
    right: Motor
    invert_left: bool = False
    invert_right: bool = True
    encoder_index: int = 0x6063
    encoder_sub: int = 0x00

    def init(self, mode: int = -3, fix_din3: bool = True) -> None:
        self.left.init_speed_mode(mode=mode, fix_din3=fix_din3)
        self.right.init_speed_mode(mode=mode, fix_din3=fix_din3)

    def _apply_inversion(self, left_rpm: int, right_rpm: int) -> tuple[int, int]:
        out_left = -left_rpm if self.invert_left else left_rpm
        out_right = -right_rpm if self.invert_right else right_rpm
        return out_left, out_right

    def set_pair(self, left_rpm: int, right_rpm: int, precise: bool = True) -> None:
        out_left, out_right = self._apply_inversion(left_rpm, right_rpm)
        self.left.set_rpm(out_left, precise=precise)
        self.right.set_rpm(out_right, precise=precise)

    def read_encoders(self) -> tuple[int, int]:
        return (
            self.left.read_encoder(self.encoder_index, self.encoder_sub),
            self.right.read_encoder(self.encoder_index, self.encoder_sub),
        )

    def stop(self) -> None:
        self.set_pair(0, 0, precise=True)

    def disable(self) -> None:
        self.left.disable()
        self.right.disable()


@dataclass
class BalanceState:
    ref_left: int = 0
    ref_right: int = 0


class KeyboardMotorTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_motor_teleop")

        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("left_node_id", 1)
        self.declare_parameter("right_node_id", 2)
        self.declare_parameter("start_rpm", 10)
        self.declare_parameter("speed_step", 10)
        self.declare_parameter("max_rpm", 20)
        self.declare_parameter("sample_ms", 80)
        self.declare_parameter("invert_left", False)
        self.declare_parameter("invert_right", True)
        self.declare_parameter("balance", True)
        self.declare_parameter("balance_kp", 0.008)
        self.declare_parameter("max_correction_rpm", 2)
        self.declare_parameter("safe_balance_ratio", 0.15)
        self.declare_parameter("timeout_sec", 0.35)
        self.declare_parameter("retries", 3)
        self.declare_parameter("mode",3)
        self.declare_parameter("fix_din3", True)
        self.declare_parameter("precise_speed_command", True)
        self.declare_parameter("nmt_start", True)
        self.declare_parameter("keep_enabled", False)
        self.declare_parameter("debug_frames", False)
        
        # Motor command topic parameters
        self.declare_parameter("motor_command_timeout", 0.5)  # seconds before stopping if no motor command

        self.can_interface = str(self.get_parameter("can_interface").value)
        self.left_node_id = int(self.get_parameter("left_node_id").value)
        self.right_node_id = int(self.get_parameter("right_node_id").value)
        self.start_rpm = int(self.get_parameter("start_rpm").value)
        self.speed_step = int(self.get_parameter("speed_step").value)
        self.max_rpm = int(self.get_parameter("max_rpm").value)
        self.sample_ms = int(self.get_parameter("sample_ms").value)
        self.invert_left = bool(self.get_parameter("invert_left").value)
        self.invert_right = bool(self.get_parameter("invert_right").value)
        self.balance_enabled = bool(self.get_parameter("balance").value)
        self.balance_kp = float(self.get_parameter("balance_kp").value)
        self.max_correction_rpm = int(self.get_parameter("max_correction_rpm").value)
        self.safe_balance_ratio = float(self.get_parameter("safe_balance_ratio").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.retries = int(self.get_parameter("retries").value)
        self.mode = int(self.get_parameter("mode").value)
        self.fix_din3 = bool(self.get_parameter("fix_din3").value)
        self.precise_speed_command = bool(self.get_parameter("precise_speed_command").value)
        self.nmt_start = bool(self.get_parameter("nmt_start").value)
        self.keep_enabled = bool(self.get_parameter("keep_enabled").value)
        self.debug_frames = bool(self.get_parameter("debug_frames").value)
        
        # Motor command topic parameters
        self.motor_command_timeout = float(self.get_parameter("motor_command_timeout").value)

        self.status_pub = self.create_publisher(String, "keyboard_motor_teleop/status", 10)
        self.encoder_pub = self.create_publisher(Int64MultiArray, "keyboard_motor_teleop/encoders", 10)
        self.command_pub = self.create_publisher(Int32MultiArray, "keyboard_motor_teleop/command_rpm", 10)
        
        # Motor command subscriber - using Int32MultiArray for left and right wheel speeds
        # The array should contain [left_wheel_speed, right_wheel_speed]
        self.motor_command_sub = self.create_subscription(
            Int32MultiArray, 
            "motor_command", 
            self.motor_command_callback, 
            10
        )

        self.io = SocketCanBus(interface=self.can_interface, debug=self.debug_frames)
        left_motor = Motor(
            CanOpenNode(self.io, node_id=self.left_node_id, timeout=self.timeout_sec, retries=self.retries, debug=self.debug_frames)
        )
        right_motor = Motor(
            CanOpenNode(self.io, node_id=self.right_node_id, timeout=self.timeout_sec, retries=self.retries, debug=self.debug_frames)
        )
        self.controller = TwoMotorController(
            left=left_motor,
            right=right_motor,
            invert_left=self.invert_left,
            invert_right=self.invert_right,
        )
        self.balance = BalanceState()

        self._sample_period = max(0.02, self.sample_ms / 1000.0)
        self._speed = clamp(abs(self.start_rpm), 0, self.max_rpm)
        self._direction = "stop"
        self._last_cmd = (None, None)
        self._last_status_time = 0.0
        self._input_stream = None
        self._input_fd = None
        
        # Motor command state
        self._motor_command_mode = False
        self._last_motor_command_time = 0.0
        self._motor_command_left_rpm = 0
        self._motor_command_right_rpm = 0

    def motor_command_callback(self, msg: Int32MultiArray) -> None:
        """Handle incoming motor command messages.
        
        The message should contain exactly 2 integers:
        - First integer: left wheel speed (RPM)
        - Second integer: right wheel speed (RPM)
        
        For example: [1, 2] means left wheel at 1 RPM, right wheel at 2 RPM
        """
        if len(msg.data) < 2:
            self.get_logger().warn(f"Motor command message must have at least 2 integers. Got {len(msg.data)}")
            return
        
        # Extract left and right wheel speeds
        left_rpm = msg.data[0]
        right_rpm = msg.data[1]
        
        # Clamp to max RPM
        left_rpm = clamp(left_rpm, -self.max_rpm, self.max_rpm)
        right_rpm = clamp(right_rpm, -self.max_rpm, self.max_rpm)
        
        self._motor_command_left_rpm = left_rpm
        self._motor_command_right_rpm = right_rpm
        self._last_motor_command_time = time.monotonic()
        self._motor_command_mode = True
        
        # Disable keyboard mode when receiving motor commands
        self._direction = "motor_command"
        
        self.get_logger().debug(f"Motor command received: L={left_rpm} RPM, R={right_rpm} RPM")

    def publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def setup_input(self) -> None:
        if sys.stdin.isatty():
            self._input_stream = sys.stdin
            self._input_fd = sys.stdin.fileno()
            return

        try:
            input_stream = open("/dev/tty", "r", encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise RuntimeError(
                "keyboard_motor_teleop needs terminal access. Run it from an interactive shell or launch it from one."
            ) from exc

        self._input_stream = input_stream
        self._input_fd = input_stream.fileno()

    def cleanup_input(self) -> None:
        if self._input_stream is not None and self._input_stream is not sys.stdin:
            self._input_stream.close()
        self._input_stream = None
        self._input_fd = None

    def initialize_motors(self) -> None:
        if self.nmt_start:
            self.io.send_nmt_start(0)
            time.sleep(0.15)
        self.controller.init(mode=self.mode, fix_din3=self.fix_din3)
        self.controller.stop()
        self.reset_balance_reference()
        self.publish_status(
            "Keyboard teleop ready: w/s/a/d move, +/- speed, space stop, r reset encoder balance, q quit\n"
            "Also subscribing to /motor_command (Int32MultiArray) - send [left_rpm, right_rpm]"
        )

    def reset_balance_reference(self) -> None:
        left_enc, right_enc = self.controller.read_encoders()
        self.balance.ref_left = left_enc
        self.balance.ref_right = right_enc

    @staticmethod
    def direction_to_pair(direction: str, rpm_mag: int) -> tuple[int, int]:
        if direction == "forward":
            return rpm_mag, rpm_mag
        if direction == "backward":
            return -rpm_mag, -rpm_mag
        if direction == "left":
            return -rpm_mag, rpm_mag
        if direction == "right":
            return rpm_mag, -rpm_mag
        if direction == "motor_command":
            return None, None  # Special case for motor command mode
        return 0, 0

    def compute_balanced_pair(self, enc_l: int, enc_r: int, base_left: int, base_right: int) -> tuple[int, int, int]:
        if not self.balance_enabled or self._direction == "stop" or (self._direction != "motor_command" and self._speed <= 0):
            return base_left, base_right, 0

        progress_left = abs(enc_l - self.balance.ref_left)
        progress_right = abs(enc_r - self.balance.ref_right)
        err = progress_left - progress_right

        # For motor command mode, use the commanded speeds to determine the dynamic limit
        if self._direction == "motor_command":
            avg_speed = (abs(base_left) + abs(base_right)) / 2
            dynamic_limit = int(max(5, avg_speed * max(0.0, min(1.0, self.safe_balance_ratio))))
        else:
            dynamic_limit = int(max(5, self._speed * max(0.0, min(1.0, self.safe_balance_ratio))))
            
        corr_limit = min(self.max_correction_rpm, dynamic_limit)
        correction = clamp(int(self.balance_kp * err), -corr_limit, corr_limit)

        left_dir = 1 if base_left > 0 else -1
        right_dir = 1 if base_right > 0 else -1
        left_mag = clamp(abs(base_left) - correction, 0, self.max_rpm)
        right_mag = clamp(abs(base_right) + correction, 0, self.max_rpm)
        return left_dir * left_mag, right_dir * right_mag, correction

    def publish_telemetry(self, enc_l: int, enc_r: int, cmd_left: int, cmd_right: int, corr: int) -> None:
        self.encoder_pub.publish(Int64MultiArray(data=[enc_l, enc_r, enc_l - enc_r]))
        self.command_pub.publish(Int32MultiArray(data=[cmd_left, cmd_right, corr, self._speed]))

        now = time.monotonic()
        if now - self._last_status_time < 0.25:
            return
        mode_str = "motor_cmd" if self._motor_command_mode else self._direction
        status = (
            f"mode={mode_str:<10} cmdL={cmd_left:<5} cmdR={cmd_right:<5} "
            f"encL={enc_l:<10} encR={enc_r:<10} diff={enc_l - enc_r:<10} corr={corr:<4}"
        )
        self.status_pub.publish(String(data=status))
        print(status)
        self._last_status_time = now

    def apply_keyboard(self, key: str | None) -> bool:
        if not key:
            return True

        k = key.lower()
        old_direction = self._direction
        old_speed = self._speed

        # Any keyboard input disables motor command mode
        self._motor_command_mode = False

        if k == "w":
            self._direction = "forward"
        elif k == "s":
            self._direction = "backward"
        elif k == "a":
            self._direction = "left"
        elif k == "d":
            self._direction = "right"
        elif k in (" ", "x"):
            self._direction = "stop"
        elif k in ("+", "="):
            self._speed = clamp(self._speed + self.speed_step, 0, self.max_rpm)
        elif k in ("-", "_"):
            self._speed = clamp(self._speed - self.speed_step, 0, self.max_rpm)
        elif k == "r":
            self.reset_balance_reference()
            self.publish_status("Balance reference reset")
        elif k == "q":
            return False

        if self._direction != old_direction or self._speed != old_speed:
            self.reset_balance_reference()
        return True

    def run(self) -> None:
        self.setup_input()
        assert self._input_stream is not None
        assert self._input_fd is not None

        old_term = termios.tcgetattr(self._input_fd)
        try:
            self.initialize_motors()
            tty.setcbreak(self._input_fd)

            running = True
            while running and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                key = read_key(self._input_stream, self._sample_period)
                running = self.apply_keyboard(key)

                # Check motor command timeout
                if self._motor_command_mode:
                    time_since_cmd = time.monotonic() - self._last_motor_command_time
                    if time_since_cmd > self.motor_command_timeout:
                        self._motor_command_mode = False
                        self._direction = "stop"
                        self.get_logger().warn(f"Motor command timeout after {time_since_cmd:.1f}s")

                # Determine base RPM commands
                if self._motor_command_mode:
                    base_left = self._motor_command_left_rpm
                    base_right = self._motor_command_right_rpm
                else:
                    base_left, base_right = self.direction_to_pair(self._direction, self._speed)

                enc_l, enc_r = self.controller.read_encoders()
                cmd_left, cmd_right, corr = self.compute_balanced_pair(enc_l, enc_r, base_left, base_right)

                if (cmd_left, cmd_right) != self._last_cmd:
                    self.controller.set_pair(cmd_left, cmd_right, precise=self.precise_speed_command)
                    self._last_cmd = (cmd_left, cmd_right)

                self.publish_telemetry(enc_l, enc_r, cmd_left, cmd_right, corr)
        finally:
            try:
                termios.tcsetattr(self._input_fd, termios.TCSADRAIN, old_term)
            except Exception:
                pass
            self.cleanup_input()

    def shutdown_motors(self) -> None:
        try:
            self.controller.stop()
            if not self.keep_enabled:
                self.controller.disable()
        finally:
            self.io.close()


def main() -> None:
    rclpy.init()
    node = KeyboardMotorTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown_motors()
        except Exception as exc:
            node.get_logger().warning(f"Shutdown warning: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
