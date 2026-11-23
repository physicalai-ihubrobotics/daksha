#!/usr/bin/env python3
# ============================================================
# BRAKE -> PUSH MODE BRIDGE
# ============================================================
#
# Reads the STM32 Black Pill serial output (main.py) and
# switches the robot's push mode from the brake button:
#
#   BRAKE: RELEASED -> push mode ON  (set_torque_disable true)
#   BRAKE: PRESSED  -> push mode OFF (set_torque_disable false)
#
# Matches both the event lines ("BRAKE: PRESSED") and the
# periodic status lines ("... | BRAKE: PRESSED | ..."), so the
# robot picks up the current state right after the node starts.
#
# Parameters:
#   port     : serial port of the STM32 (default /dev/ttyACM0)
#   baud     : baud rate (default 115200)
#   service  : SetBool service (default /drax/set_torque_disable)
#
# Run:
#   python3 brake_push_mode_node.py --ros-args -p port:=/dev/ttyACM0
# ============================================================

import re

import rclpy
import serial
from rclpy.node import Node
from std_srvs.srv import SetBool

BRAKE_RE = re.compile(r"BRAKE:\s*(PRESSED|RELEASED)")

POLL_PERIOD_S = 0.02
RECONNECT_PERIOD_S = 2.0


class BrakePushModeNode(Node):

    def __init__(self):
        super().__init__("brake_push_mode")

        self.port = self.declare_parameter("port", "/dev/ttyACM0").value
        self.baud = self.declare_parameter("baud", 115200).value
        service = self.declare_parameter(
            "service", "/drax/set_torque_disable"
        ).value

        self.client = self.create_client(SetBool, service)

        self.ser = None
        self.rx_buffer = b""
        self.last_connect_attempt = None

        # Push mode wanted by the brake / last one the robot accepted
        self.desired_push = None
        self.applied_push = None
        self.pending_call = None

        self.create_timer(POLL_PERIOD_S, self.poll)

        self.get_logger().info(
            f"Listening on {self.port} @ {self.baud}, calling {service}"
        )

    # ---------------- Serial ----------------

    def open_serial(self):
        now = self.get_clock().now()

        if self.last_connect_attempt is not None:
            elapsed = (now - self.last_connect_attempt).nanoseconds / 1e9
            if elapsed < RECONNECT_PERIOD_S:
                return

        self.last_connect_attempt = now

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0)
            self.rx_buffer = b""
            self.get_logger().info(f"Serial connected: {self.port}")
        except serial.SerialException as e:
            self.ser = None
            self.get_logger().warn(f"Serial open failed: {e}")

    def read_lines(self):
        try:
            data = self.ser.read(self.ser.in_waiting or 1)
        except serial.SerialException as e:
            self.get_logger().error(f"Serial lost: {e}")
            self.ser.close()
            self.ser = None
            return []

        self.rx_buffer += data
        *lines, self.rx_buffer = self.rx_buffer.split(b"\n")

        return [
            line.decode("utf-8", errors="ignore").strip()
            for line in lines
        ]

    # ---------------- Main loop ----------------

    def poll(self):
        if self.ser is None:
            self.open_serial()

        if self.ser is not None:
            for line in self.read_lines():
                self.handle_line(line)

        self.apply_push_mode()

    def handle_line(self, line):
        match = BRAKE_RE.search(line)

        if not match:
            return

        # RELEASED -> push mode, PRESSED -> wheels locked
        push = match.group(1) == "RELEASED"

        if push != self.desired_push:
            self.desired_push = push
            self.get_logger().info(
                f"BRAKE: {match.group(1)} -> push mode "
                f"{'ON' if push else 'OFF'}"
            )

    # ---------------- Service ----------------

    def apply_push_mode(self):
        if self.desired_push is None:
            return

        if self.desired_push == self.applied_push:
            return

        if self.pending_call is not None:
            return

        if not self.client.service_is_ready():
            return

        request = SetBool.Request()
        request.data = self.desired_push

        self.pending_call = self.client.call_async(request)
        self.pending_call.add_done_callback(
            lambda future, push=self.desired_push:
                self.on_response(future, push)
        )

    def on_response(self, future, push):
        self.pending_call = None

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return

        if response.success:
            self.applied_push = push
            self.get_logger().info(
                f"Push mode {'ON' if push else 'OFF'}"
                f" (set_torque_disable {str(push).lower()})"
            )
        else:
            self.get_logger().warn(
                f"Service rejected request: {response.message}"
            )

    def destroy_node(self):
        if self.ser is not None:
            self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = BrakePushModeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
