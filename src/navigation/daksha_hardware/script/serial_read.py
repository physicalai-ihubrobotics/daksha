#!/usr/bin/env python3
import serial
import sys

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

print(f"Reading {port} @ {baud} — Ctrl+C to stop")

ser = serial.Serial(port, baud, timeout=1)
try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(line)
except KeyboardInterrupt:
    pass
finally:
    ser.close()
