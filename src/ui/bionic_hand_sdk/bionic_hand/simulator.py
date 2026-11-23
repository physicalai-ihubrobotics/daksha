"""In-memory stand-in for the serial port, for testing without hardware.

Connect to the port name ``"sim://"`` to use it. It answers PING and READ
packets for motor IDs 1-16 and remembers every goal position written.
"""
from .protocol import INST_PING, INST_READ, INST_WRITE, REG_GOAL_POSITION, checksum, make_packet

SIM_PORT = "sim://"


class SimulatedSerial:
    def __init__(self, ids=range(1, 17), start_raw=512):
        self.is_open = True
        self.positions = {sid: start_raw for sid in ids}
        self._out = bytearray()

    def reset_input_buffer(self):
        self._out.clear()

    def flush(self):
        pass

    def close(self):
        self.is_open = False

    def write(self, packet):
        sid, instruction, params = packet[2], packet[4], list(packet[5:-1])
        if checksum(list(packet[2:-1])) != packet[-1] or sid not in self.positions:
            return len(packet)
        if instruction == INST_WRITE and params[0] == REG_GOAL_POSITION:
            self.positions[sid] = int.from_bytes(bytes(params[1:3]), "big")
        elif instruction == INST_PING:
            self._out += make_packet(sid, 0)
        elif instruction == INST_READ:
            self._out += make_packet(sid, 0, list(self.positions[sid].to_bytes(2, "big")))
        return len(packet)

    def read(self, n=1):
        data = bytes(self._out[:n])
        del self._out[:n]
        return data
