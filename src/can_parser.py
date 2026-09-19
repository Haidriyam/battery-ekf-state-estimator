"""
CAN-Bus Frame Decoder with strict boundary validation against buffer overflow
and spoofed telemetric injection.
"""
from typing import Dict, Any
import struct


class CANTelemetryParser:
    STANDARD_BMS_ARB_ID = 0x3B4  # Standard identifier for cell pack readings

    @staticmethod
    def decode_frame(arb_id: int, payload: bytes) -> Dict[str, Any]:
        """
        Parse 8-byte CAN frame:
        Bytes 0-3: Voltage in millivolts (uint32_t big-endian)
        Bytes 4-7: Current in milliamperes (int32_t big-endian, signed)
        """
        if arb_id != CANTelemetryParser.STANDARD_BMS_ARB_ID:
            raise ValueError(f"Unrecognized arbitration ID: {hex(arb_id)}")

        if len(payload) != 8:
            raise ValueError(f"Invalid payload length: expected 8 bytes, got {len(payload)}")

        voltage_mv, current_ma = struct.unpack(">Ii", payload)

        voltage_v = voltage_mv / 1000.0
        current_a = current_ma / 1000.0

        # Physical safety assertions
        if not (2.0 <= voltage_v <= 4.5):
            raise ValueError(f"Cell voltage out of physical boundary: {voltage_v}V")
        if not (-150.0 <= current_a <= 150.0):
            raise ValueError(f"Current excursion exceeded safety envelope: {current_a}A")

        return {"voltage": voltage_v, "current": current_a}