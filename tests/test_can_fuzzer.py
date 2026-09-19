import unittest
import struct
from src.can_parser import CANTelemetryParser


class TestCANIngestionFuzzer(unittest.TestCase):

    def test_nominal_frame_decode(self):
        # 3800 mV (3.8V), -15000 mA (-15.0A)
        payload = struct.pack(">Ii", 3800, -15000)
        result = CANTelemetryParser.decode_frame(0x3B4, payload)
        self.assertEqual(result["voltage"], 3.8)
        self.assertEqual(result["current"], -15.0)

    def test_fuzz_payload_lengths(self):
        """Malformed frame sizes must be cleanly rejected without unhandled crash."""
        fuzz_lengths = [b"", b"\x00", b"short", b"\x00" * 9, b"\xFF" * 128]
        for payload in fuzz_lengths:
            with self.assertRaises(ValueError):
                CANTelemetryParser.decode_frame(0x3B4, payload)

    def test_fuzz_out_of_bounds_sensor_readings(self):
        """Out-of-range sensor readings must trip security boundaries."""
        # 8000 mV (8.0V - catastrophic overvoltage)
        toxic_overvoltage = struct.pack(">Ii", 8000, 0)
        with self.assertRaises(ValueError):
            CANTelemetryParser.decode_frame(0x3B4, toxic_overvoltage)


if __name__ == "__main__":
    unittest.main()