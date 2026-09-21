import unittest
import numpy as np
from estimator.c37118_parser import C37118FrameParser
from estimator.generator_ekf import SynchronousGeneratorEKF


class TestSynchrophasorPipeline(unittest.TestCase):

    def setUp(self):
        self.parser = C37118FrameParser()
        self.ekf = SynchronousGeneratorEKF(dt=0.02, fn=50.0)

    def test_c37118_frame_unpacking_valid(self):
        frame = self.parser.pack_test_frame(
            station_id=101,
            soc=1774000000,
            fracsec=500000,
            voltage_mag=1.02,
            voltage_phase_rad=0.05,
            freq_hz=50.01,
            rocof=0.002,
            sync_valid=True
        )

        record = self.parser.unpack_data_frame(frame, num_phasors=1, is_floating=True)
        self.assertEqual(record.station_id, 101)
        self.assertEqual(record.soc_timestamp, 1774000000)
        self.assertTrue(record.time_synchronized)
        self.assertTrue(record.data_valid)
        self.assertAlmostEqual(record.phasors[0][0], 1.02, places=2)
        self.assertAlmostEqual(record.frequency_hz, 50.01, places=2)

    def test_c37118_corrupted_header_rejection(self):
        raw = bytearray(self.parser.pack_test_frame(1, 0, 0, 1.0, 0.0, 50.0))
        raw[0] = 0xFF
        with self.assertRaises(ValueError) as ctx:
            self.parser.unpack_data_frame(bytes(raw), num_phasors=1)
        self.assertIn("Invalid frame sync header", str(ctx.exception))

    def test_c37118_undersized_frame_rejection(self):
        with self.assertRaises(ValueError) as ctx:
            self.parser.unpack_data_frame(b"\xAA\x01\x00", num_phasors=1)
        self.assertIn("Packet undersized", str(ctx.exception))

    def test_ekf_tracking_convergence(self):
        np.random.seed(42)
        Vt = 1.0
        true_delta = 0.40
        true_w_dev = 0.0
        P_max = (self.ekf.E_p * Vt) / self.ekf.Xd_p
        Pm_eq = P_max * np.sin(true_delta)
        true_Pe = Pm_eq

        # Start with an offset initial state estimate
        self.ekf.x = np.array([0.60, 0.20], dtype=np.float64)

        out = {}
        for _ in range(80):
            measured_Pe = float(true_Pe + np.random.normal(0, 0.005))
            out = self.ekf.step(Pe_measured=measured_Pe, Vt=Vt, Pm=Pm_eq)

        # Assert convergence to true states
        self.assertAlmostEqual(out["rotor_angle_rad"], true_delta, delta=0.03)
        self.assertAlmostEqual(out["speed_deviation_rad_s"], true_w_dev, delta=0.02)
        self.assertLess(abs(out["innovation"]), 0.05)


if __name__ == "__main__":
    unittest.main()