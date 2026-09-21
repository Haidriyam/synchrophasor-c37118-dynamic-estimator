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

        # True physical state
        true_delta = 0.35
        true_omega_dev = 0.0
        true_Eq_p = 1.05
        true_Ed_p = 0.0
        true_state = np.array([true_delta, true_omega_dev, true_Eq_p, true_Ed_p])

        # Compute consistent steady-state equilibrium power and field voltage
        Id0 = (true_Eq_p - Vt * np.cos(true_delta)) / self.ekf.Xdp
        Iq0 = (-true_Ed_p + Vt * np.sin(true_delta)) / self.ekf.Xqp
        Pm_eq = Vt * np.sin(true_delta) * Id0 + Vt * np.cos(true_delta) * Iq0
        Efd_eq = true_Eq_p + (self.ekf.Xd - self.ekf.Xdp) * Id0

        z_nominal = self.ekf.compute_measurement_h(true_state, Vt=Vt)

        # Perturb the initial state guess in the estimator
        self.ekf.x = np.array([0.45, 0.01, 0.95, 0.05], dtype=np.float64)

        # Run EKF across 60 steps
        out = {}
        for _ in range(60):
            noisy_z = (
                float(z_nominal[0] + np.random.normal(0, 0.002)),
                float(z_nominal[1] + np.random.normal(0, 0.002))
            )
            out = self.ekf.step(z_measured=noisy_z, Vt=Vt, Pm=Pm_eq, Efd=Efd_eq)

        # Assert convergence to true state within tight error bounds
        self.assertAlmostEqual(out["rotor_angle_rad"], true_delta, delta=0.03)
        self.assertAlmostEqual(out["speed_deviation_pu"], 0.0, delta=0.01)
        self.assertLess(out["innovation_norm"], 0.05)


if __name__ == "__main__":
    unittest.main()