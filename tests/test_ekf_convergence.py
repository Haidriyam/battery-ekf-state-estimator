import unittest
import numpy as np
from src.battery_model import LiIonCellModel
from src.ekf_core import ExtendedKalmanFilterBMS


class TestEKFEstimator(unittest.TestCase):

    def setUp(self):
        self.model = LiIonCellModel()
        self.ekf = ExtendedKalmanFilterBMS(self.model, dt=0.1)

    def test_ekf_tracking_convergence(self):
        """Verify that EKF converges toward true SoC despite initial state divergence."""
        true_soc = 0.50
        self.ekf.x[0, 0] = 0.90  # Initialized with severe error (+40%)
        self.ekf.P[0, 0] = 0.25   # Reflect initial state uncertainty

        current = 2.0
        np.random.seed(42)

        for _ in range(350):
            true_soc -= (current * 0.1 / self.model.capacity_coulombs)
            measured_ocv = self.model.ocv_from_soc(true_soc)
            v_measured = measured_ocv - (current * self.model.r0) + np.random.normal(0, 0.002)

            self.ekf.predict(current)
            self.ekf.update(v_measured, current)

        # Filter must converge to within 3% of ground truth
        estimated_soc = self.ekf.x[0, 0]
        self.assertAlmostEqual(estimated_soc, true_soc, delta=0.03)


if __name__ == "__main__":
    unittest.main()
