"""
Discrete Extended Kalman Filter for real-time SoC and RC overpotential estimation.
"""
import numpy as np
from src.battery_model import LiIonCellModel


class ExtendedKalmanFilterBMS:
    def __init__(self, model: LiIonCellModel, dt: float = 0.1):
        self.model = model
        self.dt = dt

        # State vector: [SoC, V_rc]^T
        self.x = np.array([[0.65], [0.0]], dtype=np.float64)

        # Initial covariance
        self.P = np.diag([0.1, 1e-4])

        # Process noise covariance (Q) and measurement noise variance (R)
        self.Q = np.diag([1e-5, 1e-5])
        self.R = np.array([[1e-3]])

    def predict(self, current: float):
        """Time update: propagate state and covariance through discretized dynamics."""
        soc, v_rc = float(self.x[0, 0]), float(self.x[1, 0])
        exp_factor = float(np.exp(-self.dt / self.model.tau))

        soc_next = soc - (current * self.dt / self.model.capacity_coulombs)
        v_rc_next = (exp_factor * v_rc) + (self.model.r1 * (1.0 - exp_factor) * current)

        self.x = np.array([[soc_next], [v_rc_next]])

        F = np.array([
            [1.0, 0.0],
            [0.0, exp_factor]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, terminal_voltage: float, current: float):
        """Measurement update: correct predicted state using terminal voltage."""
        soc, v_rc = float(self.x[0, 0]), float(self.x[1, 0])
        ocv = self.model.ocv_from_soc(soc)

        predicted_voltage = ocv - v_rc - (current * self.model.r0)
        residual = terminal_voltage - predicted_voltage

        H = np.array([[self.model.ocv_derivative(soc), -1.0]])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K * residual
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_mat = np.eye(2)
        self.P = (I_mat - K @ H) @ self.P
