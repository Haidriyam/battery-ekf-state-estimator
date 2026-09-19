"""
First-order Thevenin equivalent circuit model for Li-ion cylindrical cell.
States: x = [SoC, V_polarization]^T
"""
import numpy as np


class LiIonCellModel:
    def __init__(
        self,
        capacity_ah: float = 2.5,
        r0: float = 0.045,      # Ohmic resistance (Ohms)
        r1: float = 0.035,      # Polarization resistance (Ohms)
        c1: float = 1200.0,     # Polarization capacitance (Farads)
    ):
        self.capacity_coulombs = capacity_ah * 3600.0
        self.r0 = r0
        self.r1 = r1
        self.c1 = c1
        self.tau = r1 * c1

    def ocv_from_soc(self, soc: float) -> float:
        """Standard NMC Open Circuit Voltage (OCV vs SoC) piecewise/linear fit."""
        s = float(np.clip(soc, 0.0, 1.0))
        # Maps 0.0 SoC -> 3.20V and 1.0 SoC -> 4.20V with realistic curvature
        return 3.20 + 0.85 * s + 0.15 * (s ** 2)

    def ocv_derivative(self, soc: float) -> float:
        """Analytical derivative d(OCV)/d(SoC) for EKF measurement Jacobian."""
        s = float(np.clip(soc, 0.0, 1.0))
        return 0.85 + 0.30 * s
