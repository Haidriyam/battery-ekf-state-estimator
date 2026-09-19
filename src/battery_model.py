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
        """Empirical polynomial approximation of Open Circuit Voltage (OCV vs SoC)."""
        soc_clamped = np.clip(soc, 0.0, 1.0)
        # 6th-order empirical fit for standard NMC chemistry
        coeffs = [3.0, 1.5, -2.1, 4.2, -4.5, 2.1, 0.0]
        return float(np.polyval(coeffs, soc_clamped))

    def ocv_derivative(self, soc: float) -> float:
        """Analytical derivative d(OCV)/d(SoC) required for EKF measurement Jacobian."""
        soc_clamped = np.clip(soc, 0.0, 1.0)
        coeffs = [3.0, 1.5, -2.1, 4.2, -4.5, 2.1, 0.0]
        d_coeffs = np.polyder(coeffs)
        return float(np.polyval(d_coeffs, soc_clamped))