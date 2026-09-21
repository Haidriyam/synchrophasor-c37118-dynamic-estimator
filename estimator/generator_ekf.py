"""
Discrete Extended Kalman Filter (EKF) for Synchronous Generator Swing Dynamics.
Tracks non-linear rotor angle (delta) and angular speed deviation (omega_dev) from PMU streams.
"""
from typing import Dict
import numpy as np


class SynchronousGeneratorEKF:
    def __init__(self, dt: float = 0.02, fn: float = 50.0):
        self.dt = dt
        self.omega_0 = 2.0 * np.pi * fn

        # Physical machine parameters (p.u.)
        self.H = 3.5          # Inertia constant (s)
        self.D = 1.0          # Damping coefficient
        self.Xd_p = 0.3       # Transient reactance
        self.E_p = 1.05       # Internal transient voltage magnitude

        # State vector: [delta (rad), omega_dev (rad/s)]
        self.x = np.array([0.35, 0.0], dtype=np.float64)

        # Covariance matrices
        self.P = np.diag([1e-2, 1e-2])
        self.Q = np.diag([1e-5, 1e-4])
        self.R = np.array([[1e-3]], dtype=np.float64)

    def state_transition(self, x: np.ndarray, Pm: float, Vt: float) -> np.ndarray:
        """Non-linear generator swing dynamic state transition."""
        delta, w_dev = x
        P_max = (self.E_p * Vt) / self.Xd_p
        Pe = P_max * np.sin(delta)

        d_delta = w_dev
        d_w_dev = (self.omega_0 / (2.0 * self.H)) * (Pm - Pe - self.D * (w_dev / self.omega_0))

        delta_next = delta + self.dt * d_delta
        w_dev_next = w_dev + self.dt * d_w_dev
        return np.array([delta_next, w_dev_next], dtype=np.float64)

    def compute_jacobian_f(self, x: np.ndarray, Vt: float) -> np.ndarray:
        """Analytic Jacobian of the swing state transition."""
        delta, _ = x
        P_max = (self.E_p * Vt) / self.Xd_p
        dPe_ddelta = P_max * np.cos(delta)

        F = np.eye(2, dtype=np.float64)
        F[0, 1] = self.dt
        F[1, 0] = -self.dt * (self.omega_0 / (2.0 * self.H)) * dPe_ddelta
        F[1, 1] = 1.0 - self.dt * (self.D / (2.0 * self.H))
        return F

    def compute_measurement_h(self, x: np.ndarray, Vt: float) -> float:
        """Observation function mapping rotor angle to active power."""
        delta, _ = x
        P_max = (self.E_p * Vt) / self.Xd_p
        return float(P_max * np.sin(delta))

    def compute_jacobian_h(self, x: np.ndarray, Vt: float) -> np.ndarray:
        """Analytic Jacobian of the observation function."""
        delta, _ = x
        P_max = (self.E_p * Vt) / self.Xd_p
        return np.array([[P_max * np.cos(delta), 0.0]], dtype=np.float64)

    def step(self, Pe_measured: float, Vt: float, Pm: float) -> Dict[str, float]:
        """Execute one EKF prediction and measurement update cycle."""
        # 1. Prediction
        x_pred = self.state_transition(self.x, Pm, Vt)
        F = self.compute_jacobian_f(self.x, Vt)
        P_pred = F @ self.P @ F.T + self.Q

        # 2. Innovation
        z_hat = self.compute_measurement_h(x_pred, Vt)
        y = Pe_measured - z_hat

        # 3. Kalman Gain
        H = self.compute_jacobian_h(x_pred, Vt)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 4. State Update
        self.x = x_pred + (K @ np.array([[y]])).flatten()
        self.P = (np.eye(2) - K @ H) @ P_pred

        return {
            "rotor_angle_rad": float(self.x[0]),
            "speed_deviation_rad_s": float(self.x[1]),
            "innovation": float(y),
        }