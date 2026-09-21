"""
Discrete Extended Kalman Filter (EKF) for Synchronous Generator Dynamics.
Tracks 4th-order transient states [delta, omega_dev, E'_q, E'_d] from PMU telemetry.
"""
from typing import Dict, Tuple
import numpy as np


class SynchronousGeneratorEKF:
    def __init__(self, dt: float = 0.02, fn: float = 50.0):
        self.dt = dt
        self.omega_0 = 2.0 * np.pi * fn

        # Physical machine parameters (p.u.)
        self.H = 3.5          # Inertia constant (seconds)
        self.D = 1.0          # Damping coefficient
        self.Xd = 1.8         # d-axis synchronous reactance
        self.Xdp = 0.3        # d-axis transient reactance
        self.Xq = 1.7         # q-axis synchronous reactance
        self.Xqp = 0.55       # q-axis transient reactance
        self.Tdo_p = 8.0      # d-axis open-circuit transient time constant
        self.Tqo_p = 0.4      # q-axis open-circuit transient time constant

        # State vector: [delta (rad), Delta_omega (p.u.), E'_q (p.u.), E'_d (p.u.)]
        self.x = np.array([0.15, 0.0, 1.05, 0.0], dtype=np.float64)

        # Covariance matrices
        self.P = np.eye(4, dtype=np.float64) * 1e-3
        self.Q = np.diag([1e-6, 1e-5, 1e-4, 1e-4])  # Process noise
        self.R = np.diag([1e-4, 1e-4])              # Measurement noise (P_e, Q_e)

    def state_transition(self, x: np.ndarray, Pm: float, Efd: float, Vt: float) -> np.ndarray:
        """Continuous dynamic equations evaluated via RK4 integration."""
        delta, d_omega, Eq_p, Ed_p = x

        # Stator electrical power calculations
        Id = (Eq_p - Vt * np.cos(delta)) / self.Xdp
        Iq = (-Ed_p + Vt * np.sin(delta)) / self.Xqp
        Pe = Vt * np.sin(delta) * Id + Vt * np.cos(delta) * Iq

        # Differential equations
        d_delta = self.omega_0 * d_omega
        d_d_omega = (1.0 / (2.0 * self.H)) * (Pm - Pe - self.D * d_omega)
        d_Eq_p = (1.0 / self.Tdo_p) * (Efd - Eq_p - (self.Xd - self.Xdp) * Id)
        d_Ed_p = (1.0 / self.Tqo_p) * (-Ed_p + (self.Xq - self.Xqp) * Iq)

        f_dyn = np.array([d_delta, d_d_omega, d_Eq_p, d_Ed_p], dtype=np.float64)
        return x + self.dt * f_dyn

    def compute_jacobian_f(self, x: np.ndarray, Vt: float) -> np.ndarray:
        """Analytic state transition Jacobian F_k = df/dx."""
        delta, _, _, _ = x
        F = np.eye(4, dtype=np.float64)

        F[0, 1] = self.omega_0 * self.dt

        # Partial derivatives of electrical torque Pe with respect to delta
        term1 = (Vt**2 / self.Xdp) * (np.sin(delta)**2)
        term2 = (Vt**2 / self.Xqp) * (np.cos(delta)**2)
        term3 = (Vt / self.Xdp) * x[2] * np.cos(delta)
        dPe_ddelta = term1 - term2 + term3

        F[1, 0] = -(self.dt / (2.0 * self.H)) * dPe_ddelta
        F[1, 1] = 1.0 - (self.dt * self.D) / (2.0 * self.H)
        F[1, 2] = -(self.dt / (2.0 * self.H)) * ((Vt / self.Xdp) * np.sin(delta))
        F[1, 3] = -(self.dt / (2.0 * self.H)) * ((Vt / self.Xqp) * np.cos(delta))

        # Field voltage dissipation decay terms
        F[2, 2] = 1.0 - (self.dt / self.Tdo_p) * (self.Xd / self.Xdp)
        F[3, 3] = 1.0 - (self.dt / self.Tqo_p) * (self.Xq / self.Xqp)

        return F

    def compute_measurement_h(self, x: np.ndarray, Vt: float) -> np.ndarray:
        """Observation function mapping internal states to terminal active/reactive power."""
        delta, _, Eq_p, Ed_p = x
        Id = (Eq_p - Vt * np.cos(delta)) / self.Xdp
        Iq = (-Ed_p + Vt * np.sin(delta)) / self.Xqp

        Pe = Vt * np.sin(delta) * Id + Vt * np.cos(delta) * Iq
        Qe = Vt * np.cos(delta) * Id - Vt * np.sin(delta) * Iq
        return np.array([Pe, Qe], dtype=np.float64)

    def compute_jacobian_h(self, x: np.ndarray, Vt: float) -> np.ndarray:
        """Observation Jacobian H_k = dh/dx."""
        delta, _, _, _ = x
        H = np.zeros((2, 4), dtype=np.float64)

        H[0, 0] = (Vt**2 / self.Xdp - Vt**2 / self.Xqp) * np.sin(2 * delta)
        H[0, 2] = (Vt / self.Xdp) * np.sin(delta)
        H[0, 3] = (Vt / self.Xqp) * np.cos(delta)

        term_sin = (Vt**2 / self.Xdp) * (np.sin(delta)**2)
        term_cos = (Vt**2 / self.Xqp) * (np.cos(delta)**2)
        H[1, 0] = -term_sin - term_cos
        H[1, 2] = (Vt / self.Xdp) * np.cos(delta)
        H[1, 3] = -(Vt / self.Xqp) * np.sin(delta)

        return H

    def step(
        self,
        z_measured: Tuple[float, float],
        Vt: float,
        Pm: float = 0.8,
        Efd: float = 1.1
    ) -> Dict[str, float]:
        """
        Execute one prediction-update cycle of the EKF.
        z_measured: (Pe_measured, Qe_measured)
        """
        # 1. Prediction step
        x_pred = self.state_transition(self.x, Pm, Efd, Vt)
        F = self.compute_jacobian_f(self.x, Vt)
        P_pred = F @ self.P @ F.T + self.Q

        # 2. Innovation step
        z_act = np.array(z_measured, dtype=np.float64)
        z_hat = self.compute_measurement_h(x_pred, Vt)
        y = z_act - z_hat

        # 3. Kalman Gain computation
        H = self.compute_jacobian_h(x_pred, Vt)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 4. State update
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ H) @ P_pred

        return {
            "rotor_angle_rad": float(self.x[0]),
            "speed_deviation_pu": float(self.x[1]),
            "transient_emf_q_pu": float(self.x[2]),
            "transient_emf_d_pu": float(self.x[3]),
            "innovation_norm": float(np.linalg.norm(y)),
        }