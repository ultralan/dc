from __future__ import annotations

import numpy as np


class AzimuthKalmanTracker:
    """A lightweight 2-state Kalman tracker for azimuth and angular velocity."""

    def __init__(
        self,
        *,
        dt: float = 0.01,
        process_var: float = 1e-2,
        measurement_var: float = 5e-2,
    ) -> None:
        self.state = np.zeros((2, 1), dtype=np.float32)
        self.cov = np.eye(2, dtype=np.float32)
        self.transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        self.observation = np.array([[1.0, 0.0]], dtype=np.float32)
        self.process_noise = process_var * np.eye(2, dtype=np.float32)
        self.measurement_noise = np.array([[measurement_var]], dtype=np.float32)

    def predict(self) -> tuple[float, float]:
        self.state = self.transition @ self.state
        self.cov = self.transition @ self.cov @ self.transition.T + self.process_noise
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, theta: float) -> tuple[float, float]:
        measurement = np.array([[theta]], dtype=np.float32)
        innovation = measurement - self.observation @ self.state
        innovation_cov = self.observation @ self.cov @ self.observation.T + self.measurement_noise
        kalman_gain = self.cov @ self.observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(2, dtype=np.float32)
        self.cov = (identity - kalman_gain @ self.observation) @ self.cov
        return float(self.state[0, 0]), float(self.state[1, 0])
