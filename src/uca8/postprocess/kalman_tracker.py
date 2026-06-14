from __future__ import annotations

"""方位角 Kalman 平滑器.

该模块是推理后的可选后处理, 用一个二状态线性 Kalman filter 平滑方位角序列:
状态为 ``[theta, omega]``, 分别表示方位角和角速度. 它不参与训练,
只用于演示或传统跟踪 baseline.
"""

import numpy as np


class AzimuthKalmanTracker:
    """轻量二状态 Kalman tracker.

    状态向量:
        ``theta``: 当前方位角.
        ``omega``: 方位角速度.

    观测量只有 ``theta``. ``omega`` 通过状态转移矩阵隐式估计.
    """

    def __init__(
        self,
        *,
        dt: float = 0.01,
        process_var: float = 1e-2,
        measurement_var: float = 5e-2,
    ) -> None:
        """初始化 Kalman filter 参数."""
        # state = [theta, omega]^T.
        self.state = np.zeros((2, 1), dtype=np.float32)
        self.cov = np.eye(2, dtype=np.float32)

        # 匀速角运动假设: theta_t = theta_{t-1} + omega * dt.
        self.transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        self.observation = np.array([[1.0, 0.0]], dtype=np.float32)
        self.process_noise = process_var * np.eye(2, dtype=np.float32)
        self.measurement_noise = np.array([[measurement_var]], dtype=np.float32)

    def predict(self) -> tuple[float, float]:
        """执行预测步, 返回预测的 ``(theta, omega)``."""
        self.state = self.transition @ self.state
        self.cov = self.transition @ self.cov @ self.transition.T + self.process_noise
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, theta: float) -> tuple[float, float]:
        """用新的方位角观测更新状态, 返回更新后的 ``(theta, omega)``."""
        measurement = np.array([[theta]], dtype=np.float32)
        innovation = measurement - self.observation @ self.state
        innovation_cov = self.observation @ self.cov @ self.observation.T + self.measurement_noise
        kalman_gain = self.cov @ self.observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(2, dtype=np.float32)
        self.cov = (identity - kalman_gain @ self.observation) @ self.cov
        return float(self.state[0, 0]), float(self.state[1, 0])
