"""推理后处理工具入口.

这里的工具不参与模型训练:
- ``estimate_source_count_from_heatmap`` 用 heatmap 局部峰估计声源数;
- ``AzimuthKalmanTracker`` 对方位角序列做简单 Kalman 平滑.
"""

from .heatmap_counter import estimate_source_count_from_heatmap
from .kalman_tracker import AzimuthKalmanTracker

__all__ = ["AzimuthKalmanTracker", "estimate_source_count_from_heatmap"]
