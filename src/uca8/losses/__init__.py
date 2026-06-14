"""损失函数入口.

当前主要导出 ``TrackTrendMultiTaskLoss``, 用于把定位、计数、slot 跟踪、
未来预测和运动趋势等监督项合并成训练总损失.
"""

from .multi_task_loss import TrackTrendMultiTaskLoss

__all__ = ["TrackTrendMultiTaskLoss"]
