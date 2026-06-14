"""UCA8 声源定位/跟踪工程包.

代码按模型流水线拆分, 不是按实验脚本堆放:

- ``data``: 读取 RealMAN/UCA 风格数据, 构造训练样本和监督标签.
- ``features``: 从原始多通道波形提取 log-mel、IPD、SRP-PHAT、VAD 特征.
- ``models``: 神经网络主体, 包括编码器、TCN 骨干和预测头.
- ``losses``: 训练时使用的多任务监督目标.
- ``metrics``: 论文/实验对齐用指标, 例如 MAE 和 ACC@5.
- ``postprocess``: 推理后的可选计数、跟踪和平滑工具.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
