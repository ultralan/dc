"""模型组件.

``UCA8TrackTrendNet`` 是完整网络入口. 本包把频谱编码器、空间编码器、
TCN 骨干和预测头拆成独立文件, 这样代码结构能和论文里的架构图对应起来.
"""

from .tracktrend_net import UCA8TrackTrendNet

__all__ = ["UCA8TrackTrendNet"]
