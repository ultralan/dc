"""声学特征提取模块.

这里集中放所有手工/信号处理特征:
STFT、log-mel、IPD、SRP-PHAT 都在本包内完成.
模型文件只负责学习结构, 消融开关统一收在 ``UCAFeatureFrontend`` 里.
"""

from .ipd import compute_ipd_features
from .srp_phat import SRPPHAT
from .stft import MultiChannelSTFT, UCAFeatureFrontend

__all__ = ["compute_ipd_features", "MultiChannelSTFT", "SRPPHAT", "UCAFeatureFrontend"]
