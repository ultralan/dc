"""数据集和标签构造入口.

本包对外暴露三类数据源:
- ``RealMANRing2Dataset``: 当前主实验使用的 RealMAN ring2 8ch 数据集;
- ``RealMANRing2HybridDataset``: 真实 RealMAN + 合成 curriculum 的扩展数据集;
- ``LocataLikeTrackTrendDataset`` / ``SyntheticTrackTrendDataset``: 通用和测试数据源.
"""

from .dataset_tracktrend import (
    LocataLikeTrackTrendDataset,
    SyntheticTrackTrendDataset,
)
from .label_builder import TrackTrendLabelBuilder
from .realman_ring2_dataset import RealMANRing2Dataset
from .realman_ring2_hybrid_dataset import RealMANRing2HybridDataset

__all__ = [
    "LocataLikeTrackTrendDataset",
    "SyntheticTrackTrendDataset",
    "TrackTrendLabelBuilder",
    "RealMANRing2HybridDataset",
    "RealMANRing2Dataset",
]
