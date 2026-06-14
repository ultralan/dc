"""圆阵几何工具入口.

这里导出阵列坐标、麦克风对、方位角网格、角度 wrap、远场延迟等函数.
这些工具被特征提取、标签构造和仿真模块共同使用.
"""

from .uca8 import (
    angular_velocity,
    azimuth_grid,
    default_mic_pairs,
    make_uniform_circular_array,
    pair_delays,
    relative_source_state,
    steering_delays,
    wrap_angle,
)

__all__ = [
    "angular_velocity",
    "azimuth_grid",
    "default_mic_pairs",
    "make_uniform_circular_array",
    "pair_delays",
    "relative_source_state",
    "steering_delays",
    "wrap_angle",
]
