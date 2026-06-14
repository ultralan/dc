from __future__ import annotations

"""阵列几何和角度工具函数.

本文件只处理和麦克风阵列/声源位置有关的纯数学计算:
圆阵坐标、默认麦克风对、方位角网格、角度 wrap、远场 steering delay 等.
这些函数不依赖数据集和模型, 因此可以被特征提取、标签构造和仿真模块复用.
"""

import math
from collections.abc import Sequence

import torch


def make_uniform_circular_array(
    num_mics: int = 8,
    radius: float = 0.045,
    z: float = 0.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构造二维均匀圆阵坐标.

    mic0 放在 x 轴正方向, 其余麦克风按逆时针均匀分布.
    返回形状为 ``[num_mics, 3]`` 的坐标张量, 单位与 ``radius`` 一致.
    """
    indices = torch.arange(num_mics, device=device, dtype=dtype)
    angles = 2.0 * math.pi * indices / float(num_mics)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z_axis = torch.full_like(x, z)
    return torch.stack((x, y, z_axis), dim=-1)


def default_mic_pairs(num_mics: int = 8) -> list[tuple[int, int]]:
    """返回圆阵默认麦克风对.

    组合包括:
    - 相邻麦克风对, 捕捉局部相位差;
    - 对径麦克风对, 捕捉更大孔径的方向差异.
    当前实现要求麦克风数为偶数.
    """
    if num_mics % 2 != 0:
        raise ValueError("default_mic_pairs expects an even microphone count.")
    adjacent = [(idx, (idx + 1) % num_mics) for idx in range(num_mics)]
    diametric = [(idx, (idx + num_mics // 2) % num_mics) for idx in range(num_mics // 2)]
    return adjacent + diametric


def azimuth_grid(
    num_bins: int = 72,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """返回 ``[-pi, pi)`` 上均匀划分的方位角网格, 单位为弧度."""
    return torch.linspace(-math.pi, math.pi, steps=num_bins + 1, device=device, dtype=dtype)[:-1]


def relative_source_state(
    source_positions: torch.Tensor,
    array_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算声源相对阵列中心的方位角和水平距离.

    ``source_positions`` 和 ``array_positions`` 可以带任意前缀维度,
    但最后一维必须是 xyz 坐标. 返回 ``theta`` 和 ``rho``:
    - ``theta``: x-y 平面方位角, 单位弧度;
    - ``rho``: x-y 平面距离.
    """
    delta = source_positions[..., :2] - array_positions[..., :2]
    theta = torch.atan2(delta[..., 1], delta[..., 0])
    rho = torch.linalg.norm(delta, dim=-1)
    return theta, rho


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """把任意角度 wrap 到 ``[-pi, pi]`` 的等价范围内."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angular_velocity(theta: torch.Tensor, dt: float) -> torch.Tensor:
    """沿第 0 维计算角速度.

    相邻帧差值先经过 ``wrap_angle`` 处理, 避免从 ``pi`` 到 ``-pi`` 的周期跳变
    被误算成巨大速度.
    """
    velocity = torch.zeros_like(theta)
    if theta.shape[0] <= 1:
        return velocity
    velocity[1:] = wrap_angle(theta[1:] - theta[:-1]) / dt
    velocity[0] = velocity[1]
    return velocity


def steering_delays(
    mic_positions: torch.Tensor,
    azimuths: torch.Tensor,
    *,
    sound_speed: float = 343.0,
) -> torch.Tensor:
    """计算远场平面波到各麦克风的相对延迟.

    返回形状为 ``[num_azimuths, num_mics]``. 这里使用二维方位角,
    elevation 固定为 0, 适配当前水平面定位任务.
    """
    directions = torch.stack(
        (torch.cos(azimuths), torch.sin(azimuths), torch.zeros_like(azimuths)),
        dim=-1,
    )
    return directions @ mic_positions.T / sound_speed


def pair_delays(
    mic_positions: torch.Tensor,
    azimuths: torch.Tensor,
    mic_pairs: Sequence[tuple[int, int]],
    *,
    sound_speed: float = 343.0,
) -> torch.Tensor:
    """计算麦克风对之间的远场延迟差.

    SRP-PHAT 和 IPD 类特征通常使用麦克风对差分量, 因此这里把单麦克风延迟
    转成 ``delay_i - delay_j``. 返回形状为 ``[num_azimuths, num_pairs]``.
    """
    delays = steering_delays(mic_positions, azimuths, sound_speed=sound_speed)
    pair_values = [delays[:, i] - delays[:, j] for i, j in mic_pairs]
    return torch.stack(pair_values, dim=-1)
