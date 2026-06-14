from __future__ import annotations

"""IPD 特征计算.

IPD, Inter-channel Phase Difference, 表示两个麦克风在同一频率 bin 上的相位差.
它是深度声源定位里常用的空间特征: 模型不用直接从原始复数 STFT 里学习相位差,
而是接收 ``cos(phase)`` 和 ``sin(phase)`` 两个连续特征, 避免角度周期断点.
"""

from collections.abc import Sequence

import torch


def _select_frequency_indices(num_freqs: int, num_bins: int, device: torch.device) -> torch.Tensor:
    """选择用于 IPD 的频率 bin.

    跳过 DC 附近的第 0 个 bin, 在可用频带上均匀采样 ``num_bins`` 个频点.
    如果请求数量超过 STFT 频点数, 则直接返回全部频点.
    """
    if num_bins >= num_freqs:
        return torch.arange(num_freqs, device=device)
    return torch.linspace(1, num_freqs - 1, steps=num_bins, device=device).round().long()


def compute_ipd_features(
    stft_complex: torch.Tensor,
    mic_pairs: Sequence[tuple[int, int]],
    *,
    num_bins: int = 32,
) -> torch.Tensor:
    """计算堆叠的 cos/sin IPD 特征.

    参数:
        stft_complex: 复数 STFT, 形状 ``[B, C, T, F]``.
        mic_pairs: 需要计算相位差的麦克风对列表.
        num_bins: 每个麦克风对保留的频率 bin 数.

    返回:
        形状 ``[B, 2 * num_pairs, T, num_bins]``.
        每个麦克风对贡献两个通道: ``cos(IPD)`` 和 ``sin(IPD)``.
    """
    if stft_complex.ndim != 4:
        raise ValueError("Expected STFT shape [batch, channels, frames, freqs].")
    _, _, _, freqs = stft_complex.shape
    indices = _select_frequency_indices(freqs, num_bins, stft_complex.device)
    pair_feats: list[torch.Tensor] = []
    for left, right in mic_pairs:
        # cross 的相位等价于 left/right 两通道的相位差.
        cross = stft_complex[:, left] * torch.conj(stft_complex[:, right])
        phase = torch.angle(cross[..., indices])
        pair_feats.extend((torch.cos(phase), torch.sin(phase)))
    return torch.stack(pair_feats, dim=1)
