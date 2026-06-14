from __future__ import annotations

"""SRP-PHAT 空间谱前端.

SRP-PHAT 会对一组候选方位角逐个计算相干得分, 得到
``[time, azimuth]`` 的空间谱. 传统方法通常直接取峰值作为 DOA;
本项目把空间谱作为神经网络输入, 让后续模型学习如何结合谱形、IPD 和 log-mel.
"""

import math
from collections.abc import Sequence

import torch
from torch import nn

from uca8.geometry.uca8 import pair_delays


class SRPPHAT(nn.Module):
    """计算多麦克风 STFT 的 SRP-PHAT 方位扫描图.

    初始化阶段预计算每个候选方位、每个麦克风对、每个频率点的 steering 相位.
    前向传播阶段只需要做 PHAT 归一化互谱和 steering 匹配, 输出
    ``[B, T, num_azimuths]``.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        mic_positions: torch.Tensor,
        mic_pairs: Sequence[tuple[int, int]],
        azimuths: torch.Tensor,
        sound_speed: float = 343.0,
        eps: float = 1e-8,
    ) -> None:
        """初始化 SRP-PHAT steering 表."""
        super().__init__()
        self.mic_pairs = list(mic_pairs)
        self.eps = eps

        # rfftfreq 与 torch.stft(return_complex=True) 的正频率 bin 对齐.
        freqs = torch.fft.rfftfreq(
            n_fft,
            d=1.0 / sample_rate,
            device=mic_positions.device,
            dtype=mic_positions.dtype,
        )
        delays = pair_delays(mic_positions, azimuths, self.mic_pairs, sound_speed=sound_speed)

        # steering[a, p, f] 表示假设声源来自第 a 个方位时,
        # 第 p 个麦克风对在第 f 个频率上的理论相位补偿.
        steering = torch.exp(1j * 2.0 * math.pi * freqs[None, None, :] * delays[:, :, None])

        # 去掉 DC 频点, 因为 DC 没有有效相位方向信息.
        self.register_buffer("steering", steering[:, :, 1:], persistent=False)

    def forward(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """从复数 STFT 计算 SRP-PHAT 空间谱.

        输入:
            ``stft_complex`` 形状为 ``[B, C, T, F]``.

        返回:
            ``[B, T, A]`` 的实数空间谱, A 为候选方位数.
        """
        if stft_complex.ndim != 4:
            raise ValueError("Expected STFT shape [batch, channels, frames, freqs].")
        pair_cross: list[torch.Tensor] = []
        for left, right in self.mic_pairs:
            cross = stft_complex[:, left, :, 1:] * torch.conj(stft_complex[:, right, :, 1:])

            # PHAT 归一化只保留相位信息, 降低能量大小对空间谱的影响.
            cross = cross / cross.abs().clamp_min(self.eps)
            pair_cross.append(cross)
        normalized = torch.stack(pair_cross, dim=1)

        # bptf: batch/pair/time/freq, apf: azimuth/pair/freq -> bta.
        score = torch.einsum("bptf,apf->bta", normalized, self.steering)
        return score.real / (len(self.mic_pairs) * self.steering.shape[-1])
