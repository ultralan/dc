"""频谱编码器: 2D CNN处理log-mel频谱图.

输入: 参考麦克风和均方根平均的log-mel拼接 [B, 2, T, mel_bins]
输出: 帧级嵌入 [B, T, out_dim]
"""

from __future__ import annotations

import torch
from torch import nn


class Residual2d(nn.Module):
    """2D残差块: 两层3×3卷积 + BatchNorm + 残差连接."""

    def __init__(self, channels: int) -> None:
        """初始化 2D 残差块."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行残差卷积并返回同形状特征图."""
        return self.activation(x + self.block(x))


class SpectrogramEncoder(nn.Module):
    """频谱编码器: 逐步下采样频率维度的2D CNN.

    结构: Conv → ResBlock×2 → 下采样(stride=2) → ResBlock×2 → 下采样 → Conv
    频率维度经过两次 stride=2 下采样, 最终通过平均池化得到帧级嵌入.
    """

    def __init__(self, out_dim: int = 128) -> None:
        """初始化频谱 CNN 编码器.

        ``out_dim`` 是输出帧级 embedding 的维度.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),    # 2通道: ref_mel + rms_mel
            nn.BatchNorm2d(32),
            nn.GELU(),
            Residual2d(32),
            Residual2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),  # 频率维×0.5
            nn.BatchNorm2d(64),
            nn.GELU(),
            Residual2d(64),
            Residual2d(64),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=(1, 2), padding=1),  # 频率维×0.5
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, 2, T, mel_bins]
        """编码 log-mel 特征.

        输入 ``[B, 2, T, mel_bins]``, 输出 ``[B, T, out_dim]``.
        """
        hidden = self.net(x)          # [B, out_dim, T, mel_bins/4]
        pooled = hidden.mean(dim=-1)  # 频率维平均池化 → [B, out_dim, T]
        return pooled.transpose(1, 2) # → [B, T, out_dim]
