from __future__ import annotations

import torch
from torch import nn

from uca8.models.spec_encoder import Residual2d


class IPDEncoder(nn.Module):
    """编码麦克风间相位差特征.

    IPD frontend 输出麦克风对之间的相位线索, 形状为
    ``[B, pair_features, T, ipd_bins]``. 本编码器把频率/IPD bin 维压缩成
    帧级空间 embedding: ``[B, T, out_dim]``.
    """

    def __init__(self, in_channels: int = 24, out_dim: int = 64) -> None:
        """初始化 IPD 编码器.

        ``in_channels`` 通常等于 ``2 * mic_pair_count``.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.GELU(),
            Residual2d(32),
            Residual2d(32),
            nn.Conv2d(32, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码 IPD 特征并输出帧级 embedding."""
        hidden = self.net(x)
        return hidden.mean(dim=-1).transpose(1, 2)


class SRPEncoder(nn.Module):
    """编码 SRP-PHAT 空间谱, 并保留方位 bin 位置信息.

    SRP-PHAT 本身就是按方位角扫描得到的空间谱.
    如果在学习前直接对 azimuth 维做平均, 方位线索会被抹掉.
    所以这里在卷积后展平 ``channel x azimuth`` 维, 再投影成和其他分支一致的
    帧级 embedding.
    """

    def __init__(self, out_dim: int = 64, azimuth_bins: int = 72) -> None:
        """初始化 SRP 编码器.

        ``azimuth_bins`` 必须和 frontend 生成的 SRP 方位 bin 数一致.
        """
        super().__init__()
        self.azimuth_bins = azimuth_bins
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.project = nn.Linear(32 * azimuth_bins, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码 SRP 空间谱并输出帧级 embedding."""
        hidden = self.conv(x)
        if hidden.shape[-1] != self.azimuth_bins:
            raise ValueError(
                f"Expected SRP azimuth dimension {self.azimuth_bins}, got {hidden.shape[-1]}."
            )

        # 保留 azimuth bin 顺序: [B, C, T, A] -> [B, T, C*A].
        # 这是 SRP 分支和普通频谱池化最关键的区别.
        hidden = hidden.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.project(hidden)
