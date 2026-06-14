"""因果时序卷积网络 (Causal TCN) 骨干.

采用膨胀因果卷积 (dilated causal convolution) + 门控激活 + 残差连接,
在不使用循环结构的情况下建模长范围时序依赖.
默认膨胀率 [1,2,4,8,16,32] 各重复2次(共12层), 感受野覆盖约数百帧.
"""

from __future__ import annotations

import torch
from torch import nn


class TCNBlock(nn.Module):
    """单个TCN残差块: 膨胀因果卷积 + 门控激活 + 残差连接 + LayerNorm."""

    def __init__(self, model_dim: int, dilation: int, kernel_size: int, dropout: float) -> None:
        """初始化一个膨胀因果卷积块."""
        super().__init__()
        # 左侧填充保证因果性: 只看过去, 不看未来
        padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            model_dim,
            model_dim * 2,  # 输出通道翻倍, 一半做value, 一半做gate
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.out = nn.Conv1d(model_dim, model_dim, kernel_size=1)  # 1×1卷积降维
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """处理一段帧级序列, 输入输出形状均为 ``[B, T, D]``."""
        residual = x
        hidden = x.transpose(1, 2)     # → [B, D, T] 适配Conv1d
        hidden = self.conv(hidden)      # → [B, 2D, T+pad]
        hidden = hidden[..., : x.shape[1]]  # 裁掉右侧填充, 保证因果性
        # 门控激活: value * sigmoid(gate), 类似WaveNet/LSTM的门控机制
        value, gate = hidden.chunk(2, dim=1)
        hidden = value * torch.sigmoid(gate)
        hidden = self.out(hidden).transpose(1, 2)  # → [B, T, D]
        hidden = self.dropout(hidden)
        return self.norm(hidden + residual)  # 残差连接 + LayerNorm


class CausalTCN(nn.Module):
    """因果TCN骨干: 堆叠多个TCNBlock, 每种膨胀率重复2次以增加模型容量."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """按给定膨胀率堆叠 TCNBlock."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TCNBlock(model_dim, dilation, kernel_size, dropout)
                for dilation in dilations
                for _ in range(2)  # 每种膨胀率重复2个block
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, model_dim]
        """顺序通过所有 TCN block."""
        for block in self.blocks:
            x = block(x)
        return x
