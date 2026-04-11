from __future__ import annotations

import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, model_dim: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            model_dim,
            model_dim * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.out = nn.Conv1d(model_dim, model_dim, kernel_size=1)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = x.transpose(1, 2)
        hidden = self.conv(hidden)
        hidden = hidden[..., : x.shape[1]]
        value, gate = hidden.chunk(2, dim=1)
        hidden = value * torch.sigmoid(gate)
        hidden = self.out(hidden).transpose(1, 2)
        hidden = self.dropout(hidden)
        return self.norm(hidden + residual)


class CausalTCN(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int = 256,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TCNBlock(model_dim, dilation, kernel_size, dropout)
                for dilation in dilations
                for _ in range(2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
