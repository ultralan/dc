from __future__ import annotations

import torch
from torch import nn


class Residual2d(nn.Module):
    def __init__(self, channels: int) -> None:
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
        return self.activation(x + self.block(x))


class SpectrogramEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            Residual2d(32),
            Residual2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            Residual2d(64),
            Residual2d(64),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net(x)
        pooled = hidden.mean(dim=-1)
        return pooled.transpose(1, 2)
