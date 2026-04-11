from __future__ import annotations

import torch
from torch import nn

from uca8.models.spec_encoder import Residual2d


class IPDEncoder(nn.Module):
    def __init__(self, in_channels: int = 24, out_dim: int = 64) -> None:
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
        hidden = self.net(x)
        return hidden.mean(dim=-1).transpose(1, 2)


class SRPEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.project = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(x).mean(dim=-1).transpose(1, 2)
        return self.project(hidden)
