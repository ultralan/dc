from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CountHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeatmapHead(nn.Module):
    def __init__(self, in_dim: int, bins: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotHead(nn.Module):
    def __init__(self, in_dim: int, max_sources: int) -> None:
        super().__init__()
        self.max_sources = max_sources
        self.slot_queries = nn.Parameter(torch.randn(max_sources, in_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x[:, None, :] + self.slot_queries[None, :, :]
        return self.mlp(hidden)


class FutureDecoder(nn.Module):
    def __init__(self, model_dim: int, future_frames: int) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.temporal = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.project = nn.Linear(model_dim, model_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal(history.transpose(1, 2)).transpose(1, 2)
        if hidden.shape[1] >= self.future_frames:
            hidden = hidden[:, -self.future_frames :]
        else:
            pad = self.future_frames - hidden.shape[1]
            hidden = F.pad(hidden, (0, 0, pad, 0))
        return self.project(hidden)
