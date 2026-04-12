from __future__ import annotations

import torch
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


class TemporalSlotDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        max_sources: int,
        *,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_sources = max_sources
        self.slot_queries = nn.Parameter(torch.randn(max_sources, in_dim))
        self.query_context = nn.Linear(in_dim, in_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 5),
        )

    def forward(
        self,
        history: torch.Tensor,
        *,
        query_context: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = history.shape[0]
        query = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        if query_context is not None:
            if query_context.ndim == 2:
                query_context = query_context[:, None, :].expand(-1, self.max_sources, -1)
            query = query + self.query_context(query_context)
        attended, _ = self.attention(query, history, history, need_weights=False)
        slot_context = self.norm(query + attended)
        slot_logits = self.mlp(slot_context)
        if return_context:
            return slot_logits, slot_context
        return slot_logits


class FutureRolloutDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        future_frames: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_slot_context: bool = True,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.num_layers = num_layers
        self.use_slot_context = use_slot_context
        self.future_step_embeddings = nn.Parameter(torch.randn(future_frames, model_dim))
        self.context_proj = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
        )
        self.slot_proj = nn.Linear(model_dim, model_dim)
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.project = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )

    def forward(
        self,
        history: torch.Tensor,
        *,
        current: torch.Tensor,
        slot_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_history = history.mean(dim=1)
        if slot_context is not None and self.use_slot_context:
            pooled_slots = slot_context.mean(dim=1)
        else:
            pooled_slots = torch.zeros_like(current)
        seed = self.context_proj(torch.cat([pooled_history, current, pooled_slots], dim=-1))
        rollout_input = (
            self.future_step_embeddings.unsqueeze(0)
            + current[:, None, :]
            + pooled_history[:, None, :]
        )
        if slot_context is not None and self.use_slot_context:
            rollout_input = rollout_input + self.slot_proj(pooled_slots)[:, None, :]
        hidden0 = seed.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        rollout, _ = self.gru(rollout_input, hidden0)
        return self.project(rollout)


class FutureSlotHead(nn.Module):
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

    def forward(self, future_hidden: torch.Tensor, slot_context: torch.Tensor) -> torch.Tensor:
        hidden = (
            future_hidden[:, :, None, :]
            + slot_context[:, None, :, :]
            + self.slot_queries[None, None, :, :]
        )
        return self.mlp(hidden)
