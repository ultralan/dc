"""模型输出头: 声源计数、方位热力图、槽位解码器、未来rollout解码器。

槽位(slot)是本模型跟踪声源的核心数据结构, 每个槽位输出5维向量:
  [activity_logit, sin(θ), cos(θ), ρ(距离), ω(角速度)]
通过 activity 判断槽位是否活跃, sin/cos 编码方位角(避免周期跳变).
"""

from __future__ import annotations

import torch
from torch import nn


class CountHead(nn.Module):
    """声源数分类头: LayerNorm → Linear, 输出 num_classes 类logits."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        """初始化分类头.

        ``in_dim`` 是输入特征维度, ``num_classes`` 是声源数类别数.
        """
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把输入特征映射成类别 logits."""
        return self.net(x)


class HeatmapHead(nn.Module):
    """方位角热力图头: 输出 heatmap_bins 维logits, sigmoid后即为各方位bin的声源概率."""

    def __init__(self, in_dim: int, bins: int) -> None:
        """初始化 heatmap 预测头.

        ``bins`` 对应方位角离散 bin 数, 当前常用 72 个 bin.
        """
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出未 sigmoid 的 heatmap logits."""
        return self.net(x)


class TemporalSlotDecoder(nn.Module):
    """时序槽位解码器: 基于可学习query + 多头交叉注意力.

    每个 max_sources 对应一个可学习的 slot_query, 与TCN输出的时序特征做交叉注意力,
    从而为每个槽位聚合其关联的时序信息, 输出该槽位的状态(activity, sin, cos, rho, omega).
    """

    def __init__(
        self,
        in_dim: int,
        max_sources: int,
        *,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """初始化 slot decoder.

        每个 slot 有一个可学习 query. Query 与历史时序特征做 cross-attention,
        得到该 slot 的上下文表示.
        """
        super().__init__()
        self.max_sources = max_sources
        self.slot_queries = nn.Parameter(torch.randn(max_sources, in_dim))  # 可学习槽位查询向量
        self.query_context = nn.Linear(in_dim, in_dim)   # 将当前帧表征融入query
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_dim)  # 残差连接后的LayerNorm
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 5),  # 输出5维: [activity, sin, cos, rho, omega]
        )

    def forward(
        self,
        history: torch.Tensor,  # [B, T, in_dim] TCN输出的时序特征
        *,
        query_context: torch.Tensor | None = None,  # [B, in_dim] 当前帧表征
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """从历史时序特征中解码当前 slot 状态.

        ``history`` 形状为 ``[B, T, D]``. 如果传入 ``query_context``,
        当前帧特征会注入每个 slot query, 帮助 slot 聚焦当前状态.
        """
        batch = history.shape[0]
        query = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        if query_context is not None:
            # 将当前帧信息加到每个slot query上, 引导解码
            if query_context.ndim == 2:
                query_context = query_context[:, None, :].expand(-1, self.max_sources, -1)
            query = query + self.query_context(query_context)
        # query(Q) × history(K=V) 的交叉注意力
        attended, _ = self.attention(query, history, history, need_weights=False)
        slot_context = self.norm(query + attended)  # 残差 + LayerNorm
        slot_logits = self.mlp(slot_context)        # [B, max_sources, 5]
        if return_context:
            return slot_logits, slot_context  # 返回context供未来预测使用
        return slot_logits


class FutureRolloutDecoder(nn.Module):
    """未来帧rollout解码器: 基于GRU的自回归预测.

    核心思路:
      1. 用(历史池化, 当前帧, 槽位池化)三者拼接作为GRU初始隐状态seed
      2. GRU输入 = 可学习的步嵌入 + 当前帧 + 历史池化 + 槽位上下文
      3. GRU rollout future_frames 步, 输出未来各帧的隐藏状态
    """

    def __init__(
        self,
        model_dim: int,
        future_frames: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_slot_context: bool = True,  # 是否将槽位上下文注入未来预测
    ) -> None:
        """初始化未来 rollout decoder.

        GRU 的初始状态来自历史池化、当前帧和 slot 上下文,
        输入序列由可学习步嵌入和上下文相加得到.
        """
        super().__init__()
        self.future_frames = future_frames
        self.num_layers = num_layers
        self.use_slot_context = use_slot_context
        self.future_step_embeddings = nn.Parameter(torch.randn(future_frames, model_dim))
        # 将(历史池化, 当前帧, 槽位池化)三者融合为GRU初始隐状态
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
        history: torch.Tensor,  # [B, T, model_dim]
        *,
        current: torch.Tensor,  # [B, model_dim] 当前帧
        slot_context: torch.Tensor | None = None,  # [B, max_sources, model_dim]
    ) -> torch.Tensor:  # [B, future_frames, model_dim]
        """生成未来每一帧的隐藏状态."""
        pooled_history = history.mean(dim=1)  # 时序平均池化
        if slot_context is not None and self.use_slot_context:
            pooled_slots = slot_context.mean(dim=1)  # 跨槽位平均
        else:
            pooled_slots = torch.zeros_like(current)
        # 构建GRU初始隐状态
        seed = self.context_proj(torch.cat([pooled_history, current, pooled_slots], dim=-1))
        # 构建GRU输入序列: 步嵌入 + 当前帧 + 历史上下文
        rollout_input = (
            self.future_step_embeddings.unsqueeze(0)
            + current[:, None, :]
            + pooled_history[:, None, :]
        )
        if slot_context is not None and self.use_slot_context:
            rollout_input = rollout_input + self.slot_proj(pooled_slots)[:, None, :]
        # 初始隐状态: [num_layers, B, model_dim]
        hidden0 = seed.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        rollout, _ = self.gru(rollout_input, hidden0)
        return self.project(rollout)  # [B, future_frames, model_dim]


class FutureSlotHead(nn.Module):
    """未来帧槽位预测头: 将未来隐藏状态与槽位上下文和可学习query结合.

    输出形状 [B, future_frames, max_sources, 5].
    """

    def __init__(self, in_dim: int, max_sources: int) -> None:
        """初始化未来 slot 预测头."""
        super().__init__()
        self.max_sources = max_sources
        self.slot_queries = nn.Parameter(torch.randn(max_sources, in_dim))  # 每个槽位的可学习query
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 5),  # [activity, sin, cos, rho, omega]
        )

    def forward(self, future_hidden: torch.Tensor, slot_context: torch.Tensor) -> torch.Tensor:
        """把未来隐藏状态和当前 slot 上下文组合成未来 slot logits."""
        # 三路相加: 未来帧隐状态 + 槽位上下文(广播到所有未来帧) + 可学习query
        hidden = (
            future_hidden[:, :, None, :]      # [B, F, 1, D]
            + slot_context[:, None, :, :]     # [B, 1, S, D]
            + self.slot_queries[None, None, :, :]  # [1, 1, S, D]
        )
        return self.mlp(hidden)  # [B, F, max_sources, 5]
