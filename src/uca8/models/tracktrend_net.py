from __future__ import annotations

import torch
from torch import nn

from uca8.features.stft import UCAFeatureFrontend
from uca8.models.heads import (
    CountHead,
    FutureRolloutDecoder,
    FutureSlotHead,
    HeatmapHead,
    TemporalSlotDecoder,
)
from uca8.models.spatial_encoder import IPDEncoder, SRPEncoder
from uca8.models.spec_encoder import SpectrogramEncoder
from uca8.models.tcn_backbone import CausalTCN


class UCA8TrackTrendNet(nn.Module):
    """8 通道声源定位/跟踪端到端网络.

    这份实现按论文架构拆成四段:

    1. ``UCAFeatureFrontend``: 从原始波形提取帧级声学特征,
       包括 log-mel、IPD、SRP-PHAT 空间谱和 VAD ratio.
    2. 频谱/IPD/SRP 三个编码分支: 分别把不同来源的特征转成帧级 embedding.
    3. ``CausalTCN``: 在历史窗口上建模时序上下文.
    4. 任务头: 输出当前方位 heatmap、声源数、槽位状态和可选未来预测.

    特征消融开关放在 frontend 里. 被关闭的分支不会改变张量形状, 而是置零,
    这样 ablation 比的是输入信息差异, 不是模型容量差异.
    """

    def __init__(
        self,
        *,
        mic_positions: torch.Tensor,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        spec_bins: int = 64,
        ipd_bins: int = 32,
        heatmap_bins: int = 72,
        history_frames: int = 128,
        future_frames: int = 32,
        max_sources: int = 4,
        spec_hidden_dim: int = 128,
        spatial_hidden_dim: int = 64,
        model_dim: int = 256,
        tcn_dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        tcn_kernel_size: int = 3,
        dropout: float = 0.1,
        slot_decoder_attention_heads: int = 4,
        future_decoder_layers: int = 2,
        future_decoder_dropout: float = 0.1,
        use_slot_context_in_future_decoder: bool = True,
        num_count_classes: int = 5,
        sound_speed: float = 343.0,
        use_logmel_ref: bool = True,
        use_logmel_rms: bool = True,
        use_ipd: bool = True,
        use_srp: bool = True,
        use_vad: bool = True,
    ) -> None:
        """初始化完整模型结构.

        参数大体分为四类:
        - frontend/STFT 参数: sample_rate、n_fft、hop_length 等;
        - 特征维度参数: spec_bins、ipd_bins、heatmap_bins;
        - 时序和任务参数: history_frames、future_frames、max_sources;
        - ablation 开关: use_logmel_ref、use_ipd、use_srp 等.
        """
        super().__init__()
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.frontend = UCAFeatureFrontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            spec_bins=spec_bins,
            ipd_bins=ipd_bins,
            azimuth_bins=heatmap_bins,
            target_frames=history_frames,
            mic_positions=mic_positions,
            sound_speed=sound_speed,
            use_logmel_ref=use_logmel_ref,
            use_logmel_rms=use_logmel_rms,
            use_ipd=use_ipd,
            use_srp=use_srp,
            use_vad=use_vad,
        )
        self.spec_encoder = SpectrogramEncoder(out_dim=spec_hidden_dim)
        self.ipd_encoder = IPDEncoder(in_channels=24, out_dim=spatial_hidden_dim)
        self.srp_encoder = SRPEncoder(out_dim=spatial_hidden_dim, azimuth_bins=heatmap_bins)
        fused_dim = spec_hidden_dim + spatial_hidden_dim + spatial_hidden_dim + 1
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.backbone = CausalTCN(
            model_dim=model_dim,
            dilations=tcn_dilations,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        count_input_dim = model_dim + heatmap_bins
        self.count_head = CountHead(count_input_dim, num_count_classes)
        self.heatmap_head = HeatmapHead(model_dim, heatmap_bins)
        self.slot_head = TemporalSlotDecoder(
            model_dim,
            max_sources,
            attention_heads=slot_decoder_attention_heads,
            dropout=dropout,
        )
        self.future_decoder = FutureRolloutDecoder(
            model_dim,
            future_frames,
            num_layers=future_decoder_layers,
            dropout=future_decoder_dropout,
            use_slot_context=use_slot_context_in_future_decoder,
        )
        self.future_count_head = CountHead(count_input_dim, num_count_classes)
        self.future_heatmap_head = HeatmapHead(model_dim, heatmap_bins)
        self.future_slot_head = FutureSlotHead(model_dim, max_sources)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 3),
        )

    def encode(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """把历史波形编码成帧级时序隐藏状态.

        参数:
            waveform: 多通道音频, 形状为 ``[batch, channels, samples]``.
            vad_history: 可选 VAD 序列, 形状为 ``[batch, frames]`` 或
                ``[batch, frames, 1]``.

        返回:
            ``(features, hidden)``. 其中 ``hidden`` 形状为
            ``[batch, history_frames, model_dim]``.
        """
        features = self.frontend(waveform, vad_history=vad_history)

        # 三类信息分开编码:
        # - logmel: 单通道/多通道能量谱, 偏内容和能量分布;
        # - IPD: 麦克风对相位差, 偏空间相位线索;
        # - SRP: 按方位角扫描得到的空间谱, 偏显式方向线索.
        spec_input = torch.cat([features["logmel_ref"], features["logmel_rms"]], dim=1)
        h_spec = self.spec_encoder(spec_input)
        h_ipd = self.ipd_encoder(features["ipd_feat"])
        h_srp = self.srp_encoder(features["srp_map"])

        # frontend 已经把所有分支裁剪/补齐到同一帧数, 这里可以直接按最后一维拼接.
        fused = torch.cat([h_spec, h_ipd, h_srp, features["vad_ratio"]], dim=-1)
        h0 = self.fuse(fused)
        return features, self.backbone(h0)

    def forward(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """执行完整前向传播并返回所有监督头.

        RealMAN 定位对比当前主要使用 ``heatmap_logits``.
        slot/future 输出保留给跟踪和扩展任务, 不影响 heatmap 定位路径.
        """
        features, hidden = self.encode(waveform, vad_history=vad_history)
        current = hidden[:, -1]

        # slot_context 同时服务当前槽位预测和未来 rollout, 让未来预测知道当前有哪些声源槽位.
        slot_logits, slot_context = self.slot_head(
            hidden,
            query_context=current,
            return_context=True,
        )
        future_hidden = self.future_decoder(
            hidden,
            current=current,
            slot_context=slot_context,
        )
        current_heatmap_logits = self.heatmap_head(current)

        # 计数头额外接收 heatmap 的置信度分布, 使声源数预测和空间证据绑定.
        current_count_input = torch.cat([current, torch.sigmoid(current_heatmap_logits)], dim=-1)
        future_heatmap_logits = self.future_heatmap_head(future_hidden)
        future_count_input = torch.cat(
            [future_hidden, torch.sigmoid(future_heatmap_logits)],
            dim=-1,
        )
        future_slots = self.future_slot_head(future_hidden, slot_context)
        return {
            "features": features,
            "count_logits": self.count_head(current_count_input),
            "heatmap_logits": current_heatmap_logits,
            "slot_logits": slot_logits,
            "future_count_logits": self.future_count_head(future_count_input),
            "future_heatmap_logits": future_heatmap_logits,
            "future_slot_logits": future_slots,
            "motion_logits": self.motion_head(future_hidden.mean(dim=1)),
        }
