from __future__ import annotations

import torch
from torch import nn

from uca8.features.stft import UCAFeatureFrontend
from uca8.models.heads import CountHead, FutureDecoder, HeatmapHead, SlotHead
from uca8.models.spatial_encoder import IPDEncoder, SRPEncoder
from uca8.models.spec_encoder import SpectrogramEncoder
from uca8.models.tcn_backbone import CausalTCN


class UCA8TrackTrendNet(nn.Module):
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
        num_count_classes: int = 5,
        sound_speed: float = 343.0,
    ) -> None:
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
        )
        self.spec_encoder = SpectrogramEncoder(out_dim=spec_hidden_dim)
        self.ipd_encoder = IPDEncoder(in_channels=24, out_dim=spatial_hidden_dim)
        self.srp_encoder = SRPEncoder(out_dim=spatial_hidden_dim)
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
        self.slot_head = SlotHead(model_dim, max_sources)
        self.future_decoder = FutureDecoder(model_dim, future_frames)
        self.future_count_head = CountHead(count_input_dim, num_count_classes)
        self.future_heatmap_head = HeatmapHead(model_dim, heatmap_bins)
        self.future_slot_head = SlotHead(model_dim, max_sources)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 3),
        )

    def encode(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = self.frontend(waveform, vad_history=vad_history)
        spec_input = torch.cat([features["logmel_ref"], features["logmel_rms"]], dim=1)
        h_spec = self.spec_encoder(spec_input)
        h_ipd = self.ipd_encoder(features["ipd_feat"])
        h_srp = self.srp_encoder(features["srp_map"])
        fused = torch.cat([h_spec, h_ipd, h_srp, features["vad_ratio"]], dim=-1)
        h0 = self.fuse(fused)
        return features, self.backbone(h0)

    def forward(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features, hidden = self.encode(waveform, vad_history=vad_history)
        current = hidden[:, -1]
        future_hidden = self.future_decoder(hidden)
        current_heatmap_logits = self.heatmap_head(current)
        current_count_input = torch.cat([current, torch.sigmoid(current_heatmap_logits)], dim=-1)
        future_heatmap_logits = self.future_heatmap_head(future_hidden)
        future_count_input = torch.cat(
            [future_hidden, torch.sigmoid(future_heatmap_logits)],
            dim=-1,
        )
        future_slot_logits = self.future_slot_head(
            future_hidden.reshape(-1, future_hidden.shape[-1])
        )
        future_slots = future_slot_logits.reshape(
            future_hidden.shape[0],
            future_hidden.shape[1],
            -1,
            5,
        )
        return {
            "features": features,
            "count_logits": self.count_head(current_count_input),
            "heatmap_logits": current_heatmap_logits,
            "slot_logits": self.slot_head(current),
            "future_count_logits": self.future_count_head(future_count_input),
            "future_heatmap_logits": future_heatmap_logits,
            "future_slot_logits": future_slots,
            "motion_logits": self.motion_head(future_hidden.mean(dim=1)),
        }
