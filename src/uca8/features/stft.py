from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from uca8.features.ipd import compute_ipd_features
from uca8.features.srp_phat import SRPPHAT
from uca8.geometry.uca8 import azimuth_grid, default_mic_pairs


@dataclass(slots=True)
class STFTSpec:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    spec_bins: int = 64
    ipd_bins: int = 32
    azimuth_bins: int = 72
    target_frames: int = 128


class MultiChannelSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError("Expected waveform shape [batch, channels, samples].")
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        stft = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            return_complex=True,
        )
        return stft.transpose(1, 2).reshape(batch, channels, -1, stft.shape[1])


def _compress_frequency(spec: torch.Tensor, out_bins: int) -> torch.Tensor:
    batch, channels, frames, freqs = spec.shape
    pooled = F.adaptive_avg_pool1d(spec.reshape(batch * channels * frames, 1, freqs), out_bins)
    return pooled.reshape(batch, channels, frames, out_bins)


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, freq_mel / 2595.0) - 1.0)


def build_mel_filterbank(
    *,
    num_freqs: int,
    sample_rate: int,
    num_mels: int,
    device: torch.device,
    dtype: torch.dtype,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive.")
    if num_mels <= 0:
        raise ValueError("num_mels must be positive.")
    upper_hz = float(sample_rate) / 2.0 if f_max is None else f_max
    fft_freqs = torch.linspace(0.0, upper_hz, steps=num_freqs, device=device, dtype=dtype)
    mel_min = _hz_to_mel(torch.tensor(f_min, device=device, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(upper_hz, device=device, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, steps=num_mels + 2, device=device, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)
    filterbank = torch.zeros(num_freqs, num_mels, device=device, dtype=dtype)
    for idx in range(num_mels):
        left = hz_points[idx]
        center = hz_points[idx + 1]
        right = hz_points[idx + 2]
        up_slope = (fft_freqs - left) / (center - left).clamp_min(1e-6)
        down_slope = (right - fft_freqs) / (right - center).clamp_min(1e-6)
        filterbank[:, idx] = torch.minimum(up_slope, down_slope).clamp_min(0.0)
    return filterbank / filterbank.sum(dim=0, keepdim=True).clamp_min(1e-6)


def fit_time_dim(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = x.shape[-2]
    if current_frames == target_frames:
        return x
    if current_frames > target_frames:
        return x[..., -target_frames:, :]
    pad = target_frames - current_frames
    return F.pad(x, (0, 0, pad, 0))


def build_logmel_like_features(
    stft_complex: torch.Tensor,
    *,
    mel_filterbank: torch.Tensor,
    out_bins: int = 64,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project power spectra to mel bins while preserving the existing frontend interface."""
    if mel_filterbank.shape[1] != out_bins:
        raise ValueError("mel_filterbank does not match requested out_bins.")
    power = stft_complex.abs().pow(2.0)
    mel_filterbank = mel_filterbank.to(device=power.device, dtype=power.dtype)
    ref = torch.einsum("bctf,fm->bctm", power[:, :1], mel_filterbank)
    rms = torch.einsum("bctf,fm->bctm", power.mean(dim=1, keepdim=True), mel_filterbank)
    return torch.log(ref.clamp_min(eps)), torch.log(rms.clamp_min(eps))


class UCAFeatureFrontend(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        spec_bins: int,
        ipd_bins: int,
        azimuth_bins: int,
        target_frames: int,
        mic_positions: torch.Tensor,
        sound_speed: float = 343.0,
    ) -> None:
        super().__init__()
        self.spec = STFTSpec(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            spec_bins=spec_bins,
            ipd_bins=ipd_bins,
            azimuth_bins=azimuth_bins,
            target_frames=target_frames,
        )
        self.stft = MultiChannelSTFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(
                num_freqs=n_fft // 2 + 1,
                sample_rate=sample_rate,
                num_mels=spec_bins,
                device=mic_positions.device,
                dtype=mic_positions.dtype,
            ),
            persistent=False,
        )
        mic_pairs = default_mic_pairs(int(mic_positions.shape[0]))
        self.srp = SRPPHAT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            mic_positions=mic_positions,
            mic_pairs=mic_pairs,
            azimuths=azimuth_grid(
                azimuth_bins,
                device=mic_positions.device,
                dtype=mic_positions.dtype,
            ),
            sound_speed=sound_speed,
        )
        self.mic_pairs = mic_pairs

    def forward(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stft_complex = self.stft(waveform)
        logmel_ref, logmel_rms = build_logmel_like_features(
            stft_complex,
            mel_filterbank=self.mel_filterbank,
            out_bins=self.spec.spec_bins,
        )
        ipd_feat = compute_ipd_features(stft_complex, self.mic_pairs, num_bins=self.spec.ipd_bins)
        srp_map = self.srp(stft_complex).unsqueeze(1)
        logmel_ref = fit_time_dim(logmel_ref, self.spec.target_frames)
        logmel_rms = fit_time_dim(logmel_rms, self.spec.target_frames)
        ipd_feat = fit_time_dim(ipd_feat, self.spec.target_frames)
        srp_map = fit_time_dim(srp_map, self.spec.target_frames)
        if vad_history is None:
            vad_history = torch.zeros(
                waveform.shape[0],
                self.spec.target_frames,
                1,
                device=waveform.device,
                dtype=waveform.dtype,
            )
        elif vad_history.ndim == 2:
            vad_history = vad_history.unsqueeze(-1)
        elif vad_history.ndim != 3:
            raise ValueError("Expected vad_history shape [batch, frames] or [batch, frames, 1].")
        if vad_history.shape[1] != self.spec.target_frames:
            vad_history = fit_time_dim(vad_history.unsqueeze(1), self.spec.target_frames).squeeze(1)
        return {
            "logmel_ref": logmel_ref,
            "logmel_rms": logmel_rms,
            "ipd_feat": ipd_feat,
            "srp_map": srp_map,
            "vad_ratio": vad_history,
        }
