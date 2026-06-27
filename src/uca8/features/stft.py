from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from uca8.features.ipd import compute_ipd_features
from uca8.features.srp_phat import SRPPHAT
from uca8.geometry.uca8 import azimuth_grid, infer_mic_pairs


@dataclass(slots=True)
class STFTSpec:
    """STFT/frontend shared parameters."""

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    spec_bins: int = 64
    ipd_bins: int = 32
    azimuth_bins: int = 72
    target_frames: int = 128


FEATURE_CACHE_VERSION = 1


class MultiChannelSTFT(nn.Module):
    """Compute multi-channel complex STFT."""

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
    """Build a stable mel-like projection without torchaudio."""
    upper_hz = float(sample_rate) / 2.0 if f_max is None else f_max
    fft_freqs = torch.linspace(0.0, upper_hz, steps=num_freqs, device=device, dtype=dtype)
    mel_min = 2595.0 * torch.log10(torch.tensor(1.0 + f_min / 700.0, device=device, dtype=dtype))
    mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + upper_hz / 700.0, device=device, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, steps=num_mels + 2, device=device, dtype=dtype)
    hz_points = 700.0 * (torch.pow(10.0, mel_points / 2595.0) - 1.0)
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
    """Trim or left-pad a tensor so the time axis matches the model window."""
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
    """Return reference-mic and multi-channel RMS log-mel features."""
    power = stft_complex.abs().pow(2.0)
    mel_filterbank = mel_filterbank.to(device=power.device, dtype=power.dtype)
    ref = torch.einsum("bctf,fm->bctm", power[:, :1], mel_filterbank)
    rms = torch.einsum("bctf,fm->bctm", power.mean(dim=1, keepdim=True), mel_filterbank)
    return torch.log(ref.clamp_min(eps)), torch.log(rms.clamp_min(eps))


class UCAFeatureFrontend(nn.Module):
    """Feature frontend for the 8-mic UCA localization model."""

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
        use_logmel_ref: bool = True,
        use_logmel_rms: bool = True,
        use_ipd: bool = True,
        use_srp: bool = True,
        use_vad: bool = True,
        feature_cache_dir: str | Path | None = None,
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
        self.use_logmel_ref = use_logmel_ref
        self.use_logmel_rms = use_logmel_rms
        self.use_ipd = use_ipd
        self.use_srp = use_srp
        self.use_vad = use_vad
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
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
        self._cache_signature = self._build_cache_signature(mic_positions)
        self.mic_pairs = infer_mic_pairs(mic_positions)
        self.srp = SRPPHAT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            mic_positions=mic_positions,
            mic_pairs=self.mic_pairs,
            azimuths=azimuth_grid(
                azimuth_bins,
                device=mic_positions.device,
                dtype=mic_positions.dtype,
            ),
            sound_speed=sound_speed,
        )

    def _build_cache_signature(self, mic_positions: torch.Tensor) -> str:
        payload = {
            "version": FEATURE_CACHE_VERSION,
            "spec": {
                "sample_rate": self.spec.sample_rate,
                "n_fft": self.spec.n_fft,
                "win_length": self.spec.win_length,
                "hop_length": self.spec.hop_length,
                "spec_bins": self.spec.spec_bins,
                "ipd_bins": self.spec.ipd_bins,
                "azimuth_bins": self.spec.azimuth_bins,
                "target_frames": self.spec.target_frames,
            },
            "flags": {
                "use_logmel_ref": self.use_logmel_ref,
                "use_logmel_rms": self.use_logmel_rms,
                "use_ipd": self.use_ipd,
                "use_srp": self.use_srp,
                "use_vad": self.use_vad,
            },
            "geometry": mic_positions.detach().cpu().tolist(),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_path(self, sample_id: str) -> Path | None:
        if self.feature_cache_dir is None:
            return None
        digest = hashlib.sha1(
            f"{FEATURE_CACHE_VERSION}:{self._cache_signature}:{sample_id}".encode("utf-8")
        ).hexdigest()
        return self.feature_cache_dir / f"{digest}.pt"

    @staticmethod
    def _normalize_sample_ids(
        sample_id: str | Sequence[str] | None,
        batch_size: int,
    ) -> list[str] | None:
        if sample_id is None:
            return None
        if isinstance(sample_id, str):
            return [sample_id] * batch_size
        sample_ids = [str(item) for item in sample_id]
        if len(sample_ids) != batch_size:
            raise ValueError("sample_id length must match batch size.")
        return sample_ids

    @staticmethod
    def _merge_feature_chunks(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = items[0].keys()
        return {key: torch.cat([item[key] for item in items], dim=0) for key in keys}

    @staticmethod
    def _move_features(
        features: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {key: value.to(device=device, dtype=dtype) for key, value in features.items()}

    def _compute_features(
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
        if not self.use_logmel_ref:
            logmel_ref = torch.zeros_like(logmel_ref)
        if not self.use_logmel_rms:
            logmel_rms = torch.zeros_like(logmel_rms)
        if not self.use_ipd:
            ipd_feat = torch.zeros_like(ipd_feat)
        if not self.use_srp:
            srp_map = torch.zeros_like(srp_map)

        logmel_ref = fit_time_dim(logmel_ref, self.spec.target_frames)
        logmel_rms = fit_time_dim(logmel_rms, self.spec.target_frames)
        ipd_feat = fit_time_dim(ipd_feat, self.spec.target_frames)
        srp_map = fit_time_dim(srp_map, self.spec.target_frames)
        if vad_history is None or not self.use_vad:
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

    def _load_cached_features(self, path: Path) -> dict[str, torch.Tensor]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Cached feature payload must be a dictionary.")
        features: dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("Cached feature payload values must be tensors.")
            features[key] = value
        return features

    def _save_cached_features(self, path: Path, features: dict[str, torch.Tensor]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".{hashlib.sha1(str(path).encode('utf-8')).hexdigest()}.tmp")
        try:
            torch.save({key: value.detach().cpu() for key, value in features.items()}, tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def forward(
        self,
        waveform: torch.Tensor,
        vad_history: torch.Tensor | None = None,
        sample_id: str | Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract aligned frontend features, optionally through a disk cache."""
        normalized_sample_ids = self._normalize_sample_ids(sample_id, waveform.shape[0])
        if self.feature_cache_dir is None or normalized_sample_ids is None:
            return self._compute_features(waveform, vad_history=vad_history)

        features_by_index: dict[int, dict[str, torch.Tensor]] = {}
        missing_indices: list[int] = []
        for index, item_id in enumerate(normalized_sample_ids):
            cache_path = self._cache_path(item_id)
            if cache_path is not None and cache_path.exists():
                features_by_index[index] = self._move_features(
                    self._load_cached_features(cache_path),
                    device=waveform.device,
                    dtype=waveform.dtype,
                )
            else:
                missing_indices.append(index)

        if missing_indices:
            missing_waveform = waveform[missing_indices]
            missing_vad = vad_history[missing_indices] if vad_history is not None else None
            missing_features = self._compute_features(missing_waveform, vad_history=missing_vad)
            for offset, batch_index in enumerate(missing_indices):
                item_features = {
                    key: value[offset : offset + 1] for key, value in missing_features.items()
                }
                features_by_index[batch_index] = item_features
                cache_path = self._cache_path(normalized_sample_ids[batch_index])
                if cache_path is not None:
                    self._save_cached_features(cache_path, item_features)

        ordered_features = [features_by_index[index] for index in range(waveform.shape[0])]
        return self._move_features(
            self._merge_feature_chunks(ordered_features),
            device=waveform.device,
            dtype=waveform.dtype,
        )
