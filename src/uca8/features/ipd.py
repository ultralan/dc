from __future__ import annotations

from collections.abc import Sequence

import torch


def _select_frequency_indices(num_freqs: int, num_bins: int, device: torch.device) -> torch.Tensor:
    if num_bins >= num_freqs:
        return torch.arange(num_freqs, device=device)
    return torch.linspace(1, num_freqs - 1, steps=num_bins, device=device).round().long()


def compute_ipd_features(
    stft_complex: torch.Tensor,
    mic_pairs: Sequence[tuple[int, int]],
    *,
    num_bins: int = 32,
) -> torch.Tensor:
    """Compute stacked cos/sin IPD features with shape [batch, 2*num_pairs, frames, num_bins]."""
    if stft_complex.ndim != 4:
        raise ValueError("Expected STFT shape [batch, channels, frames, freqs].")
    _, _, _, freqs = stft_complex.shape
    indices = _select_frequency_indices(freqs, num_bins, stft_complex.device)
    pair_feats: list[torch.Tensor] = []
    for left, right in mic_pairs:
        cross = stft_complex[:, left] * torch.conj(stft_complex[:, right])
        phase = torch.angle(cross[..., indices])
        pair_feats.extend((torch.cos(phase), torch.sin(phase)))
    return torch.stack(pair_feats, dim=1)
