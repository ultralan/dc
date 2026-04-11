from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from uca8.geometry.uca8 import pair_delays


class SRPPHAT(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        mic_positions: torch.Tensor,
        mic_pairs: Sequence[tuple[int, int]],
        azimuths: torch.Tensor,
        sound_speed: float = 343.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mic_pairs = list(mic_pairs)
        self.eps = eps
        freqs = torch.fft.rfftfreq(
            n_fft,
            d=1.0 / sample_rate,
            device=mic_positions.device,
            dtype=mic_positions.dtype,
        )
        delays = pair_delays(mic_positions, azimuths, self.mic_pairs, sound_speed=sound_speed)
        steering = torch.exp(1j * 2.0 * math.pi * freqs[None, None, :] * delays[:, :, None])
        self.register_buffer("steering", steering[:, :, 1:], persistent=False)

    def forward(self, stft_complex: torch.Tensor) -> torch.Tensor:
        if stft_complex.ndim != 4:
            raise ValueError("Expected STFT shape [batch, channels, frames, freqs].")
        pair_cross: list[torch.Tensor] = []
        for left, right in self.mic_pairs:
            cross = stft_complex[:, left, :, 1:] * torch.conj(stft_complex[:, right, :, 1:])
            cross = cross / cross.abs().clamp_min(self.eps)
            pair_cross.append(cross)
        normalized = torch.stack(pair_cross, dim=1)
        score = torch.einsum("bptf,apf->bta", normalized, self.steering)
        return score.real / (len(self.mic_pairs) * self.steering.shape[-1])
