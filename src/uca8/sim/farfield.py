from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def fractional_delay(signal: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
    signal = signal.to(dtype=torch.float32)
    delay = float(delay_samples.item())
    num_samples = int(signal.shape[0])
    sample_positions = torch.arange(num_samples, device=signal.device, dtype=torch.float32) - delay
    sample_positions = sample_positions.clamp(0.0, float(num_samples - 1))
    left = torch.floor(sample_positions).long()
    right = torch.clamp(left + 1, max=num_samples - 1)
    alpha = sample_positions - left.to(dtype=torch.float32)
    return (1.0 - alpha) * signal[left] + alpha * signal[right]


def render_farfield_history_waveform(
    *,
    mono_waveforms: Sequence[torch.Tensor],
    theta_history: torch.Tensor,
    mic_positions: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    sound_speed: float,
    source_activity: torch.Tensor | None = None,
    source_gains: Sequence[float] | None = None,
    source_offsets: Sequence[int] | None = None,
) -> torch.Tensor:
    if theta_history.ndim == 1:
        theta_history = theta_history.unsqueeze(-1)
    history_frames = int(theta_history.shape[0])
    num_sources = int(theta_history.shape[1])
    if len(mono_waveforms) < num_sources:
        raise ValueError("mono_waveforms must provide at least one waveform per source.")
    if source_activity is None:
        source_activity = torch.ones(history_frames, num_sources, dtype=torch.float32)
    elif source_activity.ndim == 1:
        source_activity = source_activity.unsqueeze(-1)
    if source_activity.shape != theta_history.shape:
        raise ValueError("source_activity must match theta_history shape.")

    gains = (
        [float(value) for value in source_gains]
        if source_gains is not None
        else [1.0 / (1.0 + 0.12 * source_idx) for source_idx in range(num_sources)]
    )
    if len(gains) < num_sources:
        raise ValueError("source_gains must provide at least one gain per source.")
    offsets = (
        [int(value) for value in source_offsets]
        if source_offsets is not None
        else [source_idx * max(win_length // 3, 1) for source_idx in range(num_sources)]
    )
    if len(offsets) < num_sources:
        raise ValueError("source_offsets must provide at least one offset per source.")

    total_samples = history_frames * hop_length
    output = torch.zeros(mic_positions.shape[0], total_samples + win_length, dtype=torch.float32)
    normalizer = torch.zeros(total_samples + win_length, dtype=torch.float32)
    window = torch.hann_window(win_length)
    required_samples = total_samples + win_length + max(offsets, default=0)
    prepared_waveforms: list[torch.Tensor] = []
    for source_idx in range(num_sources):
        mono_waveform = mono_waveforms[source_idx].to(dtype=torch.float32).flatten()
        if mono_waveform.shape[-1] < required_samples:
            mono_waveform = F.pad(mono_waveform, (0, required_samples - mono_waveform.shape[-1]))
        else:
            mono_waveform = mono_waveform[:required_samples]
        prepared_waveforms.append(mono_waveform)

    for frame_idx in range(history_frames):
        start = frame_idx * hop_length
        frame_has_signal = False
        for source_idx in range(num_sources):
            if float(source_activity[frame_idx, source_idx].item()) <= 0.5:
                continue
            frame_has_signal = True
            source_start = start + offsets[source_idx]
            frame = (
                prepared_waveforms[source_idx][source_start : source_start + win_length]
                * window
                * gains[source_idx]
            )
            theta_value = float(theta_history[frame_idx, source_idx].item())
            direction = torch.tensor(
                [math.cos(theta_value), math.sin(theta_value), 0.0],
                dtype=torch.float32,
            )
            delays = (mic_positions @ direction) / sound_speed * sample_rate
            for mic_idx in range(mic_positions.shape[0]):
                delayed = fractional_delay(frame, delays[mic_idx])
                output[mic_idx, start : start + win_length] += delayed
        if frame_has_signal:
            normalizer[start : start + win_length] += window.pow(2)
    output = output[:, :total_samples] / normalizer[:total_samples].clamp_min(1e-4)
    peak = float(output.abs().amax().item())
    if peak > 1.0:
        output = output / peak
    return output
