from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def make_uniform_circular_array(
    num_mics: int = 8,
    radius: float = 0.045,
    z: float = 0.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 2D uniform circular array with mic0 on the positive x-axis."""
    indices = torch.arange(num_mics, device=device, dtype=dtype)
    angles = 2.0 * math.pi * indices / float(num_mics)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z_axis = torch.full_like(x, z)
    return torch.stack((x, y, z_axis), dim=-1)


def default_mic_pairs(num_mics: int = 8) -> list[tuple[int, int]]:
    """Return adjacent plus diametric pairs for an even circular array."""
    if num_mics % 2 != 0:
        raise ValueError("default_mic_pairs expects an even microphone count.")
    adjacent = [(idx, (idx + 1) % num_mics) for idx in range(num_mics)]
    diametric = [(idx, (idx + num_mics // 2) % num_mics) for idx in range(num_mics // 2)]
    return adjacent + diametric


def azimuth_grid(
    num_bins: int = 72,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an evenly spaced azimuth grid in radians over [-pi, pi)."""
    return torch.linspace(-math.pi, math.pi, steps=num_bins + 1, device=device, dtype=dtype)[:-1]


def relative_source_state(
    source_positions: torch.Tensor,
    array_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute azimuth and range between sources and array centers."""
    delta = source_positions[..., :2] - array_positions[..., :2]
    theta = torch.atan2(delta[..., 1], delta[..., 0])
    rho = torch.linalg.norm(delta, dim=-1)
    return theta, rho


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angular_velocity(theta: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute wrapped angular velocity along the leading time dimension."""
    velocity = torch.zeros_like(theta)
    if theta.shape[0] <= 1:
        return velocity
    velocity[1:] = wrap_angle(theta[1:] - theta[:-1]) / dt
    velocity[0] = velocity[1]
    return velocity


def steering_delays(
    mic_positions: torch.Tensor,
    azimuths: torch.Tensor,
    *,
    sound_speed: float = 343.0,
) -> torch.Tensor:
    """Return far-field delays with shape [num_azimuths, num_mics]."""
    directions = torch.stack(
        (torch.cos(azimuths), torch.sin(azimuths), torch.zeros_like(azimuths)),
        dim=-1,
    )
    return directions @ mic_positions.T / sound_speed


def pair_delays(
    mic_positions: torch.Tensor,
    azimuths: torch.Tensor,
    mic_pairs: Sequence[tuple[int, int]],
    *,
    sound_speed: float = 343.0,
) -> torch.Tensor:
    """Return pairwise delays with shape [num_azimuths, num_pairs]."""
    delays = steering_delays(mic_positions, azimuths, sound_speed=sound_speed)
    pair_values = [delays[:, i] - delays[:, j] for i, j in mic_pairs]
    return torch.stack(pair_values, dim=-1)
