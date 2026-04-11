from __future__ import annotations

import torch


def estimate_source_count_from_heatmap(
    heatmap: torch.Tensor,
    *,
    max_sources: int,
    threshold: float = 0.30,
    relative_threshold: float = 0.60,
    min_separation_bins: int = 5,
) -> torch.Tensor:
    """Estimate source count from a circular azimuth heatmap using local-peak NMS."""
    original_shape = heatmap.shape[:-1]
    bins = int(heatmap.shape[-1])
    flat = heatmap.reshape(-1, bins)
    if flat.numel() == 0:
        return torch.zeros(original_shape, dtype=torch.long, device=heatmap.device)
    if float(flat.min().item()) < 0.0 or float(flat.max().item()) > 1.0:
        flat = torch.sigmoid(flat)

    counts: list[int] = []
    for frame in flat:
        smoothed = (
            torch.roll(frame, shifts=1, dims=0)
            + 2.0 * frame
            + torch.roll(frame, shifts=-1, dims=0)
        ) / 4.0
        peak_floor = max(float(smoothed.max().item()) * relative_threshold, threshold)
        prev = torch.roll(smoothed, shifts=1, dims=0)
        next_frame = torch.roll(smoothed, shifts=-1, dims=0)
        candidate_mask = (smoothed >= peak_floor) & (smoothed >= prev) & (smoothed >= next_frame)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
        candidate_indices.sort(key=lambda index: float(smoothed[index].item()), reverse=True)
        selected: list[int] = []
        for index in candidate_indices:
            if all(
                _circular_distance(index, chosen, bins) > min_separation_bins
                for chosen in selected
            ):
                selected.append(index)
                if len(selected) >= max_sources:
                    break
        counts.append(len(selected))
    count_tensor = torch.tensor(counts, dtype=torch.long, device=heatmap.device)
    if not original_shape:
        return count_tensor.reshape(())
    return count_tensor.reshape(*original_shape)


def _circular_distance(left: int, right: int, period: int) -> int:
    direct = abs(left - right)
    return min(direct, period - direct)


__all__ = ["estimate_source_count_from_heatmap"]
