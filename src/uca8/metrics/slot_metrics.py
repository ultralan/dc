from __future__ import annotations

import math

import torch

from uca8.geometry.uca8 import wrap_angle


def _validate_slot_state_shapes(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
) -> None:
    if pred_slot_state.shape != target_slot_state.shape:
        raise ValueError("pred_slot_state and target_slot_state must have the same shape.")
    if pred_slot_state.shape[-1] < 3:
        raise ValueError("slot_state tensors must include activity, sin(theta), and cos(theta).")


def slot_activity_mask(
    slot_state: torch.Tensor,
    *,
    is_logits: bool,
    threshold: float = 0.5,
) -> torch.Tensor:
    activity = torch.sigmoid(slot_state[..., 0]) if is_logits else slot_state[..., 0]
    return activity > threshold


def slot_angles_deg(
    slot_state: torch.Tensor,
    *,
    is_logits: bool,
    threshold: float = 0.5,
) -> torch.Tensor:
    activity = torch.sigmoid(slot_state[..., 0]) if is_logits else slot_state[..., 0]
    angles_deg = torch.atan2(slot_state[..., 1], slot_state[..., 2]) * (180.0 / math.pi)
    return angles_deg.masked_fill(activity <= threshold, float("nan"))


def slot_count_from_state(
    slot_state: torch.Tensor,
    *,
    is_logits: bool,
    threshold: float = 0.5,
) -> torch.Tensor:
    return (
        slot_activity_mask(
            slot_state,
            is_logits=is_logits,
            threshold=threshold,
        )
        .sum(dim=-1)
        .to(dtype=torch.long)
    )


def slot_count_accuracy_stats(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_slot_state_shapes(pred_slot_state, target_slot_state)
    pred_count = slot_count_from_state(pred_slot_state, is_logits=True, threshold=threshold)
    target_count = slot_count_from_state(target_slot_state, is_logits=False, threshold=threshold)
    correct = (pred_count == target_count).sum().to(dtype=pred_slot_state.dtype)
    total = pred_count.new_tensor(float(pred_count.numel()), dtype=pred_slot_state.dtype)
    return correct, total


def slot_activity_confusion_stats(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_slot_state_shapes(pred_slot_state, target_slot_state)
    pred_active = slot_activity_mask(pred_slot_state, is_logits=True, threshold=threshold)
    target_active = slot_activity_mask(target_slot_state, is_logits=False, threshold=threshold)
    true_positive = (pred_active & target_active).sum().to(dtype=pred_slot_state.dtype)
    false_positive = (pred_active & ~target_active).sum().to(dtype=pred_slot_state.dtype)
    false_negative = (~pred_active & target_active).sum().to(dtype=pred_slot_state.dtype)
    return true_positive, false_positive, false_negative


def angle_range_deg(values_deg: torch.Tensor) -> torch.Tensor:
    values_deg = values_deg.to(dtype=torch.float32)
    valid = torch.isfinite(values_deg)
    if int(valid.sum().item()) < 2:
        return values_deg.new_tensor(0.0)
    valid_values = values_deg[valid]
    deltas = torch.remainder(valid_values[1:] - valid_values[:-1] + 180.0, 360.0) - 180.0
    unwrapped = torch.cat(
        [valid_values[:1], valid_values[:1] + torch.cumsum(deltas, dim=0)],
        dim=0,
    )
    return unwrapped.max() - unwrapped.min()


def primary_slot_index_from_sequence(
    slot_state: torch.Tensor,
    *,
    is_logits: bool,
    threshold: float = 0.5,
) -> int | None:
    if slot_state.ndim != 3:
        raise ValueError("slot_state must have shape [frames, slots, features].")
    activity = torch.sigmoid(slot_state[..., 0]) if is_logits else slot_state[..., 0]
    if not torch.any(activity > threshold):
        return None
    angles_deg = slot_angles_deg(slot_state, is_logits=is_logits, threshold=threshold)
    best_slot_idx = 0
    best_score = float("-inf")
    for slot_idx in range(slot_state.shape[1]):
        slot_range = float(angle_range_deg(angles_deg[:, slot_idx]).item())
        mean_activity = float(activity[:, slot_idx].mean().item())
        active_frames = int((activity[:, slot_idx] > threshold).sum().item())
        score = slot_range + 1e-3 * mean_activity
        if active_frames == 0:
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_slot_idx = slot_idx
    if best_score == float("-inf"):
        return None
    return best_slot_idx


def primary_slot_range_stats(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
    *,
    pred_is_logits: bool,
    target_is_logits: bool,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    if pred_slot_state.shape != target_slot_state.shape:
        raise ValueError("pred_slot_state and target_slot_state must have the same shape.")
    slot_idx = primary_slot_index_from_sequence(
        target_slot_state,
        is_logits=target_is_logits,
        threshold=threshold,
    )
    zero = pred_slot_state.new_tensor(0.0)
    one = pred_slot_state.new_tensor(1.0)
    if slot_idx is None:
        return one, zero, zero, None
    pred_angles = slot_angles_deg(pred_slot_state, is_logits=pred_is_logits, threshold=threshold)
    target_angles = slot_angles_deg(target_slot_state, is_logits=target_is_logits, threshold=threshold)
    pred_range = angle_range_deg(pred_angles[:, slot_idx]).to(dtype=pred_slot_state.dtype)
    target_range = angle_range_deg(target_angles[:, slot_idx]).to(dtype=pred_slot_state.dtype)
    if float(target_range.item()) <= 1e-4:
        ratio = one if float(pred_range.item()) <= 1.0 else zero
    else:
        ratio = pred_range / target_range.clamp_min(1e-4)
    return ratio, pred_range, target_range, slot_idx


def slot_trend_label_from_sequence(
    slot_state: torch.Tensor,
    *,
    is_logits: bool,
    threshold: float = 0.5,
    stable_threshold_deg: float = 5.0,
) -> int:
    slot_idx = primary_slot_index_from_sequence(slot_state, is_logits=is_logits, threshold=threshold)
    if slot_idx is None:
        return 0
    angles_deg = slot_angles_deg(slot_state, is_logits=is_logits, threshold=threshold)[:, slot_idx]
    valid = torch.isfinite(angles_deg)
    if int(valid.sum().item()) < 2:
        return 0
    valid_values = angles_deg[valid]
    deltas = torch.remainder(valid_values[1:] - valid_values[:-1] + 180.0, 360.0) - 180.0
    total_delta = float(deltas.sum().item())
    if total_delta < -stable_threshold_deg:
        return -1
    if total_delta > stable_threshold_deg:
        return 1
    return 0


def slot_angle_error_stats_deg(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_slot_state_shapes(pred_slot_state, target_slot_state)
    active_mask = target_slot_state[..., 0] > 0.5
    if not torch.any(active_mask):
        zero = pred_slot_state.new_tensor(0.0)
        return zero, zero
    pred_theta = torch.atan2(pred_slot_state[..., 1], pred_slot_state[..., 2])
    target_theta = torch.atan2(target_slot_state[..., 1], target_slot_state[..., 2])
    angle_error_deg = torch.abs(wrap_angle(pred_theta - target_theta)) * (180.0 / math.pi)
    error_sum = angle_error_deg[active_mask].sum()
    active_count = active_mask.sum().to(dtype=pred_slot_state.dtype)
    return error_sum, active_count


def future_slot_delta_error_stats_deg(
    pred_slot_state: torch.Tensor,
    target_slot_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_slot_state_shapes(pred_slot_state, target_slot_state)
    active_mask = (target_slot_state[..., 0][..., 1:, :] > 0.5) & (
        target_slot_state[..., 0][..., :-1, :] > 0.5
    )
    if not torch.any(active_mask):
        zero = pred_slot_state.new_tensor(0.0)
        return zero, zero
    pred_theta = torch.atan2(pred_slot_state[..., 1], pred_slot_state[..., 2])
    target_theta = torch.atan2(target_slot_state[..., 1], target_slot_state[..., 2])
    pred_delta = wrap_angle(pred_theta[..., 1:, :] - pred_theta[..., :-1, :])
    target_delta = wrap_angle(target_theta[..., 1:, :] - target_theta[..., :-1, :])
    delta_error_deg = torch.abs(wrap_angle(pred_delta - target_delta)) * (180.0 / math.pi)
    error_sum = delta_error_deg[active_mask].sum()
    active_count = active_mask.sum().to(dtype=pred_slot_state.dtype)
    return error_sum, active_count


def heatmap_peak_recall_stats(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_count: torch.Tensor,
    *,
    tolerance_bins: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_heatmap.shape != target_heatmap.shape:
        raise ValueError("pred_heatmap and target_heatmap must have the same shape.")
    bins = pred_heatmap.shape[-1]
    flat_pred = pred_heatmap.reshape(-1, bins)
    flat_target = target_heatmap.reshape(-1, bins)
    flat_count = target_count.reshape(-1).to(dtype=torch.long)
    recall_sum = pred_heatmap.new_tensor(0.0)
    total = pred_heatmap.new_tensor(float(flat_pred.shape[0]))
    for row_idx in range(flat_pred.shape[0]):
        peak_count = int(max(flat_count[row_idx].item(), 0))
        if peak_count <= 0:
            recall_sum = recall_sum + 1.0
            continue
        peak_count = min(peak_count, bins)
        target_bins = torch.topk(flat_target[row_idx], k=peak_count).indices.tolist()
        pred_bins = torch.topk(flat_pred[row_idx], k=peak_count).indices.tolist()
        matched = 0
        for target_bin in target_bins:
            if any(
                min(abs(pred_bin - target_bin), bins - abs(pred_bin - target_bin))
                <= tolerance_bins
                for pred_bin in pred_bins
            ):
                matched += 1
        recall_sum = recall_sum + (matched / max(peak_count, 1))
    return recall_sum, total


def heatmap_contrast(heatmap: torch.Tensor) -> torch.Tensor:
    if heatmap.numel() == 0:
        return heatmap.new_tensor(0.0)
    contrast = heatmap.amax(dim=-1) - heatmap.mean(dim=-1)
    return contrast.mean()
