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
