from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TrackTrendMultiTaskLoss(nn.Module):
    def __init__(
        self,
        *,
        count_weight: float = 1.0,
        heat_weight: float = 1.0,
        heat_pos_weight: float = 1.0,
        track_weight: float = 2.0,
        future_count_weight: float = 1.5,
        future_heat_weight: float = 1.5,
        future_heat_pos_weight: float = 1.0,
        future_track_weight: float = 1.5,
        slot_activity_pos_weight: float = 1.0,
        slot_activity_neg_weight: float = 1.0,
        future_slot_activity_pos_weight: float = 1.0,
        future_slot_activity_neg_weight: float = 1.0,
        future_slot_deactivate_weight: float = 1.0,
        slot_count_consistency_weight: float = 0.0,
        future_slot_count_consistency_weight: float = 0.0,
        future_slot_count_transition_weight: float = 1.0,
        motion_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.count_weight = count_weight
        self.heat_weight = heat_weight
        self.heat_pos_weight = heat_pos_weight
        self.track_weight = track_weight
        self.future_count_weight = future_count_weight
        self.future_heat_weight = future_heat_weight
        self.future_heat_pos_weight = future_heat_pos_weight
        self.future_track_weight = future_track_weight
        self.slot_activity_pos_weight = slot_activity_pos_weight
        self.slot_activity_neg_weight = slot_activity_neg_weight
        self.future_slot_activity_pos_weight = future_slot_activity_pos_weight
        self.future_slot_activity_neg_weight = future_slot_activity_neg_weight
        self.future_slot_deactivate_weight = future_slot_deactivate_weight
        self.slot_count_consistency_weight = slot_count_consistency_weight
        self.future_slot_count_consistency_weight = future_slot_count_consistency_weight
        self.future_slot_count_transition_weight = future_slot_count_transition_weight
        self.motion_weight = motion_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        count_loss = F.cross_entropy(predictions["count_logits"], targets["count"])
        heat_pos_weight = None
        if self.heat_pos_weight != 1.0:
            heat_pos_weight = torch.full_like(targets["heatmap"], self.heat_pos_weight)
        heat_loss = F.binary_cross_entropy_with_logits(
            predictions["heatmap_logits"],
            targets["heatmap"],
            pos_weight=heat_pos_weight,
        )
        slot_activity_loss, slot_regression_loss = self._slot_losses(
            predictions["slot_logits"],
            targets["slot_state"],
            activity_pos_weight=self.slot_activity_pos_weight,
            activity_neg_weight=self.slot_activity_neg_weight,
        )
        slot_count_consistency_loss = self._slot_count_consistency_loss(
            predictions["slot_logits"],
            targets["count"].to(dtype=predictions["slot_logits"].dtype),
        )
        track_loss = slot_activity_loss + slot_regression_loss
        future_count_logits = predictions["future_count_logits"].reshape(
            -1,
            predictions["future_count_logits"].shape[-1],
        )
        future_slot_logits = predictions["future_slot_logits"].reshape(
            -1,
            *predictions["future_slot_logits"].shape[-2:],
        )
        future_slot_target = targets["future_slot_state"].reshape(
            -1,
            *targets["future_slot_state"].shape[-2:],
        )
        future_decrease_mask = (
            targets["future_count"] < targets["count"].unsqueeze(-1)
        ).reshape(-1, 1)
        future_count_loss = F.cross_entropy(
            future_count_logits,
            targets["future_count"].reshape(-1),
        )
        future_heat_pos_weight = None
        if self.future_heat_pos_weight != 1.0:
            future_heat_pos_weight = torch.full_like(
                targets["future_heatmap"],
                self.future_heat_pos_weight,
            )
        future_heat_loss = F.binary_cross_entropy_with_logits(
            predictions["future_heatmap_logits"],
            targets["future_heatmap"],
            pos_weight=future_heat_pos_weight,
        )
        future_slot_activity_loss, future_slot_regression_loss = self._slot_losses(
            future_slot_logits,
            future_slot_target,
            activity_pos_weight=self.future_slot_activity_pos_weight,
            activity_neg_weight=self.future_slot_activity_neg_weight,
            deactivate_mask=future_decrease_mask,
            deactivate_weight=self.future_slot_deactivate_weight,
        )
        future_slot_count_consistency_loss = self._future_slot_count_consistency_loss(
            predictions["future_slot_logits"],
            targets["future_count"].to(dtype=predictions["future_slot_logits"].dtype),
            current_count=targets["count"].to(dtype=predictions["future_slot_logits"].dtype),
            transition_weight=self.future_slot_count_transition_weight,
        )
        future_track_loss = future_slot_activity_loss + future_slot_regression_loss
        motion_loss = F.cross_entropy(predictions["motion_logits"], targets["trend_class"])
        future_loss = (
            self.future_count_weight * future_count_loss
            + self.future_heat_weight * future_heat_loss
            + self.future_track_weight * future_track_loss
            + self.future_slot_count_consistency_weight * future_slot_count_consistency_loss
        )
        total = (
            self.count_weight * count_loss
            + self.heat_weight * heat_loss
            + self.track_weight * track_loss
            + self.slot_count_consistency_weight * slot_count_consistency_loss
            + future_loss
            + self.motion_weight * motion_loss
        )
        return {
            "loss": total,
            "count_loss": count_loss.detach(),
            "heat_loss": heat_loss.detach(),
            "slot_activity_loss": slot_activity_loss.detach(),
            "slot_regression_loss": slot_regression_loss.detach(),
            "slot_count_consistency_loss": slot_count_consistency_loss.detach(),
            "track_loss": track_loss.detach(),
            "future_count_loss": future_count_loss.detach(),
            "future_heat_loss": future_heat_loss.detach(),
            "future_slot_activity_loss": future_slot_activity_loss.detach(),
            "future_slot_regression_loss": future_slot_regression_loss.detach(),
            "future_slot_count_consistency_loss": future_slot_count_consistency_loss.detach(),
            "future_track_loss": future_track_loss.detach(),
            "future_loss": future_loss.detach(),
            "motion_loss": motion_loss.detach(),
        }

    def _slot_count_consistency_loss(
        self,
        pred: torch.Tensor,
        target_count: torch.Tensor,
    ) -> torch.Tensor:
        pred_count = torch.sigmoid(pred[..., 0]).sum(dim=-1)
        return F.smooth_l1_loss(pred_count, target_count)

    def _future_slot_count_consistency_loss(
        self,
        pred: torch.Tensor,
        target_count: torch.Tensor,
        *,
        current_count: torch.Tensor,
        transition_weight: float,
    ) -> torch.Tensor:
        pred_count = torch.sigmoid(pred[..., 0]).sum(dim=-1)
        frame_loss = F.smooth_l1_loss(pred_count, target_count, reduction="none")
        if transition_weight != 1.0:
            weights = torch.ones_like(frame_loss)
            transition_mask = target_count != current_count.unsqueeze(-1)
            weights = torch.where(
                transition_mask,
                weights.new_full((), transition_weight),
                weights,
            )
            frame_loss = frame_loss * weights
            return frame_loss.sum() / weights.sum().clamp_min(1.0)
        return frame_loss.mean()

    def _slot_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        activity_pos_weight: float,
        activity_neg_weight: float,
        deactivate_mask: torch.Tensor | None = None,
        deactivate_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activity_pred = pred[..., 0]
        activity_target = target[..., 0]
        activity_loss = F.binary_cross_entropy_with_logits(
            activity_pred,
            activity_target,
            reduction="none",
        )
        activity_weights = torch.ones_like(activity_target)
        if activity_pos_weight != 1.0:
            activity_weights = torch.where(
                activity_target > 0.5,
                activity_weights * activity_pos_weight,
                activity_weights,
            )
        if activity_neg_weight != 1.0:
            activity_weights = torch.where(
                activity_target <= 0.5,
                activity_weights * activity_neg_weight,
                activity_weights,
            )
        if deactivate_mask is not None and deactivate_weight != 1.0:
            expanded_mask = deactivate_mask.expand_as(activity_target)
            activity_weights = torch.where(
                expanded_mask & (activity_target <= 0.5),
                activity_weights * deactivate_weight,
                activity_weights,
            )
        activity_loss = (activity_loss * activity_weights).sum() / activity_weights.sum().clamp_min(
            1.0
        )
        active_mask = activity_target.unsqueeze(-1)
        regression_pred = pred[..., 1:]
        regression_target = target[..., 1:]
        if torch.any(active_mask > 0.0):
            regression_loss = F.smooth_l1_loss(
                regression_pred * active_mask,
                regression_target * active_mask,
                reduction="sum",
            )
            normalizer = active_mask.sum().clamp_min(1.0)
            regression_loss = regression_loss / normalizer
        else:
            regression_loss = pred.new_tensor(0.0)
        return activity_loss, regression_loss
