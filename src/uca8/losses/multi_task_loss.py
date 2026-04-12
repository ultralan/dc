from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from uca8.geometry.uca8 import wrap_angle


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
        current_heat_kl_weight: float = 0.5,
        future_heat_kl_weight: float = 0.75,
        future_slot_delta_weight: float = 1.0,
        slot_heat_consistency_weight: float = 0.5,
        heatmap_sigma_bins: float = 1.5,
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
        self.current_heat_kl_weight = current_heat_kl_weight
        self.future_heat_kl_weight = future_heat_kl_weight
        self.future_slot_delta_weight = future_slot_delta_weight
        self.slot_heat_consistency_weight = slot_heat_consistency_weight
        self.heatmap_sigma_bins = heatmap_sigma_bins

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
        current_heat_kl = self._heatmap_kl_loss(predictions["heatmap_logits"], targets["heatmap"])
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
        current_slot_heat_consistency = self._slot_heat_consistency_loss(
            predictions["slot_logits"],
            predictions["heatmap_logits"],
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
        future_heat_kl = self._heatmap_kl_loss(
            predictions["future_heatmap_logits"],
            targets["future_heatmap"],
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
        future_slot_delta_loss = self._future_slot_delta_loss(
            predictions["future_slot_logits"],
            targets["future_slot_state"],
        )
        future_slot_heat_consistency = self._slot_heat_consistency_loss(
            predictions["future_slot_logits"],
            predictions["future_heatmap_logits"],
        )
        slot_heat_consistency_loss = 0.5 * (
            current_slot_heat_consistency + future_slot_heat_consistency
        )
        future_track_loss = future_slot_activity_loss + future_slot_regression_loss
        motion_loss = F.cross_entropy(predictions["motion_logits"], targets["trend_class"])
        future_loss = (
            self.future_count_weight * future_count_loss
            + self.future_heat_weight * future_heat_loss
            + self.future_track_weight * future_track_loss
            + self.future_slot_count_consistency_weight * future_slot_count_consistency_loss
            + self.future_heat_kl_weight * future_heat_kl
            + self.future_slot_delta_weight * future_slot_delta_loss
        )
        total = (
            self.count_weight * count_loss
            + self.heat_weight * heat_loss
            + self.track_weight * track_loss
            + self.slot_count_consistency_weight * slot_count_consistency_loss
            + future_loss
            + self.motion_weight * motion_loss
            + self.current_heat_kl_weight * current_heat_kl
            + self.slot_heat_consistency_weight * slot_heat_consistency_loss
        )
        return {
            "loss": total,
            "count_loss": count_loss.detach(),
            "heat_loss": heat_loss.detach(),
            "current_heat_kl": current_heat_kl.detach(),
            "slot_activity_loss": slot_activity_loss.detach(),
            "slot_regression_loss": slot_regression_loss.detach(),
            "slot_count_consistency_loss": slot_count_consistency_loss.detach(),
            "slot_heat_consistency_loss": slot_heat_consistency_loss.detach(),
            "track_loss": track_loss.detach(),
            "future_count_loss": future_count_loss.detach(),
            "future_heat_loss": future_heat_loss.detach(),
            "future_heat_kl": future_heat_kl.detach(),
            "future_slot_activity_loss": future_slot_activity_loss.detach(),
            "future_slot_regression_loss": future_slot_regression_loss.detach(),
            "future_slot_count_consistency_loss": future_slot_count_consistency_loss.detach(),
            "future_slot_delta_loss": future_slot_delta_loss.detach(),
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

    def _heatmap_kl_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bins = logits.shape[-1]
        flat_logits = logits.reshape(-1, bins)
        flat_target = target.reshape(-1, bins)
        valid_mask = flat_target.sum(dim=-1) > 0.0
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)
        target_dist = flat_target[valid_mask]
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.kl_div(
            F.log_softmax(flat_logits[valid_mask], dim=-1),
            target_dist,
            reduction="batchmean",
        )

    def _future_slot_delta_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_activity = target[..., 0]
        active_pair_mask = (target_activity[..., 1:, :] > 0.5) & (target_activity[..., :-1, :] > 0.5)
        if not torch.any(active_pair_mask):
            return pred.new_tensor(0.0)
        pred_theta = torch.atan2(pred[..., 1], pred[..., 2])
        target_theta = torch.atan2(target[..., 1], target[..., 2])
        pred_theta_delta = wrap_angle(pred_theta[..., 1:, :] - pred_theta[..., :-1, :])
        target_theta_delta = wrap_angle(target_theta[..., 1:, :] - target_theta[..., :-1, :])
        pred_rho_delta = pred[..., 3][..., 1:, :] - pred[..., 3][..., :-1, :]
        target_rho_delta = target[..., 3][..., 1:, :] - target[..., 3][..., :-1, :]
        pred_omega_delta = pred[..., 4][..., 1:, :] - pred[..., 4][..., :-1, :]
        target_omega_delta = target[..., 4][..., 1:, :] - target[..., 4][..., :-1, :]
        angle_loss = F.smooth_l1_loss(
            pred_theta_delta[active_pair_mask],
            target_theta_delta[active_pair_mask],
        )
        rho_loss = F.smooth_l1_loss(
            pred_rho_delta[active_pair_mask],
            target_rho_delta[active_pair_mask],
        )
        omega_loss = F.smooth_l1_loss(
            pred_omega_delta[active_pair_mask],
            target_omega_delta[active_pair_mask],
        )
        return angle_loss + rho_loss + omega_loss

    def _slot_heat_consistency_loss(
        self,
        slot_logits: torch.Tensor,
        heatmap_logits: torch.Tensor,
    ) -> torch.Tensor:
        rendered_heatmap = self._render_slot_heatmap(
            slot_logits,
            bins=heatmap_logits.shape[-1],
        )
        predicted_heatmap = torch.sigmoid(heatmap_logits)
        return F.mse_loss(rendered_heatmap, predicted_heatmap)

    def _render_slot_heatmap(self, slot_logits: torch.Tensor, *, bins: int) -> torch.Tensor:
        theta = torch.atan2(slot_logits[..., 1], slot_logits[..., 2])
        activity = torch.sigmoid(slot_logits[..., 0]).clamp(0.0, 1.0)
        azimuth_grid = torch.linspace(
            -math.pi,
            math.pi,
            steps=bins + 1,
            device=slot_logits.device,
            dtype=slot_logits.dtype,
        )[:-1]
        delta = wrap_angle(azimuth_grid - theta.unsqueeze(-1))
        bin_offset = delta / (2.0 * math.pi / bins)
        gaussian = torch.exp(-0.5 * (bin_offset / self.heatmap_sigma_bins) ** 2)
        contributions = (activity.unsqueeze(-1) * gaussian).clamp(0.0, 1.0)
        return 1.0 - torch.prod(1.0 - contributions, dim=-2)
