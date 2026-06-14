"""多任务联合损失函数: 覆盖声源定位-跟踪-趋势预测的全部监督信号.

损失项包括:
  当前帧:
    - count_loss:            声源数分类交叉熵
    - heat_loss:             方位角热力图BCE
    - current_heat_kl:       热力图分布KL散度
    - slot_activity_loss:    槽位活动性BCE (支持正/负样本不等权重)
    - slot_regression_loss:  活跃槽位的(sin,cos,rho,omega)回归SmoothL1
    - slot_count_consistency_loss: 预测活跃槽位数与标签声源数一致性
    - slot_heat_consistency_loss:  槽位渲染热力图与预测热力图的MSE一致性

  未来帧:
    - future_count_loss:     未来声源数分类
    - future_heat_loss:      未来热力图BCE
    - future_heat_kl:        未来热力图KL散度
    - future_slot_*:         同当前帧的slot系列损失
    - future_slot_delta_loss: 相邻未来帧间的角位移/距离/角速度变化量回归
    - future_slot_count_transition_weight: 声源数发生变化的帧加权

  全局:
    - motion_loss:           运动趋势3分类交叉熵 (CW/stable/CCW)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from uca8.geometry.uca8 import wrap_angle


class TrackTrendMultiTaskLoss(nn.Module):
    """TrackTrend 网络的多任务损失聚合器.

    输入的 ``predictions`` 来自 ``UCA8TrackTrendNet.forward``,
    ``targets`` 来自数据集 ``__getitem__``. 本类不创建标签, 只负责把各监督项
    按权重组合成训练总损失, 并返回 detached 的分项损失用于日志记录.
    """

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
        """初始化各损失项权重.

        大多数权重直接控制某个监督分支在总损失中的占比;
        ``*_pos_weight`` 用于处理 heatmap/activity 正样本稀疏问题;
        ``future_slot_count_transition_weight`` 用于加强声源数变化帧.
        """
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
        predictions: dict[str, torch.Tensor],  # 模型前向输出的预测字典
        targets: dict[str, torch.Tensor],       # 数据集提供的标签字典
    ) -> dict[str, torch.Tensor]:
        """计算全部损失项, 返回包含总损失和各项损失的字典."""

        # ===== 当前帧损失 =====
        # 声源数分类交叉熵
        count_loss = F.cross_entropy(predictions["count_logits"], targets["count"])
        # 方位角热力图BCE (支持正样本加权以应对稀疏性)
        heat_pos_weight = None
        if self.heat_pos_weight != 1.0:
            heat_pos_weight = torch.full_like(targets["heatmap"], self.heat_pos_weight)
        heat_loss = F.binary_cross_entropy_with_logits(
            predictions["heatmap_logits"],
            targets["heatmap"],
            pos_weight=heat_pos_weight,
        )
        # 热力图分布KL散度: 促使预测的热力图分布形态接近标签
        current_heat_kl = self._heatmap_kl_loss(predictions["heatmap_logits"], targets["heatmap"])
        # 槽位损失: activity BCE + 回归 SmoothL1
        slot_activity_loss, slot_regression_loss = self._slot_losses(
            predictions["slot_logits"],
            targets["slot_state"],
            activity_pos_weight=self.slot_activity_pos_weight,
            activity_neg_weight=self.slot_activity_neg_weight,
        )
        # 槽位数一致性: 预测的活跃槽位数应接近标签声源数
        slot_count_consistency_loss = self._slot_count_consistency_loss(
            predictions["slot_logits"],
            targets["count"].to(dtype=predictions["slot_logits"].dtype),
        )
        # 槽位-热力图一致性: 从槽位渲染的热力图应与预测热力图一致
        current_slot_heat_consistency = self._slot_heat_consistency_loss(
            predictions["slot_logits"],
            predictions["heatmap_logits"],
        )
        track_loss = slot_activity_loss + slot_regression_loss

        # ===== 未来帧损失 =====
        # 将未来帧展平: [B, F, ...] → [B*F, ...] 以统一计算
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
        # 未来帧中声源数减少的掩码, 用于对"声源消失"场景加权
        future_decrease_mask = (
            targets["future_count"] < targets["count"].unsqueeze(-1)
        ).reshape(-1, 1)
        # 未来声源数分类
        future_count_loss = F.cross_entropy(
            future_count_logits,
            targets["future_count"].reshape(-1),
        )
        # 未来热力图BCE
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
        # 未来热力图KL散度
        future_heat_kl = self._heatmap_kl_loss(
            predictions["future_heatmap_logits"],
            targets["future_heatmap"],
        )
        # 未来槽位损失 (含deactivate加权: 声源消失帧的inactive样本权重更高)
        future_slot_activity_loss, future_slot_regression_loss = self._slot_losses(
            future_slot_logits,
            future_slot_target,
            activity_pos_weight=self.future_slot_activity_pos_weight,
            activity_neg_weight=self.future_slot_activity_neg_weight,
            deactivate_mask=future_decrease_mask,
            deactivate_weight=self.future_slot_deactivate_weight,
        )
        # 未来槽位数一致性 (含transition加权: 声源数变化帧权重更高)
        future_slot_count_consistency_loss = self._future_slot_count_consistency_loss(
            predictions["future_slot_logits"],
            targets["future_count"].to(dtype=predictions["future_slot_logits"].dtype),
            current_count=targets["count"].to(dtype=predictions["future_slot_logits"].dtype),
            transition_weight=self.future_slot_count_transition_weight,
        )
        # 未来帧间变化量(delta)回归: 角位移/距离/角速度的帧间差
        future_slot_delta_loss = self._future_slot_delta_loss(
            predictions["future_slot_logits"],
            targets["future_slot_state"],
        )
        # 未来槽位-热力图一致性
        future_slot_heat_consistency = self._slot_heat_consistency_loss(
            predictions["future_slot_logits"],
            predictions["future_heatmap_logits"],
        )
        slot_heat_consistency_loss = 0.5 * (
            current_slot_heat_consistency + future_slot_heat_consistency
        )
        future_track_loss = future_slot_activity_loss + future_slot_regression_loss
        # 运动趋势分类交叉熵
        motion_loss = F.cross_entropy(predictions["motion_logits"], targets["trend_class"])

        # ===== 加权求和 =====
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
        pred: torch.Tensor,      # slot_logits [B, max_sources, 5]
        target_count: torch.Tensor,  # 标签声源数 [B]
    ) -> torch.Tensor:
        """槽位数一致性: 活跃槽位数(对activity求和)应接近标签声源数."""
        pred_count = torch.sigmoid(pred[..., 0]).sum(dim=-1)
        return F.smooth_l1_loss(pred_count, target_count)

    def _future_slot_count_consistency_loss(
        self,
        pred: torch.Tensor,
        target_count: torch.Tensor,    # 未来帧标签声源数 [B, F]
        *,
        current_count: torch.Tensor,   # 当前帧声源数 [B]
        transition_weight: float,      # 声源数变化帧的额外权重
    ) -> torch.Tensor:
        """未来帧槽位数一致性, 对声源数发生变化的帧施加额外权重."""
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
        pred: torch.Tensor,    # [B, max_sources, 5] 或 [B*F, max_sources, 5]
        target: torch.Tensor,  # 同形状
        *,
        activity_pos_weight: float,  # 正样本(活跃)权重
        activity_neg_weight: float,  # 负样本(非活跃)权重
        deactivate_mask: torch.Tensor | None = None,  # 未来声源减少的掩码
        deactivate_weight: float = 1.0,  # deactivate场景的额外权重
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算槽位activity BCE损失 + 回归SmoothL1损失(仅对活跃槽位计算)."""
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
        """热力图KL散度: 促使预测热力图分布形态接近标签(跳过全零帧)."""
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
        """未来帧间变化量损失: 相邻帧间角位移/距离/角速度差的SmoothL1回归.

        仅在相邻帧均为活跃的槽位对上计算, 鼓励平滑的轨迹预测.
        """
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
        """槽位-热力图一致性: 从槽位预测渲染出的热力图应与直接预测的热力图一致."""
        rendered_heatmap = self._render_slot_heatmap(
            slot_logits,
            bins=heatmap_logits.shape[-1],
        )
        predicted_heatmap = torch.sigmoid(heatmap_logits)
        return F.mse_loss(rendered_heatmap, predicted_heatmap)

    def _render_slot_heatmap(self, slot_logits: torch.Tensor, *, bins: int) -> torch.Tensor:
        """从槽位预测渲染热力图: 每个活跃槽位在其预测方位角处生成高斯峰, 多源叠加."""
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
