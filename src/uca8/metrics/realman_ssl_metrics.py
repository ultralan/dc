from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from uca8.geometry.uca8 import wrap_angle


@dataclass(slots=True)
class LocalizationStats:
    """定位指标的累计量, 便于跨 batch/子集汇总."""

    error_sum_deg: torch.Tensor
    acc5_count: torch.Tensor
    total: torch.Tensor


def azimuth_grid_deg(num_bins: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return bin-center-compatible azimuth grid over [-180, 180)."""
    return torch.linspace(-180.0, 180.0, steps=num_bins + 1, device=device, dtype=dtype)[:-1]


def circular_abs_error_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """计算角度的环形绝对误差, 单位为度.

    例如预测 ``179`` 度、标签 ``-179`` 度时, 误差应为 2 度, 不是 358 度.
    本项目 RealMAN MAE 都通过这个函数计算, 避免角度周期边界造成指标失真.
    """
    pred_rad = torch.deg2rad(pred_deg)
    target_rad = torch.deg2rad(target_deg)
    return torch.abs(wrap_angle(pred_rad - target_rad)) * (180.0 / math.pi)


def heatmap_logits_to_azimuth_deg(heatmap_logits: torch.Tensor) -> torch.Tensor:
    """把 heatmap logits 转成峰值方位 bin 对应的角度."""
    if heatmap_logits.ndim < 2:
        raise ValueError("heatmap_logits must have shape [..., bins].")
    bins = heatmap_logits.shape[-1]
    grid = azimuth_grid_deg(
        bins,
        device=heatmap_logits.device,
        dtype=heatmap_logits.dtype,
    )
    peak_bins = torch.sigmoid(heatmap_logits).argmax(dim=-1)
    return grid[peak_bins]


def target_slot_primary_azimuth_deg(
    slot_state: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 slot 标签里读取主活跃声源的方位角.

    当前 RealMAN 实验经过筛选后主要是单声源, 但标签格式仍然是 slot-based.
    这里统一选最活跃的 slot, 保证 MAE/ACC@5 的目标定义一致.
    """
    if slot_state.ndim < 3:
        raise ValueError("slot_state must have shape [..., slots, features].")
    activity = slot_state[..., 0]
    valid = (activity > threshold).any(dim=-1)
    slot_idx = activity.argmax(dim=-1)
    gathered = torch.gather(
        slot_state,
        dim=-2,
        index=slot_idx[..., None, None].expand(*slot_idx.shape, 1, slot_state.shape[-1]),
    ).squeeze(-2)
    theta = torch.atan2(gathered[..., 1], gathered[..., 2]) * (180.0 / math.pi)
    return theta, valid


def slot_logits_to_primary_azimuth_deg(
    slot_logits: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从预测 slot 中读取最活跃声源的方位角."""
    if slot_logits.ndim < 3:
        raise ValueError("slot_logits must have shape [..., slots, features].")
    activity = torch.sigmoid(slot_logits[..., 0])
    valid = activity.max(dim=-1).values > threshold
    slot_idx = activity.argmax(dim=-1)
    gathered = torch.gather(
        slot_logits,
        dim=-2,
        index=slot_idx[..., None, None].expand(*slot_idx.shape, 1, slot_logits.shape[-1]),
    ).squeeze(-2)
    theta = torch.atan2(gathered[..., 1], gathered[..., 2]) * (180.0 / math.pi)
    return theta, valid


def localization_stats_from_angles(
    pred_deg: torch.Tensor,
    target_deg: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    acc_threshold_deg: float = 5.0,
    eps: float = 1e-5,
) -> LocalizationStats:
    """从角度张量累计 MAE 分子和 ACC@threshold 计数."""
    if pred_deg.shape != target_deg.shape:
        raise ValueError("pred_deg and target_deg must have the same shape.")
    if valid is None:
        valid = torch.ones_like(pred_deg, dtype=torch.bool)
    if valid.shape != pred_deg.shape:
        raise ValueError("valid must have the same shape as pred_deg.")
    error = circular_abs_error_deg(pred_deg, target_deg)
    selected = error[valid]
    if selected.numel() == 0:
        zero = error.new_tensor(0.0)
        return LocalizationStats(zero, zero, zero)
    return LocalizationStats(
        error_sum_deg=selected.sum(),
        acc5_count=(selected <= acc_threshold_deg + eps).sum().to(dtype=error.dtype),
        total=selected.new_tensor(float(selected.numel())),
    )


def heatmap_localization_stats(
    heatmap_logits: torch.Tensor,
    target_slot_state: torch.Tensor,
    *,
    acc_threshold_deg: float = 5.0,
) -> LocalizationStats:
    """从当前帧 heatmap logits 计算定位指标.

    这是当前 RealMAN 8 通道对比实验的主指标路径.
    """
    pred_deg = heatmap_logits_to_azimuth_deg(heatmap_logits)
    target_deg, valid = target_slot_primary_azimuth_deg(target_slot_state)
    return localization_stats_from_angles(
        pred_deg,
        target_deg,
        valid,
        acc_threshold_deg=acc_threshold_deg,
    )


def slot_primary_localization_stats(
    slot_logits: torch.Tensor,
    target_slot_state: torch.Tensor,
    *,
    acc_threshold_deg: float = 5.0,
    missing_prediction_error_deg: float = 180.0,
) -> LocalizationStats:
    """从主预测 slot 计算定位指标."""
    pred_deg, pred_valid = slot_logits_to_primary_azimuth_deg(slot_logits)
    target_deg, target_valid = target_slot_primary_azimuth_deg(target_slot_state)
    error = circular_abs_error_deg(pred_deg, target_deg)
    error = torch.where(
        pred_valid,
        error,
        torch.full_like(error, missing_prediction_error_deg),
    )
    selected = error[target_valid]
    if selected.numel() == 0:
        zero = error.new_tensor(0.0)
        return LocalizationStats(zero, zero, zero)
    return LocalizationStats(
        error_sum_deg=selected.sum(),
        acc5_count=(selected <= acc_threshold_deg + 1e-5).sum().to(dtype=error.dtype),
        total=selected.new_tensor(float(selected.numel())),
    )


def merge_localization_stats(*stats_items: LocalizationStats) -> LocalizationStats:
    """合并多个 ``LocalizationStats`` 累计量."""
    if not stats_items:
        zero = torch.tensor(0.0)
        return LocalizationStats(zero, zero, zero)
    return LocalizationStats(
        error_sum_deg=sum((item.error_sum_deg for item in stats_items), stats_items[0].error_sum_deg.new_tensor(0.0)),
        acc5_count=sum((item.acc5_count for item in stats_items), stats_items[0].acc5_count.new_tensor(0.0)),
        total=sum((item.total for item in stats_items), stats_items[0].total.new_tensor(0.0)),
    )


def localization_mae_deg(stats: LocalizationStats) -> float:
    """从累计量计算 MAE, 单位为度."""
    total = float(stats.total.item())
    if total <= 0.0:
        return 0.0
    return float(stats.error_sum_deg.item()) / total


def localization_acc5(stats: LocalizationStats) -> float:
    """从累计量计算 ACC@5."""
    total = float(stats.total.item())
    if total <= 0.0:
        return 0.0
    return float(stats.acc5_count.item()) / total
