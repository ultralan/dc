"""训练和评估共用的指标工具.

``realman_ssl_metrics`` 放和 baseline 对比最直接相关的定位指标:
环形角度 MAE、ACC@5, 以及 heatmap/slot 到方位角的转换.
``slot_metrics`` 放辅助的槽位跟踪、运动趋势和热力图诊断指标.
"""

from __future__ import annotations

from .realman_ssl_metrics import (
    azimuth_grid_deg,
    circular_abs_error_deg,
    heatmap_localization_stats,
    heatmap_logits_to_azimuth_deg,
    localization_acc5,
    localization_mae_deg,
    localization_stats_from_angles,
    merge_localization_stats,
    slot_logits_to_primary_azimuth_deg,
    slot_primary_localization_stats,
    target_slot_primary_azimuth_deg,
)
from .slot_metrics import (
    angle_range_deg,
    future_slot_delta_error_stats_deg,
    heatmap_contrast,
    heatmap_peak_recall_stats,
    primary_slot_index_from_sequence,
    primary_slot_range_stats,
    slot_activity_confusion_stats,
    slot_activity_mask,
    slot_angles_deg,
    slot_angle_error_stats_deg,
    slot_count_accuracy_stats,
    slot_count_from_state,
    slot_trend_label_from_sequence,
)

__all__ = [
    "angle_range_deg",
    "azimuth_grid_deg",
    "circular_abs_error_deg",
    "future_slot_delta_error_stats_deg",
    "heatmap_contrast",
    "heatmap_localization_stats",
    "heatmap_logits_to_azimuth_deg",
    "heatmap_peak_recall_stats",
    "localization_acc5",
    "localization_mae_deg",
    "localization_stats_from_angles",
    "merge_localization_stats",
    "primary_slot_index_from_sequence",
    "primary_slot_range_stats",
    "slot_logits_to_primary_azimuth_deg",
    "slot_activity_confusion_stats",
    "slot_activity_mask",
    "slot_angles_deg",
    "slot_angle_error_stats_deg",
    "slot_count_accuracy_stats",
    "slot_count_from_state",
    "slot_primary_localization_stats",
    "slot_trend_label_from_sequence",
    "target_slot_primary_azimuth_deg",
]
