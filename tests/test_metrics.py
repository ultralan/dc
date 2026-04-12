from __future__ import annotations

import unittest

import torch

from uca8.metrics import (
    future_slot_delta_error_stats_deg,
    heatmap_peak_recall_stats,
    primary_slot_range_stats,
    slot_activity_confusion_stats,
    slot_angle_error_stats_deg,
    slot_count_accuracy_stats,
    slot_count_from_state,
    slot_trend_label_from_sequence,
)


class SlotMetricTests(unittest.TestCase):
    @staticmethod
    def _sequence_from_angles(angles_deg: list[float], *, is_logits: bool = False) -> torch.Tensor:
        states = torch.zeros(len(angles_deg), 2, 5)
        for frame_idx, angle_deg in enumerate(angles_deg):
            angle_rad = torch.deg2rad(torch.tensor(float(angle_deg)))
            states[frame_idx, 0, 0] = 4.0 if is_logits else 1.0
            states[frame_idx, 0, 1] = torch.sin(angle_rad)
            states[frame_idx, 0, 2] = torch.cos(angle_rad)
        return states

    def test_slot_angle_error_uses_active_targets_only(self) -> None:
        pred = torch.zeros(2, 2, 5)
        target = torch.zeros(2, 2, 5)
        target[0, 0, 0] = 1.0
        target[0, 0, 1] = 0.0
        target[0, 0, 2] = 1.0
        pred[0, 0, 1] = 1.0
        pred[0, 0, 2] = 0.0

        error_sum, active_count = slot_angle_error_stats_deg(pred, target)

        self.assertEqual(float(active_count.item()), 1.0)
        self.assertAlmostEqual(float(error_sum.item()), 90.0, places=4)

    def test_slot_count_and_activity_stats_use_activity_logits(self) -> None:
        pred = torch.zeros(2, 3, 5)
        target = torch.zeros(2, 3, 5)

        pred[0, 0, 0] = 4.0
        pred[0, 1, 0] = 3.0
        pred[0, 2, 0] = -4.0
        pred[1, 0, 0] = 3.0
        pred[1, 1, 0] = -3.0
        pred[1, 2, 0] = 2.0

        target[0, 0, 0] = 1.0
        target[0, 1, 0] = 1.0
        target[1, 0, 0] = 1.0
        target[1, 1, 0] = 1.0

        pred_count = slot_count_from_state(pred, is_logits=True)
        target_count = slot_count_from_state(target, is_logits=False)
        count_correct, count_total = slot_count_accuracy_stats(pred, target)
        true_positive, false_positive, false_negative = slot_activity_confusion_stats(pred, target)

        self.assertTrue(torch.equal(pred_count, torch.tensor([2, 2])))
        self.assertTrue(torch.equal(target_count, torch.tensor([2, 2])))
        self.assertEqual(float(count_correct.item()), 2.0)
        self.assertEqual(float(count_total.item()), 2.0)
        self.assertEqual(float(true_positive.item()), 3.0)
        self.assertEqual(float(false_positive.item()), 1.0)
        self.assertEqual(float(false_negative.item()), 1.0)

    def test_future_range_ratio_and_trend_penalize_flat_prediction(self) -> None:
        target = self._sequence_from_angles([0.0, 20.0, 40.0, 60.0])
        pred = self._sequence_from_angles([15.0, 15.0, 15.0, 15.0], is_logits=True)

        range_ratio, pred_range, target_range, slot_idx = primary_slot_range_stats(
            pred,
            target,
            pred_is_logits=True,
            target_is_logits=False,
        )

        self.assertEqual(slot_idx, 0)
        self.assertLess(float(pred_range.item()), float(target_range.item()))
        self.assertLess(float(range_ratio.item()), 0.35)
        self.assertEqual(slot_trend_label_from_sequence(pred, is_logits=True), 0)
        self.assertEqual(slot_trend_label_from_sequence(target, is_logits=False), 1)

    def test_future_slot_delta_error_detects_frozen_geometry(self) -> None:
        target = self._sequence_from_angles([0.0, 30.0, 60.0, 90.0]).unsqueeze(0)
        pred = self._sequence_from_angles([5.0, 5.0, 5.0, 5.0], is_logits=True).unsqueeze(0)

        error_sum, active_count = future_slot_delta_error_stats_deg(pred, target)

        self.assertGreater(float(active_count.item()), 0.0)
        self.assertGreater(float(error_sum.item()), 0.0)

    def test_heatmap_peak_recall_allows_small_bin_offset(self) -> None:
        pred = torch.tensor(
            [
                [0.1, 0.2, 0.8, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.9, 0.1],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [
                [0.0, 0.7, 1.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8, 1.0, 0.3],
            ],
            dtype=torch.float32,
        )
        target_count = torch.tensor([1, 1], dtype=torch.long)

        recall_sum, total = heatmap_peak_recall_stats(
            pred,
            target,
            target_count,
            tolerance_bins=1,
        )

        self.assertEqual(float(total.item()), 2.0)
        self.assertAlmostEqual(float(recall_sum.item()) / float(total.item()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
