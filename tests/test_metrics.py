from __future__ import annotations

import unittest

import torch

from uca8.metrics import (
    slot_activity_confusion_stats,
    slot_angle_error_stats_deg,
    slot_count_accuracy_stats,
    slot_count_from_state,
)


class SlotMetricTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
