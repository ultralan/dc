from __future__ import annotations

import unittest

import torch

from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.losses.multi_task_loss import TrackTrendMultiTaskLoss
from uca8.models.tracktrend_net import UCA8TrackTrendNet


class ModelShapeTests(unittest.TestCase):
    def test_forward_and_loss_shapes(self) -> None:
        batch = 2
        history_frames = 128
        future_frames = 32
        channels = 8
        samples = history_frames * 160
        waveform = torch.randn(batch, channels, samples)
        vad_history = torch.rand(batch, history_frames, 1)
        model = UCA8TrackTrendNet(
            mic_positions=make_uniform_circular_array(channels, radius=0.045),
            history_frames=history_frames,
            future_frames=future_frames,
        )
        predictions = model(waveform, vad_history)
        self.assertEqual(
            tuple(predictions["features"]["logmel_ref"].shape),
            (batch, 1, history_frames, 64),
        )
        self.assertEqual(
            tuple(predictions["features"]["logmel_rms"].shape),
            (batch, 1, history_frames, 64),
        )
        self.assertTrue(torch.isfinite(predictions["features"]["logmel_ref"]).all())
        self.assertTrue(torch.isfinite(predictions["features"]["logmel_rms"]).all())
        self.assertEqual(tuple(predictions["count_logits"].shape), (batch, 5))
        self.assertEqual(tuple(predictions["heatmap_logits"].shape), (batch, 72))
        self.assertEqual(tuple(predictions["slot_logits"].shape), (batch, 4, 5))
        self.assertEqual(tuple(predictions["future_count_logits"].shape), (batch, future_frames, 5))
        self.assertEqual(
            tuple(predictions["future_heatmap_logits"].shape),
            (batch, future_frames, 72),
        )
        self.assertEqual(
            tuple(predictions["future_slot_logits"].shape),
            (batch, future_frames, 4, 5),
        )
        _, hidden = model.encode(waveform, vad_history=vad_history)
        _, slot_context = model.slot_head(hidden, query_context=hidden[:, -1], return_context=True)
        future_hidden = model.future_decoder(hidden, current=hidden[:, -1], slot_context=slot_context)
        self.assertGreater(
            float((future_hidden[:, 1:] - future_hidden[:, :-1]).abs().sum().item()),
            0.0,
        )
        criterion = TrackTrendMultiTaskLoss()
        targets = {
            "count": torch.zeros(batch, dtype=torch.long),
            "heatmap": torch.zeros(batch, 72),
            "slot_state": torch.zeros(batch, 4, 5),
            "future_count": torch.zeros(batch, future_frames, dtype=torch.long),
            "future_heatmap": torch.zeros(batch, future_frames, 72),
            "future_slot_state": torch.zeros(batch, future_frames, 4, 5),
            "trend_class": torch.ones(batch, dtype=torch.long),
        }
        losses = criterion(predictions, targets)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIn("slot_activity_loss", losses)
        self.assertIn("slot_regression_loss", losses)
        self.assertIn("slot_count_consistency_loss", losses)
        self.assertIn("current_heat_kl", losses)
        self.assertIn("slot_heat_consistency_loss", losses)
        self.assertIn("future_slot_activity_loss", losses)
        self.assertIn("future_slot_regression_loss", losses)
        self.assertIn("future_slot_count_consistency_loss", losses)
        self.assertIn("future_heat_kl", losses)
        self.assertIn("future_slot_delta_loss", losses)

    def test_future_deactivate_weight_increases_false_positive_penalty(self) -> None:
        predictions = {
            "count_logits": torch.zeros(1, 5),
            "heatmap_logits": torch.zeros(1, 72),
            "slot_logits": torch.zeros(1, 4, 5),
            "future_count_logits": torch.zeros(1, 1, 5),
            "future_heatmap_logits": torch.zeros(1, 1, 72),
            "future_slot_logits": torch.zeros(1, 1, 4, 5),
            "motion_logits": torch.zeros(1, 3),
        }
        predictions["future_slot_logits"][0, 0, 0, 0] = 5.0
        predictions["future_slot_logits"][0, 0, 1, 0] = 5.0
        targets = {
            "count": torch.tensor([2], dtype=torch.long),
            "heatmap": torch.zeros(1, 72),
            "slot_state": torch.zeros(1, 4, 5),
            "future_count": torch.tensor([[1]], dtype=torch.long),
            "future_heatmap": torch.zeros(1, 1, 72),
            "future_slot_state": torch.zeros(1, 1, 4, 5),
            "trend_class": torch.ones(1, dtype=torch.long),
        }
        targets["future_slot_state"][0, 0, 0, 0] = 1.0
        base_loss = TrackTrendMultiTaskLoss(
            count_weight=0.0,
            heat_weight=0.0,
            track_weight=0.0,
            future_count_weight=0.0,
            future_heat_weight=0.0,
            future_track_weight=1.0,
            motion_weight=0.0,
            future_slot_activity_neg_weight=1.0,
            future_slot_deactivate_weight=1.0,
        )(predictions, targets)
        weighted_loss = TrackTrendMultiTaskLoss(
            count_weight=0.0,
            heat_weight=0.0,
            track_weight=0.0,
            future_count_weight=0.0,
            future_heat_weight=0.0,
            future_track_weight=1.0,
            motion_weight=0.0,
            future_slot_activity_neg_weight=1.0,
            future_slot_deactivate_weight=4.0,
        )(predictions, targets)

        self.assertGreater(
            float(weighted_loss["future_slot_activity_loss"].item()),
            float(base_loss["future_slot_activity_loss"].item()),
        )


if __name__ == "__main__":
    unittest.main()
