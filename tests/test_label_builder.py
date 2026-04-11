from __future__ import annotations

import unittest

import torch

from uca8.data.label_builder import TrackTrendLabelBuilder


class LabelBuilderTests(unittest.TestCase):
    def test_sequence_targets_shapes(self) -> None:
        builder = TrackTrendLabelBuilder(num_heatmap_bins=72, max_sources=4, frame_hop_seconds=0.01)
        frames = 12
        array_positions = torch.zeros(frames, 3)
        source_positions = torch.zeros(frames, 4, 3)
        source_positions[:, 0, 0] = torch.linspace(1.0, 0.5, steps=frames)
        source_positions[:, 0, 1] = torch.linspace(-0.5, 0.5, steps=frames)
        source_activity = torch.zeros(frames, 4)
        source_activity[:, 0] = 1.0
        output = builder.build_sequence_targets(
            source_positions=source_positions,
            array_positions=array_positions,
            source_activity=source_activity,
        )
        self.assertEqual(tuple(output.count.shape), (frames,))
        self.assertEqual(tuple(output.heatmap.shape), (frames, 72))
        self.assertEqual(tuple(output.slot_state.shape), (frames, 4, 5))
        self.assertTrue(torch.all(output.count == 1))

    def test_future_motion_classification(self) -> None:
        builder = TrackTrendLabelBuilder()
        slot_state = torch.zeros(8, 4, 5)
        theta = torch.linspace(0.0, 0.4, steps=8)
        slot_state[:, 0, 0] = 1.0
        slot_state[:, 0, 1] = torch.sin(theta)
        slot_state[:, 0, 2] = torch.cos(theta)
        trend = builder.classify_future_motion(slot_state)
        self.assertEqual(int(trend.item()), 2)

    def test_slot_memory_preserves_assignment_across_short_gaps(self) -> None:
        builder = TrackTrendLabelBuilder(max_sources=2, max_inactive_frames=3)
        frames = 3
        array_positions = torch.zeros(frames, 3)
        source_positions = torch.zeros(frames, 2, 3)
        source_activity = torch.zeros(frames, 2)

        theta_neg = torch.tensor(-0.8)
        theta_pos = torch.tensor(0.8)
        source_positions[0, 0, 0] = torch.cos(theta_neg)
        source_positions[0, 0, 1] = torch.sin(theta_neg)
        source_positions[0, 1, 0] = torch.cos(theta_pos)
        source_positions[0, 1, 1] = torch.sin(theta_pos)
        source_activity[0] = 1.0

        source_positions[2, 0, 0] = torch.cos(theta_pos)
        source_positions[2, 0, 1] = torch.sin(theta_pos)
        source_activity[2, 0] = 1.0

        output = builder.build_sequence_targets(
            source_positions=source_positions,
            array_positions=array_positions,
            source_activity=source_activity,
        )

        self.assertEqual(float(output.slot_state[2, 1, 0].item()), 1.0)
        self.assertEqual(float(output.slot_state[2, 0, 0].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
