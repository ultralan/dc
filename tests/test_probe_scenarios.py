from __future__ import annotations

import unittest

import torch

from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.sim import build_probe_rollout_samples, build_probe_sample


class ProbeScenarioTests(unittest.TestCase):
    def test_probe_samples_have_expected_transition_counts(self) -> None:
        mic_positions = make_uniform_circular_array(8, radius=0.06)
        mono_waveform = torch.randn(64000, dtype=torch.float32)
        common_kwargs = {
            "mono_waveform": mono_waveform,
            "mic_positions": mic_positions,
            "sample_rate": 16000,
            "hop_length": 160,
            "win_length": 400,
            "sound_speed": 343.0,
            "history_frames": 64,
            "future_frames": 16,
            "num_heatmap_bins": 72,
            "max_sources": 4,
        }

        dual_cross = build_probe_sample(scenario="dual_cross", **common_kwargs)
        source_enter = build_probe_sample(scenario="source_enter", **common_kwargs)
        source_leave = build_probe_sample(scenario="source_leave", **common_kwargs)

        self.assertEqual(int(dual_cross.count.item()), 2)
        self.assertTrue(torch.all(dual_cross.future_count == 2))

        self.assertEqual(int(source_enter.count.item()), 1)
        self.assertEqual(int(source_enter.future_count[0].item()), 1)
        self.assertEqual(int(source_enter.future_count[-1].item()), 2)
        self.assertEqual(source_enter.transition_start_index, 4)

        self.assertEqual(int(source_leave.count.item()), 2)
        self.assertEqual(int(source_leave.future_count[0].item()), 2)
        self.assertEqual(int(source_leave.future_count[-1].item()), 1)
        self.assertEqual(source_leave.transition_start_index, 4)

    def test_probe_rollout_tracks_post_transition_windows(self) -> None:
        mic_positions = make_uniform_circular_array(8, radius=0.06)
        mono_waveform = torch.randn(64000, dtype=torch.float32)
        rollout = build_probe_rollout_samples(
            scenario="source_leave",
            mono_waveform=mono_waveform,
            mic_positions=mic_positions,
            sample_rate=16000,
            hop_length=160,
            win_length=400,
            sound_speed=343.0,
            history_frames=64,
            future_frames=16,
            num_heatmap_bins=72,
            max_sources=4,
            animation_steps=24,
        )

        self.assertEqual(len(rollout), 24)
        self.assertEqual(int(rollout[0].count.item()), 2)
        self.assertEqual(int(rollout[-1].count.item()), 1)
        self.assertIsNotNone(rollout[0].transition_start_index)
        self.assertIsNone(rollout[-1].transition_start_index)


if __name__ == "__main__":
    unittest.main()
