from __future__ import annotations

import unittest

import torch

from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.sim import build_probe_rollout_samples, build_probe_sample, evaluate_probe_suite


def _state_to_logits(state: torch.Tensor) -> torch.Tensor:
    logits = state.clone()
    logits[..., 0] = torch.where(state[..., 0] > 0.5, torch.full_like(state[..., 0], 4.0), -4.0)
    return logits


class ReplayProbeModel(torch.nn.Module):
    def __init__(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.outputs = outputs
        self.cursor = 0

    def forward(self, waveform: torch.Tensor, vad_history: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.outputs[self.cursor]
        self.cursor += 1
        device = waveform.device
        return {key: value.to(device) for key, value in output.items()}


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

    def test_geometry_checkpoint_score_prefers_dynamic_probe_model(self) -> None:
        mic_positions = make_uniform_circular_array(8, radius=0.06)
        mono_waveform = torch.randn(64000, dtype=torch.float32)
        scenarios = ("dual_cross", "source_enter", "source_leave")
        probe_samples = {
            scenario: build_probe_rollout_samples(
                scenario=scenario,
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
                animation_steps=8,
            )
            for scenario in scenarios
        }

        good_outputs: list[dict[str, torch.Tensor]] = []
        flat_outputs: list[dict[str, torch.Tensor]] = []
        for scenario in scenarios:
            for sample in probe_samples[scenario]:
                current_heat = sample.heatmap.clamp(1e-4, 1.0 - 1e-4)
                future_heat = sample.future_heatmap.clamp(1e-4, 1.0 - 1e-4)
                base_output = {
                    "count_logits": torch.zeros(1, 5),
                    "heatmap_logits": torch.logit(current_heat).unsqueeze(0),
                    "slot_logits": _state_to_logits(sample.slot_state).unsqueeze(0),
                    "future_count_logits": torch.zeros(1, sample.future_count.shape[0], 5),
                    "future_heatmap_logits": torch.logit(future_heat).unsqueeze(0),
                    "future_slot_logits": _state_to_logits(sample.future_slot_state).unsqueeze(0),
                    "motion_logits": torch.zeros(1, 3),
                }
                good_outputs.append(base_output)

                flat_future_slot = sample.future_slot_state.clone()
                first_frame = sample.future_slot_state[0].clone()
                flat_future_slot[:] = first_frame
                flat_future_slot[..., 0] = sample.future_slot_state[..., 0]
                flat_future_heat = sample.future_heatmap[0:1].repeat(sample.future_heatmap.shape[0], 1)
                flat_outputs.append(
                    {
                        "count_logits": torch.zeros(1, 5),
                        "heatmap_logits": torch.logit(current_heat).unsqueeze(0),
                        "slot_logits": _state_to_logits(sample.slot_state).unsqueeze(0),
                        "future_count_logits": torch.zeros(1, sample.future_count.shape[0], 5),
                        "future_heatmap_logits": torch.logit(
                            flat_future_heat.clamp(1e-4, 1.0 - 1e-4)
                        ).unsqueeze(0),
                        "future_slot_logits": _state_to_logits(flat_future_slot).unsqueeze(0),
                        "motion_logits": torch.zeros(1, 3),
                    }
                )

        good_metrics = evaluate_probe_suite(
            model=ReplayProbeModel(good_outputs),
            probe_samples=probe_samples,
            device=torch.device("cpu"),
        )
        flat_metrics = evaluate_probe_suite(
            model=ReplayProbeModel(flat_outputs),
            probe_samples=probe_samples,
            device=torch.device("cpu"),
        )

        self.assertGreater(
            good_metrics["probe/geometry_checkpoint_score"],
            flat_metrics["probe/geometry_checkpoint_score"],
        )
        self.assertGreaterEqual(
            flat_metrics["probe/checkpoint_score"],
            0.8,
        )
        self.assertLess(
            flat_metrics["probe/trend_from_slots_score"],
            good_metrics["probe/trend_from_slots_score"],
        )


if __name__ == "__main__":
    unittest.main()
