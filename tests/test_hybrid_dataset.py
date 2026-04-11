from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from uca8.data.realman_ring2_hybrid_dataset import RealMANRing2HybridDataset


def write_location_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["filename", "angle(fake)", "distance", "ele"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fake_load_audio_file(*args: object, **kwargs: object) -> tuple[torch.Tensor, int]:
    return torch.randn(1, 48000), 16000


class RealMANHybridDatasetTests(unittest.TestCase):
    def test_curriculum_sample_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ring2_8ch"
            moving_csv = Path(temp_dir) / "moving.csv"
            static_csv = Path(temp_dir) / "static.csv"
            rows: list[dict[str, str]] = []
            for scene_idx in range(3):
                scene = f"Scene{scene_idx}"
                speaker_dir = root / scene / "moving" / "SPK1"
                speaker_dir.mkdir(parents=True, exist_ok=True)
                utterance_id = f"VAL_M_{scene}_SPK1_0001"
                dp_path = speaker_dir / f"{utterance_id}.flac"
                dp_path.touch()
                for channel_id in range(9, 17):
                    (speaker_dir / f"{utterance_id}_CH{channel_id}.flac").touch()
                rows.append(
                    {
                        "filename": f"val/ma_noisy_speech/{scene}/moving/SPK1/{utterance_id}.flac",
                        "angle(fake)": "10.0,20.0,30.0",
                        "distance": "1.2,1.2,1.2",
                        "ele": "0.0,0.0,0.0",
                    }
                )
            write_location_csv(moving_csv, rows)
            write_location_csv(static_csv, [])

            with patch(
                "uca8.data.realman_ring2_hybrid_dataset.load_audio_file",
                side_effect=fake_load_audio_file,
            ):
                dataset = RealMANRing2HybridDataset(
                    root_dir=root,
                    moving_csv=moving_csv,
                    static_csv=static_csv,
                    split="train",
                    val_ratio=0.2,
                    curriculum_size=4,
                    history_frames=64,
                    future_frames=16,
                    hop_length=160,
                    win_length=400,
                )
                sample = dataset[len(dataset.real_dataset)]

            self.assertEqual(tuple(sample["waveform"].shape), (8, 64 * 160))
            self.assertEqual(tuple(sample["vad_history"].shape), (64, 1))
            self.assertEqual(tuple(sample["future_count"].shape), (16,))
            self.assertEqual(tuple(sample["future_heatmap"].shape), (16, 72))
            self.assertIn("curriculum:", sample["sample_id"])

    def test_dual_enter_curriculum_transitions_from_one_to_two_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ring2_8ch"
            moving_csv = Path(temp_dir) / "moving.csv"
            static_csv = Path(temp_dir) / "static.csv"
            rows: list[dict[str, str]] = []
            speaker_dir = root / "SceneA" / "moving" / "SPK1"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            utterance_id = "VAL_M_SceneA_SPK1_0001"
            dp_path = speaker_dir / f"{utterance_id}.flac"
            dp_path.touch()
            for channel_id in range(9, 17):
                (speaker_dir / f"{utterance_id}_CH{channel_id}.flac").touch()
            rows.append(
                {
                    "filename": f"val/ma_noisy_speech/SceneA/moving/SPK1/{utterance_id}.flac",
                    "angle(fake)": "10.0,20.0,30.0",
                    "distance": "1.2,1.2,1.2",
                    "ele": "0.0,0.0,0.0",
                }
            )
            write_location_csv(moving_csv, rows)
            write_location_csv(static_csv, [])

            with patch(
                "uca8.data.realman_ring2_hybrid_dataset.load_audio_file",
                side_effect=fake_load_audio_file,
            ):
                dataset = RealMANRing2HybridDataset(
                    root_dir=root,
                    moving_csv=moving_csv,
                    static_csv=static_csv,
                    split="train",
                    val_ratio=0.2,
                    curriculum_size=1,
                    history_frames=64,
                    future_frames=16,
                    hop_length=160,
                    win_length=400,
                    curriculum_mode_weights={
                        "single_arc": 0.0,
                        "single_reverse": 0.0,
                        "dual_overlap": 0.0,
                        "dual_enter": 1.0,
                        "dual_leave": 0.0,
                    },
                )
                sample = dataset[len(dataset.real_dataset)]

            self.assertEqual(int(sample["count"].item()), 1)
            self.assertEqual(int(sample["future_count"][0].item()), 1)
            self.assertEqual(int(sample["future_count"][-1].item()), 2)
            self.assertIn("curriculum:dual_enter", sample["sample_id"])

    def test_dual_leave_rollout_exposes_post_transition_current_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ring2_8ch"
            moving_csv = Path(temp_dir) / "moving.csv"
            static_csv = Path(temp_dir) / "static.csv"
            rows: list[dict[str, str]] = []
            speaker_dir = root / "SceneA" / "moving" / "SPK1"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            utterance_id = "VAL_M_SceneA_SPK1_0001"
            dp_path = speaker_dir / f"{utterance_id}.flac"
            dp_path.touch()
            for channel_id in range(9, 17):
                (speaker_dir / f"{utterance_id}_CH{channel_id}.flac").touch()
            rows.append(
                {
                    "filename": f"val/ma_noisy_speech/SceneA/moving/SPK1/{utterance_id}.flac",
                    "angle(fake)": "10.0,20.0,30.0",
                    "distance": "1.2,1.2,1.2",
                    "ele": "0.0,0.0,0.0",
                }
            )
            write_location_csv(moving_csv, rows)
            write_location_csv(static_csv, [])

            with patch(
                "uca8.data.realman_ring2_hybrid_dataset.load_audio_file",
                side_effect=fake_load_audio_file,
            ):
                dataset = RealMANRing2HybridDataset(
                    root_dir=root,
                    moving_csv=moving_csv,
                    static_csv=static_csv,
                    split="train",
                    val_ratio=0.2,
                    curriculum_size=24,
                    history_frames=64,
                    future_frames=16,
                    hop_length=160,
                    win_length=400,
                    curriculum_rollout_steps=24,
                    curriculum_mode_weights={
                        "single_arc": 0.0,
                        "single_reverse": 0.0,
                        "dual_overlap": 0.0,
                        "dual_enter": 0.0,
                        "dual_leave": 1.0,
                    },
                )
                counts = [
                    int(dataset[len(dataset.real_dataset) + idx]["count"].item())
                    for idx in range(dataset.curriculum_size)
                ]

            self.assertIn(1, counts)
            self.assertIn(2, counts)


if __name__ == "__main__":
    unittest.main()
