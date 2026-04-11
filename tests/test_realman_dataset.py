from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from uca8.data.realman_ring2_dataset import RealMANRing2Dataset


def write_location_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["filename", "angle(fake)", "distance", "ele"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RealMANDatasetTests(unittest.TestCase):
    def test_manifest_cache_and_stable_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ring2_8ch"
            moving_csv = Path(temp_dir) / "moving.csv"
            static_csv = Path(temp_dir) / "static.csv"
            manifest_path = Path(temp_dir) / "cache" / "realman_manifest.json"
            moving_rows: list[dict[str, str]] = []
            static_rows: list[dict[str, str]] = []
            file_specs = [
                ("SceneA", "moving", "SPK1", "VAL_M_SCNA_SPK1_0001"),
                ("SceneA", "moving", "SPK1", "VAL_M_SCNA_SPK1_0002"),
                ("SceneA", "moving", "SPK1", "VAL_M_SCNA_SPK1_0003"),
                ("SceneA", "moving", "SPK1", "VAL_M_SCNA_SPK1_0004"),
                ("SceneA", "static", "SPK1", "VAL_S_SCNA_SPK1_0001"),
                ("SceneA", "static", "SPK1", "VAL_S_SCNA_SPK1_0002"),
            ]
            for scene, motion, speaker, utterance_id in file_specs:
                speaker_dir = root / scene / motion / speaker
                speaker_dir.mkdir(parents=True, exist_ok=True)
                dp_path = speaker_dir / f"{utterance_id}.flac"
                dp_path.touch()
                for channel_id in range(9, 17):
                    (speaker_dir / f"{utterance_id}_CH{channel_id}.flac").touch()
                filename = (
                    f"val/ma_noisy_speech/{scene}/{motion}/{speaker}/{utterance_id}.flac"
                )
                row = {
                    "filename": filename,
                    "angle(fake)": "30.0",
                    "distance": "1.5",
                    "ele": "0.0",
                }
                if motion == "moving":
                    moving_rows.append(row)
                else:
                    static_rows.append(row)
            write_location_csv(moving_csv, moving_rows)
            write_location_csv(static_csv, static_rows)

            train_dataset = RealMANRing2Dataset(
                root_dir=root,
                moving_csv=moving_csv,
                static_csv=static_csv,
                split="train",
                val_ratio=0.25,
                split_seed=7,
                use_manifest_cache=True,
                manifest_path=manifest_path,
            )
            val_dataset = RealMANRing2Dataset(
                root_dir=root,
                moving_csv=moving_csv,
                static_csv=static_csv,
                split="val",
                val_ratio=0.25,
                split_seed=7,
                use_manifest_cache=True,
                manifest_path=manifest_path,
            )

            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(train_dataset) + len(val_dataset), len(file_specs))
            self.assertEqual(train_dataset.summary()["split"], "train")
            self.assertEqual(val_dataset.summary()["split"], "val")

            train_dataset_again = RealMANRing2Dataset(
                root_dir=root,
                moving_csv=moving_csv,
                static_csv=static_csv,
                split="train",
                val_ratio=0.25,
                split_seed=7,
                use_manifest_cache=True,
                manifest_path=manifest_path,
            )
            train_ids = [record.sample_id for record in train_dataset.records]
            train_ids_again = [record.sample_id for record in train_dataset_again.records]
            self.assertEqual(train_ids, train_ids_again)


if __name__ == "__main__":
    unittest.main()
