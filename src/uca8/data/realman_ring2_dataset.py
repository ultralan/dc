from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.utils.audio_io import load_audio_file

MANIFEST_VERSION = 1


@dataclass(slots=True)
class RealMANRecord:
    scene: str
    motion: str
    speaker: str
    utterance_id: str
    dp_path: Path
    channel_paths: list[Path]
    row: dict[str, str]

    @property
    def sample_id(self) -> str:
        return f"{self.scene}:{self.motion}:{self.utterance_id}"

    def to_manifest_entry(self, root_dir: Path) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "motion": self.motion,
            "speaker": self.speaker,
            "utterance_id": self.utterance_id,
            "dp_rel": self.dp_path.relative_to(root_dir).as_posix(),
            "channel_rels": [path.relative_to(root_dir).as_posix() for path in self.channel_paths],
            "row": self.row,
        }

    @classmethod
    def from_manifest_entry(cls, root_dir: Path, entry: dict[str, Any]) -> RealMANRecord:
        return cls(
            scene=str(entry["scene"]),
            motion=str(entry["motion"]),
            speaker=str(entry["speaker"]),
            utterance_id=str(entry["utterance_id"]),
            dp_path=root_dir / str(entry["dp_rel"]),
            channel_paths=[root_dir / str(path) for path in entry["channel_rels"]],
            row={str(key): str(value) for key, value in dict(entry["row"]).items()},
        )


def _parse_sequence_cell(value: str) -> np.ndarray:
    if "," in value:
        return np.asarray([float(part) for part in value.split(",") if part], dtype=np.float32)
    return np.asarray([float(value)], dtype=np.float32)


def _read_location_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_angle_column(row: dict[str, str]) -> str:
    for key in row:
        if key.startswith("angle"):
            return key
    raise KeyError("Could not find angle column in RealMAN CSV row.")


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode()).hexdigest()


def _interpolate_sequence(values: np.ndarray, target_length: int) -> torch.Tensor:
    values = np.atleast_1d(values).astype(np.float32)
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if values.shape[0] == target_length:
        return torch.from_numpy(values)
    if values.shape[0] == 1:
        return torch.full((target_length,), float(values[0]), dtype=torch.float32)
    source_grid = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return torch.from_numpy(np.interp(target_grid, source_grid, values).astype(np.float32))


def _rms_vad(waveform: torch.Tensor, hop_length: int) -> torch.Tensor:
    frames = max(1, math.ceil(waveform.shape[-1] / hop_length))
    total = frames * hop_length
    padded = F.pad(waveform, (0, total - waveform.shape[-1]))
    rms = padded.reshape(frames, hop_length).pow(2.0).mean(dim=-1).sqrt()
    threshold = max(float(rms.max().item()) * 0.05, 1e-4)
    return (rms >= threshold).float()


def _pad_last(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[0] >= target_length:
        return sequence[:target_length]
    pad_length = target_length - sequence.shape[0]
    if sequence.shape[0] == 0:
        return torch.zeros(target_length, *sequence.shape[1:], dtype=sequence.dtype)
    pad_value = sequence[-1:].repeat(pad_length, *([1] * (sequence.ndim - 1)))
    return torch.cat([sequence, pad_value], dim=0)


class RealMANRing2Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        moving_csv: str | Path,
        static_csv: str | Path,
        channel_ids: Iterable[int] = (9, 10, 11, 12, 13, 14, 15, 16),
        model_sample_rate: int = 16000,
        history_frames: int = 128,
        future_frames: int = 32,
        hop_length: int = 160,
        max_sources: int = 4,
        num_heatmap_bins: int = 72,
        split: str = "all",
        val_ratio: float = 0.15,
        split_seed: int = 42,
        use_manifest_cache: bool = True,
        manifest_path: str | Path | None = None,
        audio_cache_dir: str | Path | None = None,
        max_items: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.moving_csv = Path(moving_csv)
        self.static_csv = Path(static_csv)
        self.model_sample_rate = model_sample_rate
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.hop_length = hop_length
        self.history_samples = history_frames * hop_length
        self.max_sources = max_sources
        self.channel_ids = tuple(channel_ids)
        self.split = split
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.use_manifest_cache = use_manifest_cache
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.root_dir / ".cache" / "realman_ring2_manifest.json"
        )
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir is not None else None
        self.label_builder = TrackTrendLabelBuilder(
            num_heatmap_bins=num_heatmap_bins,
            max_sources=max_sources,
            frame_hop_seconds=hop_length / float(model_sample_rate),
        )
        base_records = self._load_or_build_records()
        split_records = self._apply_split(base_records)
        self.records = split_records if max_items is None else split_records[:max_items]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, dp_waveform = self._load_record_audio(record)
        total_available_frames = max(1, math.ceil(waveform.shape[-1] / self.hop_length))
        azimuth = self._build_trajectory(
            record.row[_find_angle_column(record.row)],
            total_available_frames,
            degrees=True,
        )
        distance = self._build_trajectory(
            record.row["distance"],
            total_available_frames,
            degrees=False,
        )
        elevation = self._build_trajectory(
            record.row["ele"],
            total_available_frames,
            degrees=True,
        )
        activity = _rms_vad(dp_waveform[0], self.hop_length)
        source_positions = torch.zeros(
            total_available_frames,
            self.max_sources,
            3,
            dtype=torch.float32,
        )
        source_positions[:, 0, 0] = distance * torch.cos(elevation) * torch.cos(azimuth)
        source_positions[:, 0, 1] = distance * torch.cos(elevation) * torch.sin(azimuth)
        source_positions[:, 0, 2] = distance * torch.sin(elevation)
        source_activity = torch.zeros(total_available_frames, self.max_sources, dtype=torch.float32)
        source_activity[:, 0] = activity
        frame_start = max(0, (total_available_frames - self.total_frames) // 2)
        frame_end = min(total_available_frames, frame_start + self.total_frames)
        sample_start = frame_start * self.hop_length
        history_waveform = waveform[:, sample_start : sample_start + self.history_samples]
        history_waveform = F.pad(
            history_waveform,
            (0, self.history_samples - history_waveform.shape[-1]),
        )
        crop_positions = _pad_last(source_positions[frame_start:frame_end], self.total_frames)
        crop_activity = _pad_last(source_activity[frame_start:frame_end], self.total_frames)
        array_positions = torch.zeros(self.total_frames, 3, dtype=torch.float32)
        targets = self.label_builder.build_sequence_targets(
            source_positions=crop_positions,
            array_positions=array_positions,
            source_activity=crop_activity,
        )
        current_idx = self.history_frames - 1
        future_slice = slice(self.history_frames, self.total_frames)
        future_slot_state = targets.slot_state[future_slice]
        return {
            "waveform": history_waveform,
            "vad_history": targets.vad_ratio[: self.history_frames].unsqueeze(-1),
            "count": targets.count[current_idx],
            "heatmap": targets.heatmap[current_idx],
            "slot_state": targets.slot_state[current_idx],
            "future_count": targets.count[future_slice],
            "future_heatmap": targets.heatmap[future_slice],
            "future_slot_state": future_slot_state,
            "trend_class": self.label_builder.classify_future_motion(future_slot_state),
            "sample_id": record.sample_id,
        }

    def summary(self) -> dict[str, Any]:
        scenes = Counter(record.scene for record in self.records)
        motions = Counter(record.motion for record in self.records)
        return {
            "num_records": len(self.records),
            "split": self.split,
            "audio_cache_enabled": self.audio_cache_dir is not None,
            "audio_cache_dir": (
                str(self.audio_cache_dir) if self.audio_cache_dir is not None else None
            ),
            "scenes": dict(scenes),
            "motions": dict(motions),
        }

    def _load_or_build_records(self) -> list[RealMANRecord]:
        if self.use_manifest_cache:
            manifest_records = self._read_manifest()
            if manifest_records is not None:
                return manifest_records
        records = self._scan_records()
        if self.use_manifest_cache:
            self._write_manifest(records)
        return records

    def _scan_records(self) -> list[RealMANRecord]:
        moving_rows = {row["filename"]: row for row in _read_location_csv(self.moving_csv)}
        static_rows = {row["filename"]: row for row in _read_location_csv(self.static_csv)}
        records: list[RealMANRecord] = []
        for dp_path in sorted(
            path for path in self.root_dir.rglob("*.flac") if "_CH" not in path.name
        ):
            rel_path = dp_path.relative_to(self.root_dir)
            rel = rel_path.as_posix()
            key = f"val/ma_noisy_speech/{rel}"
            motion = "moving" if "/moving/" in key else "static"
            row = moving_rows.get(key) if motion == "moving" else static_rows.get(key)
            if row is None:
                continue
            if len(rel_path.parts) < 4:
                continue
            utterance_id = dp_path.stem
            scene = rel_path.parts[0]
            speaker = rel_path.parts[2]
            channel_paths = [
                dp_path.with_name(f"{utterance_id}_CH{channel_id}.flac")
                for channel_id in self.channel_ids
            ]
            if not all(path.exists() for path in channel_paths):
                continue
            records.append(
                RealMANRecord(
                    scene=scene,
                    motion=motion,
                    speaker=speaker,
                    utterance_id=utterance_id,
                    dp_path=dp_path,
                    channel_paths=channel_paths,
                    row=row,
                )
            )
        if not records:
            raise FileNotFoundError(f"No RealMAN records found under {self.root_dir}.")
        return records

    def _read_manifest(self) -> list[RealMANRecord] | None:
        if not self.manifest_path.exists():
            return None
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("version") != MANIFEST_VERSION:
            return None
        if metadata.get("root_dir") != str(self.root_dir.resolve()):
            return None
        if metadata.get("channel_ids") != list(self.channel_ids):
            return None
        if metadata.get("moving_csv") != self._csv_signature(self.moving_csv):
            return None
        if metadata.get("static_csv") != self._csv_signature(self.static_csv):
            return None
        return [
            RealMANRecord.from_manifest_entry(self.root_dir, entry)
            for entry in payload.get("records", [])
        ]

    def _write_manifest(self, records: list[RealMANRecord]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "version": MANIFEST_VERSION,
                "root_dir": str(self.root_dir.resolve()),
                "channel_ids": list(self.channel_ids),
                "moving_csv": self._csv_signature(self.moving_csv),
                "static_csv": self._csv_signature(self.static_csv),
            },
            "records": [record.to_manifest_entry(self.root_dir) for record in records],
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _apply_split(self, records: list[RealMANRecord]) -> list[RealMANRecord]:
        if self.split == "all":
            return records
        if self.split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1) for train/val splitting.")
        grouped: dict[tuple[str, str], list[RealMANRecord]] = {}
        for record in records:
            grouped.setdefault((record.scene, record.motion), []).append(record)
        selected: list[RealMANRecord] = []
        for group_records in grouped.values():
            ordered = sorted(
                group_records,
                key=lambda record: _stable_hash(record.sample_id, self.split_seed),
            )
            if len(ordered) < 2:
                val_count = 0
            else:
                val_count = int(round(len(ordered) * self.val_ratio))
                val_count = min(max(val_count, 1), len(ordered) - 1)
            if self.split == "val":
                selected.extend(ordered[:val_count])
            else:
                selected.extend(ordered[val_count:])
        return sorted(selected, key=lambda record: record.sample_id)

    def _csv_signature(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "mtime_ns": path.stat().st_mtime_ns,
        }

    def _load_record_audio(self, record: RealMANRecord) -> tuple[torch.Tensor, torch.Tensor]:
        channels: list[torch.Tensor] = []
        target_length: int | None = None
        for path in record.channel_paths:
            waveform, sample_rate = load_audio_file(
                path,
                target_sample_rate=self.model_sample_rate,
                cache_dir=self.audio_cache_dir,
            )
            mono = waveform[0]
            target_length = (
                mono.shape[-1]
                if target_length is None
                else min(target_length, mono.shape[-1])
            )
            channels.append(mono)
        assert target_length is not None
        stacked = torch.stack([channel[:target_length] for channel in channels], dim=0)
        dp_waveform, dp_rate = load_audio_file(
            record.dp_path,
            target_sample_rate=self.model_sample_rate,
            cache_dir=self.audio_cache_dir,
        )
        dp_waveform = dp_waveform[:, :target_length]
        return stacked.to(dtype=torch.float32), dp_waveform.to(dtype=torch.float32)

    def _build_trajectory(self, cell: str, target_length: int, *, degrees: bool) -> torch.Tensor:
        sequence = _interpolate_sequence(_parse_sequence_cell(cell), target_length)
        if degrees:
            sequence = torch.deg2rad(sequence)
        return sequence
