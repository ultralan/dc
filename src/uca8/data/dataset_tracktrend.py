from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.utils.audio_io import load_audio_file


@dataclass(slots=True)
class PackageRecord:
    package_dir: Path
    array_name: str
    audio_path: Path
    timestamps: torch.Tensor
    targets: dict[str, torch.Tensor]
    num_channels: int


@dataclass(slots=True)
class WindowRecord:
    package_index: int
    start_frame: int


def _read_tsv(path: Path) -> np.ndarray:
    return np.atleast_1d(
        np.genfromtxt(path, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    )


def _seconds_column(table: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(table["second"], dtype=np.float32))


def _xyz_columns(table: np.ndarray, prefix: str = "") -> torch.Tensor:
    return torch.stack(
        (
            torch.as_tensor(np.asarray(table[f"{prefix}x"], dtype=np.float32)),
            torch.as_tensor(np.asarray(table[f"{prefix}y"], dtype=np.float32)),
            torch.as_tensor(np.asarray(table[f"{prefix}z"], dtype=np.float32)),
        ),
        dim=-1,
    )


def _count_mics(table: np.ndarray) -> int:
    names = table.dtype.names or ()
    mic_x = [name for name in names if name.startswith("mic") and name.endswith("_x")]
    return len(mic_x)


def _read_vad_series(path: Path, target_length: int) -> torch.Tensor:
    values = np.loadtxt(path, skiprows=1, dtype=np.float32)
    values = np.atleast_1d(values)
    if values.shape[0] == target_length:
        return torch.from_numpy(values)
    if values.shape[0] == 0:
        return torch.zeros(target_length, dtype=torch.float32)
    indices = np.linspace(0, values.shape[0] - 1, num=target_length).round().astype(np.int64)
    return torch.from_numpy(values[indices])


class LocataLikeTrackTrendDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        history_frames: int,
        future_frames: int,
        window_stride_frames: int,
        model_sample_rate: int,
        num_input_channels: int,
        max_sources: int,
        frame_hop_seconds: float = 0.01,
        num_heatmap_bins: int = 72,
        audio_cache_dir: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_stride_frames = window_stride_frames
        self.model_sample_rate = model_sample_rate
        self.num_input_channels = num_input_channels
        self.max_sources = max_sources
        self.frame_hop_seconds = frame_hop_seconds
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir is not None else None
        self.label_builder = TrackTrendLabelBuilder(
            num_heatmap_bins=num_heatmap_bins,
            max_sources=max_sources,
            frame_hop_seconds=frame_hop_seconds,
        )
        self.packages = self._load_packages()
        self.windows = self._build_windows()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        package = self.packages[window.package_index]
        current_idx = window.start_frame + self.history_frames - 1
        future_slice = slice(current_idx + 1, current_idx + 1 + self.future_frames)
        waveform = self._load_waveform(package, window.start_frame, current_idx + 1)
        vad_history = package.targets["vad_ratio"][window.start_frame : current_idx + 1]
        vad_history = vad_history.unsqueeze(-1)
        future_slot_state = package.targets["slot_state"][future_slice]
        return {
            "waveform": waveform,
            "vad_history": vad_history,
            "count": package.targets["count"][current_idx],
            "heatmap": package.targets["heatmap"][current_idx],
            "slot_state": package.targets["slot_state"][current_idx],
            "future_count": package.targets["count"][future_slice],
            "future_heatmap": package.targets["heatmap"][future_slice],
            "future_slot_state": future_slot_state,
            "trend_class": self.label_builder.classify_future_motion(future_slot_state),
            "sample_id": f"{package.array_name}:{window.start_frame}",
        }

    def _load_packages(self) -> list[PackageRecord]:
        packages: list[PackageRecord] = []
        for package_dir in sorted(self.root_dir.glob("task*/recording*/*")):
            if not package_dir.is_dir():
                continue
            audio_paths = sorted(package_dir.glob("audio_array_*.wav"))
            if not audio_paths:
                continue
            array_name = package_dir.name
            array_table = _read_tsv(next(package_dir.glob("position_array_*.txt")))
            required_time = _read_tsv(package_dir / "required_time.txt")
            valid = torch.as_tensor(np.asarray(required_time["valid_flag"], dtype=np.float32))
            source_tracks = sorted(package_dir.glob("position_source_*.txt"))
            if not source_tracks:
                continue
            frame_count = min(len(array_table), len(required_time))
            num_channels = _count_mics(array_table)
            if num_channels < self.num_input_channels:
                continue
            timestamps = _seconds_column(array_table)[:frame_count]
            array_positions = _xyz_columns(array_table)[:frame_count]
            source_positions = torch.zeros(frame_count, self.max_sources, 3, dtype=torch.float32)
            source_activity = torch.zeros(frame_count, self.max_sources, dtype=torch.float32)
            for slot_idx, source_path in enumerate(source_tracks[: self.max_sources]):
                source_name = source_path.stem.replace("position_source_", "")
                source_table = _read_tsv(source_path)
                vad_path = package_dir / f"VAD_{array_name}_{source_name}.txt"
                if not vad_path.exists():
                    vad_path = package_dir / f"VAD_source_{source_name}.txt"
                limit = min(frame_count, len(source_table))
                source_positions[:limit, slot_idx] = _xyz_columns(source_table)[:limit]
                source_activity[:limit, slot_idx] = _read_vad_series(vad_path, limit)
            source_activity = source_activity * valid[:frame_count].unsqueeze(-1)
            targets = self.label_builder.build_sequence_targets(
                source_positions=source_positions,
                array_positions=array_positions,
                source_activity=source_activity,
            )
            packages.append(
                PackageRecord(
                    package_dir=package_dir,
                    array_name=array_name,
                    audio_path=audio_paths[0],
                    timestamps=timestamps,
                    targets={
                        "count": targets.count,
                        "vad_ratio": targets.vad_ratio,
                        "heatmap": targets.heatmap,
                        "slot_state": targets.slot_state,
                    },
                    num_channels=num_channels,
                )
            )
        if not packages:
            raise FileNotFoundError(f"No valid dataset packages found under {self.root_dir}.")
        return packages

    def _build_windows(self) -> list[WindowRecord]:
        windows: list[WindowRecord] = []
        usable = self.history_frames + self.future_frames
        for package_index, package in enumerate(self.packages):
            total_frames = int(package.targets["count"].shape[0])
            last_start = total_frames - usable
            for start in range(0, max(last_start + 1, 0), self.window_stride_frames):
                windows.append(WindowRecord(package_index=package_index, start_frame=start))
        return windows

    def _load_waveform(
        self,
        package: PackageRecord,
        start_frame: int,
        stop_frame: int,
    ) -> torch.Tensor:
        waveform, sample_rate = load_audio_file(
            package.audio_path,
            target_sample_rate=self.model_sample_rate,
            cache_dir=self.audio_cache_dir,
        )
        waveform = waveform[: self.num_input_channels]
        start_sec = float(package.timestamps[start_frame].item())
        stop_index = min(stop_frame - 1, len(package.timestamps) - 1)
        stop_sec = float(package.timestamps[stop_index].item())
        stop_sec += self.frame_hop_seconds
        relative_start = max(0.0, start_sec - float(package.timestamps[0].item()))
        relative_stop = max(
            relative_start + self.frame_hop_seconds,
            stop_sec - float(package.timestamps[0].item()),
        )
        sample_start = int(round(relative_start * sample_rate))
        sample_stop = int(round(relative_stop * sample_rate))
        sliced = waveform[:, sample_start:sample_stop]
        return sliced.to(dtype=torch.float32)


class SyntheticTrackTrendDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        size: int,
        num_channels: int,
        history_frames: int,
        future_frames: int,
        sample_rate: int,
        hop_length: int,
        max_sources: int,
        heatmap_bins: int,
    ) -> None:
        self.size = size
        self.num_channels = num_channels
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_sources = max_sources
        self.heatmap_bins = heatmap_bins
        self.waveform_samples = history_frames * hop_length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        waveform = torch.randn(self.num_channels, self.waveform_samples, generator=generator)
        vad_history = torch.rand(self.history_frames, 1, generator=generator)
        count = torch.randint(0, self.max_sources + 1, (1,), generator=generator).squeeze(0)
        heatmap = torch.rand(self.heatmap_bins, generator=generator)
        slot_state = torch.zeros(self.max_sources, 5)
        active = int(min(count.item(), self.max_sources))
        if active > 0:
            theta = torch.linspace(-1.0, 1.0, steps=active)
            slot_state[:active, 0] = 1.0
            slot_state[:active, 1] = torch.sin(theta)
            slot_state[:active, 2] = torch.cos(theta)
            slot_state[:active, 3] = torch.linspace(0.5, 1.5, steps=active)
            slot_state[:active, 4] = torch.linspace(-0.2, 0.2, steps=active)
        future_count = torch.randint(
            0, self.max_sources + 1, (self.future_frames,), generator=generator
        )
        future_heatmap = torch.rand(self.future_frames, self.heatmap_bins, generator=generator)
        future_slot_state = slot_state.unsqueeze(0).repeat(self.future_frames, 1, 1)
        future_slot_state[..., 4] += 0.02
        trend_class = torch.tensor(2, dtype=torch.long)
        return {
            "waveform": waveform,
            "vad_history": vad_history,
            "count": count.long(),
            "heatmap": heatmap,
            "slot_state": slot_state,
            "future_count": future_count.long(),
            "future_heatmap": future_heatmap,
            "future_slot_state": future_slot_state,
            "trend_class": trend_class,
        }
