from __future__ import annotations

"""RealMAN 第二圈 8 麦克风数据集封装.

当前主实验使用 RealMAN 的 ring2 8ch 子阵列, 默认通道为 CH9-CH16.
本文件负责:

1. 从 RealMAN 目录和官方 CSV 中匹配音频样本;
2. 读取多通道麦克风音频和对应的 dp_speech;
3. 根据 CSV 中的 angle/distance/elevation 构造声源三维轨迹;
4. 使用 ``TrackTrendLabelBuilder`` 生成模型需要的 count/heatmap/slot 标签.

注意: RealMAN 当前定位任务主要是单声源, 因此标签只填充第 0 个 source slot,
其余 slot 保留给模型结构和后续多声源扩展.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
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
    """一个 RealMAN utterance 的文件和标注索引.

    ``dp_path`` 是干净/近讲参考语音, 用于估计 VAD;
    ``channel_paths`` 是选定麦克风通道的多通道观测音频;
    ``row`` 保存 CSV 中角度、距离等定位标注.
    """

    scene: str
    motion: str
    speaker: str
    utterance_id: str
    dp_path: Path
    channel_paths: list[Path]
    row: dict[str, str]

    @property
    def sample_id(self) -> str:
        """返回稳定样本 id, 用于 split、日志和评估输出."""
        return f"{self.scene}:{self.motion}:{self.utterance_id}"

    def to_manifest_entry(self, root_dir: Path) -> dict[str, Any]:
        """序列化成 manifest 缓存条目."""
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
        """从 manifest 缓存条目恢复 ``RealMANRecord``."""
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
    """解析 CSV 中可能是单值或逗号序列的标注单元格."""
    if "," in value:
        return np.asarray([float(part) for part in value.split(",") if part], dtype=np.float32)
    return np.asarray([float(value)], dtype=np.float32)


def _read_location_csv(path: Path) -> list[dict[str, str]]:
    """读取 RealMAN 位置标注 CSV."""
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_angle_column(row: dict[str, str]) -> str:
    """查找角度列名.

    RealMAN CSV 的角度列可能带后缀, 因此按 ``angle`` 前缀匹配.
    """
    for key in row:
        if key.startswith("angle"):
            return key
    raise KeyError("Could not find angle column in RealMAN CSV row.")


def _canonical_rel_key_from_csv_filename(filename: str) -> str:
    """把 CSV filename 规整成可和本地路径匹配的相对 key.

    官方 CSV 里 filename 可能包含 ``ma_noisy_speech`` / ``ma_speech`` /
    ``dp_speech`` 等前缀. 本地扫描时使用数据根目录下的相对路径, 所以这里统一
    截掉这些锚点之前的部分.
    """
    parts = PurePosixPath(filename.replace("\\", "/")).parts
    for anchor in ("ma_noisy_speech", "ma_speech", "dp_speech"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            rel_parts = parts[anchor_index + 1 :]
            if rel_parts:
                return "/".join(rel_parts)
    if len(parts) >= 4:
        return "/".join(parts[-4:])
    return "/".join(parts)


def _stable_hash(value: str, seed: int) -> str:
    """生成稳定 hash, 用于可复现 train/val split."""
    return hashlib.sha1(f"{seed}:{value}".encode()).hexdigest()


def _interpolate_sequence(values: np.ndarray, target_length: int) -> torch.Tensor:
    """把标注序列插值到音频帧数.

    静态样本通常只有一个标注值, 移动样本可能是一段序列. 统一插值后,
    每个 STFT/VAD 帧都有对应的 angle/distance/elevation.
    """
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
    """用 dp_speech 的帧级 RMS 估计活动状态."""
    frames = max(1, math.ceil(waveform.shape[-1] / hop_length))
    total = frames * hop_length
    padded = F.pad(waveform, (0, total - waveform.shape[-1]))
    rms = padded.reshape(frames, hop_length).pow(2.0).mean(dim=-1).sqrt()
    threshold = max(float(rms.max().item()) * 0.05, 1e-4)
    return (rms >= threshold).float()


def _pad_last(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    """把序列补到固定长度, 补充值使用最后一帧."""
    if sequence.shape[0] >= target_length:
        return sequence[:target_length]
    pad_length = target_length - sequence.shape[0]
    if sequence.shape[0] == 0:
        return torch.zeros(target_length, *sequence.shape[1:], dtype=sequence.dtype)
    pad_value = sequence[-1:].repeat(pad_length, *([1] * (sequence.ndim - 1)))
    return torch.cat([sequence, pad_value], dim=0)


class RealMANRing2Dataset(Dataset[dict[str, Any]]):
    """RealMAN ring2 8ch PyTorch Dataset.

    ``__getitem__`` 返回一个训练样本:
    - ``waveform``: 历史窗口多通道音频, ``[8, history_samples]``;
    - ``vad_history``: 历史窗口 VAD ratio, ``[history_frames, 1]``;
    - 当前帧标签: ``count``、``heatmap``、``slot_state``;
    - 未来窗口标签: ``future_count``、``future_heatmap``、``future_slot_state``.
    """

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
        """初始化 RealMAN ring2 数据集.

        ``channel_ids`` 默认取 CH9-CH16, 对应当前 8 麦克风子阵列.
        初始化时只扫描/缓存文件索引; 音频读取和标签窗口构造在 ``__getitem__`` 中完成.
        """
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
        """返回当前 split 下的样本数量."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取一个 RealMAN 样本并构造模型训练字典."""
        record = self.records[index]
        waveform, dp_waveform = self._load_record_audio(record)
        total_available_frames = max(1, math.ceil(waveform.shape[-1] / self.hop_length))

        # CSV 标注被插值到音频帧数; angle/elevation 使用弧度, distance 使用米.
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

        # RealMAN 当前样本是单声源, 因此只填第 0 个 source slot.
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

        # 取中间连续窗口用于当前训练. history 对应模型输入, future 只用于监督标签.
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
        """返回数据集概览, 用于日志和实验记录."""
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
        """优先读取 manifest 缓存, 不存在或失效时重新扫描文件系统."""
        if self.use_manifest_cache:
            manifest_records = self._read_manifest()
            if manifest_records is not None:
                return manifest_records
        records = self._scan_records()
        if self.use_manifest_cache:
            self._write_manifest(records)
        return records

    def _scan_records(self) -> list[RealMANRecord]:
        """扫描 RealMAN 目录并和 CSV 标注行匹配."""
        moving_rows = {
            _canonical_rel_key_from_csv_filename(row["filename"]): row
            for row in _read_location_csv(self.moving_csv)
        }
        static_rows = {
            _canonical_rel_key_from_csv_filename(row["filename"]): row
            for row in _read_location_csv(self.static_csv)
        }
        records: list[RealMANRecord] = []
        for dp_path in sorted(
            path for path in self.root_dir.rglob("*.flac") if "_CH" not in path.name
        ):
            rel_path = dp_path.relative_to(self.root_dir)
            rel = rel_path.as_posix()
            motion = rel_path.parts[1] if len(rel_path.parts) >= 2 else ""
            row = moving_rows.get(rel) if motion == "moving" else static_rows.get(rel)
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
            # 缺任意一个目标通道就跳过, 保证模型输入通道数固定.
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
        """读取并校验 manifest 缓存."""
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
        """写入 manifest 缓存, 加速下一次启动."""
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
        """按 scene/motion 分组做稳定 train/val split."""
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
        """返回 CSV 文件签名, 用于判断 manifest 是否过期."""
        return {
            "path": str(path.resolve()),
            "mtime_ns": path.stat().st_mtime_ns,
        }

    def _load_record_audio(self, record: RealMANRecord) -> tuple[torch.Tensor, torch.Tensor]:
        """读取选定麦克风通道和 dp_speech, 并裁成相同长度."""
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
        """从 CSV 单元格构造逐帧轨迹."""
        sequence = _interpolate_sequence(_parse_sequence_cell(cell), target_length)
        if degrees:
            sequence = torch.deg2rad(sequence)
        return sequence
