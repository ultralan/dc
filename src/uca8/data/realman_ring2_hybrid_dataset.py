from __future__ import annotations

"""RealMAN + 解析合成 curriculum 的混合数据集.

RealMAN 主数据集中以单声源为主, 对多声源、进入/离开、反向运动等扩展任务覆盖不足.
本文件在真实 RealMAN 样本之外, 使用真实 dp_speech 作为源音频, 再用远场阵列模型
渲染可控的多声源场景. 这样既保留真实语音内容, 又能补足跟踪扩展所需的标签形态.
"""

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.data.realman_ring2_dataset import RealMANRing2Dataset
from uca8.geometry.uca8 import make_uniform_circular_array, wrap_angle
from uca8.sim import render_farfield_history_waveform
from uca8.utils.audio_io import load_audio_file


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    """用指定 generator 采样均匀分布浮点数, 保证样本可复现."""
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    """用指定 generator 采样整数, 区间为 ``[low, high)``."""
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _choice_sign(generator: torch.Generator) -> float:
    """随机返回 -1 或 1, 用于决定运动方向."""
    return -1.0 if bool(torch.randint(0, 2, (1,), generator=generator).item()) else 1.0


def _linspace_rad(start_deg: float, stop_deg: float, steps: int) -> torch.Tensor:
    """按角度端点生成弧度序列."""
    if steps <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if steps == 1:
        return torch.tensor([math.radians(start_deg)], dtype=torch.float32)
    return torch.linspace(math.radians(start_deg), math.radians(stop_deg), steps=steps)


class RealMANRing2HybridDataset(Dataset[dict[str, Any]]):
    """混合真实 RealMAN 样本和解析合成 curriculum 样本.

    前 ``len(real_dataset)`` 个 index 直接返回真实 RealMAN 样本;
    后续 index 返回合成 curriculum 样本. 合成样本仍使用真实 dp_speech 作为源,
    只是方位轨迹和多声源活动由本类生成.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        moving_csv: str | Path,
        static_csv: str | Path,
        channel_ids: tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15, 16),
        model_sample_rate: int = 16000,
        history_frames: int = 128,
        future_frames: int = 32,
        hop_length: int = 160,
        win_length: int = 400,
        max_sources: int = 4,
        num_heatmap_bins: int = 72,
        num_input_channels: int = 8,
        array_radius_m: float = 0.06,
        sound_speed: float = 343.0,
        split: str = "train",
        val_ratio: float = 0.15,
        split_seed: int = 42,
        use_manifest_cache: bool = True,
        manifest_path: str | Path | None = None,
        audio_cache_dir: str | Path | None = None,
        max_items: int | None = None,
        curriculum_ratio: float = 1.0,
        curriculum_size: int | None = None,
        curriculum_seed: int = 42,
        curriculum_rollout_steps: int = 1,
        curriculum_mode_weights: dict[str, float] | None = None,
    ) -> None:
        """初始化混合数据集.

        前半部分复用 ``RealMANRing2Dataset`` 的真实样本;
        curriculum 部分根据模式权重动态合成, 不预生成到磁盘.
        """
        self.real_dataset = RealMANRing2Dataset(
            root_dir=root_dir,
            moving_csv=moving_csv,
            static_csv=static_csv,
            channel_ids=channel_ids,
            model_sample_rate=model_sample_rate,
            history_frames=history_frames,
            future_frames=future_frames,
            hop_length=hop_length,
            max_sources=max_sources,
            num_heatmap_bins=num_heatmap_bins,
            split=split,
            val_ratio=val_ratio,
            split_seed=split_seed,
            use_manifest_cache=use_manifest_cache,
            manifest_path=manifest_path,
            audio_cache_dir=audio_cache_dir,
            max_items=max_items,
        )
        self.model_sample_rate = model_sample_rate
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_sources = max_sources
        self.num_heatmap_bins = num_heatmap_bins
        self.num_input_channels = num_input_channels
        self.array_radius_m = array_radius_m
        self.sound_speed = sound_speed
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir is not None else None
        self.curriculum_seed = int(curriculum_seed)
        default_curriculum_size = max(
            1,
            int(round(len(self.real_dataset) * float(curriculum_ratio))),
        )
        self.curriculum_size = (
            int(curriculum_size) if curriculum_size is not None else default_curriculum_size
        )
        self.curriculum_rollout_steps = max(1, int(curriculum_rollout_steps))
        self.curriculum_scene_frames = self.total_frames + self.curriculum_rollout_steps - 1
        self.history_samples = history_frames * hop_length
        self.label_builder = TrackTrendLabelBuilder(
            num_heatmap_bins=num_heatmap_bins,
            max_sources=max_sources,
            frame_hop_seconds=hop_length / float(model_sample_rate),
        )
        self.mic_positions = make_uniform_circular_array(
            num_mics=num_input_channels,
            radius=array_radius_m,
        )
        self.curriculum_mode_names = (
            "single_arc",
            "single_reverse",
            "dual_overlap",
            "dual_enter",
            "dual_leave",
        )
        weights = curriculum_mode_weights or {
            "single_arc": 0.15,
            "single_reverse": 0.15,
            "dual_overlap": 0.35,
            "dual_enter": 0.15,
            "dual_leave": 0.20,
        }
        self.curriculum_mode_probs = torch.tensor(
            [float(weights.get(name, 0.0)) for name in self.curriculum_mode_names],
            dtype=torch.float32,
        )
        self.curriculum_mode_probs = (
            self.curriculum_mode_probs / self.curriculum_mode_probs.sum().clamp_min(1e-6)
        )
        self.curriculum_records = [
            record for record in self.real_dataset.records if record.motion == "moving"
        ]
        if not self.curriculum_records:
            self.curriculum_records = list(self.real_dataset.records)
        if not self.curriculum_records:
            raise FileNotFoundError("Hybrid dataset requires at least one RealMAN record.")

    def __len__(self) -> int:
        """返回真实样本数 + curriculum 样本数."""
        return len(self.real_dataset) + self.curriculum_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按 index 选择真实样本或 curriculum 样本."""
        if index < len(self.real_dataset):
            return self.real_dataset[index]
        return self._build_curriculum_sample(index - len(self.real_dataset))

    def summary(self) -> dict[str, Any]:
        """返回混合数据集概览, 包括 curriculum 模式权重."""
        real_summary = self.real_dataset.summary()
        return {
            "dataset_kind": "realman_ring2_hybrid",
            "num_records": len(self),
            "real_records": len(self.real_dataset),
            "curriculum_records": self.curriculum_size,
            "split": real_summary["split"],
            "audio_cache_enabled": real_summary["audio_cache_enabled"],
            "audio_cache_dir": real_summary["audio_cache_dir"],
            "curriculum_rollout_steps": self.curriculum_rollout_steps,
            "scenes": real_summary["scenes"],
            "motions": real_summary["motions"],
            "curriculum_modes": {
                name: float(prob)
                for name, prob in zip(
                    self.curriculum_mode_names,
                    self.curriculum_mode_probs.tolist(),
                    strict=True,
                )
            },
        }

    def _build_curriculum_sample(self, index: int) -> dict[str, Any]:
        """构造一个解析合成训练样本.

        流程:
        1. 按权重选择场景模式;
        2. 采样声源轨迹和活动状态;
        3. 从 RealMAN dp_speech 中裁真实源音频;
        4. 用远场模型渲染 8 通道阵列观测;
        5. 构造和真实数据一致的标签字典.
        """
        generator = torch.Generator().manual_seed(self.curriculum_seed + index)
        mode_index = int(
            torch.multinomial(
                self.curriculum_mode_probs,
                num_samples=1,
                generator=generator,
            ).item()
        )
        mode = self.curriculum_mode_names[mode_index]
        source_positions, source_activity = self._sample_curriculum_scene(
            mode=mode,
            generator=generator,
        )
        source_count = int((source_activity.sum(dim=0) > 0.0).sum().item())

        # 使用真实 dp_speech 作为源内容, 只合成空间传播和轨迹.
        mono_waveforms = [
            self._sample_source_audio(
                generator,
                required_samples=self.curriculum_scene_frames * self.hop_length + self.win_length,
            )
            for _ in range(source_count)
        ]
        theta = torch.atan2(source_positions[..., 1], source_positions[..., 0])
        full_waveform = render_farfield_history_waveform(
            mono_waveforms=mono_waveforms,
            theta_history=theta[:, :source_count],
            source_activity=source_activity[:, :source_count],
            mic_positions=self.mic_positions,
            sample_rate=self.model_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sound_speed=self.sound_speed,
            source_gains=[_uniform(generator, 0.75, 1.1) for _ in range(source_count)],
        )
        targets = self.label_builder.build_sequence_targets(
            source_positions=source_positions,
            array_positions=torch.zeros(self.curriculum_scene_frames, 3, dtype=torch.float32),
            source_activity=source_activity,
        )
        window_start = _randint(generator, 0, self.curriculum_rollout_steps)
        sample_start = window_start * self.hop_length
        waveform = full_waveform[:, sample_start : sample_start + self.history_samples]
        waveform = torch.nn.functional.pad(
            waveform,
            (0, self.history_samples - waveform.shape[-1]),
        )
        current_idx = window_start + self.history_frames - 1
        future_slice = slice(current_idx + 1, current_idx + 1 + self.future_frames)
        future_slot_state = targets.slot_state[future_slice]
        return {
            "waveform": waveform,
            "vad_history": targets.vad_ratio[
                window_start : window_start + self.history_frames
            ].unsqueeze(-1),
            "count": targets.count[current_idx],
            "heatmap": targets.heatmap[current_idx],
            "slot_state": targets.slot_state[current_idx],
            "future_count": targets.count[future_slice],
            "future_heatmap": targets.heatmap[future_slice],
            "future_slot_state": future_slot_state,
            "trend_class": self.label_builder.classify_future_motion(future_slot_state),
            "sample_id": f"curriculum:{mode}:{index}:offset{window_start:02d}",
        }

    def _sample_source_audio(
        self,
        generator: torch.Generator,
        *,
        required_samples: int,
    ) -> torch.Tensor:
        """从真实 RealMAN dp_speech 中随机裁一段单声道源音频."""
        record = self.curriculum_records[_randint(generator, 0, len(self.curriculum_records))]
        waveform, _ = load_audio_file(
            record.dp_path,
            target_sample_rate=self.model_sample_rate,
            cache_dir=self.audio_cache_dir,
        )
        mono = waveform[0].to(dtype=torch.float32).flatten()
        peak = mono.abs().amax().clamp_min(1e-4)
        mono = mono / peak
        if mono.shape[-1] > required_samples:
            start = _randint(generator, 0, mono.shape[-1] - required_samples + 1)
            mono = mono[start : start + required_samples]
        elif mono.shape[-1] < required_samples:
            mono = torch.nn.functional.pad(mono, (0, required_samples - mono.shape[-1]))
        return mono

    def _sample_curriculum_scene(
        self,
        *,
        mode: str,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """采样一个 curriculum 场景的声源轨迹和活动矩阵.

        返回:
            ``source_positions``: ``[T, max_sources, 3]``.
            ``source_activity``: ``[T, max_sources]``.
        """
        source_positions = torch.zeros(
            self.curriculum_scene_frames,
            self.max_sources,
            3,
            dtype=torch.float32,
        )
        source_activity = torch.zeros(
            self.curriculum_scene_frames,
            self.max_sources,
            dtype=torch.float32,
        )
        primary_theta = self._sample_track(generator, reverse=mode == "single_reverse")
        self._fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=primary_theta,
            distance_m=_uniform(generator, 0.9, 1.4),
            active=torch.ones(self.curriculum_scene_frames, dtype=torch.float32),
        )
        if mode in {"dual_overlap", "dual_enter", "dual_leave"}:
            secondary_theta = self._sample_dual_partner(
                primary_theta=primary_theta,
                generator=generator,
            )
            secondary_activity = torch.ones(self.curriculum_scene_frames, dtype=torch.float32)
            if mode == "dual_enter":
                # 第二个声源在未来窗口中进入.
                secondary_activity = torch.zeros(self.curriculum_scene_frames, dtype=torch.float32)
                enter_after = self.history_frames + _randint(
                    generator,
                    max(1, self.future_frames // 6),
                    max(self.future_frames // 2, 2),
                )
                secondary_activity[enter_after:] = 1.0
            if mode == "dual_leave":
                # 第二个声源在未来窗口中离开.
                leave_after = self.history_frames + _randint(
                    generator,
                    max(2, self.future_frames // 4),
                    max(self.future_frames // 2, 3),
                )
                secondary_activity[leave_after:] = 0.0
            self._fill_source(
                source_positions=source_positions,
                source_activity=source_activity,
                slot_idx=1,
                theta=secondary_theta,
                distance_m=_uniform(generator, 1.0, 1.6),
                active=secondary_activity,
            )
        return source_positions, source_activity

    def _sample_track(self, generator: torch.Generator, *, reverse: bool) -> torch.Tensor:
        """采样主声源轨迹.

        ``reverse=True`` 时, 历史段到边界后在未来段反向运动, 用于测试趋势预测.
        """
        boundary_deg = _uniform(generator, -120.0, 120.0)
        direction = _choice_sign(generator)
        hist_span = _uniform(generator, 18.0, 70.0)
        future_span = _uniform(generator, 18.0, 75.0)
        start_deg = boundary_deg - direction * hist_span
        end_deg = (
            boundary_deg - direction * future_span
            if reverse
            else boundary_deg + direction * future_span
        )
        history = _linspace_rad(start_deg, boundary_deg, self.history_frames)
        future = _linspace_rad(
            boundary_deg,
            end_deg,
            self.curriculum_scene_frames - self.history_frames + 1,
        )[1:]
        return torch.cat([history, future], dim=0)

    def _sample_dual_partner(
        self,
        *,
        primary_theta: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """基于主声源边界位置采样第二个声源轨迹."""
        primary_boundary_deg = float(torch.rad2deg(primary_theta[self.history_frames - 1]).item())
        separation = _uniform(generator, 55.0, 120.0) * _choice_sign(generator)
        secondary_boundary_deg = primary_boundary_deg + separation
        secondary_boundary_deg = float(
            torch.rad2deg(wrap_angle(torch.tensor(math.radians(secondary_boundary_deg)))).item()
        )
        approach_direction = -1.0 if separation > 0.0 else 1.0
        hist_span = _uniform(generator, 20.0, 65.0)
        future_span = _uniform(generator, 20.0, 65.0)
        start_deg = secondary_boundary_deg - approach_direction * hist_span
        end_deg = secondary_boundary_deg + approach_direction * future_span
        history = _linspace_rad(start_deg, secondary_boundary_deg, self.history_frames)
        future = _linspace_rad(
            secondary_boundary_deg,
            end_deg,
            self.curriculum_scene_frames - self.history_frames + 1,
        )[1:]
        return torch.cat([history, future], dim=0)

    def _fill_source(
        self,
        *,
        source_positions: torch.Tensor,
        source_activity: torch.Tensor,
        slot_idx: int,
        theta: torch.Tensor,
        distance_m: float,
        active: torch.Tensor,
    ) -> None:
        """把极坐标轨迹写入 source_positions/source_activity 的指定 slot."""
        source_positions[:, slot_idx, 0] = distance_m * torch.cos(theta)
        source_positions[:, slot_idx, 1] = distance_m * torch.sin(theta)
        source_activity[:, slot_idx] = active.to(dtype=torch.float32)


__all__ = ["RealMANRing2HybridDataset"]
