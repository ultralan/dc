from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.metrics import slot_count_from_state
from uca8.sim.farfield import render_farfield_history_waveform
from uca8.utils.audio_io import load_audio_file

SCENARIO_CHOICES = (
    "static_front",
    "moving_arc",
    "moving_cross",
    "moving_reverse",
    "dual_cross",
    "source_enter",
    "source_leave",
)


@dataclass(slots=True)
class ProbeSample:
    sample_id: str
    waveform: torch.Tensor
    vad_history: torch.Tensor
    count: torch.Tensor
    heatmap: torch.Tensor
    slot_state: torch.Tensor
    future_count: torch.Tensor
    future_heatmap: torch.Tensor
    future_slot_state: torch.Tensor
    trend_class: torch.Tensor
    source_positions: torch.Tensor | None
    array_positions: torch.Tensor | None
    source_activity: torch.Tensor | None
    transition_start_index: int | None


def default_source_audio(root_dir: Path) -> Path:
    candidates = sorted(path for path in root_dir.rglob("*.flac") if "_CH" not in path.name)
    if not candidates:
        raise FileNotFoundError(f"No direct-path source audio found under {root_dir}.")
    return candidates[0]


def load_probe_mono_audio(
    source_audio: str | Path,
    *,
    sample_rate: int,
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    waveform, _ = load_audio_file(
        source_audio,
        target_sample_rate=sample_rate,
        cache_dir=cache_dir,
    )
    mono_waveform = waveform[0].to(dtype=torch.float32).flatten()
    peak = mono_waveform.abs().amax().clamp_min(1e-4)
    return mono_waveform / peak


def _linspace_deg(start_deg: float, stop_deg: float, steps: int) -> torch.Tensor:
    if steps <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if steps == 1:
        return torch.tensor([math.radians(start_deg)], dtype=torch.float32)
    return torch.linspace(math.radians(start_deg), math.radians(stop_deg), steps=steps)


def _fill_source(
    *,
    source_positions: torch.Tensor,
    source_activity: torch.Tensor,
    slot_idx: int,
    theta: torch.Tensor,
    distance_m: float,
    active: torch.Tensor | None = None,
) -> None:
    if theta.shape[0] != source_positions.shape[0]:
        raise ValueError("theta length must match total frames.")
    if active is None:
        active = torch.ones(theta.shape[0], dtype=torch.float32)
    source_positions[:, slot_idx, 0] = distance_m * torch.cos(theta)
    source_positions[:, slot_idx, 1] = distance_m * torch.sin(theta)
    source_activity[:, slot_idx] = active.to(dtype=torch.float32)


def transition_start_index(
    scenario: str,
    *,
    future_frames: int,
) -> int | None:
    if scenario not in {"source_enter", "source_leave"}:
        return None
    return max(future_frames // 4, 1)


def infer_transition_start_index(
    current_count: torch.Tensor,
    future_count: torch.Tensor,
) -> int | None:
    transitions = torch.nonzero(future_count != current_count, as_tuple=False).flatten()
    if transitions.numel() == 0:
        return None
    return int(transitions[0].item())


def build_scenario(
    *,
    scenario: str,
    history_frames: int,
    future_frames: int,
    max_sources: int = 4,
    distance_m: float = 1.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_frames = history_frames + future_frames
    array_positions = torch.zeros(total_frames, 3, dtype=torch.float32)
    source_positions = torch.zeros(total_frames, max_sources, 3, dtype=torch.float32)
    source_activity = torch.zeros(total_frames, max_sources, dtype=torch.float32)

    if scenario == "static_front":
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=torch.full((total_frames,), math.radians(30.0), dtype=torch.float32),
            distance_m=distance_m,
        )
    elif scenario == "moving_cross":
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=_linspace_deg(-80.0, 80.0, total_frames),
            distance_m=distance_m,
        )
    elif scenario == "moving_reverse":
        theta = torch.cat(
            [
                _linspace_deg(-110.0, -30.0, history_frames),
                _linspace_deg(-30.0, -100.0, future_frames),
            ]
        )
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=theta,
            distance_m=distance_m,
        )
    elif scenario == "dual_cross":
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=_linspace_deg(-110.0, 20.0, total_frames),
            distance_m=distance_m,
        )
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=1,
            theta=_linspace_deg(115.0, -25.0, total_frames),
            distance_m=distance_m + 0.35,
        )
    elif scenario == "source_enter":
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=torch.cat(
                [
                    _linspace_deg(-35.0, -20.0, history_frames),
                    _linspace_deg(-20.0, 15.0, future_frames),
                ]
            ),
            distance_m=distance_m,
        )
        enter_active = torch.zeros(total_frames, dtype=torch.float32)
        enter_frame = history_frames + transition_start_index(
            scenario,
            future_frames=future_frames,
        )
        enter_active[enter_frame:] = 1.0
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=1,
            theta=_linspace_deg(120.0, 70.0, total_frames),
            distance_m=distance_m + 0.25,
            active=enter_active,
        )
    elif scenario == "source_leave":
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=torch.cat(
                [
                    _linspace_deg(20.0, 35.0, history_frames),
                    _linspace_deg(35.0, 45.0, future_frames),
                ]
            ),
            distance_m=distance_m,
        )
        leave_active = torch.ones(total_frames, dtype=torch.float32)
        leave_frame = history_frames + transition_start_index(
            scenario,
            future_frames=future_frames,
        )
        leave_active[leave_frame:] = 0.0
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=1,
            theta=_linspace_deg(-120.0, -70.0, total_frames),
            distance_m=distance_m + 0.3,
            active=leave_active,
        )
    else:
        _fill_source(
            source_positions=source_positions,
            source_activity=source_activity,
            slot_idx=0,
            theta=torch.cat(
                [
                    _linspace_deg(-130.0, -90.0, history_frames),
                    _linspace_deg(-90.0, -40.0, future_frames),
                ]
            ),
            distance_m=distance_m,
        )
    return source_positions, array_positions, source_activity


def render_history_waveform(
    *,
    mono_waveform: torch.Tensor,
    theta_history: torch.Tensor,
    source_activity: torch.Tensor | None,
    mic_positions: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    sound_speed: float,
) -> torch.Tensor:
    num_sources = int(theta_history.shape[-1]) if theta_history.ndim > 1 else 1
    return render_farfield_history_waveform(
        mono_waveforms=[mono_waveform for _ in range(num_sources)],
        theta_history=theta_history,
        source_activity=source_activity,
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        sound_speed=sound_speed,
    )


def build_probe_sample(
    *,
    scenario: str,
    mono_waveform: torch.Tensor,
    mic_positions: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    sound_speed: float,
    history_frames: int,
    future_frames: int,
    num_heatmap_bins: int,
    max_sources: int,
) -> ProbeSample:
    source_positions, array_positions, source_activity = build_scenario(
        scenario=scenario,
        history_frames=history_frames,
        future_frames=future_frames,
        max_sources=max_sources,
    )
    theta = torch.atan2(source_positions[..., 1], source_positions[..., 0])
    waveform = render_history_waveform(
        mono_waveform=mono_waveform,
        theta_history=theta[:history_frames],
        source_activity=source_activity[:history_frames],
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        sound_speed=sound_speed,
    )
    label_builder = TrackTrendLabelBuilder(
        num_heatmap_bins=num_heatmap_bins,
        max_sources=max_sources,
        frame_hop_seconds=hop_length / float(sample_rate),
    )
    targets = label_builder.build_sequence_targets(
        source_positions=source_positions,
        array_positions=array_positions,
        source_activity=source_activity,
    )
    current_idx = history_frames - 1
    future_slice = slice(history_frames, history_frames + future_frames)
    future_slot_state = targets.slot_state[future_slice]
    current_count = targets.count[current_idx]
    future_count = targets.count[future_slice]
    return ProbeSample(
        sample_id=f"probe:{scenario}",
        waveform=waveform,
        vad_history=targets.vad_ratio[:history_frames].unsqueeze(-1),
        count=current_count,
        heatmap=targets.heatmap[current_idx],
        slot_state=targets.slot_state[current_idx],
        future_count=future_count,
        future_heatmap=targets.heatmap[future_slice],
        future_slot_state=future_slot_state,
        trend_class=label_builder.classify_future_motion(future_slot_state),
        source_positions=source_positions,
        array_positions=array_positions,
        source_activity=source_activity,
        transition_start_index=infer_transition_start_index(
            current_count,
            future_count,
        ),
    )


def build_probe_rollout_samples(
    *,
    scenario: str,
    mono_waveform: torch.Tensor,
    mic_positions: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    sound_speed: float,
    history_frames: int,
    future_frames: int,
    num_heatmap_bins: int,
    max_sources: int,
    animation_steps: int,
) -> list[ProbeSample]:
    total_frames = history_frames + future_frames + animation_steps - 1
    source_positions, array_positions, source_activity = build_scenario(
        scenario=scenario,
        history_frames=history_frames,
        future_frames=future_frames + animation_steps - 1,
        max_sources=max_sources,
    )
    theta = torch.atan2(source_positions[..., 1], source_positions[..., 0])
    full_waveform = render_history_waveform(
        mono_waveform=mono_waveform,
        theta_history=theta[:total_frames],
        source_activity=source_activity[:total_frames],
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        sound_speed=sound_speed,
    )
    label_builder = TrackTrendLabelBuilder(
        num_heatmap_bins=num_heatmap_bins,
        max_sources=max_sources,
        frame_hop_seconds=hop_length / float(sample_rate),
    )
    targets = label_builder.build_sequence_targets(
        source_positions=source_positions[:total_frames],
        array_positions=array_positions[:total_frames],
        source_activity=source_activity[:total_frames],
    )
    history_samples = history_frames * hop_length
    samples: list[ProbeSample] = []
    for step in range(animation_steps):
        current_idx = step + history_frames - 1
        future_slice = slice(current_idx + 1, current_idx + 1 + future_frames)
        sample_start = step * hop_length
        waveform = full_waveform[:, sample_start : sample_start + history_samples]
        waveform = torch.nn.functional.pad(
            waveform,
            (0, history_samples - waveform.shape[-1]),
        )
        current_count = targets.count[current_idx]
        future_count = targets.count[future_slice]
        future_slot_state = targets.slot_state[future_slice]
        samples.append(
            ProbeSample(
                sample_id=f"probe:{scenario}:step{step:02d}",
                waveform=waveform,
                vad_history=targets.vad_ratio[step : step + history_frames].unsqueeze(-1),
                count=current_count,
                heatmap=targets.heatmap[current_idx],
                slot_state=targets.slot_state[current_idx],
                future_count=future_count,
                future_heatmap=targets.heatmap[future_slice],
                future_slot_state=future_slot_state,
                trend_class=label_builder.classify_future_motion(future_slot_state),
                source_positions=None,
                array_positions=None,
                source_activity=None,
                transition_start_index=infer_transition_start_index(
                    current_count,
                    future_count,
                ),
            )
        )
    return samples


def evaluate_probe_suite(
    *,
    model: torch.nn.Module,
    probe_samples: dict[str, list[ProbeSample]],
    device: torch.device,
) -> dict[str, float]:
    if not probe_samples:
        return {}
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    scenario_scores: list[float] = []
    transition_scores: list[float] = []
    count_scores: list[float] = []
    with torch.no_grad():
        for scenario, samples in probe_samples.items():
            current_correct = 0.0
            current_total = 0.0
            future_correct = 0.0
            future_total = 0.0
            transition_correct = 0.0
            transition_total = 0.0
            for sample in samples:
                predictions = model(
                    sample.waveform.unsqueeze(0).to(device),
                    sample.vad_history.unsqueeze(0).to(device),
                )
                current_count_pred = int(
                    slot_count_from_state(predictions["slot_logits"][0], is_logits=True).item()
                )
                future_count_pred = (
                    slot_count_from_state(predictions["future_slot_logits"][0], is_logits=True)
                    .cpu()
                    .to(dtype=torch.long)
                )
                future_count_target = sample.future_count.to(dtype=torch.long)
                current_correct += float(current_count_pred == int(sample.count.item()))
                current_total += 1.0
                future_correct += float((future_count_pred == future_count_target).sum().item())
                future_total += float(future_count_target.numel())
                if sample.transition_start_index is not None:
                    transition_slice = slice(sample.transition_start_index, None)
                    transition_pred = future_count_pred[transition_slice]
                    transition_target = future_count_target[transition_slice]
                    transition_correct += float((transition_pred == transition_target).sum().item())
                    transition_total += float(transition_target.numel())
            current_acc = current_correct / max(current_total, 1.0)
            future_frame_acc = future_correct / max(future_total, 1.0)
            metrics[f"probe/{scenario}/current_count_acc"] = current_acc
            metrics[f"probe/{scenario}/future_count_frame_acc"] = future_frame_acc
            if transition_total > 0.0:
                transition_acc = transition_correct / transition_total
                transition_scores.append(transition_acc)
                scenario_score = transition_acc
            else:
                transition_acc = future_frame_acc
                scenario_score = future_frame_acc
            metrics[f"probe/{scenario}/transition_count_frame_acc"] = transition_acc
            metrics[f"probe/{scenario}/scenario_score"] = scenario_score
            count_scores.extend([current_acc, future_frame_acc])
            scenario_scores.append(scenario_score)
    if was_training:
        model.train()
    metrics["probe/checkpoint_score"] = (
        sum(scenario_scores) / len(scenario_scores) if scenario_scores else 0.0
    )
    metrics["probe/count_score"] = sum(count_scores) / len(count_scores) if count_scores else 0.0
    metrics["probe/transition_count_score"] = (
        sum(transition_scores) / len(transition_scores) if transition_scores else 0.0
    )
    return metrics


def summarize_probe_suite(probe_samples: dict[str, list[ProbeSample]]) -> dict[str, Any]:
    return {
        scenario: {
            "num_windows": len(samples),
            "first_sample_id": samples[0].sample_id,
            "first_current_count": int(samples[0].count.item()),
            "last_current_count": int(samples[-1].count.item()),
            "first_future_count_start": int(samples[0].future_count[0].item()),
            "last_future_count_end": int(samples[-1].future_count[-1].item()),
            "transition_window_count": sum(
                1 for sample in samples if sample.transition_start_index is not None
            ),
        }
        for scenario, samples in probe_samples.items()
    }


__all__ = [
    "ProbeSample",
    "SCENARIO_CHOICES",
    "build_probe_rollout_samples",
    "build_probe_sample",
    "build_scenario",
    "default_source_audio",
    "evaluate_probe_suite",
    "infer_transition_start_index",
    "load_probe_mono_audio",
    "render_history_waveform",
    "summarize_probe_suite",
    "transition_start_index",
]
