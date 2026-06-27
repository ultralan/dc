# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.geometry.uca8 import make_uniform_circular_array, wrap_angle
from uca8.metrics import (
    heatmap_contrast,
    heatmap_peak_recall_stats,
    primary_slot_range_stats,
    slot_count_from_state,
    slot_trend_label_from_sequence,
)
from uca8.postprocess import estimate_source_count_from_heatmap
from uca8.utils.audio_io import load_audio_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simulate_trained_scene import (
    SCENARIO_CHOICES,
    TREND_LABELS,
    build_model,
    build_scenario,
    default_source_audio,
    find_latest_run,
    render_history_waveform,
    resolve_device,
    slot_angles_deg,
    smooth_slot_angles_deg,
)  # noqa: E402

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
SLOT_TREND_LABELS = {-1: "clockwise", 0: "stable", 1: "counter_clockwise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a rolling-window animated simulation for a trained 8-mic model."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-audio", type=Path, default=None)
    parser.add_argument(
        "--scenario",
        type=str,
        default="moving_arc",
        choices=SCENARIO_CHOICES,
    )
    parser.add_argument("--animation-steps", type=int, default=48)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args()


def wrap_angle_deg(value: float) -> float:
    angle = torch.tensor(math.radians(value), dtype=torch.float32)
    return float(torch.rad2deg(wrap_angle(angle)).item())


def extract_current_pred_slot_angles_deg(predictions: dict[str, torch.Tensor]) -> np.ndarray:
    return slot_angles_deg(predictions["slot_logits"][0], is_logits=True).astype(np.float32)


def extract_future_pred_slot_angles_deg(predictions: dict[str, torch.Tensor]) -> np.ndarray:
    slot_angles = slot_angles_deg(predictions["future_slot_logits"][0], is_logits=True).astype(
        np.float32
    )
    return smooth_slot_angles_deg(slot_angles)


def slot_angle_mae_deg(target_deg: np.ndarray, pred_deg: np.ndarray) -> float:
    target_deg = np.asarray(target_deg, dtype=np.float32)
    pred_deg = np.asarray(pred_deg, dtype=np.float32)
    if target_deg.shape != pred_deg.shape:
        raise ValueError("target_deg and pred_deg must have the same shape.")
    valid_mask = np.isfinite(target_deg) & np.isfinite(pred_deg)
    if not valid_mask.any():
        if np.isfinite(target_deg).any() or np.isfinite(pred_deg).any():
            return 180.0
        return 0.0
    errors = [
        abs(wrap_angle_deg(float(pred_value - target_value)))
        for target_value, pred_value in zip(
            target_deg[valid_mask].reshape(-1),
            pred_deg[valid_mask].reshape(-1),
            strict=True,
        )
    ]
    return float(np.mean(errors))


def unwrap_angles_deg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        result = np.full_like(values, np.nan, dtype=np.float32)
        valid_mask = np.isfinite(values)
        if not valid_mask.any():
            return result
        valid_values = np.deg2rad(values[valid_mask])
        result[valid_mask] = np.rad2deg(np.unwrap(valid_values)).astype(np.float32)
        return result
    if values.ndim == 2:
        return np.stack(
            [unwrap_angles_deg(values[:, slot_idx]) for slot_idx in range(values.shape[1])], axis=1
        )
    raise ValueError("unwrap_angles_deg expects a 1D or 2D array.")


def format_angle_list(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float32)
    finite = [f"{float(value):.1f}" for value in values.reshape(-1) if np.isfinite(value)]
    return "[" + ", ".join(finite) + "]" if finite else "[]"


def build_frame_audit(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(item["step"]),
        "current_index": int(item["current_index"]),
        "current_count_target": int(item["current_count_target"]),
        "current_count_pred": int(item["current_count_pred"]),
        "current_count_pred_head": int(item["current_count_pred_head"]),
        "current_count_pred_slot": int(item["current_count_pred_slot"]),
        "current_count_pred_peak": int(item["current_count_pred_peak"]),
        "trend_target": str(item["trend_target"]),
        "trend_pred": str(item["trend_pred"]),
        "future_count_target": [int(value) for value in item["future_count_target"]],
        "future_count_pred": [int(value) for value in item["future_count_pred"]],
        "future_count_pred_head": [int(value) for value in item["future_count_pred_head"]],
        "future_count_pred_slot": [int(value) for value in item["future_count_pred_slot"]],
        "future_count_pred_peak": [int(value) for value in item["future_count_pred_peak"]],
        "current_target_slots_deg": [
            None if not np.isfinite(value) else float(value)
            for value in item["current_target_slots_deg"]
        ],
        "current_pred_slots_deg": [
            None if not np.isfinite(value) else float(value)
            for value in item["current_pred_slots_deg"]
        ],
        "current_angle_mae_deg": slot_angle_mae_deg(
            item["current_target_slots_deg"],
            item["current_pred_slots_deg"],
        ),
        "future_angle_mae_deg": slot_angle_mae_deg(
            item["future_target_slots_deg"],
            item["future_pred_slots_deg"],
        ),
        "future_target_slots_deg": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in item["future_target_slots_deg"]
        ],
        "future_pred_slots_deg": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in item["future_pred_slots_deg"]
        ],
        "current_primary_roll_range_deg": float(item["current_primary_roll_range_deg"]),
        "future_primary_range_ratio": float(item["future_primary_range_ratio"]),
        "future_heatmap_peak_recall_at_2": float(item["future_heatmap_peak_recall_at_2"]),
        "future_heatmap_contrast": float(item["future_heatmap_contrast"]),
        "trend_from_future_slots": str(item["trend_from_future_slots"]),
        "collapse_warning": bool(item["collapse_warning"]),
    }


def save_keyframe_contact_sheet(frames: list[Image.Image], output_path: Path) -> None:
    if not frames:
        return
    key_indices = sorted({0, len(frames) // 2, len(frames) - 1})
    keyframes = [frames[index] for index in key_indices]
    width, height = keyframes[0].size
    canvas = Image.new("RGB", (width * len(keyframes), height), color="white")
    for offset, image in enumerate(keyframes):
        canvas.paste(image, (offset * width, 0))
    canvas.save(output_path)


def build_rollout(
    *,
    cfg: Any,
    scenario: str,
    source_audio: Path,
    model: torch.nn.Module,
    device: torch.device,
    animation_steps: int,
) -> list[dict[str, Any]]:
    history_frames = int(cfg.model.history_frames)
    future_frames = int(cfg.model.future_frames)
    sample_rate = int(cfg.data.model_sample_rate)
    hop_length = int(cfg.feature.hop_length)
    win_length = int(cfg.feature.win_length)
    sound_speed = float(cfg.model.sound_speed)
    total_frames = history_frames + future_frames + animation_steps - 1

    mono_waveform, _ = load_audio_file(source_audio, target_sample_rate=sample_rate)
    mono_waveform = mono_waveform[0]
    mic_positions = make_uniform_circular_array(
        num_mics=int(cfg.model.num_input_channels),
        radius=float(cfg.model.array_radius_m),
    )
    source_positions, array_positions, source_activity = build_scenario(
        scenario=scenario,
        history_frames=history_frames,
        future_frames=future_frames + animation_steps - 1,
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
        num_heatmap_bins=int(cfg.model.heatmap_bins),
        max_sources=int(cfg.model.max_sources),
        frame_hop_seconds=hop_length / float(sample_rate),
    )
    targets = label_builder.build_sequence_targets(
        source_positions=source_positions[:total_frames],
        array_positions=array_positions[:total_frames],
        source_activity=source_activity[:total_frames],
    )

    history_samples = history_frames * hop_length
    frames: list[dict[str, Any]] = []
    for step in range(animation_steps):
        current_idx = step + history_frames - 1
        future_slice = slice(current_idx + 1, current_idx + 1 + future_frames)
        sample_start = step * hop_length
        history_waveform = full_waveform[:, sample_start : sample_start + history_samples]
        history_waveform = F.pad(
            history_waveform, (0, history_samples - history_waveform.shape[-1])
        )
        vad_history = targets.vad_ratio[step : step + history_frames]
        batch_waveform = history_waveform.unsqueeze(0).to(device)
        batch_vad = vad_history.unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            predictions = model(
                batch_waveform,
                batch_vad,
                sample_id=f"{args.scenario}:step{step:02d}",
            )

        current_heat_pred = torch.sigmoid(predictions["heatmap_logits"][0]).cpu()
        future_heat_pred = torch.sigmoid(predictions["future_heatmap_logits"][0]).cpu()
        current_count_pred_head = int(predictions["count_logits"][0].argmax().item())
        future_count_pred_head = predictions["future_count_logits"][0].argmax(dim=-1).cpu().numpy()
        current_count_pred_peak = int(
            estimate_source_count_from_heatmap(
                current_heat_pred,
                max_sources=int(cfg.model.max_sources),
            ).item()
        )
        future_count_pred_peak = (
            estimate_source_count_from_heatmap(
                future_heat_pred,
                max_sources=int(cfg.model.max_sources),
            )
            .cpu()
            .numpy()
        )
        current_target_slots_deg = slot_angles_deg(
            targets.slot_state[current_idx], is_logits=False
        ).astype(np.float32)
        current_pred_slots_deg = extract_current_pred_slot_angles_deg(predictions)
        future_target_slots_deg = slot_angles_deg(
            targets.slot_state[future_slice], is_logits=False
        ).astype(np.float32)
        future_pred_slots_deg = extract_future_pred_slot_angles_deg(predictions)
        current_count_pred_slot = int(
            slot_count_from_state(predictions["slot_logits"][0], is_logits=True).item()
        )
        future_count_pred_slot = (
            slot_count_from_state(predictions["future_slot_logits"][0], is_logits=True)
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        future_range_ratio, _, _, _ = primary_slot_range_stats(
            predictions["future_slot_logits"][0].cpu(),
            targets.slot_state[future_slice],
            pred_is_logits=True,
            target_is_logits=False,
        )
        future_heat_recall_sum, future_heat_recall_total = heatmap_peak_recall_stats(
            future_heat_pred,
            targets.heatmap[future_slice],
            targets.count[future_slice],
            tolerance_bins=2,
        )
        future_heat_contrast = float(heatmap_contrast(future_heat_pred).item())
        future_slot_trend = SLOT_TREND_LABELS[
            slot_trend_label_from_sequence(
                predictions["future_slot_logits"][0].cpu(),
                is_logits=True,
            )
        ]
        current_count_pred = current_count_pred_slot
        future_count_pred = future_count_pred_slot

        frames.append(
            {
                "step": step,
                "current_index": current_idx,
                "current_count_target": int(targets.count[current_idx].item()),
                "current_count_pred": current_count_pred,
                "current_count_pred_head": current_count_pred_head,
                "current_count_pred_slot": current_count_pred_slot,
                "current_count_pred_peak": current_count_pred_peak,
                "trend_target": TREND_LABELS[
                    int(
                        label_builder.classify_future_motion(
                            targets.slot_state[future_slice]
                        ).item()
                    )
                ],
                "trend_pred": TREND_LABELS[int(predictions["motion_logits"][0].argmax().item())],
                "current_heat_target": targets.heatmap[current_idx].cpu().numpy(),
                "current_heat_pred": current_heat_pred.numpy(),
                "future_heat_target": targets.heatmap[future_slice].cpu().numpy(),
                "future_heat_pred": future_heat_pred.numpy(),
                "future_count_target": targets.count[future_slice].cpu().numpy(),
                "future_count_pred": future_count_pred,
                "future_count_pred_head": future_count_pred_head,
                "future_count_pred_slot": future_count_pred_slot,
                "future_count_pred_peak": future_count_pred_peak,
                "current_target_slots_deg": current_target_slots_deg,
                "current_pred_slots_deg": current_pred_slots_deg,
                "future_target_slots_deg": future_target_slots_deg,
                "future_pred_slots_deg": future_pred_slots_deg,
                "current_slot_state_target": targets.slot_state[current_idx].cpu(),
                "current_slot_state_pred": predictions["slot_logits"][0].cpu(),
                "future_slot_state_target": targets.slot_state[future_slice].cpu(),
                "future_slot_state_pred": predictions["future_slot_logits"][0].cpu(),
                "future_primary_range_ratio": float(future_range_ratio.item()),
                "future_heatmap_peak_recall_at_2": float(
                    (future_heat_recall_sum / future_heat_recall_total.clamp_min(1.0)).item()
                ),
                "future_heatmap_contrast": future_heat_contrast,
                "trend_from_future_slots": future_slot_trend,
                "collapse_warning": bool(
                    float(future_range_ratio.item()) < 0.2 or future_heat_contrast < 0.15
                ),
                "source_xy": source_positions[:, :, :2].cpu().numpy(),
                "source_activity": source_activity.cpu().numpy(),
            }
        )
    return frames


def render_animation(
    *,
    frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
    array_radius_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gif_frames: list[Image.Image] = []
    frame_audits: list[dict[str, Any]] = []
    current_count_correct = 0
    future_count_correct = 0
    future_count_total = 0
    trend_correct = 0
    current_angle_errors: list[float] = []
    future_angle_errors: list[float] = []
    future_range_ratios: list[float] = []
    future_heat_recalls: list[float] = []
    future_heat_contrasts: list[float] = []
    collapse_warning_count = 0
    slot_colors = plt.get_cmap("tab10")(np.arange(10))
    _, current_roll_pred_range, _, _ = primary_slot_range_stats(
        torch.stack([item["current_slot_state_pred"] for item in frames], dim=0),
        torch.stack([item["current_slot_state_target"] for item in frames], dim=0),
        pred_is_logits=True,
        target_is_logits=False,
    )
    current_primary_roll_range_deg = float(current_roll_pred_range.item())
    for item in frames:
        item["current_primary_roll_range_deg"] = current_primary_roll_range_deg

    for item in frames:
        frame_audits.append(build_frame_audit(item))
        current_count_correct += int(item["current_count_target"] == item["current_count_pred"])
        future_count_correct += int(
            (item["future_count_target"] == item["future_count_pred"]).sum()
        )
        future_count_total += int(item["future_count_target"].shape[0])
        trend_correct += int(item["trend_target"] == item["trend_pred"])
        future_range_ratios.append(float(item["future_primary_range_ratio"]))
        future_heat_recalls.append(float(item["future_heatmap_peak_recall_at_2"]))
        future_heat_contrasts.append(float(item["future_heatmap_contrast"]))
        collapse_warning_count += int(bool(item["collapse_warning"]))
        current_angle_errors.append(
            slot_angle_mae_deg(item["current_target_slots_deg"], item["current_pred_slots_deg"])
        )
        future_angle_errors.append(
            slot_angle_mae_deg(item["future_target_slots_deg"], item["future_pred_slots_deg"])
        )
        target_future_unwrapped = unwrap_angles_deg(item["future_target_slots_deg"])
        pred_future_unwrapped = unwrap_angles_deg(item["future_pred_slots_deg"])

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        ax_scene = axes[0, 0]
        ax_current = axes[0, 1]
        ax_text = axes[0, 2]
        ax_future_target = axes[1, 0]
        ax_future_pred = axes[1, 1]
        ax_future_line = axes[1, 2]

        source_xy = item["source_xy"]
        source_activity = item["source_activity"]
        current_idx = item["current_index"]
        active_xy = source_xy[source_activity > 0.5]
        ray_len = max(
            1.5,
            float(np.linalg.norm(active_xy, axis=-1).max()) * 1.25 if active_xy.size > 0 else 0.0,
        )

        ax_scene.scatter([0.0], [0.0], color="black", s=60, label="array_center")
        mic_angles = np.linspace(0.0, 2.0 * math.pi, num=8, endpoint=False)
        ax_scene.scatter(
            array_radius_m * np.cos(mic_angles),
            array_radius_m * np.sin(mic_angles),
            color="black",
            s=20,
            alpha=0.7,
        )
        for slot_idx in range(source_xy.shape[1]):
            color = slot_colors[slot_idx % len(slot_colors)]
            past_xy = source_xy[: current_idx + 1, slot_idx].copy()
            past_active = source_activity[: current_idx + 1, slot_idx] > 0.5
            past_xy[~past_active] = np.nan
            if np.isfinite(past_xy).any():
                ax_scene.plot(past_xy[:, 0], past_xy[:, 1], color=color, linewidth=2.0)

            future_xy = source_xy[
                current_idx + 1 : current_idx + 1 + item["future_target_slots_deg"].shape[0],
                slot_idx,
            ].copy()
            future_active = (
                source_activity[
                    current_idx + 1 : current_idx + 1 + item["future_target_slots_deg"].shape[0],
                    slot_idx,
                ]
                > 0.5
            )
            future_xy[~future_active] = np.nan
            if np.isfinite(future_xy).any():
                ax_scene.plot(
                    future_xy[:, 0],
                    future_xy[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                )

            if source_activity[current_idx, slot_idx] > 0.5:
                current_xy = source_xy[current_idx, slot_idx]
                ax_scene.scatter([current_xy[0]], [current_xy[1]], color=color, s=75)

            current_pred_angle = item["current_pred_slots_deg"][slot_idx]
            if np.isfinite(current_pred_angle):
                pred_rad = math.radians(float(current_pred_angle))
                ax_scene.plot(
                    [0.0, ray_len * math.cos(pred_rad)],
                    [0.0, ray_len * math.sin(pred_rad)],
                    color=color,
                    linewidth=2.0,
                    alpha=0.95,
                )

            pred_future_slot = item["future_pred_slots_deg"][:, slot_idx]
            finite_future = np.flatnonzero(np.isfinite(pred_future_slot))
            if finite_future.size > 0:
                sampled = np.linspace(
                    0,
                    finite_future.size - 1,
                    num=min(6, finite_future.size),
                    dtype=int,
                )
                for rank, future_index in enumerate(finite_future[np.unique(sampled)]):
                    future_angle_rad = math.radians(float(pred_future_slot[future_index]))
                    alpha = 0.15 + 0.55 * (rank + 1) / max(len(np.unique(sampled)), 1)
                    ax_scene.plot(
                        [0.0, ray_len * math.cos(future_angle_rad)],
                        [0.0, ray_len * math.sin(future_angle_rad)],
                        color=color,
                        linewidth=1.0,
                        alpha=alpha,
                    )
        ax_scene.set_title("Top-Down Scene")
        ax_scene.set_aspect("equal", adjustable="box")
        ax_scene.grid(alpha=0.3)

        bins = np.arange(len(item["current_heat_target"]))
        ax_current.plot(bins, item["current_heat_target"], label="target")
        ax_current.plot(bins, item["current_heat_pred"], label="pred")
        ax_current.set_title("Current Heatmap")
        ax_current.set_xlabel("Azimuth Bin")
        ax_current.set_ylabel("Activation")
        ax_current.grid(alpha=0.3)
        ax_current.legend()

        ax_text.axis("off")
        future_frame_correct = int(
            (item["future_count_target"] == item["future_count_pred"]).sum()
        )
        future_frame_total = len(item["future_count_target"])
        current_count_line = (
            "current count target/pred: "
            f"{item['current_count_target']}/{item['current_count_pred']}"
        )
        current_slot_peak_line = (
            "current count slot/peak: "
            f"{item['current_count_pred_slot']}/{item['current_count_pred_peak']}"
        )
        future_frame_acc_line = (
            "future count frame acc(frame): "
            f"{future_frame_correct}/{future_frame_total}"
        )
        text = "\n".join(
            [
                f"frame: {item['step'] + 1}/{len(frames)}",
                current_count_line,
                f"current count head raw: {item['current_count_pred_head']}",
                current_slot_peak_line,
                f"trend target/pred: {item['trend_target']}/{item['trend_pred']}",
                f"future-slot trend: {item['trend_from_future_slots']}",
                future_frame_acc_line,
                f"current slot target: {format_angle_list(item['current_target_slots_deg'])}",
                f"current slot pred: {format_angle_list(item['current_pred_slots_deg'])}",
                f"current slot MAE: {current_angle_errors[-1]:.1f} deg",
                f"future slot MAE: {future_angle_errors[-1]:.1f} deg",
                f"current roll range: {item['current_primary_roll_range_deg']:.2f} deg",
                f"future range ratio: {item['future_primary_range_ratio']:.2f}",
                f"future heat recall@2: {item['future_heatmap_peak_recall_at_2']:.2f}",
                f"future heat contrast: {item['future_heatmap_contrast']:.2f}",
                f"collapse warning: {item['collapse_warning']}",
                "scene/horizon share slot colors; dashed target means future",
            ]
        )
        text_color = "crimson" if item["collapse_warning"] else "black"
        ax_text.text(
            0.02,
            0.98,
            text,
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
            color=text_color,
        )

        ax_future_target.imshow(
            item["future_heat_target"].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax_future_target.set_title("Future Heatmap Target")
        ax_future_target.set_xlabel("Future Frame")
        ax_future_target.set_ylabel("Azimuth Bin")

        ax_future_pred.imshow(
            item["future_heat_pred"].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax_future_pred.set_title("Future Heatmap Pred")
        ax_future_pred.set_xlabel("Future Frame")
        ax_future_pred.set_ylabel("Azimuth Bin")

        horizon = np.arange(item["future_target_slots_deg"].shape[0])
        for slot_idx in range(item["future_target_slots_deg"].shape[1]):
            color = slot_colors[slot_idx % len(slot_colors)]
            target_slot = target_future_unwrapped[:, slot_idx]
            pred_slot = pred_future_unwrapped[:, slot_idx]
            if np.isfinite(target_slot).any():
                ax_future_line.plot(
                    horizon, target_slot, color=color, linewidth=2.0, label=f"target_s{slot_idx}"
                )
            if np.isfinite(pred_slot).any():
                ax_future_line.plot(
                    horizon,
                    pred_slot,
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    label=f"pred_s{slot_idx}",
                )
        ax_future_line.set_title("Future Azimuth Horizon")
        ax_future_line.set_xlabel("Future Frame")
        ax_future_line.set_ylabel("Azimuth (deg)")
        ax_future_line.grid(alpha=0.3)
        handles, labels = ax_future_line.get_legend_handles_labels()
        if handles:
            ax_future_line.legend(handles[:8], labels[:8], fontsize=8)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        buffer.seek(0)
        gif_frames.append(Image.open(buffer).convert("RGB"))
        plt.close(fig)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(round(1000 / max(fps, 1))),
        loop=0,
        optimize=False,
    )
    save_keyframe_contact_sheet(
        gif_frames, output_path.with_name(f"{output_path.stem}_keyframes.png")
    )

    summary = {
        "frames": len(frames),
        "fps": fps,
        "current_count_acc": current_count_correct / max(len(frames), 1),
        "future_count_frame_acc": future_count_correct / max(future_count_total, 1),
        "trend_acc": trend_correct / max(len(frames), 1),
        "current_angle_mae_deg": float(np.mean(current_angle_errors))
        if current_angle_errors
        else 0.0,
        "future_angle_mae_deg": float(np.mean(future_angle_errors)) if future_angle_errors else 0.0,
        "current_primary_roll_range_deg": current_primary_roll_range_deg,
        "future_primary_range_ratio": float(np.mean(future_range_ratios)) if future_range_ratios else 0.0,
        "future_heatmap_peak_recall_at_2": float(np.mean(future_heat_recalls))
        if future_heat_recalls
        else 0.0,
        "future_heatmap_contrast": float(np.mean(future_heat_contrasts))
        if future_heat_contrasts
        else 0.0,
        "collapse_warning_rate": collapse_warning_count / max(len(frames), 1),
        "angle_metric_basis": "slot_mae_over_finite_assignments",
        "count_decoder": "slot_activity_count_primary",
        "gif": output_path.name,
    }
    return summary, frame_audits


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_run(RUNS_ROOT)
    checkpoint_path = args.checkpoint or run_dir / "best.pt"
    output_dir = args.output_dir or run_dir / "simulation_animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source_audio = args.source_audio or default_source_audio(Path(str(cfg.data.root_dir)))
    frames = build_rollout(
        cfg=cfg,
        scenario=args.scenario,
        source_audio=source_audio,
        model=model,
        device=device,
        animation_steps=args.animation_steps,
    )
    output_path = output_dir / f"{args.scenario}.gif"
    summary, frame_audits = render_animation(
        frames=frames,
        output_path=output_path,
        fps=args.fps,
        array_radius_m=float(cfg.model.array_radius_m),
    )
    (output_dir / f"{args.scenario}_frames.json").write_text(
        json.dumps(frame_audits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "scenario": args.scenario,
        "source_audio": str(source_audio),
        "animation_steps": args.animation_steps,
        "fps": args.fps,
        "summary": summary,
        "frame_audit": f"{args.scenario}_frames.json",
        "keyframes": f"{args.scenario}_keyframes.png",
    }
    (output_dir / f"{args.scenario}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"simulation_animation_saved={output_path}")


if __name__ == "__main__":
    main()
