from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from uca8.data.label_builder import TrackTrendLabelBuilder
from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.metrics import slot_count_from_state
from uca8.models.tracktrend_net import UCA8TrackTrendNet
from uca8.postprocess import AzimuthKalmanTracker
from uca8.sim import (
    SCENARIO_CHOICES,
    load_probe_mono_audio,
)
from uca8.sim import (
    build_scenario as build_probe_scenario,
)
from uca8.sim import (
    default_source_audio as default_probe_source_audio,
)
from uca8.sim import (
    render_history_waveform as render_probe_history_waveform,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
TREND_LABELS = {0: "clockwise", 1: "stable", 2: "counter_clockwise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an analytic 8-mic far-field scene and visualize trained-model predictions."
        )
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_run(root: Path) -> Path:
    candidates = [
        path.parent
        for path in root.rglob("config_resolved.yaml")
        if path.parent.is_dir()
        and ((path.parent / "best.pt").exists() or (path.parent / "last.pt").exists())
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}.")
    return max(candidates, key=lambda path: path.name)


def default_source_audio(root_dir: Path) -> Path:
    return default_probe_source_audio(root_dir)


def build_model(cfg: DictConfig, device: torch.device) -> UCA8TrackTrendNet:
    mic_positions = make_uniform_circular_array(
        num_mics=int(cfg.model.num_input_channels),
        radius=float(cfg.model.array_radius_m),
    ).to(device)
    return UCA8TrackTrendNet(
        mic_positions=mic_positions,
        sample_rate=int(cfg.feature.sample_rate),
        n_fft=int(cfg.feature.n_fft),
        win_length=int(cfg.feature.win_length),
        hop_length=int(cfg.feature.hop_length),
        spec_bins=int(cfg.feature.spec_bins),
        ipd_bins=int(cfg.feature.ipd_bins),
        heatmap_bins=int(cfg.model.heatmap_bins),
        history_frames=int(cfg.model.history_frames),
        future_frames=int(cfg.model.future_frames),
        max_sources=int(cfg.model.max_sources),
        spec_hidden_dim=int(cfg.model.spec_hidden_dim),
        spatial_hidden_dim=int(cfg.model.spatial_hidden_dim),
        model_dim=int(cfg.model.model_dim),
        tcn_dilations=list(cfg.model.tcn_dilations),
        tcn_kernel_size=int(cfg.model.tcn_kernel_size),
        dropout=float(cfg.model.dropout),
        slot_decoder_attention_heads=int(cfg.model.get("slot_decoder_attention_heads", 4)),
        future_decoder_layers=int(cfg.model.get("future_decoder_layers", 2)),
        future_decoder_dropout=float(cfg.model.get("future_decoder_dropout", cfg.model.dropout)),
        use_slot_context_in_future_decoder=bool(
            cfg.model.get("use_slot_context_in_future_decoder", True)
        ),
        num_count_classes=int(cfg.model.num_count_classes),
        sound_speed=float(cfg.model.sound_speed),
        feature_cache_dir=cfg.data.get("feature_cache_dir"),
    ).to(device)


def build_scenario(
    *,
    scenario: str,
    history_frames: int,
    future_frames: int,
    distance_m: float = 1.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_probe_scenario(
        scenario=scenario,
        history_frames=history_frames,
        future_frames=future_frames,
        max_sources=4,
        distance_m=distance_m,
    )


def render_history_waveform(
    *,
    mono_waveform: torch.Tensor,
    theta_history: torch.Tensor,
    source_activity: torch.Tensor | None = None,
    mic_positions: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    sound_speed: float,
) -> torch.Tensor:
    return render_probe_history_waveform(
        mono_waveform=mono_waveform,
        theta_history=theta_history,
        source_activity=source_activity,
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        sound_speed=sound_speed,
    )


def slot_angles_deg(slot_state: torch.Tensor, *, is_logits: bool) -> np.ndarray:
    if is_logits:
        activity = torch.sigmoid(slot_state[..., 0])
    else:
        activity = slot_state[..., 0]
    angles = torch.rad2deg(torch.atan2(slot_state[..., 1], slot_state[..., 2]))
    angles = angles.masked_fill(activity <= 0.5, float("nan"))
    return angles.cpu().numpy()


def smooth_slot_angles_deg(slot_angles: np.ndarray, *, dt: float = 0.01) -> np.ndarray:
    smoothed = np.full_like(slot_angles, np.nan, dtype=np.float32)
    for slot_idx in range(slot_angles.shape[1]):
        tracker = AzimuthKalmanTracker(dt=dt)
        initialized = False
        for frame_idx in range(slot_angles.shape[0]):
            angle_deg = slot_angles[frame_idx, slot_idx]
            if np.isnan(angle_deg):
                continue
            measurement = float(np.deg2rad(angle_deg))
            if not initialized:
                tracker.state[0, 0] = measurement
                tracker.state[1, 0] = 0.0
                initialized = True
            else:
                tracker.predict()
            estimate, _ = tracker.update(measurement)
            smoothed[frame_idx, slot_idx] = float(np.rad2deg(estimate))
    return smoothed


def render_scene_figure(
    *,
    scenario: str,
    output_path: Path,
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    current_heat_target = targets["heatmap"].cpu().numpy()
    current_heat_pred = torch.sigmoid(predictions["heatmap_logits"][0]).cpu().numpy()
    future_heat_target = targets["future_heatmap"].cpu().numpy()
    future_heat_pred = torch.sigmoid(predictions["future_heatmap_logits"][0]).cpu().numpy()
    future_count_target = targets["future_count"].cpu().numpy()
    future_count_pred = (
        slot_count_from_state(predictions["future_slot_logits"][0], is_logits=True).cpu().numpy()
    )
    future_count_head_pred = predictions["future_count_logits"][0].argmax(dim=-1).cpu().numpy()
    slot_target_deg = slot_angles_deg(targets["future_slot_state"], is_logits=False)
    slot_pred_deg = slot_angles_deg(predictions["future_slot_logits"][0], is_logits=True)
    slot_pred_smooth_deg = smooth_slot_angles_deg(slot_pred_deg)
    current_count_target = int(targets["count"].item())
    current_count_pred = int(
        slot_count_from_state(predictions["slot_logits"][0], is_logits=True).item()
    )
    current_count_head_pred = int(predictions["count_logits"][0].argmax().item())
    trend_target = int(targets["trend_class"].item())
    trend_pred = int(predictions["motion_logits"][0].argmax().item())

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    azimuth_bins = np.arange(current_heat_target.shape[0])
    future_frames = np.arange(future_count_target.shape[0])

    axes[0, 0].plot(azimuth_bins, current_heat_target, label="target")
    axes[0, 0].plot(azimuth_bins, current_heat_pred, label="pred")
    axes[0, 0].set_title("Current Heatmap")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].imshow(future_heat_target.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[0, 1].set_title("Future Heatmap Target")
    axes[0, 1].set_xlabel("Future Frame")
    axes[0, 1].set_ylabel("Azimuth Bin")

    axes[1, 0].imshow(future_heat_pred.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[1, 0].set_title("Future Heatmap Pred")
    axes[1, 0].set_xlabel("Future Frame")
    axes[1, 0].set_ylabel("Azimuth Bin")

    axes[1, 1].plot(future_frames, future_count_target, label="target")
    axes[1, 1].plot(future_frames, future_count_pred, label="pred")
    axes[1, 1].set_title("Future Source Count")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    for slot_idx in range(slot_target_deg.shape[1]):
        axes[2, 0].plot(future_frames, slot_target_deg[:, slot_idx], label=f"slot{slot_idx}")
    axes[2, 0].set_title("Future Slot Azimuth Target")
    axes[2, 0].set_xlabel("Future Frame")
    axes[2, 0].set_ylabel("Azimuth (deg)")
    axes[2, 0].grid(alpha=0.3)

    for slot_idx in range(slot_pred_deg.shape[1]):
        axes[2, 1].plot(future_frames, slot_pred_deg[:, slot_idx], label=f"slot{slot_idx}")
        if np.isfinite(slot_pred_smooth_deg[:, slot_idx]).any():
            axes[2, 1].plot(
                future_frames,
                slot_pred_smooth_deg[:, slot_idx],
                linestyle="--",
                alpha=0.8,
                label=f"slot{slot_idx}_kalman",
            )
    axes[2, 1].set_title("Future Slot Azimuth Pred")
    axes[2, 1].set_xlabel("Future Frame")
    axes[2, 1].set_ylabel("Azimuth (deg)")
    axes[2, 1].grid(alpha=0.3)

    fig.suptitle(
        " | ".join(
            [
                f"scenario={scenario}",
                f"count target/pred={current_count_target}/{current_count_pred}",
                f"count_head_raw={current_count_head_pred}",
                f"trend target/pred={TREND_LABELS[trend_target]}/{TREND_LABELS[trend_pred]}",
            ]
        )
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "scenario": scenario,
        "current_count_target": current_count_target,
        "current_count_pred": current_count_pred,
        "current_count_head_pred": current_count_head_pred,
        "future_count_head_pred": future_count_head_pred.tolist(),
        "trend_target": TREND_LABELS[trend_target],
        "trend_pred": TREND_LABELS[trend_pred],
        "figure": output_path.name,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_run(RUNS_ROOT)
    checkpoint_path = args.checkpoint or run_dir / "best.pt"
    output_dir = args.output_dir or run_dir / "simulation_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source_audio = args.source_audio or default_source_audio(Path(str(cfg.data.root_dir)))
    mono_waveform = load_probe_mono_audio(
        source_audio,
        sample_rate=int(cfg.data.model_sample_rate),
        cache_dir=cfg.data.get("audio_cache_dir"),
    )

    history_frames = int(cfg.model.history_frames)
    future_frames = int(cfg.model.future_frames)
    sample_rate = int(cfg.data.model_sample_rate)
    hop_length = int(cfg.feature.hop_length)
    win_length = int(cfg.feature.win_length)
    sound_speed = float(cfg.model.sound_speed)
    mic_positions = make_uniform_circular_array(
        num_mics=int(cfg.model.num_input_channels),
        radius=float(cfg.model.array_radius_m),
    )

    source_positions, array_positions, source_activity = build_scenario(
        scenario=args.scenario,
        history_frames=history_frames,
        future_frames=future_frames,
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
        num_heatmap_bins=int(cfg.model.heatmap_bins),
        max_sources=int(cfg.model.max_sources),
        frame_hop_seconds=hop_length / float(sample_rate),
    )
    targets = label_builder.build_sequence_targets(
        source_positions=source_positions,
        array_positions=array_positions,
        source_activity=source_activity,
    )
    current_idx = history_frames - 1
    future_slice = slice(history_frames, history_frames + future_frames)
    batch = {
        "waveform": waveform.unsqueeze(0).to(device),
        "vad_history": targets.vad_ratio[:history_frames].unsqueeze(0).unsqueeze(-1).to(device),
    }

    with torch.no_grad():
        predictions = model(
            batch["waveform"],
            batch["vad_history"],
            sample_id=f"{args.scenario}:sim",
        )

    figure_path = output_dir / f"{args.scenario}.png"
    summary = render_scene_figure(
        scenario=args.scenario,
        output_path=figure_path,
        predictions=predictions,
        targets={
            "count": targets.count[current_idx],
            "heatmap": targets.heatmap[current_idx],
            "future_count": targets.count[future_slice],
            "future_heatmap": targets.heatmap[future_slice],
            "future_slot_state": targets.slot_state[future_slice],
            "trend_class": label_builder.classify_future_motion(targets.slot_state[future_slice]),
        },
    )
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "scenario": args.scenario,
        "source_audio": str(source_audio),
        "figure": summary["figure"],
        "summary": summary,
    }
    (output_dir / f"{args.scenario}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"simulation_visualization_saved={figure_path}")


if __name__ == "__main__":
    main()
