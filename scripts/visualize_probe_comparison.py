from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.metrics import slot_count_from_state
from uca8.models.tracktrend_net import UCA8TrackTrendNet
from uca8.sim import build_probe_rollout_samples, default_source_audio, load_probe_mono_audio

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize probe-score tradeoffs and rollout failure modes across runs."
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories containing config_resolved.yaml and best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write comparison figures.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="source_leave",
        help="Probe scenario to visualize in rollout detail.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        num_count_classes=int(cfg.model.num_count_classes),
        sound_speed=float(cfg.model.sound_speed),
    ).to(device)


def build_probe_samples_for_scenario(cfg: DictConfig, scenario: str) -> list[Any]:
    root_dir = Path(str(cfg.data.root_dir))
    source_audio_cfg = cfg.train.get("probe_source_audio")
    source_audio = (
        Path(str(source_audio_cfg)) if source_audio_cfg else default_source_audio(root_dir)
    )
    mono_waveform = load_probe_mono_audio(
        source_audio,
        sample_rate=int(cfg.data.model_sample_rate),
        cache_dir=cfg.data.get("audio_cache_dir"),
    )
    mic_positions = make_uniform_circular_array(
        num_mics=int(cfg.model.num_input_channels),
        radius=float(cfg.model.array_radius_m),
    )
    return build_probe_rollout_samples(
        scenario=scenario,
        mono_waveform=mono_waveform,
        mic_positions=mic_positions,
        sample_rate=int(cfg.data.model_sample_rate),
        hop_length=int(cfg.feature.hop_length),
        win_length=int(cfg.feature.win_length),
        sound_speed=float(cfg.model.sound_speed),
        history_frames=int(cfg.model.history_frames),
        future_frames=int(cfg.model.future_frames),
        num_heatmap_bins=int(cfg.model.heatmap_bins),
        max_sources=int(cfg.model.max_sources),
        animation_steps=int(cfg.train.get("probe_animation_steps", 24)),
    )


def compute_rollout_detail(
    *,
    run_dir: Path,
    scenario: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / "config_resolved.yaml")
    model = build_model(cfg, device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    samples = build_probe_samples_for_scenario(cfg, scenario)
    current_target = np.array([int(sample.count.item()) for sample in samples], dtype=np.int64)
    current_pred = np.zeros_like(current_target)
    future_target = np.stack([sample.future_count.cpu().numpy() for sample in samples], axis=0)
    future_pred = np.zeros_like(future_target)
    transition_start = np.full(len(samples), np.nan, dtype=np.float32)
    for step, sample in enumerate(samples):
        with torch.no_grad():
            predictions = model(
                sample.waveform.unsqueeze(0).to(device),
                sample.vad_history.unsqueeze(0).to(device),
            )
        current_pred[step] = int(
            slot_count_from_state(predictions["slot_logits"][0], is_logits=True).item()
        )
        future_pred[step] = (
            slot_count_from_state(predictions["future_slot_logits"][0], is_logits=True)
            .cpu()
            .numpy()
        )
        if sample.transition_start_index is not None:
            transition_start[step] = float(sample.transition_start_index)
    return {
        "run_name": run_dir.name,
        "current_target": current_target,
        "current_pred": current_pred,
        "future_target": future_target,
        "future_pred": future_pred,
        "future_error": future_pred - future_target,
        "transition_start": transition_start,
    }


def plot_probe_tradeoff(
    *,
    run_dirs: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axis_map = {
        "probe/checkpoint_score": axes[0, 0],
        "probe/dual_cross/scenario_score": axes[0, 1],
        "probe/source_enter/scenario_score": axes[1, 0],
        "probe/source_leave/scenario_score": axes[1, 1],
    }
    title_map = {
        "probe/checkpoint_score": "Checkpoint Score",
        "probe/dual_cross/scenario_score": "Dual Cross Score",
        "probe/source_enter/scenario_score": "Source Enter Score",
        "probe/source_leave/scenario_score": "Source Leave Score",
    }
    summary: dict[str, Any] = {}
    for run_dir in run_dirs:
        history = load_history(run_dir / "train_history.jsonl")
        val_rows = [row for row in history if row.get("phase") == "val"]
        steps = [int(row["global_step"]) for row in val_rows]
        summary[run_dir.name] = {}
        for metric_name, axis in axis_map.items():
            values = [float(row.get(metric_name, np.nan)) for row in val_rows]
            axis.plot(steps, values, marker="o", linewidth=2.0, label=run_dir.name)
            summary[run_dir.name][metric_name] = values
            axis.set_title(title_map[metric_name])
            axis.set_xlabel("Global Step")
            axis.set_ylabel("Score")
            axis.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def plot_rollout_comparison(
    *,
    rollout_details: list[dict[str, Any]],
    scenario: str,
    output_path: Path,
) -> None:
    num_runs = len(rollout_details)
    fig, axes = plt.subplots(
        2,
        num_runs + 1,
        figsize=(5 * (num_runs + 1), 8),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    target = rollout_details[0]
    steps = np.arange(len(target["current_target"]))

    axes[0, 0].plot(steps, target["current_target"], color="black", linewidth=2.0)
    axes[0, 0].set_title(f"{scenario} Target Current Count")
    axes[0, 0].set_xlabel("Rolling Step")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_ylim(0.5, max(2.5, float(target["current_target"].max()) + 0.5))
    axes[0, 0].grid(alpha=0.3)

    target_im = axes[1, 0].imshow(
        target["future_target"],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=1,
        vmax=max(2, int(target["future_target"].max())),
        cmap="viridis",
    )
    transition_mask = np.isfinite(target["transition_start"])
    axes[1, 0].plot(
        target["transition_start"][transition_mask],
        steps[transition_mask],
        linestyle="--",
        color="white",
        linewidth=2.0,
    )
    axes[1, 0].set_title(f"{scenario} Target Future Count")
    axes[1, 0].set_xlabel("Future Frame")
    axes[1, 0].set_ylabel("Rolling Step")
    fig.colorbar(target_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    for column, detail in enumerate(rollout_details, start=1):
        axes[0, column].plot(
            steps,
            detail["current_target"],
            color="black",
            linewidth=2.0,
            label="target",
        )
        axes[0, column].plot(
            steps,
            detail["current_pred"],
            color="tab:red",
            linewidth=2.0,
            label="pred",
        )
        axes[0, column].set_title(f"{detail['run_name']} Current Count")
        axes[0, column].set_xlabel("Rolling Step")
        axes[0, column].set_ylabel("Count")
        axes[0, column].set_ylim(0.5, max(2.5, float(detail["current_target"].max()) + 0.5))
        axes[0, column].grid(alpha=0.3)
        axes[0, column].legend()

        error_im = axes[1, column].imshow(
            detail["future_error"],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[1, column].plot(
            detail["transition_start"][transition_mask],
            steps[transition_mask],
            linestyle="--",
            color="black",
            linewidth=2.0,
        )
        axes[1, column].set_title(f"{detail['run_name']} Future Count Error")
        axes[1, column].set_xlabel("Future Frame")
        axes[1, column].set_ylabel("Rolling Step")
        fig.colorbar(error_im, ax=axes[1, column], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Future count error heatmap: red=over-count, blue=under-count, white=correct",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    run_dirs = [path.resolve() for path in args.run_dirs]

    tradeoff_summary = plot_probe_tradeoff(
        run_dirs=run_dirs,
        output_path=output_dir / "probe_tradeoff.png",
    )
    rollout_details = [
        compute_rollout_detail(run_dir=run_dir, scenario=args.scenario, device=device)
        for run_dir in run_dirs
    ]
    plot_rollout_comparison(
        rollout_details=rollout_details,
        scenario=args.scenario,
        output_path=output_dir / f"{args.scenario}_rollout_comparison.png",
    )

    summary = {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "scenario": args.scenario,
        "figures": [
            "probe_tradeoff.png",
            f"{args.scenario}_rollout_comparison.png",
        ],
        "tradeoff_summary": tradeoff_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"probe_comparison_saved={output_dir}")


if __name__ == "__main__":
    main()
