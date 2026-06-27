from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from uca8.data.realman_ring2_dataset import RealMANRing2Dataset
from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.metrics import (
    circular_abs_error_deg,
    heatmap_localization_stats,
    heatmap_logits_to_azimuth_deg,
    slot_count_from_state,
    slot_logits_to_primary_azimuth_deg,
    target_slot_primary_azimuth_deg,
)
from uca8.models.tracktrend_net import UCA8TrackTrendNet
from uca8.features.stft import MultiChannelSTFT
from uca8.features.srp_phat import SRPPHAT
from uca8.geometry.uca8 import azimuth_grid, infer_mic_pairs, wrap_angle


def wrap_angle_deg(angles_deg: torch.Tensor) -> torch.Tensor:
    """把度数角度 wrap 到 [-180, 180)."""
    return wrap_angle(torch.deg2rad(angles_deg)) * (180.0 / math.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current model with RealMAN-style localization metrics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def update_group(
    groups: dict[str, dict[str, float]],
    group_name: str,
    *,
    heat_err: torch.Tensor,
    slot_err: torch.Tensor,
    heat_acc5: torch.Tensor,
    slot_acc5: torch.Tensor,
    count_head_correct: torch.Tensor,
    slot_count_correct: torch.Tensor,
    future_count_correct: torch.Tensor,
    future_total: int,
) -> None:
    stats = groups[group_name]
    stats["frames"] += float(heat_err.numel())
    stats["heat_abs_err_sum"] += float(heat_err.sum().item())
    stats["slot_abs_err_sum"] += float(slot_err.sum().item())
    stats["heat_acc5_sum"] += float(heat_acc5.sum().item())
    stats["slot_acc5_sum"] += float(slot_acc5.sum().item())
    stats["count_head_correct"] += float(count_head_correct.sum().item())
    stats["slot_count_correct"] += float(slot_count_correct.sum().item())
    stats["samples"] += float(count_head_correct.numel())
    stats["future_count_correct"] += float(future_count_correct.sum().item())
    stats["future_frames"] += float(future_total)


def finalize(groups: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, stats in groups.items():
        frames = max(stats["frames"], 1.0)
        samples = max(stats["samples"], 1.0)
        future_frames = max(stats["future_frames"], 1.0)
        output[name] = {
            "num_current_active_frames": stats["frames"],
            "num_samples": stats["samples"],
            "heatmap_argmax_mae_deg": stats["heat_abs_err_sum"] / frames,
            "heatmap_argmax_acc5_percent": 100.0 * stats["heat_acc5_sum"] / frames,
            "slot_primary_mae_deg": stats["slot_abs_err_sum"] / frames,
            "slot_primary_acc5_percent": 100.0 * stats["slot_acc5_sum"] / frames,
            "count_head_acc_percent": 100.0 * stats["count_head_correct"] / samples,
            "slot_count_acc_percent": 100.0 * stats["slot_count_correct"] / samples,
            "future_count_frame_acc_percent": 100.0
            * stats["future_count_correct"]
            / future_frames,
        }
    return output


def compute_srp_peak_metrics(
    dataset: RealMANRing2Dataset,
    *,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
    cfg: Any,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    include_center = bool(cfg.model.get("include_center_mic", False))
    num_input = int(cfg.model.num_input_channels)
    ring_mics = num_input - 1 if include_center else num_input
    mic_positions = make_uniform_circular_array(
        num_mics=ring_mics,
        radius=float(cfg.model.array_radius_m),
        include_center=include_center,
        device=device,
    )
    stft = MultiChannelSTFT(
        n_fft=int(cfg.feature.n_fft),
        win_length=int(cfg.feature.win_length),
        hop_length=int(cfg.feature.hop_length),
    ).to(device)
    srp = SRPPHAT(
        sample_rate=int(cfg.feature.sample_rate),
        n_fft=int(cfg.feature.n_fft),
        mic_positions=mic_positions,
        mic_pairs=infer_mic_pairs(mic_positions),
        azimuths=azimuth_grid(int(cfg.model.heatmap_bins), device=device),
        sound_speed=float(cfg.model.sound_speed),
    ).to(device)

    groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            waveform = batch["waveform"].to(device)
            target_angle, target_valid = target_slot_primary_azimuth_deg(batch["slot_state"].to(device))
            srp_map = srp(stft(waveform))
            srp_peak = heatmap_logits_to_azimuth_deg(srp_map)
            # SRP-PHAT 的远场 steering 方向约定与 dataset GT 坐标系相反 (差 180°),
            # 直接解码会得到镜像方位. 这里加 180° 并 wrap 回 [-180,180), 与 GT 对齐.
            srp_peak = wrap_angle_deg(srp_peak + 180.0)
            sample_ids = batch["sample_id"]
            for item_idx, sample_id in enumerate(sample_ids):
                motion = str(sample_id).split(":")[1] if ":" in str(sample_id) else "unknown"
                if not bool(target_valid[item_idx].item()):
                    continue
                item_err = circular_abs_error_deg(
                    srp_peak[item_idx : item_idx + 1],
                    target_angle[item_idx : item_idx + 1],
                )
                groups["all"]["frames"] += float(item_err.numel())
                groups["all"]["heat_abs_err_sum"] += float(item_err.sum().item())
                groups["all"]["heat_acc5_sum"] += float((item_err <= 5.0).sum().item())
                groups["all"]["samples"] += 1.0
                groups[motion]["frames"] += float(item_err.numel())
                groups[motion]["heat_abs_err_sum"] += float(item_err.sum().item())
                groups[motion]["heat_acc5_sum"] += float((item_err <= 5.0).sum().item())
                groups[motion]["samples"] += 1.0

    srp_output: dict[str, dict[str, float]] = {}
    for name, stats in groups.items():
        frames = max(stats["frames"], 1.0)
        srp_output[name] = {
            "num_samples": stats["samples"],
            "heatmap_argmax_mae_deg": stats["heat_abs_err_sum"] / frames,
            "heatmap_argmax_acc5_percent": 100.0 * stats["heat_acc5_sum"] / frames,
        }
    return srp_output


def build_dataset(cfg: Any, split: str) -> RealMANRing2Dataset:
    return RealMANRing2Dataset(
        root_dir=cfg.data.root_dir,
        moving_csv=cfg.data.moving_csv,
        static_csv=cfg.data.static_csv,
        channel_ids=tuple(int(value) for value in cfg.data.channel_ids),
        model_sample_rate=int(cfg.data.model_sample_rate),
        history_frames=int(cfg.model.history_frames),
        future_frames=int(cfg.model.future_frames),
        hop_length=int(cfg.feature.hop_length),
        max_sources=int(cfg.model.max_sources),
        num_heatmap_bins=int(cfg.model.heatmap_bins),
        split=split,
        split_mode=str(cfg.data.get("split_mode", "hash")),
        val_ratio=float(cfg.data.val_ratio),
        split_seed=int(cfg.data.split_seed),
        use_manifest_cache=bool(cfg.data.use_manifest_cache),
        manifest_path=cfg.data.manifest_path,
        audio_cache_dir=cfg.data.get("audio_cache_dir"),
        feature_cache_dir=cfg.data.get("feature_cache_dir"),
        sample_cache_dir=cfg.data.get("sample_cache_dir"),
        max_items=cfg.data.get("max_items"),
    )


def build_model(cfg: Any, device: torch.device) -> UCA8TrackTrendNet:
    include_center = bool(cfg.model.get("include_center_mic", False))
    num_input = int(cfg.model.num_input_channels)
    ring_mics = num_input - 1 if include_center else num_input
    mic_positions = make_uniform_circular_array(
        num_mics=ring_mics,
        radius=float(cfg.model.array_radius_m),
        include_center=include_center,
        device=device,
    )
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
        tcn_dilations=tuple(int(value) for value in cfg.model.tcn_dilations),
        tcn_kernel_size=int(cfg.model.tcn_kernel_size),
        dropout=float(cfg.model.dropout),
        slot_decoder_attention_heads=int(cfg.model.get("slot_decoder_attention_heads", 4)),
        future_decoder_layers=int(cfg.model.get("future_decoder_layers", 2)),
        future_decoder_dropout=float(cfg.model.get("future_decoder_dropout", 0.1)),
        use_slot_context_in_future_decoder=bool(
            cfg.model.get("use_slot_context_in_future_decoder", True)
        ),
        num_count_classes=int(cfg.model.num_count_classes),
        sound_speed=float(cfg.model.sound_speed),
        feature_cache_dir=cfg.data.get("feature_cache_dir"),
    ).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.run_dir / "config_resolved.yaml")
    dataset = build_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    checkpoint_path = args.checkpoint or args.run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            waveform = batch["waveform"].to(device)
            vad_history = batch["vad_history"].to(device)
            predictions = model(
                waveform,
                vad_history=vad_history,
                sample_id=batch.get("sample_id"),
            )

            target_angle, target_valid = target_slot_primary_azimuth_deg(
                batch["slot_state"].to(device)
            )
            heat_pred_angle = heatmap_logits_to_azimuth_deg(predictions["heatmap_logits"])
            slot_pred_angle, slot_pred_valid = slot_logits_to_primary_azimuth_deg(
                predictions["slot_logits"]
            )

            valid = target_valid
            heat_err = circular_abs_error_deg(heat_pred_angle[valid], target_angle[valid])
            slot_err_all = circular_abs_error_deg(slot_pred_angle[valid], target_angle[valid])
            slot_err = torch.where(
                slot_pred_valid[valid],
                slot_err_all,
                torch.full_like(slot_err_all, 180.0),
            )
            heat_acc5 = heat_err <= 5.0
            slot_acc5 = slot_err <= 5.0

            count_target = batch["count"].to(device)
            count_head_pred = predictions["count_logits"].argmax(dim=-1)
            slot_count_pred = slot_count_from_state(
                predictions["slot_logits"],
                is_logits=True,
            )
            future_count_pred = predictions["future_count_logits"].argmax(dim=-1)
            future_count_target = batch["future_count"].to(device)

            motions = batch["sample_id"]
            for item_idx, sample_id in enumerate(motions):
                motion = str(sample_id).split(":")[1] if ":" in str(sample_id) else "unknown"
                item_valid = valid[item_idx : item_idx + 1]
                if not bool(item_valid.item()):
                    continue
                item_heat_err = circular_abs_error_deg(
                    heat_pred_angle[item_idx : item_idx + 1],
                    target_angle[item_idx : item_idx + 1],
                )
                item_slot_err_all = circular_abs_error_deg(
                    slot_pred_angle[item_idx : item_idx + 1],
                    target_angle[item_idx : item_idx + 1],
                )
                item_slot_err = torch.where(
                    slot_pred_valid[item_idx : item_idx + 1],
                    item_slot_err_all,
                    torch.full_like(item_slot_err_all, 180.0),
                )
                kwargs = {
                    "heat_err": item_heat_err,
                    "slot_err": item_slot_err,
                    "heat_acc5": item_heat_err <= 5.0,
                    "slot_acc5": item_slot_err <= 5.0,
                    "count_head_correct": count_head_pred[item_idx : item_idx + 1]
                    == count_target[item_idx : item_idx + 1],
                    "slot_count_correct": slot_count_pred[item_idx : item_idx + 1]
                    == count_target[item_idx : item_idx + 1],
                    "future_count_correct": future_count_pred[item_idx]
                    == future_count_target[item_idx],
                    "future_total": int(future_count_target.shape[1]),
                }
                update_group(groups, "all", **kwargs)
                update_group(groups, motion, **kwargs)

    metrics = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "dataset_records": len(dataset),
        "max_batches": args.max_batches,
        "srp_peak_baseline": compute_srp_peak_metrics(
            dataset,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=device,
            cfg=cfg,
        ),
        "metrics": finalize(groups),
        "comparability_notes": [
            "RealMAN official localization baseline reports azimuth MAE and ACC(5deg), but on its official train/val/test protocol and baseline arrays; this run uses the local ring2_8ch subset and the repository split.",
            "Official CRNN/IPDnet metrics are current-frame azimuth localization; our model also predicts count, slot state, future state, and motion trend, so only current-frame azimuth MAE/ACC@5deg is directly analogous.",
            "This evaluation uses heatmap argmax as the closest counterpart to RealMAN spatial-spectrum localization. Slot metrics are reported separately because slot assignment is an extra task and is not directly comparable to official baseline.",
            "The available checkpoint was trained for a short local run, not a paper-grade full training schedule.",
        ],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
