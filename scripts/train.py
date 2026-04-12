from __future__ import annotations

import json
import random
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uca8.data.dataset_tracktrend import LocataLikeTrackTrendDataset, SyntheticTrackTrendDataset
from uca8.data.realman_ring2_dataset import RealMANRing2Dataset
from uca8.data.realman_ring2_hybrid_dataset import RealMANRing2HybridDataset
from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.losses.multi_task_loss import TrackTrendMultiTaskLoss
from uca8.metrics import (
    future_slot_delta_error_stats_deg,
    slot_activity_confusion_stats,
    slot_angle_error_stats_deg,
    slot_count_accuracy_stats,
)
from uca8.models.tracktrend_net import UCA8TrackTrendNet
from uca8.sim import (
    build_probe_rollout_samples,
    default_source_audio,
    evaluate_probe_suite,
    load_probe_mono_audio,
    summarize_probe_suite,
)

ROOT = Path(__file__).resolve().parents[1]

try:
    import hydra
    from lightning.fabric import Fabric
    from omegaconf import DictConfig, OmegaConf
except Exception as exc:  # pragma: no cover - exercised only in runtime environments with deps
    raise RuntimeError(
        "scripts/train.py requires hydra-core and lightning to be installed."
    ) from exc


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def is_primary_process(fabric: Fabric) -> bool:
    return bool(getattr(fabric, "is_global_zero", True))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_dataset(
    cfg: DictConfig,
    *,
    split_override: str | None = None,
) -> torch.utils.data.Dataset:
    if cfg.data.dataset_kind == "synthetic":
        return SyntheticTrackTrendDataset(
            size=cfg.data.synthetic_size,
            num_channels=cfg.model.num_input_channels,
            history_frames=cfg.model.history_frames,
            future_frames=cfg.model.future_frames,
            sample_rate=cfg.data.model_sample_rate,
            hop_length=cfg.feature.hop_length,
            max_sources=cfg.model.max_sources,
            heatmap_bins=cfg.model.heatmap_bins,
        )
    if cfg.data.dataset_kind == "realman_ring2":
        return RealMANRing2Dataset(
            root_dir=Path(cfg.data.root_dir),
            moving_csv=Path(cfg.data.moving_csv),
            static_csv=Path(cfg.data.static_csv),
            channel_ids=tuple(cfg.data.channel_ids),
            model_sample_rate=cfg.data.model_sample_rate,
            history_frames=cfg.model.history_frames,
            future_frames=cfg.model.future_frames,
            hop_length=cfg.feature.hop_length,
            max_sources=cfg.model.max_sources,
            num_heatmap_bins=cfg.model.heatmap_bins,
            split=split_override or str(cfg.data.get("split", "all")),
            val_ratio=float(cfg.data.get("val_ratio", 0.15)),
            split_seed=int(cfg.data.get("split_seed", cfg.seed)),
            use_manifest_cache=bool(cfg.data.get("use_manifest_cache", True)),
            manifest_path=cfg.data.get("manifest_path"),
            audio_cache_dir=cfg.data.get("audio_cache_dir"),
            max_items=cfg.data.get("max_items"),
        )
    if cfg.data.dataset_kind == "realman_ring2_hybrid":
        if split_override == "val":
            return RealMANRing2Dataset(
                root_dir=Path(cfg.data.root_dir),
                moving_csv=Path(cfg.data.moving_csv),
                static_csv=Path(cfg.data.static_csv),
                channel_ids=tuple(cfg.data.channel_ids),
                model_sample_rate=cfg.data.model_sample_rate,
                history_frames=cfg.model.history_frames,
                future_frames=cfg.model.future_frames,
                hop_length=cfg.feature.hop_length,
                max_sources=cfg.model.max_sources,
                num_heatmap_bins=cfg.model.heatmap_bins,
                split="val",
                val_ratio=float(cfg.data.get("val_ratio", 0.15)),
                split_seed=int(cfg.data.get("split_seed", cfg.seed)),
                use_manifest_cache=bool(cfg.data.get("use_manifest_cache", True)),
                manifest_path=cfg.data.get("manifest_path"),
                audio_cache_dir=cfg.data.get("audio_cache_dir"),
                max_items=cfg.data.get("max_items"),
            )
        return RealMANRing2HybridDataset(
            root_dir=Path(cfg.data.root_dir),
            moving_csv=Path(cfg.data.moving_csv),
            static_csv=Path(cfg.data.static_csv),
            channel_ids=tuple(cfg.data.channel_ids),
            model_sample_rate=cfg.data.model_sample_rate,
            history_frames=cfg.model.history_frames,
            future_frames=cfg.model.future_frames,
            hop_length=cfg.feature.hop_length,
            win_length=cfg.feature.win_length,
            max_sources=cfg.model.max_sources,
            num_heatmap_bins=cfg.model.heatmap_bins,
            num_input_channels=cfg.model.num_input_channels,
            array_radius_m=float(cfg.model.array_radius_m),
            sound_speed=float(cfg.model.sound_speed),
            split=split_override or str(cfg.data.get("split", "train")),
            val_ratio=float(cfg.data.get("val_ratio", 0.15)),
            split_seed=int(cfg.data.get("split_seed", cfg.seed)),
            use_manifest_cache=bool(cfg.data.get("use_manifest_cache", True)),
            manifest_path=cfg.data.get("manifest_path"),
            audio_cache_dir=cfg.data.get("audio_cache_dir"),
            max_items=cfg.data.get("max_items"),
            curriculum_ratio=float(cfg.data.get("curriculum_ratio", 1.0)),
            curriculum_size=cfg.data.get("curriculum_size"),
            curriculum_seed=int(cfg.data.get("curriculum_seed", cfg.seed)),
            curriculum_rollout_steps=int(cfg.data.get("curriculum_rollout_steps", 1)),
            curriculum_mode_weights=OmegaConf.to_container(
                cfg.data.get("curriculum_mode_weights", {}),
                resolve=True,
            ),
        )
    return LocataLikeTrackTrendDataset(
        root_dir=Path(cfg.data.root_dir),
        history_frames=cfg.model.history_frames,
        future_frames=cfg.model.future_frames,
        window_stride_frames=cfg.data.window_stride_frames,
        model_sample_rate=cfg.data.model_sample_rate,
        num_input_channels=cfg.model.num_input_channels,
        max_sources=cfg.model.max_sources,
        frame_hop_seconds=cfg.data.frame_hop_seconds,
        num_heatmap_bins=cfg.model.heatmap_bins,
        audio_cache_dir=cfg.data.get("audio_cache_dir"),
    )


def should_build_validation(cfg: DictConfig) -> bool:
    return bool(cfg.train.get("run_validation", False)) and cfg.data.dataset_kind in {
        "realman_ring2",
        "realman_ring2_hybrid",
    }


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    cfg: DictConfig,
    *,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(cfg.train.num_workers)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(
            cfg.train.batch_size if train else cfg.train.get("val_batch_size", cfg.train.batch_size)
        ),
        "shuffle": train,
        "num_workers": num_workers,
        "generator": generator,
        "pin_memory": bool(cfg.train.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.train.get("persistent_workers", False))
        prefetch_factor = cfg.train.get("prefetch_factor")
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def build_probe_samples(
    cfg: DictConfig,
    *,
    mic_positions: torch.Tensor,
) -> dict[str, list[Any]]:
    if not bool(cfg.train.get("probe_validation", False)):
        return {}
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
    probe_scenarios = [str(name) for name in cfg.train.get("probe_scenarios", [])]
    if not probe_scenarios:
        return {}
    probe_animation_steps = int(cfg.train.get("probe_animation_steps", 24))
    return {
        scenario: build_probe_rollout_samples(
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
            animation_steps=probe_animation_steps,
        )
        for scenario in probe_scenarios
    }


def prepare_run_dir(cfg: DictConfig) -> Path:
    resume_from = cfg.train.get("resume_from")
    if resume_from:
        run_dir = Path(str(resume_from)).resolve().parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    output_root = Path(str(cfg.train.get("output_root", ROOT / "runs")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / str(cfg.experiment_name) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def evaluate(
    *,
    model: torch.nn.Module,
    criterion: TrackTrendMultiTaskLoss,
    dataloader: DataLoader,
    phase: str,
) -> dict[str, float]:
    model.eval()
    metric_keys = (
        "loss",
        "count_loss",
        "heat_loss",
        "current_heat_kl",
        "slot_activity_loss",
        "slot_regression_loss",
        "slot_count_consistency_loss",
        "slot_heat_consistency_loss",
        "track_loss",
        "future_count_loss",
        "future_heat_loss",
        "future_heat_kl",
        "future_slot_activity_loss",
        "future_slot_regression_loss",
        "future_slot_count_consistency_loss",
        "future_slot_delta_loss",
        "future_track_loss",
        "future_loss",
        "motion_loss",
    )
    totals = {key: 0.0 for key in metric_keys}
    total_samples = 0
    count_correct = 0
    count_head_correct = 0
    trend_correct = 0
    future_count_correct = 0
    future_count_head_correct = 0
    future_count_total = 0
    current_heat_bce = 0.0
    current_slot_angle_error_sum = 0.0
    current_slot_angle_active = 0.0
    future_slot_angle_error_sum = 0.0
    future_slot_angle_active = 0.0
    future_slot_delta_error_sum = 0.0
    future_slot_delta_active = 0.0
    current_slot_tp = 0.0
    current_slot_fp = 0.0
    current_slot_fn = 0.0
    future_slot_tp = 0.0
    future_slot_fp = 0.0
    future_slot_fn = 0.0
    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch["waveform"], batch["vad_history"])
            loss_dict = criterion(predictions, batch)
            batch_size = int(batch["waveform"].shape[0])
            total_samples += batch_size
            for key in metric_keys:
                totals[key] += float(loss_dict[key].item()) * batch_size
            count_head_correct += int(
                (predictions["count_logits"].argmax(dim=-1) == batch["count"]).sum().item()
            )
            slot_count_correct, _ = slot_count_accuracy_stats(
                predictions["slot_logits"],
                batch["slot_state"],
            )
            count_correct += int(slot_count_correct.item())
            trend_correct += int(
                (predictions["motion_logits"].argmax(dim=-1) == batch["trend_class"]).sum().item()
            )
            future_count_pred = predictions["future_count_logits"].argmax(dim=-1)
            future_count_target = batch["future_count"]
            future_count_head_correct += int(
                (future_count_pred == future_count_target).sum().item()
            )
            future_slot_logits = predictions["future_slot_logits"].reshape(
                -1,
                *predictions["future_slot_logits"].shape[-2:],
            )
            future_slot_target = batch["future_slot_state"].reshape(
                -1,
                *batch["future_slot_state"].shape[-2:],
            )
            future_slot_count_correct, future_slot_count_total = slot_count_accuracy_stats(
                future_slot_logits,
                future_slot_target,
            )
            future_count_correct += int(future_slot_count_correct.item())
            future_count_total += int(future_slot_count_total.item())
            current_heat_bce += (
                float(
                    F.binary_cross_entropy(
                        torch.sigmoid(predictions["heatmap_logits"]),
                        batch["heatmap"],
                    ).item()
                )
                * batch_size
            )
            current_angle_sum, current_angle_count = slot_angle_error_stats_deg(
                predictions["slot_logits"],
                batch["slot_state"],
            )
            current_slot_angle_error_sum += float(current_angle_sum.item())
            current_slot_angle_active += float(current_angle_count.item())
            future_angle_sum, future_angle_count = slot_angle_error_stats_deg(
                future_slot_logits,
                future_slot_target,
            )
            future_slot_angle_error_sum += float(future_angle_sum.item())
            future_slot_angle_active += float(future_angle_count.item())
            future_delta_sum, future_delta_count = future_slot_delta_error_stats_deg(
                predictions["future_slot_logits"],
                batch["future_slot_state"],
            )
            future_slot_delta_error_sum += float(future_delta_sum.item())
            future_slot_delta_active += float(future_delta_count.item())
            current_tp, current_fp, current_fn = slot_activity_confusion_stats(
                predictions["slot_logits"],
                batch["slot_state"],
            )
            current_slot_tp += float(current_tp.item())
            current_slot_fp += float(current_fp.item())
            current_slot_fn += float(current_fn.item())
            future_tp, future_fp, future_fn = slot_activity_confusion_stats(
                future_slot_logits,
                future_slot_target,
            )
            future_slot_tp += float(future_tp.item())
            future_slot_fp += float(future_fp.item())
            future_slot_fn += float(future_fn.item())
    model.train()
    sample_denominator = max(total_samples, 1)
    frame_denominator = max(future_count_total, 1)
    metrics = {f"{phase}/{key}": totals[key] / sample_denominator for key in metric_keys}
    current_slot_precision = ratio(current_slot_tp, current_slot_tp + current_slot_fp)
    current_slot_recall = ratio(current_slot_tp, current_slot_tp + current_slot_fn)
    future_slot_precision = ratio(future_slot_tp, future_slot_tp + future_slot_fp)
    future_slot_recall = ratio(future_slot_tp, future_slot_tp + future_slot_fn)
    metrics[f"{phase}/count_acc"] = count_correct / sample_denominator
    metrics[f"{phase}/count_head_acc"] = count_head_correct / sample_denominator
    metrics[f"{phase}/current_slot_count_acc"] = count_correct / sample_denominator
    metrics[f"{phase}/trend_acc"] = trend_correct / sample_denominator
    metrics[f"{phase}/future_count_frame_acc"] = future_count_correct / frame_denominator
    metrics[f"{phase}/future_count_head_frame_acc"] = future_count_head_correct / frame_denominator
    metrics[f"{phase}/future_slot_count_frame_acc"] = future_count_correct / frame_denominator
    metrics[f"{phase}/current_heat_bce"] = current_heat_bce / sample_denominator
    metrics[f"{phase}/current_slot_activity_precision"] = current_slot_precision
    metrics[f"{phase}/current_slot_activity_recall"] = current_slot_recall
    metrics[f"{phase}/current_slot_activity_f1"] = ratio(
        2.0 * current_slot_precision * current_slot_recall,
        current_slot_precision + current_slot_recall,
    )
    metrics[f"{phase}/future_slot_activity_precision"] = future_slot_precision
    metrics[f"{phase}/future_slot_activity_recall"] = future_slot_recall
    metrics[f"{phase}/future_slot_activity_f1"] = ratio(
        2.0 * future_slot_precision * future_slot_recall,
        future_slot_precision + future_slot_recall,
    )
    metrics[f"{phase}/current_slot_angle_mae_deg"] = current_slot_angle_error_sum / max(
        current_slot_angle_active,
        1.0,
    )
    metrics[f"{phase}/future_slot_angle_mae_deg"] = future_slot_angle_error_sum / max(
        future_slot_angle_active,
        1.0,
    )
    metrics[f"{phase}/future_slot_delta_mae_deg"] = future_slot_delta_error_sum / max(
        future_slot_delta_active,
        1.0,
    )
    return metrics


def should_improve(
    current_value: float,
    best_value: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value - min_delta
    if mode == "max":
        return current_value > best_value + min_delta
    raise ValueError(f"Unsupported metric mode: {mode}")


def metrics_are_tied(
    current_value: float,
    best_value: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return False
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported metric mode: {mode}")
    return abs(current_value - best_value) <= min_delta


def save_checkpoint(
    *,
    path: Path,
    fabric: Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    last_loss: float,
    best_metric: float | None,
    best_secondary_metric: float | None,
    stale_validation_rounds: int,
) -> None:
    if not is_primary_process(fabric):
        return
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "last_loss": last_loss,
        "best_metric": best_metric,
        "best_secondary_metric": best_secondary_metric,
        "stale_validation_rounds": stale_validation_rounds,
        "rng_state": capture_rng_state(),
    }
    torch.save(state, path)


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    move_optimizer_state(optimizer, device)
    restore_rng_state(state.get("rng_state"))
    return state


def guard_resume_target(
    *,
    run_dir: Path,
    checkpoint_state: dict[str, Any],
    force_resume: bool,
) -> None:
    if force_resume:
        return
    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.exists():
        return
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    recorded_step = int(run_summary.get("global_step", 0))
    checkpoint_step = int(checkpoint_state.get("global_step", 0))
    if recorded_step > checkpoint_step:
        raise RuntimeError(
            "Refusing to resume from an older checkpoint into a run directory that already "
            "contains newer progress. Set train.force_resume=true to override."
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    deterministic = bool(cfg.runtime.get("deterministic", False))
    matmul_precision = cfg.runtime.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    seed_everything(int(cfg.seed), deterministic=deterministic)
    run_dir = prepare_run_dir(cfg)
    fabric = Fabric(
        accelerator=cfg.runtime.accelerator,
        devices=cfg.runtime.devices,
        precision=cfg.runtime.precision,
    )
    fabric.launch()
    train_split = "train" if should_build_validation(cfg) else None
    train_dataset = build_dataset(cfg, split_override=train_split)
    val_dataset = build_dataset(cfg, split_override="val") if should_build_validation(cfg) else None
    if is_primary_process(fabric):
        train_summary = getattr(
            train_dataset,
            "summary",
            lambda: {"num_records": len(train_dataset)},
        )()
        (run_dir / "config_resolved.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True),
            encoding="utf-8",
        )
        dataset_summary = {"train": train_summary}
        if val_dataset is not None:
            dataset_summary["val"] = getattr(
                val_dataset,
                "summary",
                lambda: {"num_records": len(val_dataset)},
            )()
        (run_dir / "dataset_summary.json").write_text(
            json.dumps(dataset_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    mic_positions = make_uniform_circular_array(
        num_mics=cfg.model.num_input_channels,
        radius=float(cfg.model.array_radius_m),
    )
    model = UCA8TrackTrendNet(
        mic_positions=mic_positions,
        sample_rate=cfg.feature.sample_rate,
        n_fft=cfg.feature.n_fft,
        win_length=cfg.feature.win_length,
        hop_length=cfg.feature.hop_length,
        spec_bins=cfg.feature.spec_bins,
        ipd_bins=cfg.feature.ipd_bins,
        heatmap_bins=cfg.model.heatmap_bins,
        history_frames=cfg.model.history_frames,
        future_frames=cfg.model.future_frames,
        max_sources=cfg.model.max_sources,
        spec_hidden_dim=cfg.model.spec_hidden_dim,
        spatial_hidden_dim=cfg.model.spatial_hidden_dim,
        model_dim=cfg.model.model_dim,
        tcn_dilations=list(cfg.model.tcn_dilations),
        tcn_kernel_size=cfg.model.tcn_kernel_size,
        dropout=cfg.model.dropout,
        slot_decoder_attention_heads=int(cfg.model.get("slot_decoder_attention_heads", 4)),
        future_decoder_layers=int(cfg.model.get("future_decoder_layers", 2)),
        future_decoder_dropout=float(cfg.model.get("future_decoder_dropout", cfg.model.dropout)),
        use_slot_context_in_future_decoder=bool(
            cfg.model.get("use_slot_context_in_future_decoder", True)
        ),
        num_count_classes=cfg.model.num_count_classes,
        sound_speed=cfg.model.sound_speed,
    )
    criterion = TrackTrendMultiTaskLoss(
        count_weight=float(cfg.train.get("loss_count_weight", 1.0)),
        heat_weight=float(cfg.train.get("loss_heat_weight", 1.0)),
        heat_pos_weight=float(cfg.train.get("loss_heat_pos_weight", 1.0)),
        track_weight=float(cfg.train.get("loss_track_weight", 2.0)),
        future_count_weight=float(cfg.train.get("loss_future_count_weight", 1.5)),
        future_heat_weight=float(cfg.train.get("loss_future_heat_weight", 1.5)),
        future_heat_pos_weight=float(cfg.train.get("loss_future_heat_pos_weight", 1.0)),
        future_track_weight=float(cfg.train.get("loss_future_track_weight", 1.5)),
        slot_activity_pos_weight=float(cfg.train.get("loss_slot_activity_pos_weight", 1.0)),
        slot_activity_neg_weight=float(cfg.train.get("loss_slot_activity_neg_weight", 1.0)),
        future_slot_activity_pos_weight=float(
            cfg.train.get("loss_future_slot_activity_pos_weight", 1.0)
        ),
        future_slot_activity_neg_weight=float(
            cfg.train.get("loss_future_slot_activity_neg_weight", 1.0)
        ),
        future_slot_deactivate_weight=float(
            cfg.train.get("loss_future_slot_deactivate_weight", 1.0)
        ),
        slot_count_consistency_weight=float(
            cfg.train.get("loss_slot_count_consistency_weight", 0.0)
        ),
        future_slot_count_consistency_weight=float(
            cfg.train.get("loss_future_slot_count_consistency_weight", 0.0)
        ),
        future_slot_count_transition_weight=float(
            cfg.train.get("loss_future_slot_count_transition_weight", 1.0)
        ),
        motion_weight=float(cfg.train.get("loss_motion_weight", 0.5)),
        current_heat_kl_weight=float(cfg.train.get("loss_current_heat_kl_weight", 0.5)),
        future_heat_kl_weight=float(cfg.train.get("loss_future_heat_kl_weight", 0.75)),
        future_slot_delta_weight=float(cfg.train.get("loss_future_slot_delta_weight", 1.0)),
        slot_heat_consistency_weight=float(
            cfg.train.get("loss_slot_heat_consistency_weight", 0.5)
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    model, optimizer = fabric.setup(model, optimizer)
    val_loader: DataLoader | None = None
    if val_dataset is not None:
        val_loader = fabric.setup_dataloaders(
            build_dataloader(
                val_dataset,
                cfg,
                train=False,
                seed=int(cfg.seed) + 100_000,
            )
        )
    history_path = run_dir / "train_history.jsonl"
    global_step = 0
    epoch = 0
    start_epoch = 0
    start_step_in_epoch = 0
    last_loss = 0.0
    best_metric: float | None = None
    best_secondary_metric: float | None = None
    stale_validation_rounds = 0
    resume_from = cfg.train.get("resume_from")
    if resume_from:
        checkpoint_state = load_checkpoint(
            path=Path(str(resume_from)),
            model=model,
            optimizer=optimizer,
            device=fabric.device,
        )
        guard_resume_target(
            run_dir=run_dir,
            checkpoint_state=checkpoint_state,
            force_resume=bool(cfg.train.get("force_resume", False)),
        )
        start_epoch = int(checkpoint_state.get("epoch", 0))
        start_step_in_epoch = int(checkpoint_state.get("step_in_epoch", 0))
        global_step = int(checkpoint_state.get("global_step", 0))
        last_loss = float(checkpoint_state.get("last_loss", 0.0))
        best_metric = checkpoint_state.get("best_metric")
        best_secondary_metric = checkpoint_state.get("best_secondary_metric")
        stale_validation_rounds = int(checkpoint_state.get("stale_validation_rounds", 0))
        fabric.print(f"resumed_from={Path(str(resume_from)).resolve()}")
    log_every_n_steps = int(cfg.train.get("log_every_n_steps", 1))
    save_every_n_steps = int(cfg.train.get("save_every_n_steps", 0))
    validate_every_n_steps = int(cfg.train.get("validate_every_n_steps", 0))
    limit_train_steps = int(cfg.train.limit_train_steps)
    metric_name = str(cfg.train.get("metric_for_best_checkpoint", "val/loss"))
    metric_mode = str(cfg.train.get("metric_mode", "min"))
    secondary_metric_name = cfg.train.get("secondary_metric_for_best_checkpoint")
    secondary_metric_mode = str(cfg.train.get("secondary_metric_mode", metric_mode))
    early_stopping_patience = int(cfg.train.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.train.get("early_stopping_min_delta", 0.0))
    fabric.print(f"run_dir={run_dir}")
    probe_samples = build_probe_samples(cfg, mic_positions=mic_positions.cpu())
    if is_primary_process(fabric) and probe_samples:
        (run_dir / "probe_suite_summary.json").write_text(
            json.dumps(
                summarize_probe_suite(probe_samples),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    stop_training = False
    model.train()
    last_step_in_epoch = start_step_in_epoch
    for epoch in range(start_epoch, int(cfg.train.max_epochs)):
        train_loader = fabric.setup_dataloaders(
            build_dataloader(
                train_dataset,
                cfg,
                train=True,
                seed=int(cfg.seed) + epoch,
            )
        )
        epoch_iterator = (
            islice(train_loader, start_step_in_epoch, None)
            if epoch == start_epoch and start_step_in_epoch > 0
            else train_loader
        )
        for step_in_epoch, batch in enumerate(epoch_iterator, start=start_step_in_epoch):
            last_step_in_epoch = step_in_epoch + 1
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch["waveform"], batch["vad_history"])
            loss_dict = criterion(predictions, batch)
            fabric.backward(loss_dict["loss"])
            fabric.clip_gradients(model, optimizer, max_norm=float(cfg.train.gradient_clip_val))
            optimizer.step()
            global_step += 1
            last_loss = float(loss_dict["loss"].detach().item())
            train_metrics = {
                "phase": "train",
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "loss": last_loss,
                "count_loss": float(loss_dict["count_loss"].item()),
                "heat_loss": float(loss_dict["heat_loss"].item()),
                "slot_activity_loss": float(loss_dict["slot_activity_loss"].item()),
                "slot_regression_loss": float(loss_dict["slot_regression_loss"].item()),
                "slot_count_consistency_loss": float(
                    loss_dict["slot_count_consistency_loss"].item()
                ),
                "track_loss": float(loss_dict["track_loss"].item()),
                "future_count_loss": float(loss_dict["future_count_loss"].item()),
                "future_heat_loss": float(loss_dict["future_heat_loss"].item()),
                "future_slot_activity_loss": float(loss_dict["future_slot_activity_loss"].item()),
                "future_slot_regression_loss": float(
                    loss_dict["future_slot_regression_loss"].item()
                ),
                "future_slot_count_consistency_loss": float(
                    loss_dict["future_slot_count_consistency_loss"].item()
                ),
                "future_track_loss": float(loss_dict["future_track_loss"].item()),
                "future_loss": float(loss_dict["future_loss"].item()),
                "motion_loss": float(loss_dict["motion_loss"].item()),
            }
            if is_primary_process(fabric):
                append_metrics(history_path, train_metrics)
            if global_step == 1 or global_step % log_every_n_steps == 0:
                fabric.print(
                    f"epoch={epoch} step={step_in_epoch} global_step={global_step} "
                    f"loss={train_metrics['loss']:.4f} "
                    f"count={train_metrics['count_loss']:.4f} "
                    f"slot_act={train_metrics['slot_activity_loss']:.4f} "
                    f"slot_reg={train_metrics['slot_regression_loss']:.4f} "
                    f"slot_cnt={train_metrics['future_slot_count_consistency_loss']:.4f} "
                    f"future={train_metrics['future_loss']:.4f}"
                )
            should_run_validation = val_loader is not None and (
                (validate_every_n_steps > 0 and global_step % validate_every_n_steps == 0)
                or global_step == limit_train_steps
            )
            if should_run_validation:
                val_metrics = evaluate(
                    model=model,
                    criterion=criterion,
                    dataloader=val_loader,
                    phase="val",
                )
                if probe_samples:
                    val_metrics.update(
                        evaluate_probe_suite(
                            model=model,
                            probe_samples=probe_samples,
                            device=fabric.device,
                        )
                    )
                val_metrics.update(
                    {
                        "phase": "val",
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                    }
                )
                if is_primary_process(fabric):
                    append_metrics(history_path, val_metrics)
                if metric_name not in val_metrics:
                    raise KeyError(f"Metric {metric_name} not found in validation metrics.")
                current_metric = float(val_metrics[metric_name])
                improve_best = should_improve(
                    current_metric,
                    best_metric,
                    mode=metric_mode,
                    min_delta=early_stopping_min_delta,
                )
                if (
                    not improve_best
                    and secondary_metric_name
                    and metrics_are_tied(
                        current_metric,
                        best_metric,
                        mode=metric_mode,
                        min_delta=early_stopping_min_delta,
                    )
                ):
                    if secondary_metric_name not in val_metrics:
                        raise KeyError(
                            f"Metric {secondary_metric_name} not found in validation metrics."
                        )
                    improve_best = should_improve(
                        float(val_metrics[secondary_metric_name]),
                        best_secondary_metric,
                        mode=secondary_metric_mode,
                        min_delta=early_stopping_min_delta,
                    )
                if improve_best:
                    best_metric = current_metric
                    if secondary_metric_name:
                        best_secondary_metric = float(val_metrics[secondary_metric_name])
                    stale_validation_rounds = 0
                    save_checkpoint(
                        path=run_dir / "best.pt",
                        fabric=fabric,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch + 1,
                        global_step=global_step,
                        last_loss=last_loss,
                        best_metric=best_metric,
                        best_secondary_metric=best_secondary_metric,
                        stale_validation_rounds=stale_validation_rounds,
                    )
                else:
                    stale_validation_rounds += 1
                fabric.print(
                    f"val_loss={val_metrics['val/loss']:.4f} "
                    f"val_count_acc={val_metrics['val/count_acc']:.4f} "
                    f"val_count_head_acc={val_metrics['val/count_head_acc']:.4f} "
                    f"val_future_slot_count={val_metrics['val/future_slot_count_frame_acc']:.4f} "
                    f"val_slot_f1={val_metrics['val/current_slot_activity_f1']:.4f} "
                    f"probe_score={val_metrics.get('probe/checkpoint_score', 0.0):.4f} "
                    f"probe_geom={val_metrics.get('probe/geometry_checkpoint_score', 0.0):.4f} "
                    f"val_trend_acc={val_metrics['val/trend_acc']:.4f} "
                    f"val_slot_deg={val_metrics['val/current_slot_angle_mae_deg']:.2f} "
                    f"val_future_delta_deg={val_metrics['val/future_slot_delta_mae_deg']:.2f}"
                )
                if (
                    early_stopping_patience > 0
                    and stale_validation_rounds >= early_stopping_patience
                ):
                    stop_training = True
                    break
            if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                save_checkpoint(
                    path=run_dir / f"step_{global_step:05d}.pt",
                    fabric=fabric,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch + 1,
                    global_step=global_step,
                    last_loss=last_loss,
                    best_metric=best_metric,
                    best_secondary_metric=best_secondary_metric,
                    stale_validation_rounds=stale_validation_rounds,
                )
            if global_step >= limit_train_steps:
                stop_training = True
                break
        start_step_in_epoch = 0
        if stop_training:
            break
    save_checkpoint(
        path=run_dir / "last.pt",
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step_in_epoch=last_step_in_epoch,
        global_step=global_step,
        last_loss=last_loss,
        best_metric=best_metric,
        best_secondary_metric=best_secondary_metric,
        stale_validation_rounds=stale_validation_rounds,
    )
    if is_primary_process(fabric):
        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "last_loss": last_loss,
                    "best_metric": best_metric,
                    "best_secondary_metric": best_secondary_metric,
                    "resume_from": str(resume_from) if resume_from else None,
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    fabric.print(f"saved_checkpoint={run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
