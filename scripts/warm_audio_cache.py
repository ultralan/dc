from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from uca8.data.dataset_tracktrend import LocataLikeTrackTrendDataset
from uca8.data.realman_ring2_dataset import RealMANRing2Dataset
from uca8.utils.audio_io import load_audio_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute decoded/resampled audio cache.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="realman_ring2_demo",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--limit-files", type=int, default=0)
    return parser.parse_args()


def load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT.resolve())):
        return hydra.compose(config_name=config_name, overrides=overrides)


def resolve_runtime_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value).replace("${hydra:runtime.cwd}", str(ROOT)))


def build_dataset(cfg: DictConfig) -> torch.utils.data.Dataset:
    data_cfg = dict(OmegaConf.to_container(cfg.data, resolve=False))
    if cfg.data.dataset_kind == "realman_ring2":
        return RealMANRing2Dataset(
            root_dir=resolve_runtime_path(data_cfg["root_dir"]),
            moving_csv=resolve_runtime_path(data_cfg["moving_csv"]),
            static_csv=resolve_runtime_path(data_cfg["static_csv"]),
            channel_ids=tuple(cfg.data.channel_ids),
            model_sample_rate=int(cfg.data.model_sample_rate),
            history_frames=int(cfg.model.history_frames),
            future_frames=int(cfg.model.future_frames),
            hop_length=int(cfg.feature.hop_length),
            max_sources=int(cfg.model.max_sources),
            num_heatmap_bins=int(cfg.model.heatmap_bins),
            split=str(cfg.data.get("split", "all")),
            val_ratio=float(cfg.data.get("val_ratio", 0.15)),
            split_seed=int(cfg.seed),
            use_manifest_cache=bool(cfg.data.get("use_manifest_cache", True)),
            manifest_path=resolve_runtime_path(data_cfg.get("manifest_path")),
            audio_cache_dir=resolve_runtime_path(data_cfg.get("audio_cache_dir")),
            max_items=cfg.data.get("max_items"),
        )
    if cfg.data.dataset_kind == "locata_like":
        return LocataLikeTrackTrendDataset(
            root_dir=resolve_runtime_path(data_cfg["root_dir"]),
            history_frames=int(cfg.model.history_frames),
            future_frames=int(cfg.model.future_frames),
            window_stride_frames=int(cfg.data.window_stride_frames),
            model_sample_rate=int(cfg.data.model_sample_rate),
            num_input_channels=int(cfg.model.num_input_channels),
            max_sources=int(cfg.model.max_sources),
            frame_hop_seconds=float(cfg.data.frame_hop_seconds),
            num_heatmap_bins=int(cfg.model.heatmap_bins),
            audio_cache_dir=resolve_runtime_path(data_cfg.get("audio_cache_dir")),
        )
    raise ValueError(f"Unsupported dataset_kind for cache warming: {cfg.data.dataset_kind}")


def iter_audio_files(dataset: torch.utils.data.Dataset) -> list[Path]:
    if isinstance(dataset, RealMANRing2Dataset):
        paths: set[Path] = set()
        for record in dataset.records:
            paths.add(record.dp_path)
            paths.update(record.channel_paths)
        return sorted(paths)
    if isinstance(dataset, LocataLikeTrackTrendDataset):
        return sorted({package.audio_path for package in dataset.packages})
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_name, list(args.override))
    dataset = build_dataset(cfg)
    raw_data_cfg = dict(OmegaConf.to_container(cfg.data, resolve=False))
    cache_dir = resolve_runtime_path(raw_data_cfg.get("audio_cache_dir"))
    if cache_dir is None:
        raise RuntimeError("Config does not define data.audio_cache_dir.")
    paths = iter_audio_files(dataset)
    if args.limit_files > 0:
        paths = paths[: args.limit_files]
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        load_audio_file(
            path,
            target_sample_rate=int(cfg.data.model_sample_rate),
            cache_dir=cache_dir,
        )
        if index == 1 or index % 50 == 0 or index == total:
            print(f"cached {index}/{total}: {path.name}")
    print(f"audio_cache_ready={cache_dir.resolve()}")


if __name__ == "__main__":
    main()
