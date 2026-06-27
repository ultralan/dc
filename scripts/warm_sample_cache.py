"""预计算整样本缓存: 把每个 RealMAN 样本的完整 __getitem__ 输出落盘成单文件 .pt.

背景: 训练在 ``num_workers=0`` 时, 瓶颈是 ``__getitem__`` 每步读 9 路音频 + 构建标签;
而本机内存不足以支撑多 worker (Windows spawn 每个 worker 复制 ~2GB, 易 OOM).
本脚本一次性把整样本算好存盘, 之后训练直接 ``torch.load`` 单文件, 单进程也能飞快.

用法:
  # 串行预热(默认, 绝不 OOM), 冒烟 50 条:
  uv run python scripts/warm_sample_cache.py --config-name realman_ring2_localization_only \\
    --override data.root_dir=D:/RealMAN/ring2_8ch/extracted \\
               data.sample_cache_dir=D:/RealMAN/ring2_8ch/extracted/.cache/sample_cache \\
    --limit 50

  # 全量串行预热(可断点续, 重跑自动跳过已缓存):
  uv run python scripts/warm_sample_cache.py --config-name realman_ring2_localization_only \\
    --override data.root_dir=... data.sample_cache_dir=...

  # 多进程预热(内存安全: worker 内独立构造, 不复制父进程大对象; N=2 在 6GB 可用内存可行):
  uv run python scripts/warm_sample_cache.py ... --workers 2

注意: 多进程 worker 拿到的是序列化后的单条 record dict + 小参数包, dataset 在 worker 内部
从磁盘读 manifest 构造, 父进程的大 records 列表不经 pickle 复制, 因此内存可控.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from uca8.data.realman_ring2_dataset import RealMANRecord, RealMANRing2Dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute per-sample dataset cache (.pt).")
    parser.add_argument("--config-name", type=str, default="realman_ring2_localization_only")
    parser.add_argument("--override", action="append", default=[], help="Hydra override, repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of samples; 0 = all.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers; 0 = serial.")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT.resolve())):
        return hydra.compose(config_name=config_name, overrides=overrides)


def resolve_runtime_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value).replace("${hydra:runtime.cwd}", str(ROOT)))


def build_dataset(cfg: DictConfig, *, split: str | None = None) -> RealMANRing2Dataset:
    """与 scripts/train.py 一致的 RealMANRing2Dataset 构造, 含 sample_cache_dir."""
    effective_split = split if split is not None else str(cfg.data.get("split", "all"))
    return RealMANRing2Dataset(
        root_dir=resolve_runtime_path(cfg.data.root_dir),
        moving_csv=resolve_runtime_path(cfg.data.moving_csv),
        static_csv=resolve_runtime_path(cfg.data.static_csv),
        channel_ids=tuple(cfg.data.channel_ids),
        model_sample_rate=int(cfg.data.model_sample_rate),
        history_frames=int(cfg.model.history_frames),
        future_frames=int(cfg.model.future_frames),
        hop_length=int(cfg.feature.hop_length),
        max_sources=int(cfg.model.max_sources),
        num_heatmap_bins=int(cfg.model.heatmap_bins),
        split=effective_split,
        val_ratio=float(cfg.data.get("val_ratio", 0.15)),
        split_seed=int(cfg.data.get("split_seed", cfg.seed)),
        use_manifest_cache=bool(cfg.data.get("use_manifest_cache", True)),
        manifest_path=resolve_runtime_path(cfg.data.get("manifest_path")),
        audio_cache_dir=resolve_runtime_path(cfg.data.get("audio_cache_dir")),
        sample_cache_dir=resolve_runtime_path(cfg.data.get("sample_cache_dir")),
        max_items=cfg.data.get("max_items"),
    )


def warm_serial(dataset: RealMANRing2Dataset, total: int, progress_every: int) -> int:
    """串行预热, 命中即跳过(断点续算), 返回本次新建的缓存数."""
    created = 0
    for index in range(total):
        record = dataset.records[index]
        cache_path = dataset._sample_cache_path(record.sample_id)
        if cache_path is not None and cache_path.exists():
            continue  # 命中, 断点续算跳过
        dataset[index]  # 触发 _compute_sample + 落盘
        created += 1
        if created == 1 or created % progress_every == 0 or index + 1 == total:
            print(f"cached {index + 1}/{total}: {record.sample_id}", flush=True)
    return created


def _build_ctor_kwargs(dataset: RealMANRing2Dataset) -> dict[str, Any]:
    """提取 worker 内独立构造 dataset 所需的精简参数(全是可 pickle 基本类型)."""
    return {
        "root_dir": str(dataset.root_dir),
        "moving_csv": str(dataset.moving_csv),
        "static_csv": str(dataset.static_csv),
        "channel_ids": list(dataset.channel_ids),
        "model_sample_rate": dataset.model_sample_rate,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "hop_length": dataset.hop_length,
        "max_sources": dataset.max_sources,
        "num_heatmap_bins": dataset.num_heatmap_bins,
        "val_ratio": dataset.val_ratio,
        "split_seed": dataset.split_seed,
        "use_manifest_cache": dataset.use_manifest_cache,
        "manifest_path": str(dataset.manifest_path) if dataset.manifest_path else None,
        "audio_cache_dir": str(dataset.audio_cache_dir) if dataset.audio_cache_dir else None,
        "sample_cache_dir": str(dataset.sample_cache_dir) if dataset.sample_cache_dir else None,
    }


def _warm_worker(task: tuple[dict[str, Any], dict[str, Any]]) -> str:
    """worker 进程入口: 恢复单条 record, 构造轻量 dataset(只含这一条), 计算+落盘.

    内存安全关键: task 只含序列化后的小对象(record dict + ctor 参数), dataset 在本进程内
    从磁盘读 manifest 构造后立即把 records 覆盖成单条, 不持有父进程的大列表.
    """
    record_entry, ctor_kwargs = task
    root_dir = Path(ctor_kwargs["root_dir"])
    target = RealMANRecord.from_manifest_entry(root_dir, record_entry)
    # split="all" 避免 worker 内重做 split; records 直接覆盖成目标单条, 跳过全量扫描影响.
    dataset = RealMANRing2Dataset(
        root_dir=root_dir,
        moving_csv=Path(ctor_kwargs["moving_csv"]),
        static_csv=Path(ctor_kwargs["static_csv"]),
        channel_ids=tuple(ctor_kwargs["channel_ids"]),
        model_sample_rate=ctor_kwargs["model_sample_rate"],
        history_frames=ctor_kwargs["history_frames"],
        future_frames=ctor_kwargs["future_frames"],
        hop_length=ctor_kwargs["hop_length"],
        max_sources=ctor_kwargs["max_sources"],
        num_heatmap_bins=ctor_kwargs["num_heatmap_bins"],
        split="all",
        val_ratio=ctor_kwargs["val_ratio"],
        split_seed=ctor_kwargs["split_seed"],
        use_manifest_cache=ctor_kwargs["use_manifest_cache"],
        manifest_path=Path(ctor_kwargs["manifest_path"]) if ctor_kwargs["manifest_path"] else None,
        audio_cache_dir=Path(ctor_kwargs["audio_cache_dir"]) if ctor_kwargs["audio_cache_dir"] else None,
        sample_cache_dir=Path(ctor_kwargs["sample_cache_dir"]) if ctor_kwargs["sample_cache_dir"] else None,
        max_items=1,
    )
    dataset.records = [target]
    sample = dataset._compute_sample(target)  # 已做缓存键校验, 命中会在 __getitem__ 内跳过
    cache_path = dataset._sample_cache_path(target.sample_id)
    if cache_path is not None:
        dataset._write_sample_cache(cache_path, sample)
    return target.sample_id


def warm_parallel(
    dataset: RealMANRing2Dataset,
    total: int,
    workers: int,
    progress_every: int,
) -> int:
    """多进程预热. 父进程只发派小任务, worker 内独立构造, 内存可控."""
    ctor_kwargs = _build_ctor_kwargs(dataset)
    root_dir = Path(ctor_kwargs["root_dir"])
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = 0
    for index in range(total):
        record = dataset.records[index]
        cache_path = dataset._sample_cache_path(record.sample_id)
        if cache_path is not None and cache_path.exists():
            skipped += 1  # 命中, 父进程预过滤减少派发量
            continue
        tasks.append((record.to_manifest_entry(root_dir), ctor_kwargs))
    print(f"dispatch: total={total} already_cached={skipped} to_compute={len(tasks)}", flush=True)
    created = 0
    with Pool(workers) as pool:
        for _sample_id in pool.imap_unordered(_warm_worker, tasks, chunksize=4):
            created += 1
            if created == 1 or created % progress_every == 0 or created == len(tasks):
                print(f"cached {created}/{len(tasks)}: {_sample_id}", flush=True)
    return created


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_name, list(args.override))
    raw_data_cfg = dict(OmegaConf.to_container(cfg.data, resolve=False))
    cache_dir = resolve_runtime_path(raw_data_cfg.get("sample_cache_dir"))
    if cache_dir is None:
        raise RuntimeError("Config must define data.sample_cache_dir to warm the sample cache.")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    total = len(dataset)
    if args.limit > 0:
        total = min(args.limit, total)
    print(
        f"warming sample_cache: dir={cache_dir} samples={total} "
        f"workers={args.workers} split={dataset.split}",
        flush=True,
    )

    if args.workers <= 0:
        created = warm_serial(dataset, total, args.progress_every)
    else:
        created = warm_parallel(dataset, total, args.workers, args.progress_every)

    print(f"sample_cache_ready={cache_dir.resolve()} newly_cached={created}", flush=True)


if __name__ == "__main__":
    main()
