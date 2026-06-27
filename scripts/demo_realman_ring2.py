from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from uca8.data.realman_ring2_dataset import RealMANRing2Dataset
from uca8.geometry.uca8 import make_uniform_circular_array
from uca8.losses.multi_task_loss import TrackTrendMultiTaskLoss
from uca8.models.tracktrend_net import UCA8TrackTrendNet

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke demo on the extracted RealMAN ring2 8-channel subset."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT / "realman_demo" / "extracted" / "ring2_8ch",
    )
    parser.add_argument(
        "--moving-csv",
        type=Path,
        default=ROOT / "realman_demo" / "raw" / "val" / "val_moving_source_location.csv",
    )
    parser.add_argument(
        "--static-csv",
        type=Path,
        default=ROOT / "realman_demo" / "raw" / "val" / "val_static_source_location.csv",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=ROOT / "realman_demo" / "extracted" / "ring2_8ch" / ".cache" / "audio_16k",
    )
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--radius-m", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
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


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = RealMANRing2Dataset(
        root_dir=args.root_dir,
        moving_csv=args.moving_csv,
        static_csv=args.static_csv,
        audio_cache_dir=args.audio_cache_dir,
        feature_cache_dir=args.audio_cache_dir.parent / "features_ring2_8ch",
        max_items=args.max_items,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    mic_positions = make_uniform_circular_array(num_mics=8, radius=args.radius_m).to(device)
    model = UCA8TrackTrendNet(mic_positions=mic_positions).to(device)
    criterion = TrackTrendMultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device={device}")
    print(f"dataset_summary={dataset.summary()}")
    print(f"batch_size={args.batch_size} steps={args.steps} lr={args.learning_rate}")

    model.train()
    for step, batch in zip(range(1, args.steps + 1), cycle(dataloader), strict=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch["waveform"], batch["vad_history"], sample_id=batch["sample_id"])
        losses = criterion(predictions, batch)
        losses["loss"].backward()
        optimizer.step()
        sample_ids = batch["sample_id"]
        sample_preview = sample_ids[: min(2, len(sample_ids))]
        print(
            " ".join(
                [
                    f"step={step}",
                    f"loss={losses['loss'].detach().item():.4f}",
                    f"count={losses['count_loss'].item():.4f}",
                    f"heat={losses['heat_loss'].item():.4f}",
                    f"track={losses['track_loss'].item():.4f}",
                    f"future={losses['future_loss'].item():.4f}",
                    f"motion={losses['motion_loss'].item():.4f}",
                    f"samples={list(sample_preview)}",
                ]
            )
        )


if __name__ == "__main__":
    main()
