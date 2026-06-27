"""按中文期刊惯例补两张标配图: 训练收敛曲线 + 预测vs真值时序曲线.

图A: 训练loss收敛曲线 — 从train_history.jsonl提取loss, 证明训练收敛(标配)
图B: 未来预测vs真值时序曲线 — 取真实moving样本, 画未来32帧的预测+真值叠加(时序预测身份配图)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from evaluate_realman_baseline_metrics import build_dataset, build_model  # noqa: E402
from uca8.metrics import slot_logits_to_primary_azimuth_deg, target_slot_primary_azimuth_deg  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})
P = {"blue": "#0F4D92", "red": "#B64342", "green": "#8BCF8B",
     "gray": "#767676", "gold": "#E8B73C", "teal": "#42949E"}
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def plot_training_curve() -> None:
    """图A: 训练loss收敛曲线(标配)."""
    hist = RD / "train_history.jsonl"
    steps, losses = [], []
    with hist.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("phase") == "train":
                steps.append(obj["global_step"])
                losses.append(obj["loss"])
    steps = np.array(steps)
    losses = np.array(losses)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(steps, losses, "-", color=P["blue"], lw=0.8, alpha=0.3)  # 原始
    # 滑动平均(平滑)
    if len(losses) > 100:
        window = max(len(losses) // 50, 20)
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smooth, "-", color=P["blue"], lw=1.5, label="Train loss")
    ax.set_xlabel("Training step", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.4)
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"training_curve.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved training_curve ({len(steps)} steps)")


def plot_future_prediction_curve() -> None:
    """图B: 预测vs真值时序曲线(时序预测标配).

    取2-3个真实moving样本, 画未来32帧的方位:
    真值(黑实线) + 模型预测(蓝) + 线性外推(绿虚) 叠加.
    """
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "val")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()

    # 找2个moving样本(有转弯)
    samples = []
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for idx in range(min(len(ds), 3000)):
            rec = ds.records[idx]
            if rec.motion != "moving":
                continue
            batch = ds[idx]
            fss = batch["future_slot_state"]  # [32,4,5]
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
            if int(tgt_valid.sum()) < 28:
                continue
            angles = tgt_deg[0].cpu().numpy()
            if np.max(angles) - np.min(angles) < 12:
                continue
            # 模型预测
            wav = batch["waveform"].unsqueeze(0).to(device)
            vad = batch["vad_history"].unsqueeze(0).to(device)
            pred = model(wav, vad_history=vad, sample_id=[batch["sample_id"]])
            pred_a_deg, _ = slot_logits_to_primary_azimuth_deg(pred["future_slot_logits"])
            # 线性外推
            from uca8.metrics import slot_logits_to_primary_azimuth_deg as _s
            # 当前帧
            cur_slot = pred["slot_logits"]
            act = torch.sigmoid(cur_slot[..., 0])
            si = act.argmax(dim=-1)
            gathered = torch.gather(cur_slot, dim=1, index=si[..., None, None].expand(1, 1, 5)).squeeze(1)
            theta0 = torch.atan2(gathered[..., 1], gathered[..., 2])  # rad
            omega0 = gathered[..., 4]
            K = 32
            dt = 0.01
            kk = torch.arange(1, K + 1, device=device, dtype=theta0.dtype)
            lin_deg = torch.rad2deg(theta0[:, None] + omega0[:, None] * kk[None, :] * dt).cpu().numpy()[0]

            samples.append({
                "sid": batch["sample_id"],
                "gt": tgt_deg[0].cpu().numpy(),
                "model": pred_a_deg[0].cpu().numpy(),
                "linear": lin_deg,
            })
            if len(samples) >= 2:
                break

    if not samples:
        # 退而求其次, 取任意moving
        for idx in range(min(len(ds), 3000)):
            rec = ds.records[idx]
            if rec.motion != "moving":
                continue
            batch = ds[idx]
            fss = batch["future_slot_state"]
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
            if int(tgt_valid.sum()) >= 30:
                with torch.no_grad():
                    wav = batch["waveform"].unsqueeze(0).to(device)
                    vad = batch["vad_history"].unsqueeze(0).to(device)
                    pred = model(wav, vad_history=vad, sample_id=[batch["sample_id"]])
                    pred_a_deg, _ = slot_logits_to_primary_azimuth_deg(pred["future_slot_logits"])
                samples.append({
                    "sid": batch["sample_id"],
                    "gt": tgt_deg[0].cpu().numpy(),
                    "model": pred_a_deg[0].cpu().numpy(),
                    "linear": pred_a_deg[0].cpu().numpy(),  # 占位
                })
                if len(samples) >= 2:
                    break

    ks = np.arange(1, 33)
    fig, axes = plt.subplots(1, len(samples), figsize=(4.0 * len(samples), 2.8), sharey=False)
    if len(samples) == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        ax.plot(ks, s["gt"], "k-", lw=1.8, label="Ground truth", zorder=5)
        ax.plot(ks, s["model"], "o-", color=P["blue"], lw=1.2, markersize=3,
                label="Model", alpha=0.85)
        ax.plot(ks, s["linear"], "s--", color=P["green"], lw=1.0, markersize=2.5,
                label="Linear", alpha=0.7)
        ax.set_xlabel("Future frame $k$", fontsize=8)
        ax.set_ylabel("Azimuth (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
        # 场景名做子图标
        scene = s["sid"].split(":")[0] if ":" in s["sid"] else s["sid"][:12]
        ax.set_title(scene, fontsize=7)
    axes[0].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"future_pred_vs_true.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved future_pred_vs_true ({len(samples)} samples)")


def main() -> None:
    plot_training_curve()
    plot_future_prediction_curve()


if __name__ == "__main__":
    main()
