"""高信息密度可视化(替换低密度的消融柱状和阵列几何图).

图A: 预测方位 vs 真值方位散点图 — 每点一个样本, 对角线=完美, 呈现系统偏差+离散+异常
图B: 误差随真值方位的分布 — 哪些方位误差大, 呈现方位依赖性

数据: 用ring1 loc_strong模型在val上跑预测, 提取每个样本的预测方位+真值方位.
"""

from __future__ import annotations

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
from uca8.metrics import heatmap_logits_to_azimuth_deg, target_slot_primary_azimuth_deg  # noqa: E402

# nature-figure 规范
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})
PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "neutral_mid": "#767676",
           "neutral_light": "#CFCECE", "gold": "#E8B73C", "teal": "#42949E"}
FIG_DIR = ROOT = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def extract_predictions() -> tuple[np.ndarray, np.ndarray]:
    """跑模型在val上, 提取预测方位和真值方位(度)."""
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "val")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 40:  # 40 batch ≈ 640 样本, 够画散点
                break
            wav = batch["waveform"].to(device)
            vad = batch["vad_history"].to(device)
            pred = model(wav, vad_history=vad, sample_id=batch.get("sample_id"))
            pred_deg = heatmap_logits_to_azimuth_deg(pred["heatmap_logits"])
            tgt_deg, valid = target_slot_primary_azimuth_deg(batch["slot_state"].to(device))
            mask = valid
            preds_all.append(pred_deg[mask].cpu().numpy())
            targets_all.append(tgt_deg[mask].cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(targets_all)


def plot_scatter(pred: np.ndarray, target: np.ndarray) -> None:
    """图A: 预测 vs 真值散点图."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    # 散点, 颜色按误差深浅
    err = np.abs(((pred - target + 180) % 360) - 180)
    sc = ax.scatter(target, pred, c=err, cmap="YlOrRd", s=12, alpha=0.7,
                    edgecolors="none", vmin=0, vmax=30, zorder=3)
    # 对角线(完美预测)
    ax.plot([-180, 180], [-180, 180], "--", color=PALETTE["neutral_mid"], lw=1.0, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Ground truth azimuth (°)", fontsize=8)
    ax.set_ylabel("Predicted azimuth (°)", fontsize=8)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linewidth=0.4)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Error (°)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG_DIR / f"pred_vs_true_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved pred_vs_true_scatter (n={len(pred)})")


def plot_error_by_azimuth(pred: np.ndarray, target: np.ndarray) -> None:
    """图B: 误差随真值方位的分布."""
    err = np.abs(((pred - target + 180) % 360) - 180)
    # 按方位bin(每15°一个bin)统计平均误差
    bin_edges = np.arange(-180, 181, 15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(target, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)
    mean_err = np.array([err[bin_idx == i].mean() if (bin_idx == i).any() else 0
                         for i in range(len(bin_centers))])
    count = np.array([(bin_idx == i).sum() for i in range(len(bin_centers))])

    fig, ax1 = plt.subplots(figsize=(4.5, 2.8))
    # 柱: 平均误差
    colors = [PALETTE["red_strong"] if e > 10 else PALETTE["blue_main"]
              if e < 5 else PALETTE["gold"] for e in mean_err]
    ax1.bar(bin_centers, mean_err, width=13, color=colors, edgecolor="black",
            linewidth=0.5, alpha=0.85, zorder=3)
    ax1.set_xlabel("Ground truth azimuth (°)", fontsize=8)
    ax1.set_ylabel("Mean error (°)", fontsize=8)
    ax1.set_xlim(-185, 185)
    ax1.set_ylim(0, max(mean_err) * 1.3 + 1)
    ax1.tick_params(labelsize=7)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.4)
    # 右轴: 样本数分布
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(bin_centers, count, "o-", color=PALETTE["neutral_mid"], lw=1.0,
             markersize=3, alpha=0.6, zorder=2)
    ax2.set_ylabel("Sample count", fontsize=8, color=PALETTE["neutral_mid"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["neutral_mid"], labelsize=7)
    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG_DIR / f"error_by_azimuth.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved error_by_azimuth")


def main() -> None:
    print("提取模型预测...")
    pred, target = extract_predictions()
    plot_scatter(pred, target)
    plot_error_by_azimuth(pred, target)


if __name__ == "__main__":
    main()
