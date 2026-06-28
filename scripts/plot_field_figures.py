"""高密度震撼可视化(场/云/脊形态, 非折线).

按 GSRP/SELD 行业规范, 三种形态:
图① stacked DOA field: 纵轴真实方位区间, 横轴72bin响应, 时均响应堆叠成场, 峰值点连线, 对角线GT
图② 极坐标多轨迹云团: 几十条moving轨迹半透明叠加成发光环
图③ ridgeline方位山脊: 单样本72bin分布按时间堆成山脊

数据: ring1 loc_strong, 多个moving样本的heatmap响应.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from evaluate_realman_baseline_metrics import build_dataset, build_model  # noqa: E402
from uca8.metrics import target_slot_primary_azimuth_deg  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})
WHITE = "#ffffff"
GT = "#2ca02c"; PRED = "#d62728"; FUT = "#1f77b4"
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def save_pub(fig, name):
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def collect_samples(n=120):
    """收集多个moving样本的(真实当前方位, heatmap响应, 未来GT轨迹)。"""
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "val")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    samples = []
    with torch.no_grad():
        for idx in range(min(len(ds), 6000)):
            if ds.records[idx].motion != "moving":
                continue
            batch = ds[idx]
            cur_slot = batch["slot_state"]
            act = cur_slot[:, 0]
            if not (act > 0.5).any():
                continue
            si = int(act.argmax())
            cur_gt = float(torch.atan2(cur_slot[si, 1], cur_slot[si, 2])) * 180 / np.pi
            wav = batch["waveform"].unsqueeze(0).to(device)
            vad = batch["vad_history"].unsqueeze(0).to(device)
            pred = model(wav, vad_history=vad, sample_id=[batch["sample_id"]])
            heat = torch.softmax(pred["heatmap_logits"][0], dim=-1).cpu().numpy()
            fut_heat = torch.softmax(pred["future_heatmap_logits"][0], dim=-1).cpu().numpy()
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(batch["future_slot_state"].unsqueeze(0).to(device))
            samples.append({
                "cur_gt": cur_gt, "heat": heat, "fut_heat": fut_heat,
                "fut_gt": tgt_deg[0].cpu().numpy(), "fut_valid": tgt_valid[0].cpu().numpy(),
            })
            if len(samples) >= n:
                break
    return samples


def wrap_diff(a, b):
    return (a - b + 180) % 360 - 180


def plot_stacked_field(samples):
    """图① stacked DOA field: 真实方位(bin) × 估计方位响应(bin)。"""
    bins = np.arange(72) * 5 - 180  # 估计方位中心
    # 把样本按真实方位分到72个真实方位bin
    n_bin = 72
    field = np.full((n_bin, n_bin), np.nan)
    for s in samples:
        gt_bin = int((s["cur_gt"] + 180) // 5)
        gt_bin = max(0, min(71, gt_bin))
        if np.isnan(field[gt_bin, 0]):  # 该行第一个
            field[gt_bin] = s["heat"]
        else:
            # 同真实方位多样本取平均
            mask = ~np.isnan(field[gt_bin])
            field[gt_bin][mask] = (field[gt_bin][mask] + s["heat"][mask]) / 2
            field[gt_bin][~mask] = s["heat"][~mask]
    # 峰值点
    valid_rows = ~np.isnan(field[:, 0])
    peak_bins = np.array([np.nanargmax(field[r]) if valid_rows[r] else -1 for r in range(n_bin)])
    peak_deg = peak_bins * 5 - 180 + 2.5

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # 用nan->0便于pcolormesh
    field_plot = np.where(np.isnan(field), 0, field)
    im = ax.pcolormesh(bins, bins, field_plot.T, shading="nearest", cmap="inferno",
                       norm=Normalize(0, np.nanmax(field)))
    # 峰值连线(黑)
    rows_with = np.where(valid_rows)[0]
    gt_axis = rows_with * 5 - 180 + 2.5
    ax.plot(peak_deg[rows_with], gt_axis, "o-", color=WHITE, lw=1.8, ms=4, zorder=4, label="模型峰值")
    # 对角线GT(白虚)
    ax.plot([-180, 180], [-180, 180], "--", color=WHITE, lw=1.2, alpha=0.6, zorder=3, label="理想对角线")
    cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label("时均方位置信度", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("估计方位 (°)", fontsize=8); ax.set_ylabel("真实方位 (°)", fontsize=8)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("堆叠方位响应场 (stacked DOA field)", fontsize=8.5)
    save_pub(fig, "impact_stacked_field")


def plot_polar_cloud(samples):
    """图② 极坐标多轨迹云团: 多样本未来轨迹半透明叠加。"""
    fig = plt.figure(figsize=(5.8, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    # 每个样本: 把未来轨迹(解卷绕)画到极坐标, 半透明
    for s in samples:
        fgt = s["fut_gt"]; valid = s["fut_valid"]
        if valid.sum() < 10:
            continue
        # 解卷绕
        seq = []
        for k, v in enumerate(fgt):
            if not valid[k]:
                continue
            if not seq:
                seq.append(v)
            else:
                d = wrap_diff(v, seq[-1])
                seq.append(seq[-1] + d)
        seq = np.array(seq)
        r = np.linspace(0.3, 1.0, len(seq))
        theta = np.deg2rad(seq)
        ax.plot(theta, r, "-", color=FUT, lw=1.0, alpha=0.15, zorder=2)
    # 叠一层glow感的粗线(取几个代表)
    for s in samples[:8]:
        fgt = s["fut_gt"]; valid = s["fut_valid"]
        if valid.sum() < 10:
            continue
        seq = []
        for k, v in enumerate(fgt):
            if not valid[k]:
                continue
            if not seq: seq.append(v)
            else: seq.append(seq[-1] + wrap_diff(v, seq[-1]))
        r = np.linspace(0.3, 1.0, len(seq))
        ax.plot(np.deg2rad(seq), r, "-", color=PRED, lw=2.5, alpha=0.5, zorder=3)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, -135, -90, -45]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "±180°", "-135°", "-90°", "-45°"], fontsize=7)
    ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=PRED, lw=2, alpha=0.6, label="代表性轨迹"),
               Line2D([0],[0], color=FUT, lw=1, alpha=0.3, label=f"全部轨迹 (n={len(samples)})")]
    ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("移动声源未来方位轨迹云 (极坐标)", fontsize=8.5, pad=14)
    save_pub(fig, "impact_polar_cloud")


def plot_ridgeline(samples):
    """图③ ridgeline方位山脊: 选一个moving样本的未来32帧分布堆成山脊。"""
    # 选轨迹明显变化的样本
    target = None
    for s in samples:
        fgt = s["fut_gt"]; valid = s["fut_valid"]
        if valid.sum() < 30:
            continue
        seq = []
        for v in fgt:
            if not seq: seq.append(v)
            else: seq.append(seq[-1] + wrap_diff(v, seq[-1]))
        if max(seq) - min(seq) > 25:
            target = s; break
    if target is None:
        print("无合适样本"); return
    fut_heat = target["fut_heat"]  # [32, 72]
    bins = np.arange(72) * 5 - 180

    fig, ax = plt.subplots(figsize=(7, 5))
    n_frames = fut_heat.shape[0]
    # 每帧分布往下堆叠, fill_between画山脊
    for i in range(n_frames):
        h = fut_heat[i]
        offset = (n_frames - i) * 1.0  # 从下往上时间递增
        scale = 18  # 纵向放大
        ax.fill_between(bins, offset, offset + h * scale, color=FUT, alpha=0.5, zorder=2)
        ax.plot(bins, offset + h * scale, color=FUT, lw=0.8, alpha=0.8, zorder=3)
        # GT点
        gt_deg = target["fut_gt"][i]
        if target["fut_valid"][i]:
            gt_bin = int((gt_deg + 180) // 5)
            if 0 <= gt_bin < 72:
                ax.plot(gt_deg, offset + h[gt_bin] * scale + 0.3, "o", color=GT, ms=2.5, zorder=4)
    ax.set_xlabel("方位角 (°)", fontsize=8)
    ax.set_ylabel("未来帧 k (下→上 递增)", fontsize=8)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.tick_params(labelsize=7)
    ax.set_yticks([])
    ax.set_title("未来方位响应山脊图 (ridgeline)", fontsize=8.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=FUT, lw=2, label="模型响应分布"),
               Line2D([0],[0], marker="o", color="w", markerfacecolor=GT, ms=6, label="真值方位")]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    save_pub(fig, "impact_ridgeline")


def main():
    print("收集样本...")
    samples = collect_samples(n=120)
    print(f"收集到 {len(samples)} 个moving样本")
    plot_stacked_field(samples)
    plot_polar_cloud(samples)
    plot_ridgeline(samples)


if __name__ == "__main__":
    main()
