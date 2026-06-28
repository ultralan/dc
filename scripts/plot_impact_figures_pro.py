"""震撼可视化图 v2 (符合 SELDnet/DOANet 行业规范 + 炸眼技巧).

规范要点(调研实证):
- colormap: viridis/plasma/inferno (科学可视化标准, 禁jet)
- 轨迹对比: GT粗实线2.5px, 预测细虚线1.5px+marker (粗细即层级)
- 热力图GT叠加: 白色折线压在深色colormap上
- glow发光线: 双层绘制(粗半透明底+细芯线), 不改数据, 规范内花哨
- 方位跳变: 用笛卡尔(sin/cos)避免±180°伪线

图① 方位热力图瀑布(时间×方位) + 预测轨迹双子图(含未来预测glow段)
图② 极坐标方位图(背景plasma热力图 + GT白线 + 预测glow轨迹)
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
# 行业标准色 (tab10 + 方法对比色)
GT_COLOR = "#2ca02c"      # 真值: 绿 (SELDnet WASPAA Fig3)
PRED_COLOR = "#d62728"    # 预测: 红 (SELDnet)
FUT_COLOR = "#1f77b4"     # 未来预测: 蓝
WHITE = "#ffffff"
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def save_pub(fig, name):
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def glow_line(ax, x, y, color, core_lw=1.5, glow_lw=6, alpha=1.0, **kw):
    """规范内发光线: 粗半透明底 + 细芯线, 不改数据。"""
    ax.plot(x, y, "-", color=color, linewidth=glow_lw, alpha=0.22, zorder=2, **kw)
    ax.plot(x, y, "-", color=color, linewidth=core_lw, alpha=alpha, zorder=3, **kw)


def load_sample():
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "val")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    # 选轨迹明显变化的moving样本: 用解卷绕后的真实跨度判断(避免±180跳变假象)
    TARGET_IDX = 11  # 已验证: 真实转弯 91°→79°, 信息量足
    with torch.no_grad():
        idx_list = [TARGET_IDX] + [i for i in range(min(len(ds), 4000)) if i != TARGET_IDX]
        for idx in idx_list:
            rec = ds.records[idx]
            if rec.motion != "moving":
                continue
            batch = ds[idx]
            fss = batch["future_slot_state"]
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
            if int(tgt_valid.sum()) < 30:
                continue
            fgt = tgt_deg[0].cpu().numpy()
            # 解卷绕后判断真实跨度
            fgt_unwrapped = []
            for v in fgt:
                if not fgt_unwrapped:
                    fgt_unwrapped.append(v)
                else:
                    d = (v - fgt_unwrapped[-1] + 180) % 360 - 180
                    fgt_unwrapped.append(fgt_unwrapped[-1] + d)
            fgt_u = np.array(fgt_unwrapped)
            real_spread = fgt_u.max() - fgt_u.min()
            if real_spread < 25:
                continue
            wav = batch["waveform"].unsqueeze(0).to(device)
            vad = batch["vad_history"].unsqueeze(0).to(device)
            pred = model(wav, vad_history=vad, sample_id=[batch["sample_id"]])
            cur_heat = torch.softmax(pred["heatmap_logits"][0], dim=-1).cpu().numpy()
            fut_heat = torch.softmax(pred["future_heatmap_logits"][0], dim=-1).cpu().numpy()
            cur_slot = batch["slot_state"]
            act = cur_slot[:, 0]
            si = int(act.argmax())
            cur_gt = float(torch.atan2(cur_slot[si, 1], cur_slot[si, 2])) * 180 / np.pi
            print(f"选中 idx={idx}, 真实跨度={real_spread:.1f}°, GT起始={fgt_u[0]:.1f}°→末尾={fgt_u[-1]:.1f}°")
            return {
                "sid": batch["sample_id"],
                "cur_heat": cur_heat, "fut_heat": fut_heat,
                "cur_gt": cur_gt, "fut_gt": tgt_deg[0].cpu().numpy(),
                "fut_valid": tgt_valid[0].cpu().numpy(),
            }
    return None


def wrap_diff(a, b):
    """环形角度差, 处理±180跳变。"""
    return (a - b + 180) % 360 - 180


def unwrap_deg(deg_seq):
    """解卷绕方位序列, 消除±180跳变伪线。"""
    out = [deg_seq[0]]
    for i in range(1, len(deg_seq)):
        d = wrap_diff(deg_seq[i], out[-1])
        out.append(out[-1] + d)
    return np.array(out)


def plot_heatmap_trajectory(sample):
    """图① 方位热力图瀑布 + 轨迹(含未来glow)。"""
    all_heat = np.vstack([sample["cur_heat"][None, :], sample["fut_heat"]])  # [33,72]
    bins = np.arange(72) * 5 - 180
    all_gt = np.concatenate([[sample["cur_gt"]], sample["fut_gt"]])
    valid = np.concatenate([[True], sample["fut_valid"]])
    pred_deg = np.array([bins[h.argmax()] for h in all_heat])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5),
                                    gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.35})

    # --- 子图1: 热力图瀑布 (plasma最炸, 白色GT折线压顶) ---
    frames = np.arange(33)
    im = ax1.pcolormesh(frames, bins, all_heat.T, shading="nearest",
                        cmap="plasma", norm=Normalize(0, all_heat.max()))
    # GT白色粗实线 (解卷绕避免跳变)
    gt_unwrapped = unwrap_deg(all_gt[valid])
    ax1.plot(frames[valid], gt_unwrapped, "-", color=WHITE, lw=2.5, alpha=0.95, zorder=4)
    # 预测虚线
    pred_unwrapped = unwrap_deg(pred_deg)
    ax1.plot(frames, pred_unwrapped, "--", color="#00d4ff", lw=1.2, alpha=0.85, zorder=5)
    # 当前/未来分界
    ax1.axvline(0.5, color=WHITE, lw=1.0, ls=":", alpha=0.7)
    ax1.text(16, 170, "未来预测段", fontsize=7, color=WHITE, ha="center", alpha=0.9)
    cb = plt.colorbar(im, ax=ax1, pad=0.01)
    cb.set_label("方位置信度", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax1.set_ylabel("方位角 (°)", fontsize=8)
    ax1.set_xticks([0, 8, 16, 24, 32])
    ax1.set_title("(a) 方位热力图瀑布 (白线=真值, 青虚线=预测)", fontsize=8, loc="left")
    ax1.tick_params(labelsize=7)

    # --- 子图2: 轨迹对比 (GT粗绿线 + 预测glow红线 + 未来段蓝glow) ---
    t = np.arange(33)
    gt_u = unwrap_deg(all_gt)
    pred_u = unwrap_deg(pred_deg)
    # GT粗绿实线
    glow_line(ax2, t[valid], gt_u[valid], GT_COLOR, core_lw=2.5, glow_lw=7)
    ax2.plot(t[valid], gt_u[valid], "o", color=GT_COLOR, ms=3, zorder=4)
    # 当前帧预测(红)
    glow_line(ax2, t[:1], pred_u[:1], PRED_COLOR, core_lw=2.0, glow_lw=6)
    # 未来预测段(蓝glow) - 独家
    glow_line(ax2, t[1:], pred_u[1:], FUT_COLOR, core_lw=1.8, glow_lw=6)
    ax2.plot(t[1:], pred_u[1:], "s", color=FUT_COLOR, ms=2.5, alpha=0.8, zorder=4)
    # 未来段背景
    ax2.axvspan(1, 32, color=FUT_COLOR, alpha=0.06)
    ax2.axvline(0.5, color="#999", lw=0.8, ls=":", alpha=0.6)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=GT_COLOR, lw=2.5, label="真值"),
        Line2D([0],[0], color=PRED_COLOR, lw=2.0, label="当前帧预测"),
        Line2D([0],[0], color=FUT_COLOR, lw=1.8, label="未来预测 (glow)"),
    ]
    ax2.legend(handles=handles, fontsize=7, loc="upper left", ncol=3)
    ax2.set_xlabel("帧 (0=当前, 1–32=未来)", fontsize=8)
    ax2.set_ylabel("方位角 (°)", fontsize=8)
    ax2.set_title("(b) 方位轨迹对比", fontsize=8, loc="left")
    ax2.tick_params(labelsize=7)
    save_pub(fig, "impact_heatmap_trajectory")


def plot_polar(sample):
    """图② 极坐标方位图 (plasma背景热力 + GT白线 + 预测glow)。"""
    heat = sample["cur_heat"]
    gt_deg = sample["cur_gt"]
    theta = np.deg2rad(np.arange(72) * 5 - 180 + 2.5)
    pred_deg = np.arange(72)[heat.argmax()] * 5 - 180 + 2.5

    fig = plt.figure(figsize=(5.5, 5.0))
    ax = fig.add_subplot(111, projection="polar")
    # 背景热力 (plasma, 极坐标pcolormesh)
    theta_edges = np.deg2rad(np.arange(73) * 5 - 180)
    r_edges = np.linspace(0, 1, 30)
    T, R = np.meshgrid(theta_edges, r_edges)
    H = np.tile(heat[:, None], (1, 29)).T  # [29,72]
    ax.pcolormesh(T, R, H, shading="auto", cmap="plasma", norm=Normalize(0, heat.max()))
    # GT白色径向粗线
    gt_t = np.deg2rad(gt_deg)
    ax.plot([gt_t, gt_t], [0, 1.0], "-", color=WHITE, lw=2.8, alpha=0.95, zorder=5)
    ax.scatter([gt_t], [1.02], color=WHITE, s=120, marker="*", zorder=6, edgecolors=GT_COLOR, linewidths=1.5)
    # 预测glow径向线 (蓝)
    pt = np.deg2rad(pred_deg)
    ax.plot([pt, pt], [0, heat.max()/heat.max()*0.95], "-", color=FUT_COLOR, lw=6, alpha=0.25, zorder=3)
    ax.plot([pt, pt], [0, 0.95], "--", color=FUT_COLOR, lw=1.5, zorder=4)
    ax.scatter([pt], [0.97], color=FUT_COLOR, s=70, marker="x", zorder=5, linewidths=2)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, -135, -90, -45]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "±180°", "-135°", "-90°", "-45°"], fontsize=7)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=WHITE, lw=2.5, marker="*", ms=10, label="真值方位"),
        Line2D([0],[0], color=FUT_COLOR, lw=1.5, ls="--", marker="x", ms=8, label="预测峰"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("当前帧方位响应 (圆阵极坐标, plasma谱)", fontsize=8, pad=14)
    save_pub(fig, "impact_polar_rose")


def main():
    print("加载样本...")
    s = load_sample()
    if s is None:
        print("无样本"); return
    print(f"样本 {s['sid']}, 当前GT={s['cur_gt']:.1f}°")
    plot_heatmap_trajectory(s)
    plot_polar(s)


if __name__ == "__main__":
    main()
