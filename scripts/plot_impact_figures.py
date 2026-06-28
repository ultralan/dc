"""两张震撼可视化图(按 nature-figure 规范).

图① 时频+DOA轨迹三联图(含未来预测延伸): SELDnet标志性三联升级版
   - 上: 多通道语谱图
   - 中: heatmap瀑布图(时间×方位, GT叠黑线)
   - 下: 预测轨迹(红)+GT(绿虚)+未来预测段(蓝虚+置信带)
图② 极坐标方位玫瑰图: 模型heatmap→极坐标, GT方位径向亮线

数据: ring1 loc_strong 模型在 val 上取一个 moving 样本的逐帧预测.
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
from uca8.metrics import slot_logits_to_primary_azimuth_deg, target_slot_primary_azimuth_deg  # noqa: E402

# ===== nature-figure 规范 =====
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})
P = {"blue": "#0F4D92", "red": "#B64342", "green": "#5B9B5B",
     "gray": "#767676", "gold": "#E8B73C", "teal": "#42949E", "neutral": "#CFCECE"}
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def save_pub(fig, name: str) -> None:
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load_model_and_sample():
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "val")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    # 找一个moving且轨迹有变化的样本(逐帧预测需要该样本在数据集里)
    # 注意: dataset每条样本是单窗口(history128+future32), 要逐帧轨迹需取多个窗口或用slot序列
    # 这里用一个样本的 future_slot (32帧未来) 做未来预测段, history 部分用 heatmap 做瀑布
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    sample = None
    with torch.no_grad():
        for idx in range(min(len(ds), 2500)):
            rec = ds.records[idx]
            if rec.motion != "moving":
                continue
            batch = ds[idx]
            fss = batch["future_slot_state"]
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
            if int(tgt_valid.sum()) < 28:
                continue
            wav = batch["waveform"].unsqueeze(0).to(device)
            vad = batch["vad_history"].unsqueeze(0).to(device)
            pred = model(wav, vad_history=vad, sample_id=[batch["sample_id"]])
            # 当前帧heatmap (72 bin)
            cur_heat = torch.softmax(pred["heatmap_logits"][0], dim=-1).cpu().numpy()
            # 未来heatmap序列 (32, 72)
            fut_heat = torch.softmax(pred["future_heatmap_logits"][0], dim=-1).cpu().numpy()
            # 真值当前+未来方位
            cur_tgt = tgt_deg[0].cpu().numpy()  # 实际是future的GT
            # 当前帧GT从 slot_state (4,5) 取活动slot的方位
            cur_slot = batch["slot_state"]  # [4,5]
            act = cur_slot[:, 0]
            if (act > 0.5).any():
                si = int(act.argmax())
                cur_gt = float(torch.atan2(cur_slot[si, 1], cur_slot[si, 2])) * 180 / np.pi
            else:
                continue
            if abs(cur_gt - cur_tgt[0]) > 30:  # 跳变的不选
                continue
            # 多通道语谱图(取参考通道STFT幅度)
            wav_np = batch["waveform"][0].numpy()  # [C, N]
            sample = {
                "sid": batch["sample_id"],
                "wav": wav_np,
                "cur_heat": cur_heat,
                "fut_heat": fut_heat,
                "cur_gt_deg": cur_gt,
                "fut_gt_deg": cur_tgt,  # [32]
                "fut_valid": tgt_valid[0].cpu().numpy(),
            }
            break
    return sample


def plot_triplet(sample):
    """图① 时频+DOA轨迹三联图."""
    # 语谱图(参考通道)
    from scipy.signal import stft as scipy_stft
    wav0 = sample["wav"]  # 已是参考通道一维 (load时取了waveform[0])
    f, t, Z = scipy_stft(wav0, fs=16000, nperseg=400, noverlap=240)
    mag = np.log1p(np.abs(Z))

    # heatmap瀑布: history的heatmap我们只有当前帧; 用future_heatmap(32帧)做瀑布主体
    # 构造: 当前帧heatmap + 32帧future_heatmap 拼成33帧序列
    all_heat = np.vstack([sample["cur_heat"][None, :], sample["fut_heat"]])  # [33,72]
    # 对应GT: 当前 + 32未来
    all_gt = np.concatenate([[sample["cur_gt_deg"]], sample["fut_gt_deg"]])  # [33]

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.0], hspace=0.45)

    # 子图1: 语谱图
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(t, f, mag, shading="auto", cmap="magma")
    ax1.set_ylabel("频率 (Hz)", fontsize=8)
    ax1.set_xlabel("时间 (s)", fontsize=8)
    ax1.set_title("(a) 参考通道语谱图", fontsize=8, loc="left")
    ax1.tick_params(labelsize=7)

    # 子图2: heatmap瀑布图(时间×方位)
    ax2 = fig.add_subplot(gs[1])
    frames = np.arange(all_heat.shape[0])
    bins = np.arange(72) * 5 - 180  # 方位 bin 中心
    ax2.pcolormesh(frames, bins, all_heat.T, shading="nearest", cmap="viridis")
    # GT叠黑线(只画valid帧)
    valid = np.concatenate([[True], sample["fut_valid"]])
    ax2.plot(frames[valid], all_gt[valid], "k-", lw=1.5, label="真值")
    ax2.axvline(0.5, color="white", lw=1.0, ls="--", alpha=0.8)  # 当前/未来分界
    ax2.set_ylabel("方位角 (°)", fontsize=8)
    ax2.set_xlabel("帧 (0=当前, 1–32=未来)", fontsize=8)
    ax2.set_title("(b) 方位热力图瀑布", fontsize=8, loc="left")
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6.5, loc="upper right")

    # 子图3: 轨迹(预测argmax vs GT + 未来置信带)
    ax3 = fig.add_subplot(gs[2])
    pred_deg = np.array([bins[:-1][h.argmax()] for h in all_heat])
    ax3.plot(frames[valid], all_gt[valid], "o--", color=P["green"], lw=1.2, ms=3, label="真值")
    ax3.plot(frames, pred_deg, "s-", color=P["red"], lw=1.0, ms=2.5, label="预测", alpha=0.9)
    # 未来段置信带(用heatmap峰宽近似)
    fut_pred = pred_deg[1:]
    # 用 top-3 bin 的展宽做置信带
    for i, h in enumerate(sample["fut_heat"]):
        top = np.sort(h)[-3:]
        spread = 8 * (1 - top[0])  # 越集中spread越小
        ax3.fill_between([i+1], fut_pred[i]-spread, fut_pred[i]+spread, color=P["blue"], alpha=0.12)
    ax3.axvline(0.5, color=P["gray"], lw=0.8, ls="--", alpha=0.6)
    ax3.axvspan(1, 32, color=P["blue"], alpha=0.05)
    ax3.text(16, ax3.get_ylim()[1]*0.9, "未来预测段", fontsize=6.5, color=P["blue"], ha="center")
    ax3.set_ylabel("方位角 (°)", fontsize=8)
    ax3.set_xlabel("帧", fontsize=8)
    ax3.set_title("(c) 方位轨迹与未来预测", fontsize=8, loc="left")
    ax3.tick_params(labelsize=7)
    ax3.legend(fontsize=6.5, loc="upper left", ncol=2)

    save_pub(fig, "triplet_spectra_trajectory")


def plot_polar(sample):
    """图② 极坐标方位玫瑰图."""
    heat = sample["cur_heat"]  # [72]
    gt_deg = sample["cur_gt_deg"]
    theta = np.deg2rad(np.arange(72) * 5 - 180 + 2.5)  # bin中心
    theta_full = np.concatenate([theta, [theta[0]]])
    r_full = np.concatenate([heat, [heat[0]]])

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"projection": "polar"})
    # 填充玫瑰瓣
    ax.fill(theta_full, r_full, color=P["blue"], alpha=0.35, zorder=2)
    ax.plot(theta_full, r_full, color=P["blue"], lw=1.2, zorder=3)
    # GT径向亮线
    gt_theta = np.deg2rad(gt_deg)
    ax.plot([gt_theta, gt_theta], [0, r_full.max()*1.05], color=P["red"], lw=2.0, zorder=4)
    ax.scatter([gt_theta], [r_full.max()*1.05], color=P["red"], s=40, zorder=5, marker="*")
    # 预测峰
    pred_deg = np.arange(72)[heat.argmax()]*5 - 180 + 2.5
    pt = np.deg2rad(pred_deg)
    ax.plot([pt, pt], [0, heat.max()], color=P["green"], lw=1.5, ls="--", zorder=4)

    ax.set_theta_zero_location("E")  # 0度在东(右), 对应阵列mic0
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"], fontsize=7)
    ax.set_yticklabels([])
    ax.set_title("当前帧方位响应 (圆阵极坐标)", fontsize=8, pad=12)
    # 图例
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=P["blue"], lw=2, label="模型响应"),
        Line2D([0],[0], color=P["red"], lw=2, label="真值方位"),
        Line2D([0],[0], color=P["green"], lw=1.5, ls="--", label="预测峰"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save_pub(fig, "polar_azimuth_rose")


def main():
    print("加载模型与样本...")
    sample = load_model_and_sample()
    if sample is None:
        print("未找到合适样本")
        return
    print(f"样本: {sample['sid']}, 当前GT={sample['cur_gt_deg']:.1f}°")
    plot_triplet(sample)
    plot_polar(sample)


if __name__ == "__main__":
    main()
