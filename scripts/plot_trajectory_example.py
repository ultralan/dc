"""画图5: 移动声源未来方位轨迹定性对比.

从 test set 选一个 moving 且有明显转弯的样本, 画出未来32帧的
真值方位 / 模型预测 / 线性外推 / Kalman 四条曲线, 直观展示
端到端预测能跟随非线性轨迹, 而外推基线偏离.

用法: uv run python scripts/plot_trajectory_example.py
"""
from __future__ import annotations

import math
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sys
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from evaluate_realman_baseline_metrics import build_dataset, build_model  # noqa: E402
from evaluate_future_prediction import (  # noqa: E402
    extract_current_primary_theta_omega,
    predict_linear,
    predict_kalman,
)
from uca8.metrics import slot_logits_to_primary_azimuth_deg, target_slot_primary_azimuth_deg  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RD = Path("D:/RealMAN/runs/_test_eval_tmp")
OUT = Path("paper/figures/trajectory_example.pdf")
K = 32
DT = 0.01


def main() -> None:
    cfg = OmegaConf.load(RD / "config_resolved.yaml")
    device = torch.device("cuda")
    ds = build_dataset(cfg, "all")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(RD / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()

    # 找一个 moving 且轨迹有较大角速度变化的样本(转弯明显)
    picked = None
    with torch.no_grad():
        for idx in range(0, min(len(ds), 4000)):
            rec = ds.records[idx]
            if rec.motion != "moving":
                continue
            sample = ds[idx]
            fss = sample["future_slot_state"]  # [32,4,5]
            tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
            if int(tgt_valid.sum()) < 28:  # 要大部分帧有声源
                continue
            # 看真值方位变化幅度(转弯)
            angles = tgt_deg[0][tgt_valid[0]].cpu().numpy()
            span = np.max(angles) - np.min(angles)
            # 取弯曲明显的(非单调直线): 用相邻帧差的方向变化
            if span > 15:  # 至少有15度变化
                picked = (idx, sample, tgt_deg, tgt_valid)
                break
        if picked is None:
            # 退而求其次, 取任意 moving 全 valid 的
            for idx in range(min(len(ds), 4000)):
                rec = ds.records[idx]
                if rec.motion != "moving":
                    continue
                sample = ds[idx]
                fss = sample["future_slot_state"]
                tgt_deg, tgt_valid = target_slot_primary_azimuth_deg(fss.unsqueeze(0).to(device))
                if int(tgt_valid.sum()) >= 30:
                    picked = (idx, sample, tgt_deg, tgt_valid)
                    break

    if picked is None:
        print("[skip] 未找到合适样本")
        return
    idx, sample, tgt_deg, tgt_valid = picked
    print(f"选中样本 idx={idx}: {sample['sample_id']}")

    # 模型预测
    with torch.no_grad():
        wav = sample["waveform"].unsqueeze(0).to(device)
        vad = sample["vad_history"].unsqueeze(0).to(device)
        pred = model(wav, vad_history=vad, sample_id=[sample["sample_id"]])
        theta0, omega0, _ = extract_current_primary_theta_omega(pred["slot_logits"])
        # 三方法
        gt = tgt_deg[0].cpu().numpy()  # [32]
        valid = tgt_valid[0].cpu().numpy()
        pred_a_deg, _ = slot_logits_to_primary_azimuth_deg(pred["future_slot_logits"])
        pa = pred_a_deg[0].cpu().numpy()
        pl = predict_linear(theta0, omega0, k_future=K, dt=DT)[0].cpu().numpy()
        pk = predict_kalman(theta0, omega0, k_future=K, dt=DT)[0].cpu().numpy()

    ks = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ks, gt, "ko-", linewidth=2.2, markersize=5, label="真值 (ground truth)", zorder=5)
    ax.plot(ks, pa, "^-", color="#08519c", linewidth=1.8, markersize=4.5, label="Model (端到端)")
    ax.plot(ks, pl, "s--", color="#2ca02c", linewidth=1.5, markersize=4, label="Linear 外推")
    ax.plot(ks, pk, "^:", color="#d62728", linewidth=1.5, markersize=4, label="Kalman")
    # 标注真值无效帧
    for k, v in enumerate(valid, start=1):
        if not v:
            ax.axvspan(k - 0.5, k + 0.5, color="gray", alpha=0.08)
    ax.set_xlabel("未来帧 k (10 ms/帧, k=32 即 320 ms)", fontsize=10)
    ax.set_ylabel("方位角 (°)", fontsize=10)
    sid = sample["sample_id"].replace(":", "/")
    ax.set_title(f"移动声源未来方位轨迹对比 ({sid})", fontsize=10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    ax.set_xticks([1, 8, 16, 24, 32])
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
