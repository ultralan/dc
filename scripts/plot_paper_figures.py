"""绘制论文图3(特征消融柱状)和图4(未来预测 k-MAE 曲线).

数据来源:
  图3: 特征消融结果(official val, 2epoch), 硬编码自实验记录.
  图4: D:/RealMAN/ring2_8ch/logs/eval_future_full.json 的 k_mae_deg.

输出:
  paper/figures/ablation_bar.pdf   (图3)
  paper/figures/future_kmae.pdf    (图4)

用法: uv run python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体: 优先 microsoft yahei, 回退 sans
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- 图3: 特征消融柱状 ----------


def plot_ablation() -> None:
    # (配置标签, logmel, ipd, srp, MAE, ACC5)
    rows = [
        ("log-mel", True, False, False, 48.77, 5.2),
        ("log-mel+IPD", True, True, False, 55.97, 7.6),
        ("log-mel+SRP", True, False, True, 53.21, 6.4),
        ("log-mel+IPD+SRP", True, True, True, 6.87, 87.5),
    ]
    labels = [r[0] for r in rows]
    mae = [r[4] for r in rows]
    acc5 = [r[5] for r in rows]

    x = np.arange(len(labels))
    width = 0.55
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))

    # MAE 柱: 突出第四组的骤降
    colors = ["#9ecae1", "#9ecae1", "#9ecae1", "#08519c"]
    bars = ax1.bar(x, mae, width, color=colors, edgecolor="black", linewidth=0.6, label="MAE (°)")
    ax1.set_ylabel("当前帧 MAE (°)", fontsize=11)
    ax1.set_ylim(0, max(mae) * 1.18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_xlabel("特征配置", fontsize=11)
    for xi, v in zip(x, mae):
        ax1.text(xi, v + max(mae) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # ACC@5 折线 (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, acc5, "o-", color="#d62728", linewidth=2, markersize=7, label="ACC@5 (%)")
    ax2.set_ylabel("ACC@5 (%)", fontsize=11, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 100)
    for xi, v in zip(x, acc5):
        ax2.text(xi, v + 3, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#d62728")

    # 虚线标注随机基线
    ax1.axhline(180 / 3.7, ls=":", color="gray", linewidth=1)
    ax1.text(len(labels) - 0.5, 180 / 3.7 + 1, "随机基线≈48.6°", ha="right", fontsize=8, color="gray")

    plt.title("空间先验前端特征消融 (IPD 与 SRP 的非线性协同)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "ablation_bar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------- 图4: 未来预测 k-MAE 曲线 ----------


def plot_future_kmae() -> None:
    src = Path("D:/RealMAN/ring2_8ch/logs/eval_future_full.json")
    if not src.exists():
        print(f"[skip] 未找到 {src}, 跳过图4")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    kmae = data["groups"]["all"]["k_mae_deg"]
    ks = sorted(int(k) for k in kmae["model_endtoend"].keys())

    def series(method: str) -> list[float]:
        return [kmae[method][str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(ks, series("model_endtoend"), "o-", color="#08519c", linewidth=2, markersize=6, label="Model (端到端)")
    ax.plot(ks, series("linear_extrap"), "s--", color="#2ca02c", linewidth=1.8, markersize=6, label="Linear 外推")
    ax.plot(ks, series("kalman"), "^:", color="#d62728", linewidth=1.8, markersize=6, label="Kalman")
    ax.set_xlabel("预测步数 k (未来第 k 帧)", fontsize=11)
    ax.set_ylabel("MAE (°)", fontsize=11)
    ax.set_title("未来方位预测 MAE 随预测步数衰减", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    # 标注: Model 在小k偏高是180°漏检惩罚放大
    ax.annotate("Model 在小 k 偏高:\n少量漏检被 180° 重罚放大", xy=(2, series("model_endtoend")[1]),
                xytext=(6, max(series("model_endtoend")) * 0.92),
                fontsize=8, color="#08519c",
                arrowprops=dict(arrowstyle="->", color="#08519c", lw=0.8))
    fig.tight_layout()
    out = FIG_DIR / "future_kmae.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    plot_ablation()
    plot_future_kmae()


if __name__ == "__main__":
    main()
