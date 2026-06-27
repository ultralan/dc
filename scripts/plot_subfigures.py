"""合并训练曲线+k-MAE曲线为父子图(a)(b), 符合中文期刊DL论文惯例."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})
P = {"blue": "#0F4D92", "red": "#B64342", "green": "#8BCF8B", "gray": "#767676"}
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
RD = Path("D:/RealMAN/runs/realman_ring1_loc_strong/20260625_121550")


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8))

    # (a) 训练loss
    hist = RD / "train_history.jsonl"
    steps, losses = [], []
    with hist.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("phase") == "train":
                steps.append(obj["global_step"])
                losses.append(obj["loss"])
    steps, losses = np.array(steps), np.array(losses)
    ax1.plot(steps, losses, "-", color=P["blue"], lw=0.5, alpha=0.25)
    if len(losses) > 100:
        w = max(len(losses) // 50, 20)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax1.plot(steps[w - 1:], smooth, "-", color=P["blue"], lw=1.5)
    ax1.set_xlabel("Training step", fontsize=8)
    ax1.set_ylabel("Loss", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=0.2, linewidth=0.4)
    ax1.set_title("(a) Training loss", fontsize=8)

    # (b) k-MAE曲线
    src = Path("D:/RealMAN/ring1_9ch/logs/eval_ring1_loc_strong_future.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    kmae = data["groups"]["all"]["k_mae_deg"]
    ks = sorted(int(k) for k in kmae["model_endtoend"].keys())
    ax2.plot(ks, [kmae["model_endtoend"][str(k)] for k in ks], "o-",
             color=P["blue"], lw=1.5, markersize=4, label="Model")
    ax2.plot(ks, [kmae["linear_extrap"][str(k)] for k in ks], "s--",
             color=P["green"], lw=1.2, markersize=3, label="Linear")
    ax2.plot(ks, [kmae["kalman"][str(k)] for k in ks], "^:",
             color=P["red"], lw=1.2, markersize=3, label="Kalman")
    ax2.set_xlabel("Prediction step $k$", fontsize=8)
    ax2.set_ylabel("MAE (°)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6.5, loc="upper left")
    ax2.grid(alpha=0.2, linewidth=0.4)
    ax2.set_title("(b) Future MAE vs step", fontsize=8)

    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"train_and_kmae.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved train_and_kmae")


if __name__ == "__main__":
    main()
