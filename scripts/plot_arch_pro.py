"""专业级架构图: 标注张量维度、颜色编码模块、⊕融合、TCN dilation、多任务头分叉、参数量、图例."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
})

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
C = {"input": "#E8E8E8", "spec": "#D6E4F0", "ipd": "#D6E4F0", "srp": "#D6E4F0",
     "enc": "#D5E8D4", "fuse": "#FFE6CC", "tcn": "#FFF2CC", "head": "#E1D5E7",
     "arrow": "#444", "blue": "#0F4D92", "green": "#5B9B5B", "gold": "#D4A017",
     "violet": "#7B4FA3", "gray": "#666"}


def box(ax, x, y, w, h, text, fc, ec, fs=6.5, bold=False, shape_note=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w/2, y + h/2 + (0.06 if shape_note else 0), text,
            ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
    if shape_note:
        ax.text(x + w/2, y + h/2 - 0.08, shape_note, ha="center", va="center",
                fontsize=5.5, color=C["gray"], style="italic")


def arrow(ax, x1, y1, x2, y2, lw=0.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=8, lw=lw, color=C["arrow"]))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # === 输入 ===
    box(ax, 0.1, 1.7, 0.9, 0.8, "9ch audio", C["input"], C["gray"], bold=True,
        shape_note="[9, 20480]")
    arrow(ax, 1.0, 2.1, 1.3, 2.1)
    box(ax, 1.3, 1.7, 0.6, 0.8, "STFT", "#F5F5F5", C["gray"], fs=6.5,
        shape_note="512-pt")

    # === 三路特征 ===
    fx = 2.3
    box(ax, fx, 3.2, 1.2, 0.55, "log-mel", C["spec"], C["blue"], fs=6.5, shape_note="[1,128,64]")
    box(ax, fx, 2.4, 1.2, 0.55, "IPD", C["ipd"], C["blue"], fs=6.5, shape_note="[40,128,32]")
    box(ax, fx, 1.6, 1.2, 0.55, "SRP-PHAT", C["srp"], C["blue"], fs=6.5, shape_note="[1,128,72]")
    box(ax, fx, 0.9, 1.2, 0.4, "VAD", C["spec"], C["blue"], fs=6.5, shape_note="[128,1]")
    for fy in [3.47, 2.67, 1.87]:
        arrow(ax, 1.9, 2.1, fx, fy)

    # === 三路编码器 ===
    ex = 3.9
    box(ax, ex, 3.2, 0.85, 0.55, r"$E_{\rm spec}$", C["enc"], C["green"], fs=6.5,
        shape_note="[128,192]")
    box(ax, ex, 2.4, 0.85, 0.55, r"$E_{\rm ipd}$", C["enc"], C["green"], fs=6.5,
        shape_note="[128,96]")
    box(ax, ex, 1.6, 0.85, 0.55, r"$E_{\rm srp}$", C["enc"], C["green"], fs=6.5,
        shape_note="[128,96]")
    for fy in [3.47, 2.67, 1.87]:
        arrow(ax, fx + 1.2, fy, ex, fy)

    # === 融合(用文字代替⊕符号)
    ax.text(5.15, 2.55, "Concat", fontsize=7, ha="center", va="center", fontweight="bold",
            color=C["gold"])
    arrow(ax, ex + 0.85, 3.47, 5.0, 2.65)
    arrow(ax, ex + 0.85, 2.67, 5.1, 2.6)
    arrow(ax, ex + 0.85, 1.87, 5.0, 2.55)

    # === 融合层 ===
    box(ax, 5.35, 2.1, 0.75, 0.65, "Fusion\nLN+GELU", C["fuse"], C["gold"], fs=6, bold=True,
        shape_note="[128,384]")
    arrow(ax, 5.3, 2.55, 5.35, 2.45)

    # === TCN 主干 ===
    box(ax, 6.35, 1.8, 1.15, 1.2, "Causal TCN", C["tcn"], C["gold"], fs=7, bold=True,
        shape_note="d=[1,2,4,8,16,32]\n[128,384]")
    arrow(ax, 6.1, 2.42, 6.35, 2.4)
    # TCN 内部 dilation 块示意(小竖块)
    for i, d in enumerate([1, 2, 4, 8, 16, 32]):
        bx = 6.45 + i * 0.16
        ax.add_patch(mpatches.FancyBboxPatch((bx, 1.95), 0.12, 0.9,
                     boxstyle="round,pad=0.01", fc=C["gold"], ec="white", lw=0.5, alpha=0.6))
        ax.text(bx + 0.06, 1.88, str(d), fontsize=4.5, ha="center", color=C["gray"])

    # === 多任务头(分叉) ===
    hx = 7.85
    box(ax, hx, 3.4, 0.9, 0.5, "Current\nheads", C["head"], C["violet"], fs=5.5,
        shape_note="heatmap[72]\ncount[5]")
    box(ax, hx, 2.5, 0.9, 0.5, "Future\nheads", C["head"], C["violet"], fs=5.5, bold=True,
        shape_note="fut_heat[32,72]\nfut_slot[32,4,5]")
    box(ax, hx, 1.6, 0.9, 0.5, "Motion\nhead", C["head"], C["violet"], fs=5.5,
        shape_note="[3]")
    arrow(ax, 7.5, 2.7, hx, 3.65)
    arrow(ax, 7.5, 2.4, hx, 2.75)
    arrow(ax, 7.5, 2.1, hx, 1.85)

    # === 参数量 ===
    ax.text(5.0, 0.35, "Total: 16.79M parameters", fontsize=6.5, ha="center",
            color=C["gray"], style="italic")

    # === 段标注 ===
    for x, txt, col in [(2.9, "Frontend", C["blue"]), (4.3, "Encoders", C["green"]),
                        (5.7, "Fusion", C["gold"]), (6.9, "TCN backbone", C["gold"]),
                        (8.3, "Multi-task heads", C["violet"])]:
        ax.text(x, 4.05, txt, fontsize=6.5, ha="center", fontweight="bold", color=col)

    # === 图例 ===
    legend_items = [
        (C["spec"], "Feature"), (C["enc"], "Encoder"), (C["fuse"], "Fusion"),
        (C["tcn"], "TCN"), (C["head"], "Task head"),
    ]
    for i, (fc, label) in enumerate(legend_items):
        lx = 0.3 + i * 1.0
        ax.add_patch(mpatches.Rectangle((lx, 0.05), 0.15, 0.12, fc=fc, ec="black", lw=0.5))
        ax.text(lx + 0.2, 0.11, label, fontsize=5.5, va="center")

    fig.tight_layout()
    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"architecture.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved architecture (professional)")


if __name__ == "__main__":
    main()
