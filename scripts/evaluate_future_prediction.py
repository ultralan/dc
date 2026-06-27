"""未来帧方位预测评估: 三方法对比(论文核心实验).

对比:
  A. model_endtoend   — 模型直接输出 future_slot_logits → 未来方位
  B. linear_extrap    — 模型当前帧 (theta, omega) 匀速外推 32 帧
  C. kalman           — 模型当前帧作单次观测, AzimuthKalmanTracker 外推 32 帧

三方法共用同一份真值掩码 (target_valid), 公平比较. 输出按未来步数 k 分桶的 MAE 曲线
+ 整体 MAE/ACC@5/ACC@10, 分 static/moving 两组. 证明端到端未来预测优于"先定位再外推".

用法:
  uv run python scripts/evaluate_future_prediction.py \
    --run-dir D:/RealMAN/runs/mvp_full_5epoch/<时间戳> \
    --checkpoint <run-dir>/best.pt --split val --device cuda --batch-size 16

冒烟:
  uv run python scripts/evaluate_future_prediction.py --run-dir ... --max-batches 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# scripts/ 不是 package, 把它加进 sys.path 以复用 build_model/build_dataset.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from evaluate_realman_baseline_metrics import build_dataset, build_model  # noqa: E402

from uca8.geometry.uca8 import wrap_angle  # noqa: E402
from uca8.metrics import (  # noqa: E402
    circular_abs_error_deg,
    slot_logits_to_primary_azimuth_deg,
    target_slot_primary_azimuth_deg,
)
from uca8.postprocess.kalman_tracker import AzimuthKalmanTracker  # noqa: E402

# 三方法标识, 与输出 JSON 的 key 一致.
METHOD_MODEL = "model_endtoend"
METHOD_LINEAR = "linear_extrap"
METHOD_KALMAN = "kalman"
ALL_METHODS = (METHOD_MODEL, METHOD_LINEAR, METHOD_KALMAN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="未来帧方位预测三方法对比评估.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="k-MAE 曲线取样的未来帧索引 (1-based).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="帧间隔 (秒). 默认从 cfg (hop_length/sample_rate) 自动推算.",
    )
    parser.add_argument(
        "--acc-thresholds", type=float, nargs="+", default=[5.0, 10.0]
    )
    return parser.parse_args()


# ---------- 当前帧主声源提取 (B/C 共用输入) ----------


def extract_current_primary_theta_omega(
    slot_logits: torch.Tensor, *, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从当前帧 slot_logits 取主声源 (theta_rad, omega, valid).

    参数:
        slot_logits: [B, max_sources, 5], 特征顺序 [activity, sin, cos, rho, omega].

    返回:
        theta_cur_rad: [B] 弧度 (atan2(sin, cos));
        omega_cur:     [B] rad/s (模型预测, 非 真值 —— baseline 公平性核心);
        cur_valid:     [B] 当前帧模型是否预测出有声源.
    """
    activity = torch.sigmoid(slot_logits[..., 0])  # [B, S]
    slot_idx = activity.argmax(dim=-1)  # [B]
    cur_valid = activity.max(dim=-1).values > threshold  # [B]
    gathered = torch.gather(
        slot_logits,
        dim=1,
        index=slot_idx[..., None, None].expand(slot_idx.shape[0], 1, slot_logits.shape[-1]),
    ).squeeze(1)  # [B, 5]
    theta_cur_rad = torch.atan2(gathered[..., 1], gathered[..., 2])  # [B]
    omega_cur = gathered[..., 4]  # [B], rad/s
    return theta_cur_rad, omega_cur, cur_valid


# ---------- 三方法预测器 ----------


def predict_model(future_slot_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """方法 A: 模型端到端输出未来方位 (度). 复用 metrics, 支持任意前缀维度."""
    return slot_logits_to_primary_azimuth_deg(future_slot_logits)  # [B,K],[B,K]


def predict_linear(
    theta_cur_rad: torch.Tensor,
    omega_cur: torch.Tensor,
    *,
    k_future: int,
    dt: float,
) -> torch.Tensor:
    """方法 B: 匀速外推. 完全向量化.

    theta_future[k] = theta_cur + omega * k * dt   (k=1..K, omega 单位 rad/s).
    返回 [B, K] 的度数, 已 wrap 到 [-180, 180). pred_valid 恒 True (外推必有值).
    """
    k = torch.arange(1, k_future + 1, device=theta_cur_rad.device, dtype=theta_cur_rad.dtype)
    theta_future_rad = theta_cur_rad[:, None] + omega_cur[:, None] * k[None, :] * dt
    pred_deg = torch.rad2deg(wrap_angle(theta_future_rad))  # [-180,180)
    return pred_deg


def predict_kalman(
    theta_cur_rad: torch.Tensor,
    omega_cur: torch.Tensor,  # noqa: ARG001 保留参数签名对称; Kalman 默认不注入 omega.
    *,
    k_future: int,
    dt: float,
) -> torch.Tensor:
    """方法 C: Kalman 外推. 逐样本 (tracker 有状态).

    每 sample 一个 AzimuthKalmanTracker, 用模型当前帧角度作单次观测 update,
    然后 predict() k_future 次. omega≈0 (单次观测估不出角速度) —— 这正是 Kalman
    与 linear 的差异来源.
    """
    batch = theta_cur_rad.shape[0]
    pred_deg = torch.zeros(batch, k_future, device=theta_cur_rad.device, dtype=torch.float32)
    for b in range(batch):
        tracker = AzimuthKalmanTracker(dt=dt)
        tracker.update(float(theta_cur_rad[b].item()))
        for k in range(k_future):
            theta_pred_rad, _omega = tracker.predict()
            pred_deg[b, k] = math.degrees(
                float(torch.atan2(torch.sin(torch.tensor(theta_pred_rad)),
                                  torch.cos(torch.tensor(theta_pred_rad))))
            )
    return pred_deg


# ---------- 指标累加器 (按 k 列, 掩码乘法) ----------


class KAccumulator:
    """按未来步数 k 列累计三方法的误差/命中数. 掩码乘法, 无 python 循环."""

    def __init__(self, k_future: int, acc_thresholds: tuple[float, ...]) -> None:
        self.k = k_future
        self.thresholds = acc_thresholds
        self.err_sum = {m: torch.zeros(k_future) for m in ALL_METHODS}
        self.err_sum_no_penalty = torch.zeros(k_future)  # 仅 model, 不带漏检惩罚
        self.acc_count = {
            m: {t: torch.zeros(k_future) for t in acc_thresholds} for m in ALL_METHODS
        }
        self.acc_count_no_penalty = {t: torch.zeros(k_future) for t in acc_thresholds}
        self.count = torch.zeros(k_future)

    def update(
        self,
        err: dict[str, torch.Tensor],  # {method: [B,K] 误差}
        err_model_raw: torch.Tensor,   # [B,K] 模型不带惩罚的误差 (仅 target_valid 帧有效)
        mask: torch.Tensor,            # [B,K] target_valid
    ) -> None:
        # 输入可能在 GPU; 累加器全在 CPU, 统一搬过来避免 device 不一致.
        mask = mask.detach().cpu()
        err = {m: e.detach().cpu() for m, e in err.items()}
        err_model_raw = err_model_raw.detach().cpu()
        m = mask.float()  # [B,K]
        self.count += m.sum(dim=0)
        for method in ALL_METHODS:
            self.err_sum[method] += (err[method] * m).sum(dim=0)
            for t in self.thresholds:
                hit = ((err[method] <= t + 1e-5) & mask).sum(dim=0).float()
                self.acc_count[method][t] += hit
        # 模型无惩罚对照 (只在 target_valid 帧统计, 不对漏检罚 180)
        self.err_sum_no_penalty += (err_model_raw * m).sum(dim=0)
        for t in self.thresholds:
            self.acc_count_no_penalty[t] += ((err_model_raw <= t + 1e-5) & mask).sum(dim=0).float()

    def finalize(self, ks: list[int]) -> dict[str, Any]:
        total = self.count.clamp_min(1.0)
        result: dict[str, Any] = {
            "num_valid_future_frames": int(float(self.count.sum())),
            "k_mae_deg": {},
            "overall_mae_deg": {},
            "overall_acc_percent": {},
        }
        for method in ALL_METHODS:
            mae_per_k = self.err_sum[method] / total
            result["k_mae_deg"][method] = {str(k): _f(mae_per_k[k - 1]) for k in ks}
            result["overall_mae_deg"][method] = _f(self.err_sum[method].sum() / total.sum().clamp_min(1.0))
            result["overall_acc_percent"][method] = {
                f"acc{int(t)}": _f(self.acc_count[method][t].sum() / total.sum().clamp_min(1.0))
                for t in self.thresholds
            }
        # 模型无惩罚对照
        mae_np = self.err_sum_no_penalty / total
        result["model_endtoend_no_miss_penalty"] = {
            "overall_mae_deg": _f(self.err_sum_no_penalty.sum() / total.sum().clamp_min(1.0)),
            "overall_acc_percent": {
                f"acc{int(t)}": _f(self.acc_count_no_penalty[t].sum() / total.sum().clamp_min(1.0))
                for t in self.thresholds
            },
        }
        return result


def _f(x: torch.Tensor) -> float:
    return round(float(x), 4)


# ---------- 主流程 ----------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.run_dir / "config_resolved.yaml")

    # 帧间隔: 默认从 cfg 推算, 与数据集帧率对齐.
    if args.dt is not None:
        dt = args.dt
    else:
        dt = float(cfg.feature.hop_length) / float(cfg.feature.sample_rate)
    k_future = int(cfg.model.future_frames)

    dataset = build_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    ckpt_path = args.checkpoint or (args.run_dir / "best.pt")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
    )
    model.eval()

    thresholds = tuple(args.acc_thresholds)
    groups: dict[str, KAccumulator] = {
        name: KAccumulator(k_future, thresholds) for name in ("all", "static", "moving")
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            waveform = batch["waveform"].to(device)
            vad_history = batch["vad_history"].to(device)
            future_slot_state = batch["future_slot_state"].to(device)  # [B,K,4,5]

            pred = model(waveform, vad_history=vad_history, sample_id=batch.get("sample_id"))

            # 当前帧主声源 (B/C 输入)
            theta_cur_rad, omega_cur, _cur_valid = extract_current_primary_theta_omega(
                pred["slot_logits"]
            )

            # 真值未来方位与掩码
            target_theta_deg, target_valid = target_slot_primary_azimuth_deg(future_slot_state)

            # 三方法预测 (度)
            pred_a_deg, pred_a_valid = predict_model(pred["future_slot_logits"])
            pred_b_deg = predict_linear(theta_cur_rad, omega_cur, k_future=k_future, dt=dt)
            pred_c_deg = predict_kalman(theta_cur_rad, omega_cur, k_future=k_future, dt=dt)

            # 误差: A 带漏检惩罚 (target_valid 且 pred_a_valid=False → 180), B/C 恒有预测
            err_a_raw = circular_abs_error_deg(pred_a_deg, target_theta_deg)
            err_a = torch.where(
                pred_a_valid, err_a_raw, torch.full_like(err_a_raw, 180.0)
            )
            err_b = circular_abs_error_deg(pred_b_deg, target_theta_deg)
            err_c = circular_abs_error_deg(pred_c_deg, target_theta_deg)
            err = {METHOD_MODEL: err_a, METHOD_LINEAR: err_b, METHOD_KALMAN: err_c}

            mask = target_valid  # [B,K], 三方法共用

            # 按 motion 分组累加
            motions = [
                str(sid).split(":")[1] if ":" in str(sid) else "unknown"
                for sid in batch["sample_id"]
            ]
            for b_idx, motion in enumerate(motions):
                row = slice(b_idx, b_idx + 1)
                row_mask = mask[row]
                groups["all"].update(
                    {m: e[row] for m, e in err.items()}, err_a_raw[row], row_mask
                )
                if motion in groups:
                    groups[motion].update(
                        {m: e[row] for m, e in err.items()}, err_a_raw[row], row_mask
                    )

    metrics = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "dt_seconds": dt,
        "future_frames": k_future,
        "ks_reported": args.ks,
        "acc_thresholds_deg": list(args.acc_thresholds),
        "methods": list(ALL_METHODS),
        "groups": {name: acc.finalize(args.ks) for name, acc in groups.items()},
        "comparability_notes": [
            "三方法都用同一份真值 target_valid 掩码筛选 (真值未来某帧无声源 → 该帧从所有方法统计剔除).",
            "baseline (linear/kalman) 的初始 theta 与 omega 均来自模型当前帧 slot_logits 预测 (非真值), 保证不泄漏未来信息.",
            "model_endtoend 的漏检 (target_valid=True 但 pred_valid=False) 按 180° 惩罚, 与现有 slot_primary_localization_stats 一致; model_endtoend_no_miss_penalty 为不带该惩罚的对照.",
            "linear 直接信任模型 omega; kalman 单次角度观测后 omega≈0, 由过程噪声缓慢估计 —— 两者差异完全来自预测机制.",
        ],
    }
    out_path = args.output or (args.run_dir / "future_prediction_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nsaved={out_path}", flush=True)


if __name__ == "__main__":
    main()
