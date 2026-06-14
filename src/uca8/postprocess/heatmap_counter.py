from __future__ import annotations

"""基于方位热力图的简单声源数估计.

这是一个后处理基线/辅助工具: 不训练额外模型, 只在 circular azimuth heatmap
上找局部峰值, 用峰值数量估计声源数. 它适合做 sanity check, 不应替代主模型的
计数头.
"""

import torch


def estimate_source_count_from_heatmap(
    heatmap: torch.Tensor,
    *,
    max_sources: int,
    threshold: float = 0.30,
    relative_threshold: float = 0.60,
    min_separation_bins: int = 5,
) -> torch.Tensor:
    """用局部峰值非极大值抑制估计声源数.

    参数:
        heatmap: 最后一维为方位 bin 的热力图. 可以是概率, 也可以是 logits.
        max_sources: 返回计数上限.
        threshold: 峰值绝对下限.
        relative_threshold: 峰值相对当前帧最大值的下限.
        min_separation_bins: 两个峰之间至少相隔多少个方位 bin.

    返回:
        与 ``heatmap.shape[:-1]`` 对齐的 long tensor, 表示每帧估计声源数.
    """
    original_shape = heatmap.shape[:-1]
    bins = int(heatmap.shape[-1])
    flat = heatmap.reshape(-1, bins)
    if flat.numel() == 0:
        return torch.zeros(original_shape, dtype=torch.long, device=heatmap.device)
    if float(flat.min().item()) < 0.0 or float(flat.max().item()) > 1.0:
        flat = torch.sigmoid(flat)

    counts: list[int] = []
    for frame in flat:
        # 轻量平滑可以减少相邻 bin 抖动造成的伪峰.
        smoothed = (
            torch.roll(frame, shifts=1, dims=0)
            + 2.0 * frame
            + torch.roll(frame, shifts=-1, dims=0)
        ) / 4.0
        peak_floor = max(float(smoothed.max().item()) * relative_threshold, threshold)
        prev = torch.roll(smoothed, shifts=1, dims=0)
        next_frame = torch.roll(smoothed, shifts=-1, dims=0)
        candidate_mask = (smoothed >= peak_floor) & (smoothed >= prev) & (smoothed >= next_frame)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
        candidate_indices.sort(key=lambda index: float(smoothed[index].item()), reverse=True)
        selected: list[int] = []
        for index in candidate_indices:
            # 方位角是环形空间, 因此 0 bin 和最后一个 bin 也是相邻的.
            if all(
                _circular_distance(index, chosen, bins) > min_separation_bins
                for chosen in selected
            ):
                selected.append(index)
                if len(selected) >= max_sources:
                    break
        counts.append(len(selected))
    count_tensor = torch.tensor(counts, dtype=torch.long, device=heatmap.device)
    if not original_shape:
        return count_tensor.reshape(())
    return count_tensor.reshape(*original_shape)


def _circular_distance(left: int, right: int, period: int) -> int:
    """计算环形序列上两个 bin 的最短距离."""
    direct = abs(left - right)
    return min(direct, period - direct)


__all__ = ["estimate_source_count_from_heatmap"]
