from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from uca8.geometry.uca8 import angular_velocity, relative_source_state, wrap_angle

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy may be intentionally absent
    linear_sum_assignment = None


@dataclass(slots=True)
class LabelBuilderOutput:
    count: torch.Tensor
    vad_ratio: torch.Tensor
    heatmap: torch.Tensor
    slot_state: torch.Tensor


@dataclass(slots=True)
class SlotMemoryState:
    valid: torch.Tensor
    theta: torch.Tensor
    rho: torch.Tensor
    omega: torch.Tensor
    stale: torch.Tensor


class TrackTrendLabelBuilder:
    def __init__(
        self,
        *,
        num_heatmap_bins: int = 72,
        max_sources: int = 4,
        heatmap_sigma_bins: float = 1.5,
        frame_hop_seconds: float = 0.01,
        max_inactive_frames: int = 8,
        assignment_distance_weight: float = 0.2,
        assignment_stale_penalty: float = 0.05,
        max_assignment_cost: float = 1.25,
    ) -> None:
        self.num_heatmap_bins = num_heatmap_bins
        self.max_sources = max_sources
        self.heatmap_sigma_bins = heatmap_sigma_bins
        self.frame_hop_seconds = frame_hop_seconds
        self.max_inactive_frames = max_inactive_frames
        self.assignment_distance_weight = assignment_distance_weight
        self.assignment_stale_penalty = assignment_stale_penalty
        self.max_assignment_cost = max_assignment_cost
        self.azimuth_grid = torch.linspace(-math.pi, math.pi, steps=num_heatmap_bins + 1)[:-1]

    def build_sequence_targets(
        self,
        *,
        source_positions: torch.Tensor,
        array_positions: torch.Tensor,
        source_activity: torch.Tensor,
    ) -> LabelBuilderOutput:
        frames, _, _ = source_positions.shape
        theta, rho = relative_source_state(source_positions, array_positions[:, None, :])
        omega = torch.stack(
            [
                angular_velocity(theta[:, idx], self.frame_hop_seconds)
                for idx in range(theta.shape[1])
            ],
            dim=1,
        )
        count = source_activity.sum(dim=1).round().clamp_max(self.max_sources).long()
        vad_ratio = source_activity.float().mean(dim=1)
        heatmap = torch.stack(
            [self._build_heatmap(theta[t], source_activity[t]) for t in range(frames)],
            dim=0,
        )
        slot_states: list[torch.Tensor] = []
        slot_memory = self._init_slot_memory(theta.device)
        for t in range(frames):
            current, slot_memory = self._build_slot_state(
                theta[t],
                rho[t],
                omega[t],
                source_activity[t],
                slot_memory,
            )
            slot_states.append(current)
        return LabelBuilderOutput(
            count=count,
            vad_ratio=vad_ratio,
            heatmap=heatmap,
            slot_state=torch.stack(slot_states, dim=0),
        )

    def classify_future_motion(self, future_slot_state: torch.Tensor) -> torch.Tensor:
        """Return 0=clockwise, 1=stable, 2=counter_clockwise."""
        if future_slot_state.shape[0] == 0:
            return torch.tensor(1, dtype=torch.long)
        activity = future_slot_state[..., 0].mean(dim=0)
        slot_index = int(torch.argmax(activity).item())
        if activity[slot_index] < 0.5:
            return torch.tensor(1, dtype=torch.long)
        theta = torch.atan2(
            future_slot_state[:, slot_index, 1],
            future_slot_state[:, slot_index, 2],
        )
        delta = wrap_angle(theta[-1] - theta[0])
        threshold = 0.05
        if delta < -threshold:
            return torch.tensor(0, dtype=torch.long)
        if delta > threshold:
            return torch.tensor(2, dtype=torch.long)
        return torch.tensor(1, dtype=torch.long)

    def _build_heatmap(self, theta: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        azimuth_grid = self.azimuth_grid.to(device=theta.device, dtype=theta.dtype)
        heatmap = torch.zeros(self.num_heatmap_bins, dtype=torch.float32, device=theta.device)
        active_indices = torch.nonzero(activity > 0.5, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return heatmap
        for idx in active_indices:
            delta = wrap_angle(azimuth_grid - theta[idx])
            bins = delta / (2.0 * math.pi / self.num_heatmap_bins)
            heatmap = torch.maximum(
                heatmap,
                torch.exp(-0.5 * (bins / self.heatmap_sigma_bins) ** 2),
            )
        return heatmap

    def _build_slot_state(
        self,
        theta: torch.Tensor,
        rho: torch.Tensor,
        omega: torch.Tensor,
        activity: torch.Tensor,
        slot_memory: SlotMemoryState,
    ) -> tuple[torch.Tensor, SlotMemoryState]:
        slots = torch.zeros(self.max_sources, 5, dtype=torch.float32, device=theta.device)
        active = torch.nonzero(activity > 0.5, as_tuple=False).flatten()
        if active.numel() == 0:
            return slots, self._update_slot_memory(slot_memory, slots)
        assignment = self._assign_slots(theta[active], rho[active], slot_memory)
        for source_idx, slot_idx in enumerate(assignment[: self.max_sources]):
            src = active[source_idx]
            slots[slot_idx, 0] = 1.0
            slots[slot_idx, 1] = torch.sin(theta[src])
            slots[slot_idx, 2] = torch.cos(theta[src])
            slots[slot_idx, 3] = rho[src]
            slots[slot_idx, 4] = omega[src]
        return slots, self._update_slot_memory(slot_memory, slots)

    def _init_slot_memory(self, device: torch.device) -> SlotMemoryState:
        return SlotMemoryState(
            valid=torch.zeros(self.max_sources, dtype=torch.bool, device=device),
            theta=torch.zeros(self.max_sources, dtype=torch.float32, device=device),
            rho=torch.zeros(self.max_sources, dtype=torch.float32, device=device),
            omega=torch.zeros(self.max_sources, dtype=torch.float32, device=device),
            stale=torch.full(
                (self.max_sources,),
                self.max_inactive_frames + 1,
                dtype=torch.float32,
                device=device,
            ),
        )

    def _update_slot_memory(
        self,
        slot_memory: SlotMemoryState,
        slots: torch.Tensor,
    ) -> SlotMemoryState:
        next_memory = SlotMemoryState(
            valid=slot_memory.valid.clone(),
            theta=slot_memory.theta.clone(),
            rho=slot_memory.rho.clone(),
            omega=slot_memory.omega.clone(),
            stale=slot_memory.stale.clone(),
        )
        next_memory.stale[next_memory.valid] += 1.0
        active_slots = torch.nonzero(slots[:, 0] > 0.5, as_tuple=False).flatten()
        for slot_idx in active_slots.tolist():
            next_memory.valid[slot_idx] = True
            next_memory.theta[slot_idx] = torch.atan2(slots[slot_idx, 1], slots[slot_idx, 2])
            next_memory.rho[slot_idx] = slots[slot_idx, 3]
            next_memory.omega[slot_idx] = slots[slot_idx, 4]
            next_memory.stale[slot_idx] = 0.0
        expired = next_memory.valid & (next_memory.stale > float(self.max_inactive_frames))
        next_memory.valid[expired] = False
        next_memory.theta[expired] = 0.0
        next_memory.rho[expired] = 0.0
        next_memory.omega[expired] = 0.0
        return next_memory

    def _assign_slots(
        self,
        theta: torch.Tensor,
        rho: torch.Tensor,
        slot_memory: SlotMemoryState,
    ) -> list[int]:
        assigned = [-1] * theta.shape[0]
        used_slots: set[int] = set()
        reusable_slots = torch.nonzero(
            slot_memory.valid & (slot_memory.stale <= float(self.max_inactive_frames)),
            as_tuple=False,
        ).flatten()
        if reusable_slots.numel() > 0:
            cost = self._build_assignment_cost(theta, rho, reusable_slots, slot_memory)
            for row_idx, col_idx in self._solve_assignment(cost):
                slot_idx = int(reusable_slots[col_idx].item())
                if float(cost[row_idx, col_idx].item()) > self.max_assignment_cost:
                    continue
                assigned[row_idx] = slot_idx
                used_slots.add(slot_idx)
        remaining_slots = self._remaining_slots(slot_memory, used_slots)
        for source_idx in range(len(assigned)):
            if assigned[source_idx] < 0 and remaining_slots:
                assigned[source_idx] = remaining_slots.pop(0)
        return [slot_idx for slot_idx in assigned if slot_idx >= 0]

    def _build_assignment_cost(
        self,
        theta: torch.Tensor,
        rho: torch.Tensor,
        reusable_slots: torch.Tensor,
        slot_memory: SlotMemoryState,
    ) -> torch.Tensor:
        remembered_theta = slot_memory.theta[reusable_slots]
        remembered_rho = slot_memory.rho[reusable_slots].clamp_min(1e-3)
        remembered_stale = slot_memory.stale[reusable_slots]
        angular_cost = torch.abs(wrap_angle(theta[:, None] - remembered_theta[None, :])) / math.pi
        distance_cost = torch.abs(rho[:, None] - remembered_rho[None, :]) / remembered_rho[None, :]
        stale_cost = remembered_stale[None, :] / max(float(self.max_inactive_frames), 1.0)
        return (
            angular_cost
            + self.assignment_distance_weight * distance_cost
            + self.assignment_stale_penalty * stale_cost
        )

    def _solve_assignment(self, cost: torch.Tensor) -> list[tuple[int, int]]:
        if cost.numel() == 0:
            return []
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            return list(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
        pairs: list[tuple[int, int]] = []
        used_cols: set[int] = set()
        for row_idx in range(cost.shape[0]):
            row_costs = cost[row_idx]
            candidate_cols = torch.argsort(row_costs)
            for col_idx in candidate_cols.tolist():
                if col_idx not in used_cols:
                    used_cols.add(col_idx)
                    pairs.append((row_idx, col_idx))
                    break
        return pairs

    def _remaining_slots(
        self,
        slot_memory: SlotMemoryState,
        used_slots: set[int],
    ) -> list[int]:
        candidates: list[tuple[int, float, int]] = []
        for slot_idx in range(self.max_sources):
            if slot_idx in used_slots:
                continue
            if not bool(slot_memory.valid[slot_idx].item()):
                priority = 0
            else:
                priority = 1
            candidates.append((priority, -float(slot_memory.stale[slot_idx].item()), slot_idx))
        return [slot_idx for _, _, slot_idx in sorted(candidates)]
