"""Deterministic cell lists and Verlet neighbor lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .potential import CUTOFF_RATIO, PotentialResult, all_pairs, analytic_potential, minimum_image
from .system import ParticleSystem


def _filter_by_radius(
    pairs: torch.Tensor,
    positions: torch.Tensor,
    box: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    if pairs.shape[1] == 0:
        return pairs
    displacement = minimum_image(positions[pairs[0]] - positions[pairs[1]], box)
    keep = torch.sum(displacement.square(), dim=1) < radius * radius
    return pairs[:, keep]


def _half_neighbor_offsets(device: torch.device) -> torch.Tensor:
    offsets: list[tuple[int, int, int]] = [(0, 0, 0)]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    offsets.append((dx, dy, dz))
    return torch.tensor(offsets, device=device, dtype=torch.int64)


def cell_list_pairs(positions: torch.Tensor, box: torch.Tensor, list_radius: float) -> torch.Tensor:
    """Build unique, lexicographically sorted candidate pairs within a radius.

    Cell occupancy is represented as a padded sorted table.  No duplicate-index
    writes or atomic reductions occur, including on MPS.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if list_radius <= 0.0:
        raise ValueError("list_radius must be positive")
    n_particles = int(positions.shape[0])
    if n_particles < 2:
        return torch.empty((2, 0), device=positions.device, dtype=torch.int64)
    n_cells = torch.floor(box / list_radius).to(torch.int64)
    # With fewer than three cells, periodic +/- neighbor offsets alias.  The
    # exact brute candidate builder is cheaper and simpler for those small boxes.
    if bool(torch.any(n_cells < 3)):
        return _filter_by_radius(all_pairs(n_particles, positions.device), positions, box, list_radius)

    cell_width = box / n_cells.to(dtype=box.dtype)
    wrapped = torch.remainder(positions, box)
    coordinates = torch.floor(wrapped / cell_width).to(torch.int64)
    coordinates = torch.minimum(coordinates, n_cells - 1)
    ny, nz = n_cells[1], n_cells[2]
    cell_ids = (coordinates[:, 0] * ny + coordinates[:, 1]) * nz + coordinates[:, 2]
    particle_order = torch.argsort(cell_ids, stable=True)
    sorted_cell_ids = cell_ids[particle_order]
    cell_count = int(torch.prod(n_cells).item())
    cells = torch.arange(cell_count, device=positions.device, dtype=torch.int64)
    starts = torch.searchsorted(sorted_cell_ids, cells, right=False)
    ends = torch.searchsorted(sorted_cell_ids, cells, right=True)
    counts = ends - starts
    max_occupancy = int(counts.max().item())
    slots = torch.arange(max_occupancy, device=positions.device, dtype=torch.int64)
    gather_positions = starts[:, None] + slots[None, :]
    valid_slots = slots[None, :] < counts[:, None]
    safe_positions = torch.clamp(gather_positions, max=n_particles - 1)
    table = torch.where(valid_slots, particle_order[safe_positions], -torch.ones_like(safe_positions))

    cx = cells // (n_cells[1] * n_cells[2])
    remainder = cells % (n_cells[1] * n_cells[2])
    cell_coordinates = torch.stack((cx, remainder // n_cells[2], remainder % n_cells[2]), dim=1)
    candidate_i: list[torch.Tensor] = []
    candidate_j: list[torch.Tensor] = []
    for offset in _half_neighbor_offsets(positions.device):
        other_coordinates = torch.remainder(cell_coordinates + offset, n_cells)
        other_ids = (other_coordinates[:, 0] * ny + other_coordinates[:, 1]) * nz + other_coordinates[:, 2]
        left = table[:, :, None].expand(-1, -1, max_occupancy)
        right = table[other_ids][:, None, :].expand(-1, max_occupancy, -1)
        valid = (left >= 0) & (right >= 0)
        low = torch.minimum(left, right)
        high = torch.maximum(left, right)
        valid &= low < high
        candidate_i.append(low[valid])
        candidate_j.append(high[valid])

    i = torch.cat(candidate_i)
    j = torch.cat(candidate_j)
    keys = i * n_particles + j
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    i, j = i[order], j[order]
    if keys.numel() == 0:
        return torch.stack((i, j))
    unique = torch.cat((torch.ones(1, device=keys.device, dtype=torch.bool), keys[1:] != keys[:-1]))
    pairs = torch.stack((i[unique], j[unique]))
    return _filter_by_radius(pairs, positions, box, list_radius)


@dataclass
class VerletList:
    skin: float
    pair_indices: torch.Tensor
    reference_positions: torch.Tensor
    list_radius: float
    rebuild_count: int = 1

    @classmethod
    def from_system(cls, system: ParticleSystem, skin: float = 0.3) -> "VerletList":
        if skin <= 0.0:
            raise ValueError("skin must be positive")
        list_radius = CUTOFF_RATIO * float(system.diameters.max().item()) + float(skin)
        pairs = cell_list_pairs(system.positions, system.box, list_radius)
        return cls(
            skin=float(skin),
            pair_indices=pairs,
            reference_positions=system.positions.detach().clone(),
            list_radius=list_radius,
        )

    def needs_rebuild(self, positions: torch.Tensor, box: torch.Tensor) -> bool:
        displacement = minimum_image(positions - self.reference_positions, box)
        maximum = torch.linalg.vector_norm(displacement, dim=1).max()
        return bool(maximum > 0.5 * self.skin)

    def update(self, system: ParticleSystem) -> bool:
        if not self.needs_rebuild(system.positions, system.box):
            return False
        self.list_radius = CUTOFF_RATIO * float(system.diameters.max().item()) + self.skin
        self.pair_indices = cell_list_pairs(system.positions, system.box, self.list_radius)
        self.reference_positions = system.positions.detach().clone()
        self.rebuild_count += 1
        return True

    def evaluate(self, system: ParticleSystem) -> PotentialResult:
        self.update(system)
        return analytic_potential(
            system.positions,
            system.diameters,
            system.box,
            pairs=self.pair_indices,
            active_mask=system.active_mask,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "skin": self.skin,
            "pair_indices": self.pair_indices.detach().clone(),
            "reference_positions": self.reference_positions.detach().clone(),
            "list_radius": self.list_radius,
            "rebuild_count": self.rebuild_count,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> "VerletList":
        target = state["pair_indices"].device if device is None else torch.device(device)
        return cls(
            skin=float(state["skin"]),
            pair_indices=state["pair_indices"].detach().clone().to(target),
            reference_positions=state["reference_positions"].detach().clone().to(target),
            list_radius=float(state["list_radius"]),
            rebuild_count=int(state["rebuild_count"]),
        )
