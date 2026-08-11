"""Seeded diameter-swap Monte Carlo and hybrid MD/swap scheduling."""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
import math

import torch

from .integrate import MDIntegrator
from .neighbors import VerletList
from .potential import CUTOFF_RATIO, analytic_potential, minimum_image, mixing_diameter, pair_potential
from .system import ParticleSystem


@dataclass(frozen=True)
class SwapStatistics:
    attempts: int
    accepted: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


def metropolis_acceptance_probability(delta_energy: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return 1.0 if delta_energy <= 0.0 else math.exp(-delta_energy / temperature)


def _diameters_with_swap(diameters: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Reference-only full-tensor swap retained for bitwise regression tests."""

    indices = torch.arange(diameters.shape[0], device=diameters.device)
    return torch.where(
        indices == i,
        diameters[j],
        torch.where(indices == j, diameters[i], diameters),
    )


def _swap_energy_delta_reference(
    system: ParticleSystem,
    i: int,
    j: int,
    neighbor_list: VerletList,
) -> torch.Tensor:
    """Exact local energy change for exchanging two particle diameters."""

    if i == j:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    if not (0 <= i < system.n_particles and 0 <= j < system.n_particles):
        raise IndexError("swap particle index out of range")
    neighbor_list.update(system)
    pairs = neighbor_list.pair_indices
    affected = (pairs[0] == i) | (pairs[1] == i) | (pairs[0] == j) | (pairs[1] == j)
    local_pairs = pairs[:, affected]
    before = analytic_potential(
        system.positions,
        system.diameters,
        system.box,
        pairs=local_pairs,
        active_mask=system.active_mask,
    ).energy
    swapped_diameters = _diameters_with_swap(system.diameters, i, j)
    after = analytic_potential(
        system.positions,
        swapped_diameters,
        system.box,
        pairs=local_pairs,
        active_mask=system.active_mask,
    ).energy
    return after - before


@dataclass(frozen=True)
class _SwapSweepWorkspace:
    """Frozen pair-list data reused by all attempts in one swap sweep."""

    pairs: torch.Tensor
    particle_pair_rows: torch.Tensor
    pair_count: int


def _particle_pair_rows(pairs: torch.Tensor, n_particles: int) -> torch.Tensor:
    """Build a padded, row-sorted particle-to-pair adjacency table once."""

    pair_count = int(pairs.shape[1])
    if pair_count == 0:
        return torch.empty((n_particles, 0), device=pairs.device, dtype=pairs.dtype)

    pair_rows = torch.arange(pair_count, device=pairs.device, dtype=pairs.dtype)
    incidence_particles = torch.cat((pairs[0], pairs[1]))
    incidence_rows = torch.cat((pair_rows, pair_rows))
    order = torch.argsort(incidence_particles, stable=True)
    sorted_particles = incidence_particles[order]
    sorted_rows = incidence_rows[order]
    particles = torch.arange(n_particles, device=pairs.device, dtype=pairs.dtype)
    starts = torch.searchsorted(sorted_particles, particles, right=False)
    ends = torch.searchsorted(sorted_particles, particles, right=True)
    counts = ends - starts
    width = int(counts.max().item())
    slots = torch.arange(width, device=pairs.device, dtype=pairs.dtype)
    gather_positions = starts[:, None] + slots[None, :]
    valid_slots = slots[None, :] < counts[:, None]
    safe_positions = torch.clamp(gather_positions, max=sorted_rows.shape[0] - 1)
    sentinel = torch.full_like(safe_positions, pair_count)
    table = torch.where(valid_slots, sorted_rows[safe_positions], sentinel)
    return torch.sort(table, dim=1).values


def _prepare_swap_sweep(system: ParticleSystem, neighbor_list: VerletList) -> _SwapSweepWorkspace:
    """Update once, then index the fixed candidate pair list for this sweep."""

    neighbor_list.update(system)
    pairs = neighbor_list.pair_indices
    return _SwapSweepWorkspace(
        pairs=pairs,
        particle_pair_rows=_particle_pair_rows(pairs, system.n_particles),
        pair_count=int(pairs.shape[1]),
    )


def _affected_pair_rows(workspace: _SwapSweepWorkspace, i: int, j: int) -> torch.Tensor:
    """Return the exact lexicographic pair rows touching either selected particle."""

    if workspace.pair_count == 0:
        return torch.empty((0,), device=workspace.pairs.device, dtype=workspace.pairs.dtype)
    candidate_rows = torch.cat((workspace.particle_pair_rows[i], workspace.particle_pair_rows[j]))
    candidate_rows = torch.sort(candidate_rows).values
    previous = torch.cat((torch.full_like(candidate_rows[:1], -1), candidate_rows[:-1]))
    keep = (candidate_rows < workspace.pair_count) & (candidate_rows != previous)
    return candidate_rows[keep]


def _energy_only(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    pairs: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match ``analytic_potential(...).energy`` without constructing forces."""

    if active_mask is not None:
        active = active_mask.to(device=positions.device, dtype=torch.bool)
        keep = active[pairs[0]] | active[pairs[1]]
        pairs = pairs[:, keep]
    i, j = pairs[0], pairs[1]
    displacement = minimum_image(positions[i] - positions[j], box)
    radius = torch.linalg.vector_norm(displacement, dim=1)
    sigma_ij = mixing_diameter(diameters[i], diameters[j])
    interacting = radius < CUTOFF_RATIO * sigma_ij
    pairs = pairs[:, interacting]
    i, j = pairs[0], pairs[1]
    displacement = displacement[interacting]
    radius = radius[interacting]
    sigma_ij = sigma_ij[interacting]
    del i, j, displacement
    pair_energy = pair_potential(radius, sigma_ij, derivatives=0)
    return pair_energy.sum()


def _swap_diameter_entries_(diameters: torch.Tensor, indices: torch.Tensor) -> None:
    """Exchange two entries in place; applying it twice restores the input."""

    values = diameters[indices].clone()
    diameters[indices] = values.flip(0)


def _local_pairs_from_mask(pairs: torch.Tensor, i: int, j: int) -> torch.Tensor:
    affected = (pairs[0] == i) | (pairs[1] == i) | (pairs[0] == j) | (pairs[1] == j)
    return pairs[:, affected]


def swap_energy_delta(
    system: ParticleSystem,
    i: int,
    j: int,
    neighbor_list: VerletList,
) -> torch.Tensor:
    """Exact local energy change for exchanging two particle diameters."""

    if i == j:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    if not (0 <= i < system.n_particles and 0 <= j < system.n_particles):
        raise IndexError("swap particle index out of range")
    neighbor_list.update(system)
    local_pairs = _local_pairs_from_mask(neighbor_list.pair_indices, i, j)
    before = _energy_only(
        system.positions,
        system.diameters,
        system.box,
        pairs=local_pairs,
        active_mask=system.active_mask,
    )
    indices = torch.tensor((i, j), device=system.device, dtype=torch.int64)
    _swap_diameter_entries_(system.diameters, indices)
    try:
        after = _energy_only(
            system.positions,
            system.diameters,
            system.box,
            pairs=local_pairs,
            active_mask=system.active_mask,
        )
    finally:
        _swap_diameter_entries_(system.diameters, indices)
    return after - before


def _proposal_draws(
    system: ParticleSystem,
    generator: torch.Generator,
    attempts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    active_indices = torch.nonzero(system.active_mask.detach().cpu(), as_tuple=False).flatten()
    n_active = int(active_indices.numel())
    if attempts == 0 or n_active < 2:
        return None
    first_slot = torch.randint(n_active, (attempts,), generator=generator, device="cpu")
    second_slot = torch.randint(n_active - 1, (attempts,), generator=generator, device="cpu")
    second_slot = second_slot + (second_slot >= first_slot)
    uniforms = torch.rand((attempts,), generator=generator, device="cpu", dtype=torch.float64)
    return active_indices[first_slot], active_indices[second_slot], uniforms


def _accepts(delta: float, uniform: torch.Tensor, temperature: float) -> bool:
    return delta <= 0.0 or math.log(float(uniform)) < -delta / temperature


def _swap_sweep_reference(
    system: ParticleSystem,
    temperature: float,
    generator: torch.Generator,
    neighbor_list: VerletList,
    *,
    n_attempts: int | None = None,
    acceptance_decisions: MutableSequence[bool] | None = None,
) -> SwapStatistics:
    """Pre-restructure sweep retained solely as a bitwise reference oracle."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if str(generator.device) != "cpu":
        raise ValueError("ButterflyCone generators must be CPU generators")
    attempts = system.n_particles if n_attempts is None else int(n_attempts)
    if attempts < 0:
        raise ValueError("n_attempts must be nonnegative")
    proposals = _proposal_draws(system, generator, attempts)
    if proposals is None:
        return SwapStatistics(0, 0)
    first_particles, second_particles, uniforms = proposals
    accepted = 0
    for attempt in range(attempts):
        i = int(first_particles[attempt])
        j = int(second_particles[attempt])
        delta = float(_swap_energy_delta_reference(system, i, j, neighbor_list))
        decision = _accepts(delta, uniforms[attempt], temperature)
        if acceptance_decisions is not None:
            acceptance_decisions.append(decision)
        if decision:
            system.diameters = _diameters_with_swap(system.diameters, i, j)
            accepted += 1
    return SwapStatistics(attempts, accepted)


def diameter_swap_sweep(
    system: ParticleSystem,
    temperature: float,
    generator: torch.Generator,
    neighbor_list: VerletList,
    *,
    n_attempts: int | None = None,
    acceptance_decisions: MutableSequence[bool] | None = None,
    workspace: _SwapSweepWorkspace | None = None,
) -> SwapStatistics:
    """Attempt sequential, symmetric Metropolis diameter exchanges exactly."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if str(generator.device) != "cpu":
        raise ValueError("ButterflyCone generators must be CPU generators")
    attempts = system.n_particles if n_attempts is None else int(n_attempts)
    if attempts < 0:
        raise ValueError("n_attempts must be nonnegative")
    proposals = _proposal_draws(system, generator, attempts)
    if proposals is None:
        return SwapStatistics(0, 0)
    first_particles, second_particles, uniforms = proposals
    prepared = _prepare_swap_sweep(system, neighbor_list) if workspace is None else workspace
    swap_indices = torch.stack((first_particles, second_particles), dim=1).to(system.device)
    accepted = 0
    for attempt in range(attempts):
        i = int(first_particles[attempt])
        j = int(second_particles[attempt])
        rows = _affected_pair_rows(prepared, i, j)
        local_pairs = prepared.pairs[:, rows]
        before = _energy_only(
            system.positions,
            system.diameters,
            system.box,
            pairs=local_pairs,
            active_mask=system.active_mask,
        )
        indices = swap_indices[attempt]
        _swap_diameter_entries_(system.diameters, indices)
        after = _energy_only(
            system.positions,
            system.diameters,
            system.box,
            pairs=local_pairs,
            active_mask=system.active_mask,
        )
        delta = float(after - before)
        decision = _accepts(delta, uniforms[attempt], temperature)
        if acceptance_decisions is not None:
            acceptance_decisions.append(decision)
        if decision:
            accepted += 1
        else:
            _swap_diameter_entries_(system.diameters, indices)
    return SwapStatistics(attempts, accepted)


class HybridSwapMD:
    """Alternate physical MD blocks with nonphysical equilibrium swap sweeps."""

    def __init__(
        self,
        integrator: MDIntegrator,
        *,
        temperature: float,
        generator: torch.Generator,
        md_steps: int,
        swap_attempts: int | None = None,
    ) -> None:
        if md_steps < 0:
            raise ValueError("md_steps must be nonnegative")
        self.integrator = integrator
        self.temperature = float(temperature)
        self.generator = generator
        self.md_steps = int(md_steps)
        self.swap_attempts = swap_attempts
        self.statistics = SwapStatistics(0, 0)

    def cycle(self, cycles: int = 1) -> SwapStatistics:
        if cycles < 0:
            raise ValueError("cycles must be nonnegative")
        attempts = self.statistics.attempts
        accepted = self.statistics.accepted
        for _ in range(cycles):
            self.integrator.step(self.md_steps)
            block = diameter_swap_sweep(
                self.integrator.system,
                self.temperature,
                self.generator,
                self.integrator.neighbor_list,
                n_attempts=self.swap_attempts,
            )
            attempts += block.attempts
            accepted += block.accepted
            # Diameter changes invalidate the force cached by velocity-Verlet.
            result = self.integrator.neighbor_list.evaluate(self.integrator.system)
            self.integrator.forces = result.forces
            self.integrator.potential_energy = result.energy
            self.integrator.virial = result.virial
        self.statistics = SwapStatistics(attempts, accepted)
        return self.statistics
