"""Smoothed repulsive pair potential and deterministic force evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

CUTOFF_RATIO = 1.25
C0 = -28.0 * CUTOFF_RATIO**-12
C2 = 48.0 * CUTOFF_RATIO**-14
C4 = -21.0 * CUTOFF_RATIO**-16
NONADDITIVITY = 0.2


@dataclass(frozen=True)
class PotentialResult:
    energy: torch.Tensor
    forces: torch.Tensor
    virial: torch.Tensor
    pair_energies: torch.Tensor
    pair_virials: torch.Tensor
    pair_indices: torch.Tensor


def minimum_image(displacements: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Apply the nearest-image convention in an orthorhombic box."""

    return displacements - box * torch.round(displacements / box)


def mixing_diameter(sigma_i: torch.Tensor, sigma_j: torch.Tensor) -> torch.Tensor:
    return 0.5 * (sigma_i + sigma_j) * (1.0 - NONADDITIVITY * torch.abs(sigma_i - sigma_j))


def pair_potential(
    radius: torch.Tensor,
    sigma_ij: torch.Tensor,
    *,
    derivatives: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Evaluate pair energy and optionally first/second radial derivatives."""

    if derivatives not in (0, 1, 2):
        raise ValueError("derivatives must be 0, 1, or 2")
    if radius.shape != sigma_ij.shape:
        raise ValueError("radius and sigma_ij must have identical shapes")
    positive_radius = torch.clamp(radius, min=torch.finfo(radius.dtype).eps)
    x = positive_radius / sigma_ij
    inside = x < CUTOFF_RATIO
    x_m12 = x.pow(-12)
    value_raw = x_m12 + C0 + C2 * x.square() + C4 * x.pow(4)
    value = torch.where(inside, value_raw, torch.zeros_like(value_raw))
    if derivatives == 0:
        return value
    dv_dx = -12.0 * x.pow(-13) + 2.0 * C2 * x + 4.0 * C4 * x.pow(3)
    first = torch.where(inside, dv_dx / sigma_ij, torch.zeros_like(value))
    if derivatives == 1:
        return value, first
    d2v_dx2 = 156.0 * x.pow(-14) + 2.0 * C2 + 12.0 * C4 * x.square()
    second = torch.where(inside, d2v_dx2 / sigma_ij.square(), torch.zeros_like(value))
    return value, first, second


def all_pairs(n_particles: int, device: torch.device | str) -> torch.Tensor:
    return torch.triu_indices(n_particles, n_particles, offset=1, device=device)


def _deterministic_particle_sum(
    particle_indices: torch.Tensor,
    contributions: torch.Tensor,
    n_particles: int,
) -> torch.Tensor:
    """Reduce by particle after sorting, without duplicate-index atomics."""

    if particle_indices.numel() == 0:
        return contributions.new_zeros((n_particles, contributions.shape[-1]))
    order = torch.argsort(particle_indices, stable=True)
    sorted_indices = particle_indices[order]
    sorted_values = contributions[order]
    particles = torch.arange(n_particles, device=particle_indices.device, dtype=particle_indices.dtype)
    starts = torch.searchsorted(sorted_indices, particles, right=False)
    ends = torch.searchsorted(sorted_indices, particles, right=True)
    counts = ends - starts
    width = int(counts.max().item())
    slots = torch.arange(width, device=particle_indices.device, dtype=particle_indices.dtype)
    gather_positions = starts[:, None] + slots[None, :]
    valid = slots[None, :] < counts[:, None]
    safe_positions = torch.clamp(gather_positions, max=sorted_values.shape[0] - 1)
    padded = sorted_values[safe_positions]
    padded = torch.where(valid[..., None], padded, torch.zeros_like(padded))
    return padded.sum(dim=1)


def analytic_potential(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    pairs: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> PotentialResult:
    """Return total energy, analytic forces, and per-pair virials.

    Pair rows are expected in deterministic lexicographic ``i<j`` order.  With
    an active mask, frozen-frozen pairs are excluded and frozen forces are
    zeroed, while active-frozen interactions remain in the energy and active
    force.
    """

    n_particles = int(positions.shape[0])
    if positions.shape != (n_particles, 3) or diameters.shape != (n_particles,):
        raise ValueError("positions must be (N,3) and diameters must be (N,)")
    if pairs is None:
        pairs = all_pairs(n_particles, positions.device)
    if pairs.shape[0] != 2:
        raise ValueError("pairs must have shape (2, P)")
    if active_mask is None:
        active = torch.ones(n_particles, dtype=torch.bool, device=positions.device)
    else:
        active = active_mask.to(device=positions.device, dtype=torch.bool)
        keep = active[pairs[0]] | active[pairs[1]]
        pairs = pairs[:, keep]
    i, j = pairs[0], pairs[1]
    displacement = minimum_image(positions[i] - positions[j], box)
    radius = torch.linalg.vector_norm(displacement, dim=1)
    sigma_ij = mixing_diameter(diameters[i], diameters[j])
    # Canonicalize every evaluator to the same ordered set of nonzero pairs.
    # This makes a Verlet candidate set bitwise-comparable with the O(N^2)
    # reference: zero-force padding pairs never alter a reduction tree.
    interacting = radius < CUTOFF_RATIO * sigma_ij
    pairs = pairs[:, interacting]
    i, j = pairs[0], pairs[1]
    displacement = displacement[interacting]
    radius = radius[interacting]
    sigma_ij = sigma_ij[interacting]
    pair_energy, derivative = pair_potential(radius, sigma_ij, derivatives=1)
    safe_radius = torch.clamp(radius, min=torch.finfo(radius.dtype).eps)
    force_i = -(derivative / safe_radius)[:, None] * displacement
    pair_virial = -derivative * radius

    contribution_indices = torch.cat((i, j))
    contribution_values = torch.cat((force_i * active[i, None], -force_i * active[j, None]), dim=0)
    forces = _deterministic_particle_sum(contribution_indices, contribution_values, n_particles)
    return PotentialResult(
        energy=pair_energy.sum(),
        forces=forces,
        virial=pair_virial.sum(),
        pair_energies=pair_energy,
        pair_virials=pair_virial,
        pair_indices=pairs,
    )


def brute_force(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> PotentialResult:
    return analytic_potential(positions, diameters, box, active_mask=active_mask)


def autograd_forces(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    pairs: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Test-only force path obtained as ``-grad(total energy)``."""

    differentiable = positions.detach().clone().requires_grad_(True)
    energy = analytic_potential(
        differentiable,
        diameters,
        box,
        pairs=pairs,
        active_mask=active_mask,
    ).energy
    forces = -torch.autograd.grad(energy, differentiable, create_graph=False)[0]
    if active_mask is not None:
        forces = forces * active_mask[:, None]
    return forces.detach()
