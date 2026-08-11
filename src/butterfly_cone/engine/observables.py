"""Thermodynamic, structural, and dynamical observables."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .potential import all_pairs, minimum_image
from .neighbors import VerletList
from .system import ParticleSystem


def degrees_of_freedom(system: ParticleSystem, *, remove_com: bool = True) -> int:
    n_active = int(system.active_mask.sum().item())
    if n_active == 0:
        return 0
    return 3 * n_active - (3 if remove_com and n_active > 1 else 0)


def kinetic_energy(system: ParticleSystem) -> torch.Tensor:
    velocities = system.velocities[system.active_mask]
    return 0.5 * velocities.square().sum()


def potential_energy(system: ParticleSystem, neighbor_list: VerletList | None = None) -> torch.Tensor:
    neighbors = VerletList.from_system(system) if neighbor_list is None else neighbor_list
    return neighbors.evaluate(system).energy


def temperature(system: ParticleSystem, *, remove_com: bool = True) -> torch.Tensor:
    ndof = degrees_of_freedom(system, remove_com=remove_com)
    if ndof == 0:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    return 2.0 * kinetic_energy(system) / ndof


def pressure(system: ParticleSystem, virial: torch.Tensor) -> torch.Tensor:
    """Instantaneous pressure ``(2 K + sum r_ij.F_ij)/(3 V)``."""

    volume = torch.prod(system.box)
    return (2.0 * kinetic_energy(system) + virial) / (3.0 * volume)


def radial_distribution(
    positions: torch.Tensor,
    box: torch.Tensor,
    *,
    bins: int = 200,
    r_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centers and finite-N normalized unique-pair ``g(r)``."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    if r_max is None:
        r_max = 0.5 * float(box.min().item())
    if r_max <= 0.0 or r_max > 0.5 * float(box.min().item()) + 1e-12:
        raise ValueError("r_max must lie in (0, min(box)/2]")
    n_particles = int(positions.shape[0])
    pairs = all_pairs(n_particles, positions.device)
    displacement = minimum_image(positions[pairs[0]] - positions[pairs[1]], box)
    distances = torch.linalg.vector_norm(displacement, dim=1)
    counts = torch.histc(distances, bins=bins, min=0.0, max=float(r_max))
    edges = torch.linspace(0.0, float(r_max), bins + 1, device=positions.device, dtype=positions.dtype)
    shell_volume = (4.0 / 3.0) * torch.pi * (edges[1:].pow(3) - edges[:-1].pow(3))
    volume = torch.prod(box)
    ideal_pairs = n_particles * max(n_particles - 1, 0) * shell_volume / (2.0 * volume)
    g = torch.where(ideal_pairs > 0, counts.to(positions.dtype) / ideal_pairs, torch.zeros_like(ideal_pairs))
    return 0.5 * (edges[1:] + edges[:-1]), g


@dataclass(frozen=True)
class DisplacementReference:
    unwrapped_positions: torch.Tensor

    @classmethod
    def capture(cls, system: ParticleSystem) -> "DisplacementReference":
        return cls(system.unwrapped_positions.detach().clone())


def _active_displacements(system: ParticleSystem, reference: DisplacementReference) -> torch.Tensor:
    if reference.unwrapped_positions.shape != system.unwrapped_positions.shape:
        raise ValueError("reference and system particle counts differ")
    return (system.unwrapped_positions - reference.unwrapped_positions)[system.active_mask]


def mean_squared_displacement(system: ParticleSystem, reference: DisplacementReference) -> torch.Tensor:
    displacement = _active_displacements(system, reference)
    if displacement.shape[0] == 0:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    return displacement.square().sum(dim=1).mean()


def overlap(
    system: ParticleSystem,
    reference: DisplacementReference,
    *,
    cutoff: float = 0.3,
) -> torch.Tensor:
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    displacement = _active_displacements(system, reference)
    if displacement.shape[0] == 0:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    return (displacement.square().sum(dim=1) < cutoff * cutoff).to(system.dtype).mean()


def self_intermediate_scattering(
    system: ParticleSystem,
    reference: DisplacementReference,
    *,
    wave_number: float = 7.1,
) -> torch.Tensor:
    """Axis-averaged self intermediate scattering function at ``|k|``."""

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    displacement = _active_displacements(system, reference)
    if displacement.shape[0] == 0:
        return torch.zeros((), device=system.device, dtype=system.dtype)
    return torch.cos(wave_number * displacement).mean()
