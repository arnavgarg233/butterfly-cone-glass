"""End-to-end CPU-float64 entropy measurement on a ButterflyCone particle system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

from butterfly_cone.engine.potential import analytic_potential
from butterfly_cone.engine.system import ParticleSystem
from butterfly_cone.mechanics.hessian import coerce_analysis_system, hessian_dense
from butterfly_cone.mechanics.inherent_structure import minimize_to_IS

from .configurational import entropy_difference
from .mixing import binned_diameter_mixing_entropy
from .thermodynamic import excess_entropy, sackur_tetrode_entropy
from .vibrational import HarmonicEntropy, harmonic_vibrational_entropy


@dataclass(frozen=True)
class BasinMeasurement:
    """Inherent-structure and harmonic-spectrum result for one configuration."""

    n_particles: int
    e_is_per_particle: float
    fmax: float
    converged: bool
    fire_steps: int
    lbfgs_steps: int
    eigenvalues: np.ndarray
    harmonic: HarmonicEntropy


@dataclass(frozen=True)
class EntropyPoint:
    """All per-particle entropy legs and diagnostics at one temperature."""

    temperature: float
    density: float
    s_ideal_reference: float
    s_excess: float
    s_total_reference: float
    s_reported_mixing: float
    s_total: float
    s_vibrational: float
    s_anharmonic: float
    s_effective_mixing: float
    s_glass: float
    s_configurational: float
    u_ex: float
    beta_f_ex: float
    head_integral: float
    head_relative_rms_residual: float
    e_is_per_particle: float
    fmax: float
    converged: bool
    n_modes: int

    def as_record(self) -> dict[str, float | int | bool]:
        """Return a flat serialization-safe record."""

        return asdict(self)


def _maximum_force(system: ParticleSystem) -> float:
    forces = analytic_potential(
        system.positions,
        system.diameters,
        system.box,
        active_mask=system.active_mask,
    ).forces
    return float(torch.linalg.vector_norm(forces, dim=1).max().detach().cpu())


def measure_harmonic_basin(
    system: ParticleSystem | Any,
    *,
    temperature: float,
    minimize: bool = True,
    minimize_kwargs: Mapping[str, object] | None = None,
    expected_zero_modes: int = 3,
    zero_tolerance: float = 1.0e-3,
    planck_constant: float = 1.0,
    mass: float = 1.0,
) -> BasinMeasurement:
    """Minimize, diagonalize the analytic Hessian, and measure harmonic entropy."""

    analysis_system = coerce_analysis_system(system)
    if not bool(torch.all(analysis_system.active_mask)):
        raise ValueError("bulk entropy requires every particle to be active")
    controls = dict(minimize_kwargs or {})
    tolerance = float(controls.get("tol", 1.0e-8))
    if minimize:
        inherent = minimize_to_IS(analysis_system, **controls)
        minimized = inherent.system
        e_is = inherent.e_is_per_particle
        fmax = inherent.fmax
        converged = inherent.converged
        fire_steps = inherent.fire_steps
        lbfgs_steps = inherent.lbfgs_steps
    else:
        minimized = analysis_system.clone()
        energy = analytic_potential(
            minimized.positions,
            minimized.diameters,
            minimized.box,
            active_mask=minimized.active_mask,
        ).energy
        e_is = float(energy.detach().cpu()) / minimized.n_particles
        fmax = _maximum_force(minimized)
        converged = bool(fmax < tolerance)
        fire_steps = 0
        lbfgs_steps = 0
    if not converged:
        raise RuntimeError(f"inherent-structure minimization did not converge: fmax={fmax:.6g}")

    hessian = hessian_dense(minimized)
    eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T)).astype(np.float64, copy=False)
    harmonic = harmonic_vibrational_entropy(
        eigenvalues,
        temperature=temperature,
        n_particles=minimized.n_particles,
        expected_zero_modes=expected_zero_modes,
        zero_tolerance=zero_tolerance,
        planck_constant=planck_constant,
        mass=mass,
    )
    return BasinMeasurement(
        n_particles=minimized.n_particles,
        e_is_per_particle=float(e_is),
        fmax=float(fmax),
        converged=converged,
        fire_steps=int(fire_steps),
        lbfgs_steps=int(lbfgs_steps),
        eigenvalues=eigenvalues,
        harmonic=harmonic,
    )


def analyze_entropy_point(
    system: ParticleSystem | Any,
    *,
    temperature: float,
    beta_grid: np.ndarray | object,
    u_grid: np.ndarray | object,
    u_at_temperature: float | None = None,
    effective_mixing_entropy: float = 0.0,
    reported_mixing_bins: int | None = 40,
    anharmonic_entropy: float = 0.0,
    planck_constant: float = 1.0,
    mass: float = 1.0,
    n_head: int = 8,
    minimize: bool = True,
    minimize_kwargs: Mapping[str, object] | None = None,
) -> EntropyPoint:
    """Run every entropy leg at one temperature using one shared convention."""

    analysis_system = coerce_analysis_system(system)
    if not math.isfinite(anharmonic_entropy):
        raise ValueError("anharmonic_entropy must be finite")
    basin = measure_harmonic_basin(
        analysis_system,
        temperature=temperature,
        minimize=minimize,
        minimize_kwargs=minimize_kwargs,
        planck_constant=planck_constant,
        mass=mass,
    )
    volume = float(torch.prod(analysis_system.box).detach().cpu())
    density = analysis_system.n_particles / volume
    ideal = sackur_tetrode_entropy(
        temperature,
        density,
        planck_constant=planck_constant,
        mass=mass,
    )
    excess = excess_entropy(
        temperature=temperature,
        beta_grid=beta_grid,
        u_grid=u_grid,
        u_at_temperature=u_at_temperature,
        n_head=n_head,
    )
    total_reference = ideal + excess.entropy_per_particle
    reported_mixing = (
        0.0
        if reported_mixing_bins is None
        else binned_diameter_mixing_entropy(
            analysis_system.diameters.detach().cpu().numpy(),
            bins=reported_mixing_bins,
        )
    )
    fixed_basin = basin.harmonic.entropy_per_particle + anharmonic_entropy
    difference = entropy_difference(
        reference_total_entropy=total_reference,
        fixed_basin_entropy=fixed_basin,
        reported_mixing_entropy=reported_mixing,
        effective_mixing_entropy=effective_mixing_entropy,
    )
    return EntropyPoint(
        temperature=float(temperature),
        density=float(density),
        s_ideal_reference=float(ideal),
        s_excess=excess.entropy_per_particle,
        s_total_reference=float(total_reference),
        s_reported_mixing=reported_mixing,
        s_total=difference.total_entropy,
        s_vibrational=basin.harmonic.entropy_per_particle,
        s_anharmonic=float(anharmonic_entropy),
        s_effective_mixing=float(effective_mixing_entropy),
        s_glass=difference.glass_entropy,
        s_configurational=difference.configurational_entropy,
        u_ex=excess.u_ex,
        beta_f_ex=excess.beta_f_ex,
        head_integral=excess.integration.head_integral,
        head_relative_rms_residual=excess.integration.head.relative_rms_residual,
        e_is_per_particle=basin.e_is_per_particle,
        fmax=basin.fmax,
        converged=basin.converged,
        n_modes=basin.harmonic.n_modes,
    )
