"""CPU-float64 FIRE plus L-BFGS inherent-structure minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from butterfly_cone.engine.neighbors import VerletList
from butterfly_cone.engine.potential import analytic_potential
from butterfly_cone.engine.system import ParticleSystem

from .hessian import coerce_analysis_system, resolve_active_mask


@dataclass(frozen=True)
class ISResult:
    """A minimized active-coordinate state and convergence diagnostics."""

    system: ParticleSystem
    e_is_per_particle: float
    fmax: float
    converged: bool
    fire_steps: int
    lbfgs_steps: int

    def __iter__(self) -> Iterable[object]:
        """Compatibility with the historical ``system, energy, fmax`` tuple."""

        yield self.system
        yield self.e_is_per_particle
        yield self.fmax


@dataclass(frozen=True)
class SpectrumGate:
    """Detailed clean-spectrum decision used by the analysis runner."""

    passed: bool
    expected_zero_modes: int
    com_eigenvalues: tuple[float, ...]
    n_negative_physical: int
    minimum_physical_eigenvalue: float


def _maximum_active_force(forces: torch.Tensor, active: torch.Tensor) -> float:
    if not bool(torch.any(active)):
        return 0.0
    norms = torch.linalg.vector_norm(forces[active], dim=1)
    return float(norms.max().detach().cpu())


def _force_result(system: ParticleSystem, neighbors: VerletList) -> tuple[torch.Tensor, torch.Tensor]:
    result = neighbors.evaluate(system)
    return result.energy, result.forces


def minimize_to_IS(
    system: ParticleSystem | Any,
    *,
    tol: float = 1.0e-8,
    dt0: float = 0.002,
    dt_max: float = 0.02,
    max_steps: int = 40_000,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    a0: float = 0.1,
    f_a: float = 0.99,
    n_min: int = 5,
    neighbor_skin: float = 0.3,
    lbfgs_max_iter: int = 500,
    lbfgs_history_size: int = 100,
    lbfgs_outer_steps: int = 6,
) -> ISResult:
    """Minimize active particles with the reference FIRE + strong-Wolfe polish.

    The FIRE defaults are ported directly from the read-only ``s_config.py``
    reference.  A deterministic engine ``VerletList`` evaluates the same
    analytic force during descent; L-BFGS then uses the exact all-pair analytic
    energy over active coordinates for its final smooth, strong-Wolfe polish.
    """

    if tol <= 0.0 or dt0 <= 0.0 or dt_max < dt0 or max_steps < 0:
        raise ValueError("invalid FIRE tolerance, time-step, or iteration controls")
    if f_inc <= 1.0 or not 0.0 < f_dec < 1.0 or not 0.0 < a0 <= 1.0 or not 0.0 < f_a <= 1.0:
        raise ValueError("invalid FIRE adaptation controls")
    if n_min < 0 or neighbor_skin <= 0.0 or lbfgs_max_iter < 0 or lbfgs_outer_steps < 0:
        raise ValueError("invalid minimizer controls")

    source = coerce_analysis_system(system)
    current = source.clone()
    active = resolve_active_mask(current)
    # CandidateState.to_system supplies the buffer mask; an explicit clone makes
    # it impossible for minimization to mutate caller-owned candidate tensors.
    current.active_mask = active.clone()
    velocities = torch.zeros_like(current.positions)
    neighbors = VerletList.from_system(current, skin=neighbor_skin)
    dt, alpha, n_positive = float(dt0), float(a0), 0
    fire_steps = 0
    fmax = float("inf")

    with torch.no_grad():
        for _ in range(max_steps):
            _, forces = _force_result(current, neighbors)
            fmax = _maximum_active_force(forces, active)
            if not np.isfinite(fmax):
                raise FloatingPointError("non-finite force encountered during FIRE minimization")
            if fmax < tol:
                break
            power = float(torch.sum(forces * velocities).detach().cpu())
            if power > 0.0:
                n_positive += 1
                if n_positive > n_min:
                    dt = min(dt * f_inc, dt_max)
                    alpha *= f_a
            else:
                n_positive = 0
                dt *= f_dec
                alpha = a0
                velocities.zero_()
            velocities = velocities + dt * forces
            force_norm = torch.linalg.vector_norm(forces)
            velocity_norm = torch.linalg.vector_norm(velocities)
            if float(force_norm) > 0.0:
                velocities = (1.0 - alpha) * velocities + alpha * velocity_norm * forces / force_norm
            velocities = torch.where(active[:, None], velocities, torch.zeros_like(velocities))
            displacement = dt * velocities
            current.positions = torch.remainder(current.positions + displacement, current.box)
            current.unwrapped_positions = current.unwrapped_positions + displacement
            fire_steps += 1

        # Evaluate after the last displacement even when FIRE exhausted its cap.
        _, forces = _force_result(current, neighbors)
        fmax = _maximum_active_force(forces, active)

    lbfgs_steps = 0
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if fmax >= tol and active_indices.numel() and lbfgs_outer_steps:
        fixed_positions = current.positions.detach().clone()
        variable = fixed_positions[active_indices].detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [variable],
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            tolerance_grad=1.0e-16,
            tolerance_change=1.0e-20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            positions = fixed_positions.index_copy(0, active_indices, variable)
            energy = analytic_potential(
                positions,
                current.diameters,
                current.box,
                active_mask=active,
            ).energy
            energy.backward()
            return energy

        for _ in range(lbfgs_outer_steps):
            optimizer.step(closure)
            lbfgs_steps += 1
            with torch.no_grad():
                displacement = variable.detach() - fixed_positions[active_indices]
                current.positions = torch.remainder(
                    fixed_positions.index_copy(0, active_indices, variable.detach()), current.box
                )
                current.unwrapped_positions[active_indices] = (
                    current.unwrapped_positions[active_indices] + displacement
                )
                fixed_positions = current.positions.detach().clone()
                # Rebase the optimization coordinate after periodic wrapping so
                # the next closure sees the current box image, not a stale one.
                variable.data.copy_(fixed_positions[active_indices])
                _, forces = _force_result(current, neighbors)
                fmax = _maximum_active_force(forces, active)
            if fmax < tol:
                break

    current.velocities = torch.zeros_like(current.velocities)
    energy = analytic_potential(
        current.positions,
        current.diameters,
        current.box,
        active_mask=active,
    ).energy
    fmax = _maximum_active_force(analytic_potential(
        current.positions,
        current.diameters,
        current.box,
        active_mask=active,
    ).forces, active)
    return ISResult(
        system=current,
        e_is_per_particle=float(energy.detach().cpu()) / current.n_particles,
        fmax=fmax,
        converged=bool(fmax < tol),
        fire_steps=fire_steps,
        lbfgs_steps=lbfgs_steps,
    )


def inspect_inherent_spectrum(
    spectrum: np.ndarray | object,
    *,
    expected_zero_modes: int = 3,
    com_tolerance: float = 1.0e-3,
    negative_tolerance: float = -1.0e-7,
) -> SpectrumGate:
    """Apply the ``s_config.py`` clean-spectrum rule without diagonalizing anew."""

    if expected_zero_modes < 0:
        raise ValueError("expected_zero_modes must be non-negative")
    if hasattr(spectrum, "raw_eigenvalues"):
        values = np.asarray(getattr(spectrum, "raw_eigenvalues"), dtype=np.float64).reshape(-1)
    elif hasattr(spectrum, "eigenvalues") and not isinstance(spectrum, np.ndarray):
        values = np.asarray(getattr(spectrum, "eigenvalues"), dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    if values.size < expected_zero_modes:
        return SpectrumGate(False, expected_zero_modes, (), 0, float("nan"))
    if not np.all(np.isfinite(values)):
        return SpectrumGate(False, expected_zero_modes, (), 0, float("nan"))
    if expected_zero_modes:
        com_indices = np.argsort(np.abs(values), kind="stable")[:expected_zero_modes]
        com_values = values[com_indices]
        physical = np.delete(values, com_indices)
        translations_clean = bool(np.all(np.abs(com_values) < com_tolerance))
    else:
        com_values = np.empty(0, dtype=np.float64)
        physical = values
        translations_clean = True
    n_negative = int(np.count_nonzero(physical <= negative_tolerance))
    minimum = float(physical.min()) if physical.size else float("inf")
    return SpectrumGate(
        passed=translations_clean and n_negative == 0,
        expected_zero_modes=expected_zero_modes,
        com_eigenvalues=tuple(float(value) for value in com_values),
        n_negative_physical=n_negative,
        minimum_physical_eigenvalue=minimum,
    )


def validate_inherent_structure(
    spectrum: np.ndarray | object,
    *,
    expected_zero_modes: int = 3,
) -> bool:
    """Return whether a supplied spectrum passes the frozen clean-IS gate."""

    return inspect_inherent_spectrum(spectrum, expected_zero_modes=expected_zero_modes).passed
