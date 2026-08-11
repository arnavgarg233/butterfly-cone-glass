"""Harmonic Born/nonaffine simple-shear elasticity on an inherent structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

try:  # pragma: no branch - required by the mechanics analysis environment.
    from scipy import sparse
    from scipy.sparse.linalg import splu
except ModuleNotFoundError as error:  # pragma: no cover - explicit optional dependency error.
    raise RuntimeError(
        "butterfly_cone.mechanics.elastic requires SciPy sparse linear algebra. "
        "Run it in the dedicated CPU/float64 mechanics environment."
    ) from error

from butterfly_cone.engine.system import ParticleSystem

from .hessian import (
    _pair_stiffness,
    coerce_analysis_system,
    interacting_pair_geometry,
    resolve_active_mask,
)


Axis = tuple[int, int]


@dataclass(frozen=True)
class ElasticResult:
    """Global athermal harmonic modulus and its nonaffine decomposition."""

    G: float
    G_born: float
    G_nonaffine_correction: float
    affine_force: np.ndarray
    nonaffine_displacement: np.ndarray
    translation_nullspace_deflated: bool


@dataclass(frozen=True)
class LocalModulusMap:
    """A conservative coarse-grained decomposition of the global shear modulus."""

    values: np.ndarray
    volumes: np.ndarray
    grid: tuple[int, int, int] | None
    labels: tuple[str, ...]
    particle_born: np.ndarray
    particle_nonaffine: np.ndarray
    global_modulus: float
    core_modulus: float | None
    subregion_moduli: Mapping[str, float]


def _validate_axis(axis: Axis) -> Axis:
    if len(axis) != 2 or any(not isinstance(component, int) or component not in (0, 1, 2) for component in axis):
        raise ValueError("axis must be a pair of distinct Cartesian indices in {0, 1, 2}")
    if axis[0] == axis[1]:
        raise ValueError("simple-shear flow and gradient axes must differ")
    return axis


def _active_coordinate_indices(system: ParticleSystem) -> np.ndarray:
    active = resolve_active_mask(system).detach().cpu().numpy()
    return np.flatnonzero(np.repeat(active, 3))


def _pair_born_contributions(system: ParticleSystem, *, axis: Axis) -> tuple[np.ndarray, np.ndarray]:
    """Return total Born numerator and a conservative per-particle partition."""

    flow, gradient = _validate_axis(axis)
    geometry = interacting_pair_geometry(system)
    n_particles = system.n_particles
    per_particle = np.zeros(n_particles, dtype=np.float64)
    if geometry.radius.numel() == 0:
        return np.empty(0, dtype=np.float64), per_particle
    displacement = geometry.displacement.detach().cpu().numpy()
    radius = geometry.radius.detach().cpu().numpy()
    first = geometry.first.detach().cpu().numpy()
    second = geometry.second.detach().cpu().numpy()
    da = displacement[:, flow]
    db = displacement[:, gradient]
    # d r / d gamma = x_a x_b / r and
    # d2 r / d gamma2 = x_b^2 (r^2 - x_a^2) / r^3 for x_a <- x_a + gamma x_b.
    pair_born = second * (da * db / radius) ** 2 + first * (
        db * db / radius - (da * db) ** 2 / radius**3
    )
    i = geometry.i.detach().cpu().numpy().astype(np.int64, copy=False)
    j = geometry.j.detach().cpu().numpy().astype(np.int64, copy=False)
    active = geometry.active.detach().cpu().numpy()
    for index, value in enumerate(pair_born):
        n_active_ends = int(active[i[index]]) + int(active[j[index]])
        if n_active_ends:
            share = value / n_active_ends
            if active[i[index]]:
                per_particle[i[index]] += share
            if active[j[index]]:
                per_particle[j[index]] += share
    return pair_born, per_particle


def affine_force_field(system: ParticleSystem | Any, *, axis: Axis = (0, 1)) -> np.ndarray:
    """Return ``Xi = d2 U / (d gamma d r)`` for affine simple shear.

    ``axis=(a, b)`` means ``x_a -> x_a + gamma x_b``.  The expression is
    evaluated from the same radial derivatives and pair geometry as the
    Hessian.  Frozen coordinates remain zero while active--frozen pairs retain
    their active-coordinate mismatch contribution.
    """

    analysis_system = coerce_analysis_system(system)
    flow, gradient = _validate_axis(axis)
    geometry = interacting_pair_geometry(analysis_system)
    xi = np.zeros((analysis_system.n_particles, 3), dtype=np.float64)
    if geometry.radius.numel() == 0:
        return xi.reshape(-1)
    stiffness = _pair_stiffness(geometry)
    displacement = geometry.displacement.detach().cpu().numpy()
    radius = geometry.radius.detach().cpu().numpy()
    first = geometry.first.detach().cpu().numpy()
    affine_displacement = np.zeros_like(displacement)
    affine_displacement[:, flow] = displacement[:, gradient]
    pair_gradient = first[:, None] * displacement / radius[:, None]
    transpose_affine_gradient = np.zeros_like(pair_gradient)
    transpose_affine_gradient[:, gradient] = pair_gradient[:, flow]
    pair_xi = np.einsum("pab,pb->pa", stiffness, affine_displacement) + transpose_affine_gradient
    i = geometry.i.detach().cpu().numpy().astype(np.int64, copy=False)
    j = geometry.j.detach().cpu().numpy().astype(np.int64, copy=False)
    active = geometry.active.detach().cpu().numpy()
    for index, contribution in enumerate(pair_xi):
        if active[i[index]]:
            xi[i[index]] += contribution
        if active[j[index]]:
            xi[j[index]] -= contribution
    return xi.reshape(-1)


def born_modulus(system: ParticleSystem | Any, *, axis: Axis = (0, 1)) -> float:
    """Return the affine (Cauchy--Born) xy shear modulus ``(d2U/dgamma2)/V``."""

    analysis_system = coerce_analysis_system(system)
    pair_born, _ = _pair_born_contributions(analysis_system, axis=axis)
    volume = float(torch.prod(analysis_system.box).detach().cpu())
    return float(pair_born.sum() / volume) if pair_born.size else 0.0


def _translation_basis(n_active_particles: int) -> np.ndarray:
    basis = np.zeros((3 * n_active_particles, 3), dtype=np.float64)
    normalization = np.sqrt(n_active_particles)
    for axis in range(3):
        basis[axis::3, axis] = 1.0 / normalization
    return basis


def _has_translation_nullspace(matrix: sparse.csr_matrix, n_active_particles: int) -> bool:
    if n_active_particles == 0:
        return False
    translations = _translation_basis(n_active_particles)
    residual = np.linalg.norm(matrix @ translations)
    scale = max(1.0, float(np.linalg.norm(matrix.data)) if matrix.nnz else 1.0)
    return bool(residual <= 1.0e-9 * scale)


def _solve_active_deflated(system: ParticleSystem, hessian: sparse.spmatrix, rhs: np.ndarray) -> tuple[np.ndarray, bool]:
    """Solve the active-coordinate system, removing constrained/frozen nulls first."""

    if not sparse.issparse(hessian):
        raise TypeError("H must be a SciPy sparse matrix")
    expected = 3 * system.n_particles
    if hessian.shape != (expected, expected):
        raise ValueError("H shape must match system's 3N coordinate dimension")
    coordinates = _active_coordinate_indices(system)
    solution = np.zeros(expected, dtype=np.float64)
    if coordinates.size == 0:
        return solution, False
    active_matrix = hessian.tocsr()[coordinates][:, coordinates].astype(np.float64, copy=False)
    active_rhs = np.asarray(rhs, dtype=np.float64)[coordinates]
    n_active_particles = coordinates.size // 3
    try:
        deflated = _has_translation_nullspace(active_matrix, n_active_particles)
        if deflated:
            translations = _translation_basis(n_active_particles)
            # The KKT system enforces T^T u = 0 without forming a dense
            # projector.  It is one sparse LU solve and leaves physical
            # eigenvalues untouched.
            augmented = sparse.bmat(
                [[active_matrix, sparse.csr_matrix(translations)], [sparse.csr_matrix(translations.T), None]],
                format="csc",
            )
            augmented_rhs = np.concatenate((active_rhs, np.zeros(3, dtype=np.float64)))
            active_solution = splu(augmented).solve(augmented_rhs)[: coordinates.size]
        else:
            active_solution = splu(active_matrix.tocsc()).solve(active_rhs)
    except RuntimeError as error:
        raise RuntimeError(
            "nonaffine solve is singular after removing frozen coordinates; the candidate is not a stable "
            "active-coordinate inherent structure"
        ) from error
    solution[coordinates] = active_solution
    return solution, deflated


def nonaffine_modulus(
    system: ParticleSystem | Any,
    hessian: sparse.spmatrix,
    *,
    axis: Axis = (0, 1),
) -> ElasticResult:
    """Compute ``G = G_Born - Xi.T H^-1 Xi / V`` with nullspace deflation."""

    analysis_system = coerce_analysis_system(system)
    xi = affine_force_field(analysis_system, axis=axis)
    displacement, deflated = _solve_active_deflated(analysis_system, hessian, xi)
    volume = float(torch.prod(analysis_system.box).detach().cpu())
    correction = float(np.dot(xi, displacement) / volume)
    born = born_modulus(analysis_system, axis=axis)
    numerical_scale = max(1.0, abs(born))
    if correction < 0.0 and correction > -1.0e-10 * numerical_scale:
        correction = 0.0
    if correction < 0.0:
        raise ValueError("nonaffine correction is negative; spectrum is not a stable inherent structure")
    return ElasticResult(
        G=born - correction,
        G_born=born,
        G_nonaffine_correction=correction,
        affine_force=xi,
        nonaffine_displacement=displacement,
        translation_nullspace_deflated=deflated,
    )


def _normalise_grid(grid: tuple[int, int, int] | int) -> tuple[int, int, int]:
    if isinstance(grid, int):
        grid = (grid, grid, grid)
    if len(grid) != 3 or any(isinstance(value, bool) or int(value) <= 0 for value in grid):
        raise ValueError("grid must be a positive integer or a length-three tuple of positive integers")
    return tuple(int(value) for value in grid)


def _subregion_records(
    subregions: Mapping[str, object] | None,
    contributions: np.ndarray,
    n_particles: int,
) -> dict[str, float]:
    records: dict[str, float] = {}
    if subregions is None:
        return records
    for name, specification in subregions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("subregion names must be non-empty strings")
        if isinstance(specification, tuple) and len(specification) == 2:
            mask, volume = specification
        elif isinstance(specification, Mapping) and "mask" in specification and "volume" in specification:
            mask, volume = specification["mask"], specification["volume"]
        else:
            raise ValueError("each subregion must be (mask, volume) or {'mask': ..., 'volume': ...}")
        array_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if array_mask.shape != (n_particles,):
            raise ValueError(f"subregion {name!r} mask must have shape (N,)")
        numeric_volume = float(volume)
        if not np.isfinite(numeric_volume) or numeric_volume <= 0.0:
            raise ValueError(f"subregion {name!r} volume must be positive and finite")
        records[name] = float(contributions[array_mask].sum() / numeric_volume)
    return records


def local_modulus_map(
    system: ParticleSystem | Any,
    hessian: sparse.spmatrix,
    *,
    grid: tuple[int, int, int] | int | None = (4, 4, 4),
    subregions: Mapping[str, object] | None = None,
    core_mask: np.ndarray | torch.Tensor | None = None,
    core_volume: float | None = None,
    axis: Axis = (0, 1),
) -> LocalModulusMap:
    """Coarse-grain conservative per-particle Born/nonaffine contributions.

    A pair Born term is divided equally among its active endpoints, while the
    nonaffine numerator is partitioned as ``Xi_i dot u_i``.  Their sum is
    exactly the global numerator, so the returned map obeys
    ``sum(G_cell * V_cell) / V == G`` up to floating-point summation order.
    """

    analysis_system = coerce_analysis_system(system)
    result = nonaffine_modulus(analysis_system, hessian, axis=axis)
    _, particle_born = _pair_born_contributions(analysis_system, axis=axis)
    particle_nonaffine = np.einsum(
        "ij,ij->i",
        result.affine_force.reshape(analysis_system.n_particles, 3),
        result.nonaffine_displacement.reshape(analysis_system.n_particles, 3),
    )
    contributions = particle_born - particle_nonaffine
    volume = float(torch.prod(analysis_system.box).detach().cpu())
    subregion_values = _subregion_records(subregions, contributions, analysis_system.n_particles)
    core_value: float | None = None
    if core_mask is not None:
        if core_volume is None:
            raise ValueError("core_volume is required when core_mask is supplied")
        mask = np.asarray(core_mask, dtype=bool).reshape(-1)
        if mask.shape != (analysis_system.n_particles,):
            raise ValueError("core_mask must have shape (N,)")
        if not np.isfinite(float(core_volume)) or float(core_volume) <= 0.0:
            raise ValueError("core_volume must be positive and finite")
        core_value = float(contributions[mask].sum() / float(core_volume))
        subregion_values["core"] = core_value

    if grid is None:
        if not subregion_values:
            raise ValueError("provide grid or at least one subregion")
        labels = tuple(sorted(subregion_values))
        values = np.asarray([subregion_values[label] for label in labels], dtype=np.float64)
        # Explicit subregion volumes are deliberately not assumed to tessellate
        # the box; this array is metadata only in the non-grid form.
        volumes = np.full(values.shape, np.nan, dtype=np.float64)
        return LocalModulusMap(
            values=values,
            volumes=volumes,
            grid=None,
            labels=labels,
            particle_born=particle_born,
            particle_nonaffine=particle_nonaffine,
            global_modulus=result.G,
            core_modulus=core_value,
            subregion_moduli=dict(subregion_values),
        )

    shape = _normalise_grid(grid)
    cell_volume = volume / float(np.prod(shape))
    numerators = np.zeros(shape, dtype=np.float64)
    positions = torch.remainder(analysis_system.positions, analysis_system.box).detach().cpu().numpy()
    box = analysis_system.box.detach().cpu().numpy()
    cells = np.floor(positions / box * np.asarray(shape)).astype(np.int64)
    cells = np.minimum(np.maximum(cells, 0), np.asarray(shape) - 1)
    for particle, contribution in enumerate(contributions):
        numerators[tuple(cells[particle])] += contribution
    values = numerators / cell_volume
    volumes = np.full(shape, cell_volume, dtype=np.float64)
    return LocalModulusMap(
        values=values,
        volumes=volumes,
        grid=shape,
        labels=(),
        particle_born=particle_born,
        particle_nonaffine=particle_nonaffine,
        global_modulus=result.G,
        core_modulus=core_value,
        subregion_moduli=dict(subregion_values),
    )
