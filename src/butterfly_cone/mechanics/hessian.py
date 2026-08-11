"""Analytic, sparse inherent-structure Hessians for the ButterflyCone pair potential.

The assembler deliberately follows :func:`butterfly_cone.engine.potential.analytic_potential`:
it starts from its deterministic ``i < j`` pair enumeration, applies the same
minimum-image and cutoff predicates, and obtains radial derivatives from the
engine primitive.  The diagonal blocks are accumulated from the off-diagonal
pair blocks, so a fully active system obeys the acoustic sum rule by
construction.

For an active mask, this module returns a *clamped* 3N-by-3N matrix.  Frozen
degrees of freedom have zero rows and columns; an active--frozen interaction
still contributes its curvature to the active particle's diagonal block.  This
is exactly the Hessian of the engine's active-coordinate energy with frozen
coordinates held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:  # Keep the error actionable on minimal branch-engine environments.
    from scipy import sparse
except ModuleNotFoundError as error:  # pragma: no cover - exercised without SciPy installed.
    raise RuntimeError(
        "butterfly_cone.mechanics requires SciPy (scipy.sparse and scipy.sparse.linalg). "
        "Run the float64/CPU analysis with the project mechanics environment."
    ) from error

from butterfly_cone.engine.potential import (
    CUTOFF_RATIO,
    all_pairs,
    minimum_image,
    mixing_diameter,
    pair_potential,
)
from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class PairGeometry:
    """In-cutoff pair geometry in the engine's deterministic order.

    The tensors are CPU float64 (indices are int64) and are intentionally kept
    internal to the mechanics package.  Elastic observables reuse this exact
    geometry to avoid a second, subtly different interaction predicate.
    """

    i: torch.Tensor
    j: torch.Tensor
    displacement: torch.Tensor
    radius: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    active: torch.Tensor


def coerce_analysis_system(system: ParticleSystem | Any) -> ParticleSystem:
    """Return a CPU-float64 system or fail before mechanics runs on MPS.

    ``CandidateState`` is accepted by duck type and converted through its
    public ``to_system`` method.  A ``ParticleSystem`` is intentionally *not*
    silently copied from MPS/float32: that would make it too easy to execute
    mechanics in the branch engine rather than in its separate CPU analysis
    process.
    """

    if isinstance(system, ParticleSystem):
        analysis_system = system
    elif hasattr(system, "to_system"):
        analysis_system = system.to_system(device="cpu", dtype=torch.float64)
    else:
        raise TypeError("system must be a ParticleSystem or CandidateState-like object with to_system()")
    if not isinstance(analysis_system, ParticleSystem):
        raise TypeError("to_system() did not return a ParticleSystem")
    if analysis_system.device.type != "cpu" or analysis_system.dtype != torch.float64:
        raise ValueError(
            "butterfly_cone.mechanics is CPU/float64 only; pass a CPU float64 ParticleSystem "
            "or a CandidateState (which is converted by to_system)."
        )
    return analysis_system


def resolve_active_mask(system: ParticleSystem, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Resolve the engine-compatible active-coordinate mask on CPU."""

    active = system.active_mask if active_mask is None else active_mask
    active = active.detach().to(device="cpu", dtype=torch.bool)
    if active.shape != (system.n_particles,):
        raise ValueError("active_mask must have shape (N,)")
    return active


def interacting_pair_geometry(
    system: ParticleSystem | Any,
    *,
    active_mask: torch.Tensor | None = None,
) -> PairGeometry:
    """Build in-cutoff pairs using the same path as ``analytic_potential``."""

    analysis_system = coerce_analysis_system(system)
    active = resolve_active_mask(analysis_system, active_mask)
    pairs = all_pairs(analysis_system.n_particles, analysis_system.device)
    if pairs.numel() == 0:
        empty_index = torch.empty(0, dtype=torch.int64)
        empty_scalar = torch.empty(0, dtype=torch.float64)
        return PairGeometry(
            i=empty_index,
            j=empty_index,
            displacement=torch.empty((0, 3), dtype=torch.float64),
            radius=empty_scalar,
            first=empty_scalar,
            second=empty_scalar,
            active=active,
        )

    # Matches analytic_potential's active-mask pair filter exactly: only
    # frozen--frozen interactions are absent from the constrained energy.
    keep = active[pairs[0]] | active[pairs[1]]
    pairs = pairs[:, keep]
    i, j = pairs[0], pairs[1]
    displacement = minimum_image(analysis_system.positions[i] - analysis_system.positions[j], analysis_system.box)
    radius = torch.linalg.vector_norm(displacement, dim=1)
    sigma = mixing_diameter(analysis_system.diameters[i], analysis_system.diameters[j])
    interacting = radius < CUTOFF_RATIO * sigma
    i = i[interacting]
    j = j[interacting]
    displacement = displacement[interacting]
    radius = radius[interacting]
    sigma = sigma[interacting]
    if radius.numel() and bool(torch.any(radius <= torch.finfo(torch.float64).eps)):
        raise ValueError("analytic Hessian is undefined for coincident interacting particles")
    _, first, second = pair_potential(radius, sigma, derivatives=2)
    return PairGeometry(
        i=i,
        j=j,
        displacement=displacement,
        radius=radius,
        first=first,
        second=second,
        active=active,
    )


def _pair_stiffness(geometry: PairGeometry) -> np.ndarray:
    """Return ``K = phi'' nn^T + phi'/r (I - nn^T)`` for every pair."""

    if geometry.radius.numel() == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    direction = geometry.displacement / geometry.radius[:, None]
    outer = direction[:, :, None] * direction[:, None, :]
    identity = torch.eye(3, dtype=torch.float64)[None, :, :]
    stiffness = geometry.second[:, None, None] * outer + (geometry.first / geometry.radius)[:, None, None] * (
        identity - outer
    )
    return stiffness.detach().cpu().numpy()


def analytic_hessian(
    system: ParticleSystem | Any,
    *,
    active_mask: torch.Tensor | None = None,
) -> sparse.csr_matrix:
    """Assemble the analytic 3N-by-3N dynamical matrix as a CSR matrix.

    The input must be CPU float64 (or a ``CandidateState`` convertible to it).
    Matrix rows/columns retain the original particle labels even when a mask is
    present, allowing QLM vectors and local maps to remain label-aligned with
    candidate cores.
    """

    analysis_system = coerce_analysis_system(system)
    geometry = interacting_pair_geometry(analysis_system, active_mask=active_mask)
    n_particles = analysis_system.n_particles
    size = 3 * n_particles
    if n_particles == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    stiffness = _pair_stiffness(geometry)
    i = geometry.i.detach().cpu().numpy().astype(np.int64, copy=False)
    j = geometry.j.detach().cpu().numpy().astype(np.int64, copy=False)
    active = geometry.active.detach().cpu().numpy()
    diagonal = np.zeros((n_particles, 3, 3), dtype=np.float64)

    # Sequential accumulation follows the canonical engine pair order.  Each
    # diagonal is the negative sum of emitted off-diagonal blocks, rather than
    # an independently-evaluated expression.
    active_active = active[i] & active[j]
    for pair_index in range(len(i)):
        block = stiffness[pair_index]
        if active[i[pair_index]]:
            diagonal[i[pair_index]] += block
        if active[j[pair_index]]:
            diagonal[j[pair_index]] += block

    axes = np.arange(3, dtype=np.int64)
    active_particles = np.flatnonzero(active)
    diagonal_rows = (3 * active_particles[:, None, None] + axes[None, :, None]).repeat(3, axis=2).reshape(-1)
    diagonal_cols = (3 * active_particles[:, None, None] + axes[None, None, :]).repeat(3, axis=1).reshape(-1)
    diagonal_values = diagonal[active_particles].reshape(-1)

    if np.any(active_active):
        left = i[active_active]
        right = j[active_active]
        blocks = -stiffness[active_active]
        rows_ij = (3 * left[:, None, None] + axes[None, :, None]).repeat(3, axis=2).reshape(-1)
        cols_ij = (3 * right[:, None, None] + axes[None, None, :]).repeat(3, axis=1).reshape(-1)
        rows_ji = (3 * right[:, None, None] + axes[None, :, None]).repeat(3, axis=2).reshape(-1)
        cols_ji = (3 * left[:, None, None] + axes[None, None, :]).repeat(3, axis=1).reshape(-1)
        off_values = blocks.reshape(-1)
        rows = np.concatenate((diagonal_rows, rows_ij, rows_ji))
        cols = np.concatenate((diagonal_cols, cols_ij, cols_ji))
        values = np.concatenate((diagonal_values, off_values, off_values))
    else:
        rows, cols, values = diagonal_rows, diagonal_cols, diagonal_values

    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=np.float64).tocsr()
    # No duplicate pair blocks are generated, but this makes the matrix's CSR
    # canonical layout explicit and stable for deterministic regression tests.
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def hessian_dense(
    system: ParticleSystem | Any,
    *,
    active_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Return a dense analytic Hessian for tiny test systems only."""

    return analytic_hessian(system, active_mask=active_mask).toarray()
