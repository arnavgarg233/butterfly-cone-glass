"""Shift-invert low-frequency modes and gauge-invariant QLM observables."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:  # pragma: no branch - normal mechanics environments have SciPy.
    from scipy import linalg, sparse
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh
except ModuleNotFoundError as error:  # pragma: no cover - clear failure without optional dependency.
    raise RuntimeError(
        "butterfly_cone.mechanics.modes requires SciPy's sparse eigensolvers. "
        "Use the dedicated CPU/float64 mechanics environment."
    ) from error


@dataclass(frozen=True)
class ModeSet:
    """Physical low modes plus the raw shift-invert diagnostic spectrum.

    ``eigenvalues`` and ``eigenvectors`` exclude detected uniform translation
    modes.  ``raw_*`` retain the requested Lanczos output for IS-gate
    diagnostics; consumers should use the physical fields for QLM observables.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    participation_ratios: np.ndarray
    raw_eigenvalues: np.ndarray
    raw_eigenvectors: np.ndarray
    com_mask: np.ndarray
    sigma_used: float

    @property
    def n_com_modes(self) -> int:
        return int(np.count_nonzero(self.com_mask))


@dataclass(frozen=True)
class DOSLowTail:
    """A descriptive histogram of positive low frequencies.

    The fitted slope is intentionally descriptive only.  Finite-size low-mode
    data are reported as a soft-tail diagnostic and are never used here as an
    assertion that a configuration obeys a particular power law.
    """

    frequencies: np.ndarray
    bin_edges: np.ndarray
    density: np.ndarray
    loglog_slope: float


def _as_mode_vector(mode: np.ndarray | object) -> np.ndarray:
    vector = np.asarray(mode, dtype=np.float64).reshape(-1)
    if vector.size == 0 or vector.size % 3:
        raise ValueError("a mode must be a non-empty flat vector with length 3N")
    return vector


def participation_ratio(mode: np.ndarray | object) -> float:
    """Return the standard per-particle participation ratio of a mode."""

    vector = _as_mode_vector(mode).reshape(-1, 3)
    weights = np.einsum("ij,ij->i", vector, vector)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("participation ratio is undefined for a zero mode vector")
    return total * total / (len(weights) * float(np.dot(weights, weights)))


def cavity_projection(mode: np.ndarray | object, core_mask: np.ndarray | object) -> float:
    """Return the fraction of a mode's squared norm inside an RCCE core."""

    vector = _as_mode_vector(mode).reshape(-1, 3)
    mask = np.asarray(core_mask, dtype=bool).reshape(-1)
    if mask.shape != (vector.shape[0],):
        raise ValueError("core_mask must have shape (N,)")
    weights = np.einsum("ij,ij->i", vector, vector)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("cavity projection is undefined for a zero mode vector")
    return float(weights[mask].sum() / total)


def _translation_overlap(vectors: np.ndarray) -> np.ndarray:
    """Squared overlap with the three uniform translations for each column."""

    n_coordinates, n_modes = vectors.shape
    if n_coordinates % 3:
        raise ValueError("Hessian dimension must be divisible by three")
    n_particles = n_coordinates // 3
    reshaped = vectors.reshape(n_particles, 3, n_modes)
    amplitudes = reshaped.sum(axis=0) / math.sqrt(n_particles)
    projection_norm = np.einsum("ij,ij->j", amplitudes, amplitudes)
    vector_norm = np.einsum("ij,ij->j", vectors, vectors)
    return projection_norm / np.maximum(vector_norm, np.finfo(np.float64).tiny)


def _tiny_dense_eigensystem(matrix: sparse.spmatrix, requested: int) -> tuple[np.ndarray, np.ndarray]:
    """Test-only fallback when ARPACK's k < N contract cannot be satisfied."""

    dense = matrix.toarray()
    values, vectors = linalg.eigh(0.5 * (dense + dense.T), check_finite=True)
    # Keep the modes closest to zero, matching shift-invert selection, before
    # sorting by eigenvalue for the public result.
    indices = np.argsort(np.abs(values))[:requested]
    return values[indices], vectors[:, indices]


def _eigsh_near_zero(
    matrix: sparse.csr_matrix,
    requested: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run deterministic shift-invert Lanczos, retrying exact-null spectra."""

    n_coordinates = matrix.shape[0]
    ncv = min(n_coordinates, max(2 * requested + 1, 20))
    initial = np.sin(np.arange(1, n_coordinates + 1, dtype=np.float64))
    initial /= np.linalg.norm(initial)
    kwargs = {"k": requested, "sigma": sigma, "which": "LM", "v0": initial, "tol": 1e-10, "ncv": ncv}
    try:
        values, vectors = eigsh(matrix, **kwargs)
        return values, vectors, float(sigma)
    except (RuntimeError, ValueError, ArpackNoConvergence) as original_error:
        # Sigma=0 is singular for a translation-invariant Hessian.  A positive
        # 1e-10 shift remains far below the 1e-7 physical-resolution gate while
        # making the sparse LU in ARPACK's shift-invert operator nonsingular.
        if sigma != 0.0:
            raise RuntimeError(f"shift-invert Lanczos failed at sigma={sigma}: {original_error}") from original_error
        safe_sigma = 1.0e-10
        try:
            values, vectors = eigsh(matrix, **{**kwargs, "sigma": safe_sigma})
        except (RuntimeError, ValueError, ArpackNoConvergence) as retry_error:
            raise RuntimeError(
                "shift-invert Lanczos failed both at sigma=0 and at the nullspace-safe "
                f"shift {safe_sigma}: {retry_error}"
            ) from retry_error
        return values, vectors, safe_sigma


def low_modes(
    hessian: sparse.spmatrix,
    *,
    k: int = 40,
    sigma: float = 0.0,
    active_mask: np.ndarray | object | None = None,
    com_eigenvalue_tolerance: float = 1.0e-7,
    com_overlap_tolerance: float = 1.0 - 1.0e-7,
) -> ModeSet:
    """Extract the lowest physical modes with sparse shift-invert Lanczos.

    Production-sized Hessians always go through ``scipy.sparse.linalg.eigsh``;
    full diagonalization is restricted to matrices of dimension at most 96 for
    test-scale edge cases where ARPACK cannot legally request ``k >= N``.
    """

    if not sparse.issparse(hessian):
        raise TypeError("hessian must be a SciPy sparse matrix")
    matrix = hessian.tocsr().astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 3:
        raise ValueError("hessian must be a square 3N-by-3N sparse matrix")
    if k <= 0:
        raise ValueError("k must be positive")
    full_coordinates = matrix.shape[0]
    if active_mask is None:
        coordinate_indices = np.arange(full_coordinates, dtype=np.int64)
    else:
        particle_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if particle_mask.shape != (full_coordinates // 3,):
            raise ValueError("active_mask must have shape (N,) for the 3N-by-3N Hessian")
        coordinate_indices = np.flatnonzero(np.repeat(particle_mask, 3))
        if coordinate_indices.size == 0:
            raise ValueError("cannot extract modes when no particle degrees of freedom are active")
    solver_matrix = matrix[coordinate_indices][:, coordinate_indices].tocsr()
    n_coordinates = solver_matrix.shape[0]

    # Ask for three additional candidates so the COM modes can be removed while
    # still returning k physical modes.  A clamped active-mask Hessian has no
    # such nullspace; the surplus candidates are simply discarded after sort.
    requested = min(k + 3, max(1, n_coordinates - 1))
    if requested >= n_coordinates - 1:
        if n_coordinates > 96:
            raise ValueError(
                "requested too many modes for production sparse Lanczos; choose k well below 3N "
                "rather than triggering a full diagonalization"
            )
        raw_values, raw_vectors = _tiny_dense_eigensystem(solver_matrix, n_coordinates)
        sigma_used = float("nan")
    else:
        raw_values, raw_vectors, sigma_used = _eigsh_near_zero(solver_matrix, requested, float(sigma))

    order = np.argsort(raw_values, kind="stable")
    raw_values = np.asarray(raw_values[order], dtype=np.float64)
    raw_vectors = np.asarray(raw_vectors[:, order], dtype=np.float64)
    overlaps = _translation_overlap(raw_vectors)
    com_mask = (np.abs(raw_values) < com_eigenvalue_tolerance) & (overlaps >= com_overlap_tolerance)
    physical = ~com_mask
    physical_values = raw_values[physical][:k]
    physical_solver_vectors = raw_vectors[:, physical][:, :k]
    raw_full_vectors = np.zeros((full_coordinates, raw_vectors.shape[1]), dtype=np.float64)
    raw_full_vectors[coordinate_indices, :] = raw_vectors
    physical_vectors = np.zeros((full_coordinates, physical_solver_vectors.shape[1]), dtype=np.float64)
    physical_vectors[coordinate_indices, :] = physical_solver_vectors
    ratios = np.asarray([participation_ratio(physical_vectors[:, index]) for index in range(physical_vectors.shape[1])])
    return ModeSet(
        eigenvalues=physical_values,
        eigenvectors=physical_vectors,
        participation_ratios=ratios,
        raw_eigenvalues=raw_values,
        raw_eigenvectors=raw_full_vectors,
        com_mask=com_mask,
        sigma_used=sigma_used,
    )


def dos_low_tail(eigenvalues: np.ndarray | object) -> DOSLowTail:
    """Report a compact low-frequency DOS histogram and optional log--log slope."""

    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    frequencies = np.sqrt(np.sort(values[values > 0.0]))
    if frequencies.size == 0:
        return DOSLowTail(
            frequencies=frequencies,
            bin_edges=np.asarray([0.0, 1.0]),
            density=np.asarray([0.0]),
            loglog_slope=float("nan"),
        )
    if frequencies.size == 1 or np.isclose(frequencies.min(), frequencies.max()):
        half_width = max(abs(float(frequencies[0])) * 0.05, 1.0e-12)
        edges = np.asarray([frequencies[0] - half_width, frequencies[0] + half_width])
    else:
        n_bins = max(2, min(12, int(np.ceil(np.sqrt(frequencies.size)))))
        edges = np.linspace(float(frequencies.min()), float(frequencies.max()), n_bins + 1)
    density, edges = np.histogram(frequencies, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = (density > 0.0) & (centers > 0.0)
    slope = float("nan")
    if int(np.count_nonzero(valid)) >= 2:
        slope = float(np.polyfit(np.log(centers[valid]), np.log(density[valid]), deg=1)[0])
    return DOSLowTail(frequencies=frequencies, bin_edges=edges, density=density, loglog_slope=slope)
