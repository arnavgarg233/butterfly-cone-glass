"""Structural-integrity diagnostics for the ButterflyCone bulk pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemixingStatistic:
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    n_pairs: int


def demixing_passes(statistic: DemixingStatistic, *, positive_z_threshold: float = 4.0) -> bool:
    """Flag only significant *positive* like-diameter clustering as demixing.

    A negative neighbour correlation is retained as a diagnostic of local
    unlike-size packing; it is not macroscopic size segregation.
    """

    return bool(np.isfinite(statistic.z_score) and statistic.z_score <= positive_z_threshold)


def empirical_ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the two-sample empirical CDF distance without SciPy."""

    left = np.sort(np.asarray(first, dtype=float))
    right = np.sort(np.asarray(second, dtype=float))
    if left.ndim != 1 or right.ndim != 1 or len(left) == 0 or len(right) == 0:
        raise ValueError("both samples must be non-empty one-dimensional arrays")
    values = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, values, side="right") / len(left)
    right_cdf = np.searchsorted(right, values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def low_k_structure_factor(
    positions: np.ndarray,
    box: float | np.ndarray,
) -> dict[str, float | list[float]]:
    """Return the three smallest-axis-wavevector collective structure factors."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("positions must have shape (particles, 3)")
    sides = np.asarray(box, dtype=float)
    if sides.ndim == 0:
        sides = np.full(3, float(sides))
    if sides.shape != (3,) or np.any(sides <= 0.0) or not np.isfinite(sides).all():
        raise ValueError("box must be a positive scalar or length-three vector")
    wavevectors = np.diag(2.0 * np.pi / sides)
    values = []
    for vector in wavevectors:
        amplitude = np.exp(1j * (points @ vector)).sum()
        values.append(float((amplitude.conjugate() * amplitude).real / len(points)))
    return {
        "wave_numbers": [float(value) for value in np.linalg.norm(wavevectors, axis=1)],
        "values": values,
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _minimum_image(displacements: np.ndarray, box_length: float) -> np.ndarray:
    return displacements - box_length * np.rint(displacements / box_length)


def _neighbor_pairs(positions: np.ndarray, box_length: float, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    displacements = _minimum_image(positions[:, None, :] - positions[None, :, :], box_length)
    squared_distance = np.einsum("ijk,ijk->ij", displacements, displacements)
    return np.nonzero(np.triu(squared_distance < cutoff * cutoff, k=1))


def local_q6(
    positions: np.ndarray,
    *,
    box_length: float,
    neighbor_cutoff: float = 1.4,
) -> np.ndarray:
    """Compute local Steinhardt ``q_6`` via the Legendre addition theorem."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (particles, 3)")
    if box_length <= 0.0 or neighbor_cutoff <= 0.0:
        raise ValueError("box length and neighbour cutoff must be positive")
    displacements = _minimum_image(points[:, None, :] - points[None, :, :], float(box_length))
    distances = np.linalg.vector_norm(displacements, axis=2)
    output = np.zeros(points.shape[0], dtype=float)
    for particle in range(points.shape[0]):
        neighbors = np.flatnonzero((distances[particle] < neighbor_cutoff) & (distances[particle] > 0.0))
        if len(neighbors) < 4:
            continue
        directions = displacements[particle, neighbors] / distances[particle, neighbors, None]
        cosine = np.clip(directions @ directions.T, -1.0, 1.0)
        cosine2 = cosine * cosine
        legendre6 = (231.0 * cosine2 * cosine2 * cosine2 - 315.0 * cosine2 * cosine2 + 105.0 * cosine2 - 5.0) / 16.0
        output[particle] = float(np.sqrt(max(float(legendre6.mean()), 0.0)))
    return output


def q6_values(
    positions: np.ndarray,
    box: float | np.ndarray,
    *,
    neighbor_cutoff: float = 1.4,
) -> np.ndarray:
    """Return local ``q_6`` values for the cubic ButterflyCone simulation box."""

    sides = np.asarray(box, dtype=float)
    if sides.ndim == 0:
        length = float(sides)
    elif sides.shape == (3,) and np.allclose(sides, sides[0], rtol=0.0, atol=1e-12):
        length = float(sides[0])
    else:
        raise ValueError("the bulk pilot requires a cubic box")
    return local_q6(positions, box_length=length, neighbor_cutoff=neighbor_cutoff)


def _pair_diameter_correlation(diameters: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    centered = np.asarray(diameters, dtype=float) - float(np.mean(diameters))
    variance = float(np.mean(centered * centered))
    if variance <= np.finfo(float).eps or len(left) == 0:
        return float("nan")
    return float(np.mean(centered[left] * centered[right]) / variance)


def diameter_demixing_statistic(
    positions: np.ndarray,
    diameters: np.ndarray,
    box_length: float | np.ndarray | None = None,
    *,
    neighbor_cutoff: float = 1.4,
    n_shuffles: int = 128,
    seed: int = 0,
) -> DemixingStatistic:
    """Compare nearest-neighbour diameter correlations to shuffled labels."""

    if box_length is None:
        raise ValueError("box_length is required")
    sides = np.asarray(box_length, dtype=float)
    if sides.ndim == 0:
        length = float(sides)
    elif sides.shape == (3,) and np.allclose(sides, sides[0], rtol=0.0, atol=1e-12):
        length = float(sides[0])
    else:
        raise ValueError("the bulk pilot requires a cubic box")
    points = np.asarray(positions, dtype=float)
    sizes = np.asarray(diameters, dtype=float)
    if points.shape != (len(sizes), 3):
        raise ValueError("positions and diameters have incompatible shapes")
    if n_shuffles < 2:
        raise ValueError("at least two shuffled null samples are required")
    left, right = _neighbor_pairs(points, length, float(neighbor_cutoff))
    observed = _pair_diameter_correlation(sizes, left, right)
    generator = np.random.default_rng(seed)
    null = np.array(
        [_pair_diameter_correlation(generator.permutation(sizes), left, right) for _ in range(n_shuffles)],
        dtype=float,
    )
    null_mean = float(np.nanmean(null))
    null_std = float(np.nanstd(null, ddof=1))
    if not np.isfinite(observed) or not np.isfinite(null_std) or null_std <= np.finfo(float).eps:
        z_score = float("nan")
    else:
        z_score = float((observed - null_mean) / null_std)
    return DemixingStatistic(observed, null_mean, null_std, z_score, int(len(left)))
