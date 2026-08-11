"""Small, deterministic reductions used by the bulk-temperature pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TauAlphaResult:
    """A 1/e crossing or an honest finite-window lower bound."""

    value: float | None
    crossed: bool
    lower_bound: float | None
    threshold: float


def extract_tau_alpha(
    times: np.ndarray | Iterable[float],
    fs: np.ndarray | Iterable[float],
    *,
    threshold: float = float(np.exp(-1.0)),
) -> TauAlphaResult:
    """Find the first ``F_s=1/e`` crossing by linear interpolation in log time."""

    time_values = np.asarray(times, dtype=float)
    fs_values = np.asarray(fs, dtype=float)
    if time_values.ndim != 1 or fs_values.ndim != 1 or len(time_values) != len(fs_values):
        raise ValueError("times and fs must be equal-length one-dimensional arrays")
    if len(time_values) == 0:
        raise ValueError("at least one time point is required")
    if not np.isfinite(time_values).all() or not np.isfinite(fs_values).all():
        raise ValueError("times and fs must be finite")
    if np.any(time_values < 0.0) or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing and non-negative")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    if fs_values[0] <= threshold:
        return TauAlphaResult(float(time_values[0]), True, None, float(threshold))
    for index in range(1, len(fs_values)):
        before, after = fs_values[index - 1], fs_values[index]
        if after <= threshold <= before:
            weight = (before - threshold) / max(before - after, np.finfo(float).eps)
            if time_values[index - 1] == 0.0:
                crossing_time = time_values[index - 1] + weight * (
                    time_values[index] - time_values[index - 1]
                )
            else:
                log_time = np.log(time_values[index - 1]) + weight * (
                    np.log(time_values[index]) - np.log(time_values[index - 1])
                )
                crossing_time = np.exp(log_time)
            return TauAlphaResult(float(crossing_time), True, None, float(threshold))
    return TauAlphaResult(None, False, float(time_values[-1]), float(threshold))


def _minimum_image(displacements: np.ndarray, box_length: float) -> np.ndarray:
    return displacements - box_length * np.rint(displacements / box_length)


def _neighbor_pairs(initial_positions: np.ndarray, box_length: float, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    displacement = _minimum_image(
        initial_positions[:, None, :] - initial_positions[None, :, :], box_length
    )
    squared_distance = np.einsum("ijk,ijk->ij", displacement, displacement)
    mask = np.triu(squared_distance < cutoff * cutoff, k=1)
    return np.nonzero(mask)


def cage_relative_event_fractions(
    unwrapped_trajectory: np.ndarray,
    times: np.ndarray | Iterable[float],
    *,
    box_length: float,
    horizons: Iterable[float],
    threshold: float = 0.6,
    persistence: float = 5.0,
    neighbor_cutoff: float = 1.4,
) -> dict[float, float]:
    """Return a pre-freeze sustained cage-relative mobility fraction per horizon.

    A particle is counted at a horizon once its initial-neighbour cage-relative
    displacement exceeds ``threshold`` continuously for at least
    ``persistence`` time units.  Particles with no initial neighbours are never
    called events by this proxy.
    """

    positions = np.asarray(unwrapped_trajectory, dtype=float)
    time_values = np.asarray(times, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("unwrapped_trajectory must have shape (frames, particles, 3)")
    if positions.shape[0] != len(time_values) or len(time_values) < 2:
        raise ValueError("trajectory and times must contain the same at least-two frames")
    if not np.isfinite(positions).all() or not np.isfinite(time_values).all():
        raise ValueError("trajectory and times must be finite")
    if box_length <= 0.0 or threshold <= 0.0 or persistence <= 0.0 or neighbor_cutoff <= 0.0:
        raise ValueError("box length, threshold, persistence, and cutoff must be positive")
    intervals = np.diff(time_values)
    if np.any(intervals <= 0.0):
        raise ValueError("times must be strictly increasing")
    timestep = float(np.median(intervals))
    if not np.allclose(intervals, timestep, rtol=1e-6, atol=1e-10):
        raise ValueError("event proxy requires regularly sampled trajectory frames")

    horizon_values = tuple(sorted({float(value) for value in horizons}))
    if any(value <= 0.0 for value in horizon_values):
        raise ValueError("horizons must be positive")
    n_frames, n_particles, _ = positions.shape
    left, right = _neighbor_pairs(positions[0], float(box_length), float(neighbor_cutoff))
    neighbour_count = np.bincount(
        np.concatenate((left, right)), minlength=n_particles
    ).astype(float)
    displacements = positions - positions[0:1]
    active = np.zeros((n_frames, n_particles), dtype=bool)
    chunk_size = 64
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        chunk = displacements[start:stop]
        neighbour_sum = np.zeros_like(chunk)
        if len(left):
            frame_index = np.arange(stop - start)[:, None]
            np.add.at(neighbour_sum, (frame_index, left[None, :]), chunk[:, right, :])
            np.add.at(neighbour_sum, (frame_index, right[None, :]), chunk[:, left, :])
        neighbour_mean = np.divide(
            neighbour_sum,
            neighbour_count[None, :, None],
            out=np.zeros_like(neighbour_sum),
            where=neighbour_count[None, :, None] > 0.0,
        )
        cage_relative = chunk - neighbour_mean
        cage_relative[:, neighbour_count == 0.0, :] = 0.0
        active[start:stop] = np.linalg.vector_norm(cage_relative, axis=2) > threshold

    required_frames = max(1, int(np.ceil(persistence / timestep)))
    run_length = np.zeros(n_particles, dtype=np.int64)
    seen = np.zeros(n_particles, dtype=bool)
    results: dict[float, float] = {}
    next_horizon = 0
    for frame_index, time_value in enumerate(time_values):
        run_length = np.where(active[frame_index], run_length + 1, 0)
        seen |= run_length >= required_frames
        while next_horizon < len(horizon_values) and time_value >= horizon_values[next_horizon]:
            results[horizon_values[next_horizon]] = float(seen.mean())
            next_horizon += 1
    while next_horizon < len(horizon_values):
        results[horizon_values[next_horizon]] = float(seen.mean())
        next_horizon += 1
    return results


def event_proxy(
    frames: np.ndarray,
    reference_positions: np.ndarray,
    box: float | np.ndarray,
    *,
    sample_interval: float,
    horizons: Iterable[float],
    threshold: float = 0.6,
    persistence_time: float = 5.0,
    neighbor_cutoff: float = 1.4,
) -> dict[float, float]:
    """Evaluate the advance-declaration-labelled sustained event proxy.

    ``persistence_time`` is discretized as the minimum number of consecutive
    stored frames, ``ceil(persistence_time / sample_interval)``.  The model
    box is cubic, so a length-three box vector is accepted after checking that
    all sides match.
    """

    trajectory = np.asarray(frames, dtype=float).copy()
    reference = np.asarray(reference_positions, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[2] != 3 or reference.shape != trajectory.shape[1:]:
        raise ValueError("frames and reference_positions have incompatible shapes")
    if sample_interval <= 0.0:
        raise ValueError("sample_interval must be positive")
    sides = np.asarray(box, dtype=float)
    if sides.ndim == 0:
        box_length = float(sides)
    elif sides.shape == (3,) and np.allclose(sides, sides[0], rtol=0.0, atol=1e-12):
        box_length = float(sides[0])
    else:
        raise ValueError("the bulk pilot requires a cubic box")
    trajectory[0] = reference
    times = np.arange(trajectory.shape[0], dtype=float) * float(sample_interval)
    return cage_relative_event_fractions(
        trajectory,
        times,
        box_length=box_length,
        horizons=horizons,
        threshold=threshold,
        persistence=persistence_time,
        neighbor_cutoff=neighbor_cutoff,
    )
