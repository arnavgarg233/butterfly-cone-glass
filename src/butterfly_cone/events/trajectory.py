"""Trajectory container and shared periodic-geometry helpers.

This package operates purely on saved trajectory arrays; it runs no simulation
and does not import the torch engine at runtime, so it can also process inherited
configurations' trajectories.  The minimum-image and pair-diameter conventions
here are numpy reimplementations of ``butterfly_cone.engine.potential`` and match its
constants exactly (nonadditivity 0.2, mixing rule 0.5*(si+sj)*(1-0.2|si-sj|)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONADDITIVITY = 0.2


def as_float_array(value) -> np.ndarray:
    """Convert numpy arrays or CPU torch tensors to a float64 numpy array."""

    return np.asarray(value, dtype=float)


def as_box(box) -> np.ndarray:
    """Return the box as a length-3 float array (scalar broadcast to a cube)."""

    sides = np.asarray(box, dtype=float)
    if sides.ndim == 0:
        sides = np.full(3, float(sides))
    if sides.shape != (3,) or np.any(sides <= 0.0) or not np.isfinite(sides).all():
        raise ValueError("box must be a positive scalar or length-three vector")
    return sides


def minimum_image(displacements: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Nearest-image convention in an orthorhombic box (matches the engine)."""

    return displacements - box * np.rint(displacements / box)


def mixing_diameter(sigma_i: np.ndarray, sigma_j: np.ndarray) -> np.ndarray:
    """Nonadditive pair diameter sigma_ij (matches engine.potential)."""

    return 0.5 * (sigma_i + sigma_j) * (1.0 - NONADDITIVITY * np.abs(sigma_i - sigma_j))


def wrap(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap positions into the primary periodic cell [0, L)."""

    return np.remainder(positions, box)


@dataclass
class Trajectory:
    """A saved trajectory: unwrapped positions plus static per-particle data.

    Parameters
    ----------
    unwrapped_positions : (T, N, 3) array
        Continuous (unwrapped) positions; the engine convention for dynamics.
        Displacements over time are read directly from these.
    times : (T,) array
        Strictly increasing sample times.
    sigma : (N,) array
        Per-particle diameters.
    box : scalar or (3,)
        Periodic box lengths.  Distances (neighbours, clustering, strings) use
        the minimum image of wrapped positions.
    positions : optional (T, N, 3) array
        Wrapped positions.  Derived from ``unwrapped_positions`` if omitted.
    """

    unwrapped_positions: np.ndarray
    times: np.ndarray
    sigma: np.ndarray
    box: np.ndarray
    positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.unwrapped_positions = as_float_array(self.unwrapped_positions)
        self.times = as_float_array(self.times)
        self.sigma = as_float_array(self.sigma)
        self.box = as_box(self.box)
        if self.unwrapped_positions.ndim != 3 or self.unwrapped_positions.shape[2] != 3:
            raise ValueError("unwrapped_positions must have shape (T, N, 3)")
        n_frames, n_particles, _ = self.unwrapped_positions.shape
        if self.times.shape != (n_frames,):
            raise ValueError("times must have shape (T,)")
        if n_frames < 2:
            raise ValueError("a trajectory needs at least two frames")
        if self.sigma.shape != (n_particles,):
            raise ValueError("sigma must have shape (N,)")
        if not np.isfinite(self.unwrapped_positions).all() or not np.isfinite(self.times).all():
            raise ValueError("positions and times must be finite")
        intervals = np.diff(self.times)
        if np.any(intervals <= 0.0):
            raise ValueError("times must be strictly increasing")
        self._dt = float(np.median(intervals))
        if self.positions is None:
            self.positions = wrap(self.unwrapped_positions, self.box)
        else:
            self.positions = as_float_array(self.positions)
            if self.positions.shape != self.unwrapped_positions.shape:
                raise ValueError("positions must match unwrapped_positions shape")

    @property
    def n_frames(self) -> int:
        return int(self.unwrapped_positions.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.unwrapped_positions.shape[1])

    @property
    def dt(self) -> float:
        return self._dt

    def frames_for_duration(self, duration: float) -> int:
        """Number of frames spanning ``duration`` time units (at least one)."""

        if duration <= 0.0:
            raise ValueError("duration must be positive")
        return max(1, int(np.ceil(duration / self._dt)))

    def frame_at_or_after(self, frame: int, duration: float) -> int:
        """Index of the frame >= ``duration`` after ``frame`` (clamped to end)."""

        target_time = self.times[frame] + duration
        idx = int(np.searchsorted(self.times, target_time, side="left"))
        return min(idx, self.n_frames - 1)


def neighbor_pairs(
    positions: np.ndarray,
    sigma: np.ndarray,
    box: np.ndarray,
    first_shell_factor: float,
    *,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Undirected first-shell neighbour pairs (i < j) at a single frame.

    Pair ``(i, j)`` is a neighbour when the minimum-image distance is below
    ``first_shell_factor * sigma_ij``.  Chunked over rows to bound memory for
    large N (O(N^2) work, O(chunk*N) memory).
    """

    positions = as_float_array(positions)
    sigma = as_float_array(sigma)
    box = as_box(box)
    n = positions.shape[0]
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        disp = positions[start:stop, None, :] - positions[None, :, :]
        disp = minimum_image(disp, box)
        squared = np.einsum("ijk,ijk->ij", disp, disp)
        sig_ij = mixing_diameter(sigma[start:stop, None], sigma[None, :])
        cutoff = first_shell_factor * sig_ij
        mask = squared < cutoff * cutoff
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        mask &= rows < cols
        local_i, local_j = np.nonzero(mask)
        lefts.append((local_i + start).astype(np.int64))
        rights.append(local_j.astype(np.int64))
    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lefts), np.concatenate(rights)


def neighbor_sets(
    positions: np.ndarray,
    sigma: np.ndarray,
    box: np.ndarray,
    first_shell_factor: float,
) -> list[np.ndarray]:
    """Per-particle first-shell neighbour index arrays at a single frame."""

    left, right = neighbor_pairs(positions, sigma, box, first_shell_factor)
    n = as_float_array(positions).shape[0]
    buckets: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(left.tolist(), right.tolist()):
        buckets[i].append(j)
        buckets[j].append(i)
    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def minimum_image_centroid(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image-aware centroid of a set of points in a periodic box.

    Unwraps every point relative to the first point before averaging, then wraps
    the mean back into the cell.  Correct for clusters straddling the boundary.
    """

    points = as_float_array(points)
    box = as_box(box)
    if points.shape[0] == 0:
        raise ValueError("cannot take the centroid of zero points")
    reference = points[0]
    relative = minimum_image(points - reference, box)
    mean_relative = relative.mean(axis=0)
    return wrap(reference + mean_relative, box)


def radius_of_gyration(points: np.ndarray, box: np.ndarray) -> float:
    """Minimum-image radius of gyration of a set of points."""

    points = as_float_array(points)
    box = as_box(box)
    if points.shape[0] == 0:
        return 0.0
    reference = points[0]
    relative = minimum_image(points - reference, box)
    mean_relative = relative.mean(axis=0)
    deviations = relative - mean_relative
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", deviations, deviations))))
