"""Cage-relative displacement machinery.

For particle i between frames t0 and t1 the cage-relative displacement is its
plain displacement minus the mean displacement of its t0 first-shell neighbours.
Plain displacements are read directly from unwrapped positions (the engine's
dynamics convention); neighbour sets and any distance use the minimum image of
wrapped positions.  Cage-relative subtraction removes uniform drift and slow
affine shear, isolating genuine cage escape.
"""

from __future__ import annotations

import numpy as np

from .config import DisplacementConfig
from .trajectory import Trajectory, neighbor_pairs


def plain_displacement(traj: Trajectory, t0_frame: int, t1_frame: int) -> np.ndarray:
    """Plain per-particle displacement vector between two frames, shape (N, 3)."""

    return traj.unwrapped_positions[t1_frame] - traj.unwrapped_positions[t0_frame]


def _neighbor_mean_displacement(
    displacement: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Mean displacement of each particle's neighbours, per frame.

    ``displacement`` has shape (F, N, 3); returns the same shape.
    """

    n_frames = displacement.shape[0]
    neighbour_sum = np.zeros_like(displacement)
    if left.size:
        frame_index = np.arange(n_frames)[:, None]
        np.add.at(neighbour_sum, (frame_index, left[None, :]), displacement[:, right, :])
        np.add.at(neighbour_sum, (frame_index, right[None, :]), displacement[:, left, :])
    return np.divide(
        neighbour_sum,
        counts[None, :, None],
        out=np.zeros_like(neighbour_sum),
        where=counts[None, :, None] > 0.0,
    )


def cage_relative_field(
    traj: Trajectory,
    config: DisplacementConfig = DisplacementConfig(),
    *,
    reference_frame: int = 0,
    chunk: int = 64,
) -> np.ndarray:
    """Cage-relative displacement of every particle at every frame.

    Returns an array of shape (T, N, 3): the displacement of each particle
    relative to ``reference_frame`` minus the mean displacement of its
    reference-frame first-shell neighbours.
    """

    reference = traj.positions[reference_frame]
    left, right = neighbor_pairs(reference, traj.sigma, traj.box, config.first_shell_factor)
    counts = np.bincount(
        np.concatenate((left, right)) if left.size else np.empty(0, dtype=np.int64),
        minlength=traj.n_particles,
    ).astype(float)
    plain = traj.unwrapped_positions - traj.unwrapped_positions[reference_frame][None, :, :]
    field = np.empty_like(plain)
    for start in range(0, traj.n_frames, chunk):
        stop = min(start + chunk, traj.n_frames)
        block = plain[start:stop]
        neighbour_mean = _neighbor_mean_displacement(block, left, right, counts)
        cage = block - neighbour_mean
        if config.isolated_fallback == "zero":
            cage[:, counts == 0.0, :] = 0.0
        elif config.isolated_fallback != "plain":
            raise ValueError("isolated_fallback must be 'plain' or 'zero'")
        # For "plain" the neighbour mean is already zero where counts == 0, so
        # cage equals the plain displacement there and no action is needed.
        field[start:stop] = cage
    return field


def cage_relative_displacement(
    traj: Trajectory,
    t0_frame: int,
    t1_frame: int,
    config: DisplacementConfig = DisplacementConfig(),
) -> np.ndarray:
    """Cage-relative displacement between a single pair of frames, shape (N, 3).

    Neighbour sets are taken at ``t0_frame`` (the reference).
    """

    reference = traj.positions[t0_frame]
    left, right = neighbor_pairs(reference, traj.sigma, traj.box, config.first_shell_factor)
    counts = np.bincount(
        np.concatenate((left, right)) if left.size else np.empty(0, dtype=np.int64),
        minlength=traj.n_particles,
    ).astype(float)
    plain = plain_displacement(traj, t0_frame, t1_frame)[None, :, :]
    neighbour_mean = _neighbor_mean_displacement(plain, left, right, counts)
    cage = (plain - neighbour_mean)[0]
    if config.isolated_fallback == "zero":
        cage[counts == 0.0, :] = 0.0
    return cage


def overlap_indicator(
    traj: Trajectory,
    t0_frame: int,
    t1_frame: int,
    config: DisplacementConfig = DisplacementConfig(),
) -> np.ndarray:
    """Per-particle overlap indicator w_i = 1 iff |plain displacement| < a."""

    disp = plain_displacement(traj, t0_frame, t1_frame)
    magnitude = np.linalg.norm(disp, axis=1)
    return (magnitude < config.overlap_a).astype(np.int64)


def magnitude_field(field: np.ndarray) -> np.ndarray:
    """Per-frame per-particle magnitude of a displacement field (T, N)."""

    return np.linalg.norm(field, axis=2)
