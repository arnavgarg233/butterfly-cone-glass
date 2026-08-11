"""Rearrangement (hop) detection.

Two interchangeable detectors sit behind one interface and emit the same
per-particle event record:

* persistent-displacement detector -- cage-relative (default) displacement
  magnitude sustained above a threshold for a persistence window;
* p_hop detector -- the Candelier two-window mean-separation statistic.

Every threshold lives in :class:`HopConfig` and is provisional until the
advance-declaration freeze.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DisplacementConfig, HopConfig
from .displacements import cage_relative_field, magnitude_field
from .trajectory import Trajectory


@dataclass(frozen=True)
class HopEvent:
    """A detected per-particle rearrangement."""

    particle: int
    onset_frame: int
    onset_time: float
    magnitude: float
    detector: str


def _displacement_field(traj: Trajectory, config: HopConfig, cage_relative: bool) -> np.ndarray:
    """Displacement field relative to the reference frame (T, N, 3)."""

    if cage_relative:
        disp_config = DisplacementConfig(first_shell_factor=config.first_shell_factor)
        return cage_relative_field(traj, disp_config, reference_frame=config.reference_frame)
    return traj.unwrapped_positions - traj.unwrapped_positions[config.reference_frame][None, :, :]


def detect_persistent(traj: Trajectory, config: HopConfig = HopConfig()) -> list[HopEvent]:
    """Persistent-displacement detector.

    A particle is active at a frame when its (cage-relative) displacement from
    the reference frame exceeds ``persistent_threshold``.  Its onset is the start
    of the first run of consecutive active frames that lasts at least
    ``persist_time``; the magnitude is the displacement magnitude at that onset.
    A rattler that never sustains the threshold produces no event.
    """

    field = _displacement_field(traj, config, config.persistent_cage_relative)
    magnitude = magnitude_field(field)  # (T, N)
    active = magnitude > config.persistent_threshold
    required = traj.frames_for_duration(config.persist_time)

    events: list[HopEvent] = []
    n_frames, n_particles = active.shape
    run = np.zeros(n_particles, dtype=np.int64)
    onset_frame = np.full(n_particles, -1, dtype=np.int64)
    seen = np.zeros(n_particles, dtype=bool)
    for frame in range(n_frames):
        run = np.where(active[frame], run + 1, 0)
        just_reached = (run == required) & (~seen)
        if np.any(just_reached):
            starts = frame - required + 1
            onset_frame[just_reached] = starts
            seen |= just_reached
    for particle in np.nonzero(seen)[0]:
        start = int(onset_frame[particle])
        events.append(
            HopEvent(
                particle=int(particle),
                onset_frame=start,
                onset_time=float(traj.times[start]),
                magnitude=float(magnitude[start, particle]),
                detector="persistent",
            )
        )
    events.sort(key=lambda e: (e.onset_frame, e.particle))
    return events


def phop_statistic(traj: Trajectory, config: HopConfig = HopConfig()) -> np.ndarray:
    """Candelier p_hop(i, t) statistic for every particle and frame, shape (T, N).

    With half-window w = phop_window_time/2 and windows A=[t-w, t], B=[t, t+w],

        p_hop(i, t) = sqrt( <(r_i - <r_i>_B)^2>_A * <(r_i - <r_i>_A)^2>_B )

    where means are over the window frames.  Frames without both full windows
    yield zero.
    """

    positions = traj.unwrapped_positions
    if config.phop_cage_relative:
        disp_config = DisplacementConfig(first_shell_factor=config.first_shell_factor)
        positions = cage_relative_field(traj, disp_config, reference_frame=config.reference_frame)

    half = traj.frames_for_duration(config.phop_window_time / 2.0)
    n_frames, n_particles = positions.shape[0], positions.shape[1]
    phop = np.zeros((n_frames, n_particles), dtype=float)
    for t in range(half, n_frames - half):
        window_a = positions[t - half : t + 1]  # inclusive of t
        window_b = positions[t : t + half + 1]
        mean_a = window_a.mean(axis=0)
        mean_b = window_b.mean(axis=0)
        term_a = np.mean(np.sum((window_a - mean_b) ** 2, axis=2), axis=0)
        term_b = np.mean(np.sum((window_b - mean_a) ** 2, axis=2), axis=0)
        phop[t] = np.sqrt(term_a * term_b)
    return phop


def detect_phop(traj: Trajectory, config: HopConfig = HopConfig()) -> list[HopEvent]:
    """p_hop detector: a particle hops if max_t p_hop(i, t) exceeds the threshold.

    The onset is the frame of the peak p_hop; the magnitude is that peak value.
    """

    phop = phop_statistic(traj, config)
    peak_frame = np.argmax(phop, axis=0)
    peak_value = phop[peak_frame, np.arange(phop.shape[1])]
    events: list[HopEvent] = []
    for particle in np.nonzero(peak_value > config.phop_threshold)[0]:
        frame = int(peak_frame[particle])
        events.append(
            HopEvent(
                particle=int(particle),
                onset_frame=frame,
                onset_time=float(traj.times[frame]),
                magnitude=float(peak_value[particle]),
                detector="phop",
            )
        )
    events.sort(key=lambda e: (e.onset_frame, e.particle))
    return events


_DETECTORS = {"persistent": detect_persistent, "phop": detect_phop}


def detect_hops(
    traj: Trajectory,
    config: HopConfig = HopConfig(),
    *,
    detector: str = "persistent",
) -> list[HopEvent]:
    """Unified interface: dispatch to the named detector.

    ``detector`` is one of ``"persistent"`` or ``"phop"``.
    """

    if detector not in _DETECTORS:
        raise ValueError(f"detector must be one of {sorted(_DETECTORS)}")
    return _DETECTORS[detector](traj, config)
