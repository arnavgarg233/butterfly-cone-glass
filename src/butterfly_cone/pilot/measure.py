"""Bulk-pilot stationarity and time-origin dynamical measurements.

The functions in this module deliberately operate on CPU NumPy arrays after a
trajectory has been sampled.  Integration remains on the selected torch device;
the reductions are small enough that keeping them deterministic and easy to
audit is preferable to adding another device-specific implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import TauAlphaResult, extract_tau_alpha


@dataclass(frozen=True)
class StationarityResult:
    """First-half/second-half potential-energy stationarity diagnostic."""

    first_half_mean: float
    second_half_mean: float
    absolute_drift: float
    standard_error: float
    threshold: float
    passed: bool
    n_samples: int


@dataclass(frozen=True)
class TimeOriginDynamics:
    """Self-dynamics and overlap fluctuations evaluated at selected lags."""

    times: np.ndarray
    fs: np.ndarray
    msd: np.ndarray
    overlap: np.ndarray
    chi4: np.ndarray
    tau_alpha: TauAlphaResult
    cage_plateau_present: bool
    cage_plateau_height: float | None
    cage_plateau_window: tuple[float, float] | None


def stationarity_check(
    potential_energy_per_particle: np.ndarray,
    *,
    absolute_drift_floor: float = 0.01,
    standard_error_multiplier: float = 2.0,
) -> StationarityResult:
    """Apply the declared in advance-style energy-drift gate.

    The trace is split in time.  A state passes when the absolute difference
    between the two half means is no larger than the greater of 0.01 reduced
    energy units per particle and two independent-sample standard errors.  The
    absolute floor prevents a highly correlated, oversampled energy trace from
    making an implausibly strict numerical gate.
    """

    values = np.asarray(potential_energy_per_particle, dtype=float)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("stationarity requires at least four scalar samples")
    if not np.isfinite(values).all():
        raise ValueError("stationarity energies must be finite")
    if absolute_drift_floor <= 0.0 or standard_error_multiplier <= 0.0:
        raise ValueError("stationarity thresholds must be positive")

    split = values.size // 2
    first, second = values[:split], values[split:]
    first_mean = float(first.mean())
    second_mean = float(second.mean())
    first_variance = float(first.var(ddof=1))
    second_variance = float(second.var(ddof=1))
    standard_error = float(np.sqrt(first_variance / first.size + second_variance / second.size))
    threshold = max(float(absolute_drift_floor), float(standard_error_multiplier) * standard_error)
    absolute_drift = abs(second_mean - first_mean)
    return StationarityResult(
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        absolute_drift=absolute_drift,
        standard_error=standard_error,
        threshold=threshold,
        passed=bool(absolute_drift <= threshold),
        n_samples=int(values.size),
    )


def _lag_indices(n_frames: int, n_lags: int) -> np.ndarray:
    if n_lags < 3:
        raise ValueError("n_lags must be at least three")
    if n_frames < 2:
        raise ValueError("at least two trajectory frames are required")
    if n_lags >= n_frames:
        return np.arange(n_frames, dtype=np.int64)
    logarithmic = np.rint(np.geomspace(1.0, float(n_frames - 1), n_lags // 2)).astype(np.int64)
    linear = np.rint(np.linspace(0.0, float(n_frames - 1), n_lags - n_lags // 2)).astype(np.int64)
    return np.unique(np.concatenate((np.array([0], dtype=np.int64), logarithmic, linear)))


def _origin_indices(n_available: int, max_origins: int) -> np.ndarray:
    if max_origins <= 0:
        raise ValueError("max_origins must be positive")
    return np.unique(
        np.rint(np.linspace(0.0, float(n_available - 1), min(n_available, max_origins))).astype(np.int64)
    )


def _cage_plateau(
    times: np.ndarray,
    msd: np.ndarray,
) -> tuple[bool, float | None, tuple[float, float] | None]:
    """Locate a low-log-slope plateau between 0.5 and 10 time units."""

    candidate = (times >= 0.5) & (times <= min(10.0, float(times[-1]))) & (msd > 0.0)
    indices = np.flatnonzero(candidate)
    if indices.size < 3:
        return False, None, None
    candidate_times = times[indices]
    candidate_msd = msd[indices]
    slopes = np.gradient(np.log(candidate_msd), np.log(candidate_times))
    acceptable = np.abs(slopes) <= 0.25
    start = 0
    while start < acceptable.size:
        if not acceptable[start]:
            start += 1
            continue
        stop = start + 1
        while stop < acceptable.size and acceptable[stop]:
            stop += 1
        if stop - start >= 3:
            plateau_values = candidate_msd[start:stop]
            return (
                True,
                float(np.median(plateau_values)),
                (float(candidate_times[start]), float(candidate_times[stop - 1])),
            )
        start = stop
    return False, None, None


def analyze_time_origin_dynamics(
    unwrapped_trajectory: np.ndarray,
    times: np.ndarray,
    *,
    wave_number: float = 7.1,
    overlap_cutoff: float = 0.3,
    n_lags: int = 120,
    max_origins: int = 16,
) -> TimeOriginDynamics:
    """Measure ``F_s``, MSD, ``Q``, and ``chi4`` with time-origin averaging."""

    frames = np.asarray(unwrapped_trajectory, dtype=float)
    time_values = np.asarray(times, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError("unwrapped_trajectory must have shape (frames, particles, 3)")
    if time_values.shape != (frames.shape[0],):
        raise ValueError("times must contain one value per trajectory frame")
    if frames.shape[1] < 1 or frames.shape[0] < 2:
        raise ValueError("trajectory must include at least two frames and one particle")
    if not np.isfinite(frames).all() or not np.isfinite(time_values).all():
        raise ValueError("trajectory and times must be finite")
    if time_values[0] != 0.0 or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must start at zero and be strictly increasing")
    if wave_number <= 0.0 or overlap_cutoff <= 0.0:
        raise ValueError("wave_number and overlap_cutoff must be positive")

    lag_indices = _lag_indices(frames.shape[0], n_lags)
    lag_times = time_values[lag_indices] - time_values[0]
    fs = np.empty(lag_indices.size, dtype=float)
    msd = np.empty(lag_indices.size, dtype=float)
    mean_overlap = np.empty(lag_indices.size, dtype=float)
    chi4 = np.empty(lag_indices.size, dtype=float)
    cutoff_squared = overlap_cutoff * overlap_cutoff
    n_particles = frames.shape[1]

    for output_index, lag in enumerate(lag_indices):
        origins = _origin_indices(frames.shape[0] - int(lag), max_origins)
        displacement = frames[origins + lag] - frames[origins]
        squared = np.sum(displacement * displacement, axis=2)
        fs[output_index] = float(np.cos(wave_number * displacement).mean())
        per_origin_msd = squared.mean(axis=1)
        per_origin_overlap = (squared < cutoff_squared).mean(axis=1)
        msd[output_index] = float(per_origin_msd.mean())
        mean_overlap[output_index] = float(per_origin_overlap.mean())
        chi4[output_index] = (
            float(n_particles * per_origin_overlap.var(ddof=1)) if per_origin_overlap.size > 1 else 0.0
        )

    tau_alpha = extract_tau_alpha(lag_times, fs)
    plateau_present, plateau_height, plateau_window = _cage_plateau(lag_times, msd)
    return TimeOriginDynamics(
        times=lag_times,
        fs=fs,
        msd=msd,
        overlap=mean_overlap,
        chi4=chi4,
        tau_alpha=tau_alpha,
        cage_plateau_present=plateau_present,
        cage_plateau_height=plateau_height,
        cage_plateau_window=plateau_window,
    )
