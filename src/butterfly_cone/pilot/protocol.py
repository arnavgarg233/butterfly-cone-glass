"""Execution and reduction primitives for the bulk-temperature pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch

from butterfly_cone.engine.integrate import BussiThermostat, MDIntegrator, maxwell_boltzmann_velocities
from butterfly_cone.engine.observables import temperature as instantaneous_temperature
from butterfly_cone.engine.swap import HybridSwapMD
from butterfly_cone.engine.system import ParticleSystem, make_generator

from .dynamics import TauAlphaResult, event_proxy, extract_tau_alpha
from .structure import (
    demixing_passes,
    diameter_demixing_statistic,
    empirical_ks_distance,
    local_q6,
    low_k_structure_factor,
)


@dataclass(frozen=True)
class PilotProtocol:
    """Frozen numerical controls for one bulk-pilot execution."""

    dt: float = 0.005
    thermostat_tau: float = 0.5
    reequilibration_cycles: int = 8
    reequilibration_md_steps: int = 400
    swap_attempts_per_cycle: int = 64
    production_time: float = 250.0
    sample_interval: float = 0.25
    origin_interval: float = 5.0
    n_lags: int = 64
    wave_number: float = 7.1
    overlap_cutoff: float = 0.3
    event_threshold: float = 0.6
    event_persistence: float = 5.0
    event_neighbor_cutoff: float = 1.4
    q6_neighbor_cutoff: float = 1.4
    demixing_neighbor_cutoff: float = 1.4
    demixing_shuffles: int = 128

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.thermostat_tau <= 0.0:
            raise ValueError("dt and thermostat_tau must be positive")
        if self.reequilibration_cycles < 4 or self.reequilibration_md_steps <= 0:
            raise ValueError("at least four positive reequilibration blocks are required")
        if self.swap_attempts_per_cycle < 0 or self.production_time < 0.0:
            raise ValueError("swap attempts and production time must be non-negative")
        if self.sample_interval <= 0.0 or self.origin_interval <= 0.0 or self.n_lags < 4:
            raise ValueError("sampling controls are invalid")


@dataclass(frozen=True)
class StationarityDiagnostic:
    """Predeclared first-half/second-half potential-energy drift diagnostic."""

    first_half_mean: float
    second_half_mean: float
    absolute_drift: float
    standard_error: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class ReequilibrationResult:
    potential_energy_per_particle: np.ndarray
    swap_attempts: int
    swap_accepted: int
    swap_acceptance: float
    stationarity: StationarityDiagnostic


@dataclass(frozen=True)
class ProductionTrajectory:
    times: np.ndarray
    unwrapped_positions: np.ndarray
    potential_energy_per_particle: np.ndarray
    kinetic_temperature: np.ndarray
    simulated_steps: int


@dataclass(frozen=True)
class CagePlateau:
    present: bool
    height: float | None
    time: float | None
    minimum_log_slope: float | None


@dataclass(frozen=True)
class TrajectoryObservables:
    lag_times: np.ndarray
    fs: np.ndarray
    msd: np.ndarray
    overlap: np.ndarray
    chi4: np.ndarray
    q_samples: np.ndarray
    tau_alpha: TauAlphaResult
    cage_plateau: CagePlateau
    event_fractions: dict[float, float]


STATIONARITY_ABSOLUTE_TOLERANCE = 0.0025
STATIONARITY_STANDARD_ERROR_MULTIPLIER = 2.0


def engine_seed(issued_seed: int) -> int:
    """Project a harness SHA-256 seed into PyTorch's accepted generator range."""

    if isinstance(issued_seed, bool) or not isinstance(issued_seed, int):
        raise TypeError("issued_seed must be an integer")
    return int(issued_seed) % (2**63 - 1)


def stationarity_diagnostic(
    potential_energy_per_particle: np.ndarray | Iterable[float],
    *,
    absolute_tolerance: float = STATIONARITY_ABSOLUTE_TOLERANCE,
    standard_error_multiplier: float = STATIONARITY_STANDARD_ERROR_MULTIPLIER,
) -> StationarityDiagnostic:
    """Apply the declared in advance-style two-half energy-drift gate.

    The gate accepts a difference no larger than ``max(0.0025, 2 * SE_delta)``
    in reduced potential energy per particle.  The nonzero absolute floor avoids
    treating a nearly noiseless but physically negligible float32 change as
    evidence of aging.
    """

    values = np.asarray(potential_energy_per_particle, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("at least four finite energy samples are required")
    if absolute_tolerance < 0.0 or standard_error_multiplier < 0.0:
        raise ValueError("stationarity thresholds must be non-negative")
    split = len(values) // 2
    first, second = values[:split], values[split:]
    first_mean, second_mean = float(first.mean()), float(second.mean())
    first_variance = float(np.var(first, ddof=1)) if len(first) > 1 else 0.0
    second_variance = float(np.var(second, ddof=1)) if len(second) > 1 else 0.0
    standard_error = math.sqrt(first_variance / len(first) + second_variance / len(second))
    drift = abs(second_mean - first_mean)
    threshold = max(float(absolute_tolerance), float(standard_error_multiplier) * standard_error)
    return StationarityDiagnostic(
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        absolute_drift=float(drift),
        standard_error=float(standard_error),
        threshold=float(threshold),
        passed=bool(drift <= threshold),
    )


def set_maxwell_boltzmann_velocities(system: ParticleSystem, temperature: float, seed: int) -> None:
    """Draw production or reequilibration velocities from an explicit seed."""

    system.velocities = maxwell_boltzmann_velocities(
        system.n_particles,
        temperature,
        make_generator(engine_seed(seed)),
        device=system.device,
        dtype=system.dtype,
        active_mask=system.active_mask,
    )


def reequilibrate_with_swaps(
    system: ParticleSystem,
    *,
    temperature: float,
    protocol: PilotProtocol,
    thermostat_seed: int,
    swap_seed: int,
) -> ReequilibrationResult:
    """Run short Bussi-NVT/diameter-swap blocks and measure energy drift."""

    thermostat = BussiThermostat(
        temperature=temperature,
        tau=protocol.thermostat_tau,
        generator=make_generator(engine_seed(thermostat_seed)),
    )
    integrator = MDIntegrator(system, dt=protocol.dt, thermostat=thermostat)
    hybrid = HybridSwapMD(
        integrator,
        temperature=temperature,
        generator=make_generator(engine_seed(swap_seed)),
        md_steps=protocol.reequilibration_md_steps,
        swap_attempts=protocol.swap_attempts_per_cycle,
    )
    energies = []
    for _ in range(protocol.reequilibration_cycles):
        hybrid.cycle()
        energies.append(float(integrator.potential_energy.detach().cpu()) / system.n_particles)
    statistics = hybrid.statistics
    energy_values = np.asarray(energies, dtype=float)
    return ReequilibrationResult(
        potential_energy_per_particle=energy_values,
        swap_attempts=statistics.attempts,
        swap_accepted=statistics.accepted,
        swap_acceptance=float(statistics.acceptance_rate),
        stationarity=stationarity_diagnostic(energy_values),
    )


def _steps_for_time(duration: float, dt: float) -> int:
    steps = int(round(duration / dt))
    if not math.isclose(steps * dt, duration, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    return steps


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def run_nvt_production(
    system: ParticleSystem,
    *,
    temperature: float,
    protocol: PilotProtocol,
    velocity_seed: int,
    thermostat_seed: int,
) -> ProductionTrajectory:
    """Run swap-free physical NVT dynamics and retain uniformly sampled frames."""

    total_steps = _steps_for_time(protocol.production_time, protocol.dt)
    sample_steps = _steps_for_time(protocol.sample_interval, protocol.dt)
    if total_steps % sample_steps:
        raise ValueError("production time must be divisible by sample interval")
    set_maxwell_boltzmann_velocities(system, temperature, velocity_seed)
    thermostat = BussiThermostat(
        temperature=temperature,
        tau=protocol.thermostat_tau,
        generator=make_generator(engine_seed(thermostat_seed)),
    )
    integrator = MDIntegrator(system, dt=protocol.dt, thermostat=thermostat)
    n_frames = total_steps // sample_steps + 1
    positions = np.empty((n_frames, system.n_particles, 3), dtype=np.float32)
    energies = np.empty(n_frames, dtype=float)
    temperatures = np.empty(n_frames, dtype=float)
    for frame in range(n_frames):
        positions[frame] = _cpu_array(system.unwrapped_positions).astype(np.float32, copy=False)
        energies[frame] = float(integrator.potential_energy.detach().cpu()) / system.n_particles
        temperatures[frame] = float(instantaneous_temperature(system).detach().cpu())
        if frame + 1 < n_frames:
            integrator.step(sample_steps)
    return ProductionTrajectory(
        times=np.arange(n_frames, dtype=float) * protocol.sample_interval,
        unwrapped_positions=positions,
        potential_energy_per_particle=energies,
        kinetic_temperature=temperatures,
        simulated_steps=total_steps,
    )


def _lag_indices(n_frames: int, n_lags: int) -> np.ndarray:
    if n_frames < 2:
        raise ValueError("at least two sampled frames are required")
    positive = np.unique(np.rint(np.geomspace(1.0, n_frames - 1, n_lags - 1)).astype(int))
    return np.unique(np.concatenate((np.array([0], dtype=int), positive, np.array([n_frames - 1]))))


def _cage_plateau(lag_times: np.ndarray, msd: np.ndarray) -> CagePlateau:
    positive = (lag_times > 0.0) & (msd > 0.0)
    if int(positive.sum()) < 5:
        return CagePlateau(False, None, None, None)
    max_search_time = min(50.0, float(lag_times[-1]) / 2.0)
    usable = positive & (lag_times >= 0.5) & (lag_times <= max_search_time)
    indices = np.flatnonzero(usable)
    if len(indices) < 3:
        return CagePlateau(False, None, None, None)
    log_time = np.log(lag_times[indices])
    log_msd = np.log(msd[indices])
    slopes = np.gradient(log_msd, log_time)
    local = int(np.argmin(slopes))
    selected = int(indices[local])
    height_indices = indices[max(0, local - 1) : min(len(indices), local + 2)]
    return CagePlateau(
        present=bool(slopes[local] <= 0.6),
        height=float(np.median(msd[height_indices])),
        time=float(lag_times[selected]),
        minimum_log_slope=float(slopes[local]),
    )


def analyze_physical_trajectory(
    trajectory: ProductionTrajectory,
    *,
    box: float | np.ndarray,
    protocol: PilotProtocol,
    event_horizons: Iterable[float] = (10.0, 50.0, 250.0),
) -> TrajectoryObservables:
    """Time-origin-average ``F_s``, MSD, overlap, and ``chi4`` for one trajectory."""

    positions = trajectory.unwrapped_positions
    if positions.ndim != 3 or positions.shape[0] != len(trajectory.times):
        raise ValueError("trajectory positions and times are inconsistent")
    lag_indices = _lag_indices(positions.shape[0], protocol.n_lags)
    origin_steps = max(1, _steps_for_time(protocol.origin_interval, protocol.sample_interval))
    origins = np.arange(0, positions.shape[0], origin_steps, dtype=int)
    n_particles = positions.shape[1]
    fs = np.empty(len(lag_indices), dtype=float)
    msd = np.empty(len(lag_indices), dtype=float)
    overlap = np.empty(len(lag_indices), dtype=float)
    chi4 = np.full(len(lag_indices), np.nan, dtype=float)
    q_samples = np.full((len(lag_indices), len(origins)), np.nan, dtype=float)
    for output_index, lag in enumerate(lag_indices):
        valid_origins = origins[origins + lag < positions.shape[0]]
        displacement = positions[valid_origins + lag] - positions[valid_origins]
        squared = np.einsum("ijk,ijk->ij", displacement, displacement)
        fs[output_index] = float(np.cos(protocol.wave_number * displacement).mean())
        msd[output_index] = float(squared.mean())
        q_values = (squared < protocol.overlap_cutoff * protocol.overlap_cutoff).mean(axis=1)
        q_samples[output_index, : len(q_values)] = q_values
        overlap[output_index] = float(q_values.mean())
        if len(q_values) > 1:
            chi4[output_index] = float(n_particles * np.var(q_values, ddof=1))
    lag_times = lag_indices.astype(float) * protocol.sample_interval
    tau = extract_tau_alpha(lag_times, fs)
    events = event_proxy(
        positions,
        positions[0],
        box,
        sample_interval=protocol.sample_interval,
        horizons=event_horizons,
        threshold=protocol.event_threshold,
        persistence_time=protocol.event_persistence,
        neighbor_cutoff=protocol.event_neighbor_cutoff,
    )
    return TrajectoryObservables(
        lag_times=lag_times,
        fs=fs,
        msd=msd,
        overlap=overlap,
        chi4=chi4,
        q_samples=q_samples,
        tau_alpha=tau,
        cage_plateau=_cage_plateau(lag_times, msd),
        event_fractions=events,
    )


def structural_integrity_diagnostic(
    positions: np.ndarray,
    diameters: np.ndarray,
    *,
    box: float | np.ndarray,
    baseline_q6: np.ndarray,
    protocol: PilotProtocol,
    shuffle_seed: int,
) -> dict[str, float | bool | list[float]]:
    """Evaluate crystallization, demixing, and low-wavevector sanity checks."""

    q6 = local_q6(positions, box_length=float(np.asarray(box).reshape(-1)[0]), neighbor_cutoff=protocol.q6_neighbor_cutoff)
    baseline = np.asarray(baseline_q6, dtype=float)
    if baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError("baseline_q6 must be a non-empty one-dimensional array")
    q6_mean = float(q6.mean())
    q6_high_fraction = float((q6 >= 0.45).mean())
    baseline_mean = float(baseline.mean())
    baseline_high_fraction = float((baseline >= 0.45).mean())
    q6_ks = empirical_ks_distance(q6, baseline)
    crystallized = bool(
        q6_high_fraction > max(0.12, baseline_high_fraction + 0.05)
        and q6_mean > baseline_mean + 0.05
        and q6_ks > 0.15
    )
    demixing = diameter_demixing_statistic(
        positions,
        diameters,
        box,
        neighbor_cutoff=protocol.demixing_neighbor_cutoff,
        n_shuffles=protocol.demixing_shuffles,
        seed=shuffle_seed,
    )
    low_k = low_k_structure_factor(positions, box)
    demixing_pass = demixing_passes(demixing, positive_z_threshold=4.0)
    low_k_pass = bool(np.isfinite(float(low_k["max"])) and float(low_k["max"]) < 10.0)
    return {
        "q6_mean": q6_mean,
        "q6_std": float(q6.std(ddof=1)) if len(q6) > 1 else 0.0,
        "q6_p90": float(np.quantile(q6, 0.90)),
        "q6_p99": float(np.quantile(q6, 0.99)),
        "q6_high_fraction": q6_high_fraction,
        "baseline_q6_mean": baseline_mean,
        "baseline_q6_high_fraction": baseline_high_fraction,
        "q6_ks_vs_t015": q6_ks,
        "crystallization_pass": not crystallized,
        "demixing_observed_correlation": demixing.observed,
        "demixing_null_mean": demixing.null_mean,
        "demixing_null_std": demixing.null_std,
        "demixing_z_score": demixing.z_score,
        "demixing_pairs": float(demixing.n_pairs),
        "demixing_pass": demixing_pass,
        "low_k_wave_numbers": list(low_k["wave_numbers"]),
        "low_k_structure_factor": list(low_k["values"]),
        "low_k_max": float(low_k["max"]),
        "low_k_pass": low_k_pass,
        "structural_pass": bool((not crystallized) and demixing_pass and low_k_pass),
    }
