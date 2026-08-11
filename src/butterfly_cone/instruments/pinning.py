"""Cavity-free random-pinning operator and its retrospective pinning postdiction.

Random pinning freezes a random fraction ``f_p`` of the particles at their
equilibrium positions and evolves the rest, then measures how the frozen
*scaffold* reshapes the dynamics of the mobile particles.  This is ButterflyCone's
matched-CONTROL arm realised as a real physical experiment: it is exactly the
Cammarota--Biroli / Kob--Berthier pinning geometry whose interventional trends
were reported by Nagamanasa *et al.* (Nat. Phys. 2015) and Gokhale *et al.*
(Nat. Commun. 2014).

Mechanism (why this reuses the engine ``active_mask`` verbatim)
--------------------------------------------------------------
The engine already carries the entire pinning machinery in its ``active_mask``:

* :class:`butterfly_cone.engine.integrate.MDIntegrator` zeroes the displacement of every
  inactive particle each step, so a frozen particle is held *bitwise*-static in
  positions, unwrapped positions and velocities.
* :func:`butterfly_cone.engine.potential.analytic_potential` drops frozen--frozen pairs
  but keeps every active--frozen interaction in the energy and in the *active*
  force, zeroing only the frozen force.  A mobile particle therefore feels the
  full force of the frozen scaffold around it, while the scaffold never moves.

Random pinning is thus nothing more than "set ``active_mask`` so a fraction
``f_p`` is frozen".  We never touch the integrator or the potential.

Dimensionless observables emitted for the published-data comparison
-------------------------------------------------------------------
For a grid of pinned fractions ``f_p`` the postdiction emits, all made
dimensionless for a trend-only comparison with the published (arbitrary-unit)
curves:

* ``tau_alpha_ratio = tau_alpha(f_p) / tau_alpha(0)`` -- the relaxation slowdown.
  Frozen prediction: **increasing** with ``f_p``.
* ``xi_pts_over_sigma`` and ``xi_pts_ratio`` -- a point-to-set-like static
  overlap length in units of the mean diameter and relative to its smallest-f_p
  value.  Frozen prediction: **increasing** (pinning drives the static
  point-to-set length up, toward the ideal-glass limit).
* ``xi_dyn_ratio = xi_dyn(f_p) / xi_dyn(f_p_min)`` -- a dynamic correlation
  length proxy from the four-point susceptibility peak, ``sqrt(chi4_peak)``.
  Frozen prediction: **non-monotonic** (rise then fall), the signature result.
* ``q_pin`` -- the long-lag mobile self-overlap (a static overlap in ``[0, 1]``).
  Frozen prediction: **increasing**.

Also emitted, but **only as a reported diagnostic** (NOT a frozen prediction and
NOT compared like-for-like to any published curve):

* ``absolute_mobile_rearrangement_fraction`` -- the absolute fraction of mobile
  particles that undergo a sustained cage-relative rearrangement (reuses
  ``events``).  This is an *absolute* mobile-rearrangement fraction; the
  published Nagamanasa/Gokhale facilitation trend is a *conditional* facilitated
  fraction that grows with pin fraction, so the two are not the same estimand.
  We therefore report this quantity for context only and deliberately do not
  freeze it or score it against the published facilitation trend.

The predicted qualitative trends are *frozen* (hashed) by
:func:`freeze_predictions` before any measured curve is inspected; the driver
script prints the digest, then reports measured trends and deviations.

Determinism
-----------
Every stochastic step -- the pin selection, the mobile Maxwell velocities and
the thermostat -- draws on an explicit CPU generator seeded with domain
separation from the caller seed, following the ButterflyCone device protocol (draw on
CPU, cast once to the target device).  A given ``(seed, device)`` therefore
reproduces the pin mask bitwise and the measured scalar curves exactly.  The
analysis reductions are deterministic NumPy.  Cross-device float equality is
not claimed; per-device repeatability is.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from butterfly_cone.engine.integrate import (
    BussiThermostat,
    MDIntegrator,
    _normal_draw,
    maxwell_boltzmann_velocities,
)
from butterfly_cone.engine.system import ParticleSystem, make_generator
from butterfly_cone.events.displacements import cage_relative_field, magnitude_field
from butterfly_cone.events.trajectory import Trajectory
from butterfly_cone.rcce.diagnostics import cell_occupancy_overlap

__all__ = [
    "PinningProtocol",
    "PinnedTrajectory",
    "SpatialDecay",
    "OneOverECrossing",
    "PinningPoint",
    "PinningResponse",
    "DECLARED_PREDICTIONS",
    "random_pin",
    "pin_system",
    "simulate_pinned",
    "self_scattering_overlap",
    "pinning_overlap_length",
    "facilitation_fraction",
    "classify_trend",
    "freeze_predictions",
    "pinning_response",
]

# torch accepts a 64-bit manual seed; harness seeds may be larger.  Modular
# projection keeps deterministic domain separation (matches rcce.sampler and
# instruments.momentum).
_SEED_MODULUS = 2**63 - 1

# The frozen qualitative predictions for the dimensionless observables, frozen
# before any measured curve is inspected.  See the module docstring.
DECLARED_PREDICTIONS: dict[str, str] = {
    "tau_alpha_ratio": "increasing",
    "xi_pts_ratio": "increasing",
    "xi_dyn_ratio": "non-monotonic",
    "q_pin": "increasing",
}


def _derive_seed(seed: int, tag: str) -> int:
    """Domain-separated child seed from a base seed and a string tag."""

    digest = hashlib.sha256(f"{tag}:{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


# ---------------------------------------------------------------------------
# The random-pinning operator (cavity-free)
# ---------------------------------------------------------------------------


def random_pin(system: ParticleSystem, f_p: float, seed: int) -> torch.Tensor:
    """Return an ``active_mask`` freezing a random fraction ``f_p`` of particles.

    Exactly ``round(f_p * n_candidates)`` of the currently-active ("candidate")
    particles are frozen (``active=False``); the remaining ``(1 - f_p)`` stay
    mobile (``active=True``).  Freezing only ever removes particles from the
    active set, so the operator composes with any pre-existing mask without
    resurrecting a frozen particle; for the cavity-free default (all particles
    active) this pins ``f_p`` of ``N``.

    The frozen set is chosen by a random permutation drawn on a CPU generator
    seeded from ``seed`` and cast once to ``system.device`` -- bitwise
    reproducible per device.
    """

    if not (0.0 <= f_p <= 1.0):
        raise ValueError("f_p must lie in [0, 1]")
    device = system.device
    active = system.active_mask.detach().to(device=device, dtype=torch.bool)
    candidate_indices = torch.nonzero(active.cpu(), as_tuple=False).flatten()
    n_candidates = int(candidate_indices.numel())
    n_freeze = int(round(float(f_p) * n_candidates))
    n_freeze = max(0, min(n_candidates, n_freeze))

    mask = active.clone()
    if n_freeze > 0:
        generator = make_generator(_derive_seed(seed, "random_pin"))
        permutation = torch.randperm(n_candidates, generator=generator, device="cpu")
        frozen_local = candidate_indices[permutation[:n_freeze]]
        frozen = frozen_local.to(device=device)
        mask[frozen] = False
    return mask


def pin_system(system: ParticleSystem, f_p: float, seed: int) -> ParticleSystem:
    """Clone ``system`` with a random-pinning ``active_mask`` of fraction ``f_p``.

    Positions, velocities, diameters, box and unwrapped positions are copied
    verbatim; only ``active_mask`` changes.  The frozen particles are held at
    their (equilibrium) input positions.
    """

    pinned = system.clone()
    pinned.active_mask = random_pin(system, f_p, seed)
    return pinned


# ---------------------------------------------------------------------------
# Cheap deterministic NVT trajectory of a pinned configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinningProtocol:
    """Frozen numerical controls for one pinned-trajectory measurement."""

    dt: float = 0.01
    thermostat_tau: float = 0.5
    production_time: float = 30.0
    sample_interval: float = 0.5
    origin_interval: float = 2.0
    n_lags: int = 40
    wave_number: float = 7.1
    overlap_cutoff: float = 0.3
    event_threshold: float = 0.3
    first_shell_factor: float = 1.4
    cage_scale: float = 0.3

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.thermostat_tau <= 0.0:
            raise ValueError("dt and thermostat_tau must be positive")
        if self.production_time <= 0.0 or self.sample_interval <= 0.0:
            raise ValueError("production_time and sample_interval must be positive")
        if self.n_lags < 3:
            raise ValueError("n_lags must be at least three")
        if self.wave_number <= 0.0 or self.overlap_cutoff <= 0.0:
            raise ValueError("wave_number and overlap_cutoff must be positive")


@dataclass(frozen=True)
class PinnedTrajectory:
    """Sampled unwrapped-position frames of an evolved pinned configuration."""

    times: np.ndarray  # (T,)
    frames: np.ndarray  # (T, N, 3) unwrapped positions
    active_mask: np.ndarray  # (N,) bool, True = mobile
    diameters: np.ndarray  # (N,)
    box: np.ndarray  # (3,)
    f_p: float
    n_pinned: int
    sample_interval: float

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.frames.shape[1])

    @property
    def mobile_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)

    @property
    def pinned_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.active_mask)


def _cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


class _MobileScaffoldThermostat(BussiThermostat):
    """Bussi thermostat for a mobile set that exchanges momentum with a frozen scaffold.

    The unpinned :class:`~butterfly_cone.engine.integrate.BussiThermostat` removes the
    centre-of-mass mode (``ndof = 3 n_active - 3``) because an isolated, freely
    drifting system conserves total momentum.  A pinned mobile set is *not*
    isolated: every step it trades momentum with the immovable frozen scaffold
    through the active--frozen forces, so there is no conserved COM mode and no
    constraint to subtract.  The thermal ndof is therefore the full
    ``ndof = 3 * n_mobile``.  This matches the ``remove_com=False`` Maxwell draw
    used to seed the mobile velocities.  Only :meth:`apply` changes; the numerics
    are otherwise identical to the base thermostat (same CPU normal draw, same
    canonical rescaling), preserving per-device bitwise determinism.
    """

    def apply(self, system: ParticleSystem, dt: float) -> float:
        active = system.active_mask
        n_active = int(active.sum().item())
        # Mobile set exchanges momentum with the frozen scaffold: no COM removal.
        ndof = 3 * n_active
        if ndof <= 0:
            return 1.0
        kinetic_before = 0.5 * system.velocities[active].square().sum()
        if float(kinetic_before) <= 0.0:
            raise ValueError("Bussi rescaling requires nonzero kinetic energy")
        randoms = _normal_draw((ndof,), self.generator).to(system.device, system.dtype)
        gaussian = randoms[0]
        chi_square = randoms[1:].square().sum()
        c = math.exp(-float(dt) / self.tau)
        target_kinetic = 0.5 * ndof * self.temperature
        ratio = torch.as_tensor(target_kinetic, device=system.device, dtype=system.dtype) / kinetic_before
        alpha_squared = (
            c
            + (1.0 - c) * ratio * (chi_square + gaussian.square()) / ndof
            + 2.0 * gaussian * torch.sqrt(c * (1.0 - c) * ratio / ndof)
        )
        alpha = torch.sqrt(torch.clamp(alpha_squared, min=0.0))
        sign_threshold = gaussian + torch.sqrt(
            torch.as_tensor(c / (1.0 - c) * ndof, device=system.device, dtype=system.dtype) / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        system.velocities = torch.where(active[:, None], system.velocities * alpha, system.velocities)
        kinetic_after = 0.5 * system.velocities[active].square().sum()
        self.last_alpha = float(alpha)
        self.heat += float(kinetic_after - kinetic_before)
        return self.last_alpha


def simulate_pinned(
    system: ParticleSystem,
    *,
    temperature: float,
    protocol: PinningProtocol,
    seed: int,
) -> PinnedTrajectory:
    """Evolve a pinned configuration under Bussi-NVT and retain sampled frames.

    ``system`` is cloned (never mutated).  Mobile Maxwell velocities and the
    thermostat are seeded with domain separation from ``seed``.  Frozen
    particles never move.  When no particle is mobile the trajectory is the
    static configuration repeated.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    evolving = system.clone()
    device, dtype = evolving.device, evolving.dtype
    n_particles = evolving.n_particles

    n_mobile = int(evolving.active_mask.sum().item())
    # The mobile set exchanges momentum with the frozen scaffold, so its
    # centre-of-mass mode is not conserved: draw the full 3*n_mobile Maxwell
    # velocities without COM removal (matching the ndof of the thermostat below).
    evolving.velocities = maxwell_boltzmann_velocities(
        n_particles,
        float(temperature),
        make_generator(_derive_seed(seed, "pin_velocity")),
        device=device,
        dtype=dtype,
        active_mask=evolving.active_mask,
        remove_com=False,
    )

    sample_steps = max(1, int(round(protocol.sample_interval / protocol.dt)))
    total_steps = max(sample_steps, int(round(protocol.production_time / protocol.dt)))
    n_frames = total_steps // sample_steps + 1

    integrator: MDIntegrator | None = None
    if n_mobile > 0:
        # The Bussi thermostat needs a nonzero mobile kinetic energy and a
        # meaningful thermal ndof; with a single mobile particle (ndof would be
        # ill-posed and its velocity can be zero under an isolated cage) we
        # evolve NVE instead of risking a divide-by-zero rescale.  For >= 2
        # mobile particles we use the scaffold-coupled thermostat whose ndof is
        # the full 3*n_mobile (no COM subtraction), because the mobile set trades
        # momentum with the frozen scaffold and has no conserved COM mode.
        thermostat = (
            _MobileScaffoldThermostat(
                temperature=float(temperature),
                tau=protocol.thermostat_tau,
                generator=make_generator(_derive_seed(seed, "pin_thermostat")),
            )
            if n_mobile >= 2
            else None
        )
        integrator = MDIntegrator(evolving, dt=protocol.dt, thermostat=thermostat)

    frames = np.empty((n_frames, n_particles, 3), dtype=np.float64)
    for frame in range(n_frames):
        frames[frame] = _cpu(evolving.unwrapped_positions).astype(np.float64, copy=False)
        if integrator is not None and frame + 1 < n_frames:
            integrator.step(sample_steps)

    n_pinned = n_particles - n_mobile
    f_p_actual = n_pinned / n_particles if n_particles else 0.0
    return PinnedTrajectory(
        times=np.arange(n_frames, dtype=float) * float(protocol.sample_interval),
        frames=frames,
        active_mask=_cpu(evolving.active_mask).astype(bool),
        diameters=_cpu(evolving.diameters).astype(np.float64),
        box=_cpu(evolving.box).astype(np.float64),
        f_p=float(f_p_actual),
        n_pinned=int(n_pinned),
        sample_interval=float(protocol.sample_interval),
    )


# ---------------------------------------------------------------------------
# One-over-e crossings (in time for tau_alpha, in space for xi_PTS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OneOverECrossing:
    """A 1/e crossing of a decaying curve, or an honest finite-range bound."""

    value: float | None
    crossed: bool
    lower_bound: float | None

    @property
    def representative(self) -> float | None:
        """Best single scalar: the crossing if found, else the range bound."""

        return self.value if self.crossed else self.lower_bound


def _one_over_e(axis: np.ndarray, curve: np.ndarray, *, threshold: float) -> OneOverECrossing:
    """First crossing of ``curve`` below ``threshold`` by linear interpolation.

    ``axis`` is strictly increasing and non-negative; ``curve`` is expected to
    start near one and decay.  Interpolation is linear in log-axis when both
    endpoints of the bracketing interval are positive (matches the pilot's
    tau_alpha convention), else linear in the axis.
    """

    axis = np.asarray(axis, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if axis.ndim != 1 or curve.shape != axis.shape or axis.size == 0:
        raise ValueError("axis and curve must be equal-length 1D arrays")
    if curve[0] <= threshold:
        return OneOverECrossing(float(axis[0]), True, None)
    for index in range(1, curve.size):
        before, after = curve[index - 1], curve[index]
        if after <= threshold <= before:
            weight = (before - threshold) / max(before - after, np.finfo(float).eps)
            left, right = axis[index - 1], axis[index]
            if left > 0.0 and right > 0.0:
                crossing = float(np.exp(np.log(left) + weight * (np.log(right) - np.log(left))))
            else:
                crossing = float(left + weight * (right - left))
            return OneOverECrossing(crossing, True, None)
    return OneOverECrossing(None, False, float(axis[-1]))


# ---------------------------------------------------------------------------
# Self dynamics of the mobile particles (F_s, overlap Q, chi4) + tau_alpha
# ---------------------------------------------------------------------------


def _lag_indices(n_frames: int, n_lags: int) -> np.ndarray:
    if n_frames < 2:
        return np.array([0], dtype=np.int64)
    positive = np.unique(np.rint(np.geomspace(1.0, float(n_frames - 1), max(1, n_lags - 1))).astype(np.int64))
    return np.unique(np.concatenate((np.array([0], dtype=np.int64), positive)))


@dataclass(frozen=True)
class SelfDynamics:
    """Mobile-particle self dynamics with time-origin averaging."""

    lag_times: np.ndarray
    fs: np.ndarray
    overlap: np.ndarray
    chi4: np.ndarray
    msd: np.ndarray
    tau_alpha: OneOverECrossing
    q_pin: float
    per_particle_overlap: np.ndarray  # (n_mobile,) long-lag self-overlap
    long_lag_time: float


def self_scattering_overlap(
    trajectory: PinnedTrajectory,
    *,
    wave_number: float = 7.1,
    overlap_cutoff: float = 0.3,
    n_lags: int = 40,
    origin_interval: float = 2.0,
) -> SelfDynamics:
    """Time-origin-averaged ``F_s``, overlap ``Q``, ``chi4`` of mobile particles.

    ``tau_alpha`` is the ``F_s = 1/e`` crossing.  ``q_pin`` is the mobile
    self-overlap at a long lag (half the trajectory), a static overlap proxy,
    and ``per_particle_overlap`` is its per-mobile-particle field (used for the
    point-to-set length).
    """

    frames = trajectory.frames
    mobile = trajectory.mobile_indices
    n_frames = trajectory.n_frames
    times = trajectory.times
    if mobile.size == 0 or n_frames < 2:
        empty = np.zeros(1, dtype=float)
        return SelfDynamics(
            lag_times=np.zeros(1),
            fs=np.ones(1),
            overlap=np.ones(1),
            chi4=np.zeros(1),
            msd=np.zeros(1),
            tau_alpha=OneOverECrossing(None, False, 0.0),
            q_pin=1.0,
            per_particle_overlap=np.ones(max(mobile.size, 1)),
            long_lag_time=0.0,
        )

    positions = frames[:, mobile, :]  # (T, m, 3)
    n_mobile = mobile.size
    lag_indices = _lag_indices(n_frames, n_lags)
    origin_stride = max(1, int(round(origin_interval / trajectory.sample_interval)))
    origins_all = np.arange(0, n_frames, origin_stride, dtype=np.int64)

    cutoff_squared = overlap_cutoff * overlap_cutoff
    fs = np.empty(lag_indices.size, dtype=float)
    overlap = np.empty(lag_indices.size, dtype=float)
    chi4 = np.zeros(lag_indices.size, dtype=float)
    msd = np.empty(lag_indices.size, dtype=float)
    for out_index, lag in enumerate(lag_indices):
        origins = origins_all[origins_all + lag < n_frames]
        if origins.size == 0:
            origins = np.array([0], dtype=np.int64)
        displacement = positions[origins + lag] - positions[origins]  # (o, m, 3)
        squared = np.einsum("omk,omk->om", displacement, displacement)
        fs[out_index] = float(np.cos(wave_number * displacement).mean())
        msd[out_index] = float(squared.mean())
        per_origin_q = (squared < cutoff_squared).mean(axis=1)  # (o,)
        overlap[out_index] = float(per_origin_q.mean())
        if per_origin_q.size > 1:
            chi4[out_index] = float(n_mobile * per_origin_q.var(ddof=1))

    lag_times = times[lag_indices] - times[0]
    tau_alpha = _one_over_e(lag_times, fs, threshold=float(np.exp(-1.0)))

    long_lag = max(1, n_frames // 2)
    long_origins = origins_all[origins_all + long_lag < n_frames]
    if long_origins.size == 0:
        long_origins = np.array([0], dtype=np.int64)
    long_disp = positions[long_origins + long_lag] - positions[long_origins]
    long_sq = np.einsum("omk,omk->om", long_disp, long_disp)
    per_particle = (long_sq < cutoff_squared).mean(axis=0)  # (m,)
    q_pin = float(per_particle.mean())

    return SelfDynamics(
        lag_times=lag_times,
        fs=fs,
        overlap=overlap,
        chi4=chi4,
        msd=msd,
        tau_alpha=tau_alpha,
        q_pin=q_pin,
        per_particle_overlap=per_particle,
        long_lag_time=float(times[long_lag] - times[0]),
    )


# ---------------------------------------------------------------------------
# Point-to-set-like static overlap length (xi_PTS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialDecay:
    """Point-to-set overlap profile and its extracted decay length."""

    bin_centers: np.ndarray
    excess_profile: np.ndarray  # q(r) - q_bulk
    xi_pts: OneOverECrossing
    q_bulk: float
    q_near: float


def pinning_overlap_length(
    trajectory: PinnedTrajectory,
    per_particle_overlap: np.ndarray,
    *,
    n_bins: int = 8,
) -> SpatialDecay | None:
    """Extract a point-to-set-like length from the overlap-vs-pin-distance profile.

    Each mobile particle's long-lag self-overlap is binned by its minimum-image
    distance to the *nearest pinned particle* (the frozen scaffold).  Near the
    scaffold the overlap is high (the cage is anchored); far away it decays to
    the bulk value.  The point-to-set length is the ``1/e`` spatial decay of the
    excess profile ``q(r) - q_bulk``, normalised to its near-scaffold value.

    Returns ``None`` when there is no scaffold (``f_p = 0``) or too few mobile
    particles to form a profile.
    """

    mobile = trajectory.mobile_indices
    pinned = trajectory.pinned_indices
    if pinned.size == 0 or mobile.size < 4:
        return None
    box = trajectory.box
    initial = trajectory.frames[0]
    mobile_pos = initial[mobile]  # (m, 3)
    pinned_pos = initial[pinned]  # (p, 3)

    # Minimum-image distance from each mobile particle to the nearest pin.
    delta = mobile_pos[:, None, :] - pinned_pos[None, :, :]
    delta = delta - box * np.rint(delta / box)
    nearest = np.sqrt(np.einsum("mpk,mpk->mp", delta, delta).min(axis=1))  # (m,)

    order = np.argsort(nearest)
    nearest_sorted = nearest[order]
    q_sorted = np.asarray(per_particle_overlap, dtype=float)[order]

    # Equal-count bins keep every bin populated even when the pin geometry is
    # highly non-uniform.
    n_bins = max(3, min(n_bins, mobile.size // 2))
    edges = np.linspace(0, mobile.size, n_bins + 1).astype(int)
    centers = np.empty(n_bins, dtype=float)
    means = np.empty(n_bins, dtype=float)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            centers[b] = nearest_sorted[min(lo, mobile.size - 1)]
            means[b] = q_sorted[min(lo, mobile.size - 1)]
            continue
        centers[b] = float(nearest_sorted[lo:hi].mean())
        means[b] = float(q_sorted[lo:hi].mean())

    q_bulk = float(means[-1])
    q_near = float(means[0])
    excess = means - q_bulk
    denominator = excess[0]
    if denominator <= 0.0:
        # No positive near-scaffold excess: the profile carries no PTS signal.
        xi = OneOverECrossing(None, False, float(centers[-1] - centers[0]))
    else:
        normalized = excess / denominator
        # Measure decay length from the near edge outward.
        axis = centers - centers[0]
        xi = _one_over_e(axis, normalized, threshold=float(np.exp(-1.0)))
    return SpatialDecay(
        bin_centers=centers,
        excess_profile=excess,
        xi_pts=xi,
        q_bulk=q_bulk,
        q_near=q_near,
    )


# ---------------------------------------------------------------------------
# Facilitation proxy (reuses events/): sustained cage-relative rearrangements
# ---------------------------------------------------------------------------


def facilitation_fraction(
    trajectory: PinnedTrajectory,
    *,
    threshold: float = 0.3,
    first_shell_factor: float = 1.4,
) -> float:
    """Fraction of mobile particles that undergo a cage-relative rearrangement.

    Reuses :func:`butterfly_cone.events.displacements.cage_relative_field` over *all*
    particles (frozen neighbours are a genuine part of each mobile particle's
    cage), then counts a mobile particle as a facilitation event when its
    peak cage-relative displacement magnitude over the trajectory exceeds
    ``threshold``.  Pinning suppresses these events, so this proxy is expected
    to fall with ``f_p``.
    """

    mobile = trajectory.mobile_indices
    if mobile.size == 0 or trajectory.n_frames < 2:
        return 0.0
    from butterfly_cone.events.config import DisplacementConfig

    traj = Trajectory(
        unwrapped_positions=trajectory.frames,
        times=trajectory.times,
        sigma=trajectory.diameters,
        box=trajectory.box,
    )
    config = DisplacementConfig(first_shell_factor=first_shell_factor, isolated_fallback="zero")
    field = cage_relative_field(traj, config, reference_frame=0)  # (T, N, 3)
    magnitude = magnitude_field(field)  # (T, N)
    peak = magnitude.max(axis=0)  # (N,)
    events = peak[mobile] > float(threshold)
    return float(events.mean())


# ---------------------------------------------------------------------------
# Trend classification and frozen predictions
# ---------------------------------------------------------------------------


def classify_trend(values: Sequence[float], *, rel_tol: float = 0.08) -> str:
    """Classify a curve as increasing / decreasing / non-monotonic / flat.

    An interior extremum whose flanks both exceed ``rel_tol`` of the value range
    is reported as ``"non-monotonic"`` (the rise-then-fall / dip signature).
    Otherwise the curve is ``"increasing"`` / ``"decreasing"`` when monotone
    within tolerance with a net change beyond ``rel_tol`` of the range, and
    ``"flat"`` when the net change is within tolerance.  Non-finite entries are
    dropped (keeping order); fewer than two finite values give ``"undetermined"``.
    """

    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return "undetermined"
    value_range = float(arr.max() - arr.min())
    scale = value_range if value_range > 0.0 else max(float(np.abs(arr).max()), 1e-12)
    band = rel_tol * scale
    n = arr.size
    imax = int(np.argmax(arr))
    imin = int(np.argmin(arr))
    peak_interior = 0 < imax < n - 1 and (arr[imax] - arr[0]) > band and (arr[imax] - arr[-1]) > band
    valley_interior = 0 < imin < n - 1 and (arr[0] - arr[imin]) > band and (arr[-1] - arr[imin]) > band
    if peak_interior or valley_interior:
        return "non-monotonic"
    diffs = np.diff(arr)
    net = float(arr[-1] - arr[0])
    non_decreasing = bool(np.all(diffs >= -band))
    non_increasing = bool(np.all(diffs <= band))
    if abs(net) <= band:
        return "flat"
    if non_decreasing and net > band:
        return "increasing"
    if non_increasing and net < -band:
        return "decreasing"
    return "non-monotonic"


def freeze_predictions(predictions: Mapping[str, str] = DECLARED_PREDICTIONS) -> dict[str, object]:
    """Return the frozen predictions and a stable SHA-256 freeze over them.

    The freeze is a canonical-JSON hash: recording it before any measured curve is
    inspected is the postdiction's "predict, then compare" receipt.
    """

    canonical = json.dumps(dict(sorted(predictions.items())), separators=(",", ":"), sort_keys=True)
    freeze = hashlib.sha256(canonical.encode()).hexdigest()
    return {"predictions": dict(predictions), "canonical": canonical, "digest": freeze}


# ---------------------------------------------------------------------------
# Per-f_p measurement point and the full pinning response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinningPoint:
    """All observables measured at a single pinned fraction ``f_p``."""

    f_p: float
    n_pinned: int
    n_mobile: int
    tau_alpha: float | None
    tau_alpha_crossed: bool
    tau_alpha_representative: float | None
    q_pin: float
    xi_pts: float | None
    xi_pts_crossed: bool
    xi_pts_representative: float | None
    chi4_peak: float | None
    xi_dyn: float | None
    absolute_mobile_rearrangement_fraction: float
    scaffold_overlap: float
    fs_final: float
    msd_final: float

    def to_dict(self) -> dict[str, object]:
        return {
            "f_p": self.f_p,
            "n_pinned": self.n_pinned,
            "n_mobile": self.n_mobile,
            "tau_alpha": self.tau_alpha,
            "tau_alpha_crossed": self.tau_alpha_crossed,
            "tau_alpha_representative": self.tau_alpha_representative,
            "q_pin": self.q_pin,
            "xi_pts": self.xi_pts,
            "xi_pts_crossed": self.xi_pts_crossed,
            "xi_pts_representative": self.xi_pts_representative,
            "chi4_peak": self.chi4_peak,
            "xi_dyn": self.xi_dyn,
            "absolute_mobile_rearrangement_fraction": self.absolute_mobile_rearrangement_fraction,
            "scaffold_overlap": self.scaffold_overlap,
            "fs_final": self.fs_final,
            "msd_final": self.msd_final,
        }


def _measure_point(
    trajectory: PinnedTrajectory,
    protocol: PinningProtocol,
) -> PinningPoint:
    dynamics = self_scattering_overlap(
        trajectory,
        wave_number=protocol.wave_number,
        overlap_cutoff=protocol.overlap_cutoff,
        n_lags=protocol.n_lags,
        origin_interval=protocol.origin_interval,
    )
    decay = pinning_overlap_length(trajectory, dynamics.per_particle_overlap)
    if decay is None:
        xi_pts, xi_crossed, xi_repr = None, False, None
    else:
        xi_pts = decay.xi_pts.value
        xi_crossed = decay.xi_pts.crossed
        xi_repr = decay.xi_pts.representative

    finite_chi4 = dynamics.chi4[np.isfinite(dynamics.chi4)]
    # Ignore the trivial zero-lag value when locating the peak.
    peak_source = dynamics.chi4[1:] if dynamics.chi4.size > 1 else dynamics.chi4
    peak_source = peak_source[np.isfinite(peak_source)]
    chi4_peak = float(peak_source.max()) if peak_source.size and finite_chi4.size else None
    xi_dyn = float(np.sqrt(max(chi4_peak, 0.0))) if chi4_peak is not None else None

    facilitation = facilitation_fraction(
        trajectory,
        threshold=protocol.event_threshold,
        first_shell_factor=protocol.first_shell_factor,
    )

    # Identity-free scaffold overlap (reuses rcce.diagnostics), restricted to the
    # MOBILE sites: the fraction of initial *mobile* sites still occupied by any
    # particle at the final frame.  Frozen sites are excluded because a frozen
    # particle never moves, so its site is trivially self-occupied and would make
    # an all-site overlap a tautology; restricting to mobile sites measures genuine
    # mobile caging.
    mobile = trajectory.mobile_indices
    scaffold_overlap = cell_occupancy_overlap(
        torch.as_tensor(trajectory.frames[-1][mobile] % trajectory.box, dtype=torch.float64),
        torch.as_tensor(trajectory.frames[0][mobile] % trajectory.box, dtype=torch.float64),
        torch.as_tensor(trajectory.box, dtype=torch.float64),
        cage_scale=protocol.cage_scale,
    )

    tau = dynamics.tau_alpha
    return PinningPoint(
        f_p=trajectory.f_p,
        n_pinned=trajectory.n_pinned,
        n_mobile=int(trajectory.mobile_indices.size),
        tau_alpha=tau.value,
        tau_alpha_crossed=tau.crossed,
        tau_alpha_representative=tau.representative,
        q_pin=dynamics.q_pin,
        xi_pts=xi_pts,
        xi_pts_crossed=xi_crossed,
        xi_pts_representative=xi_repr,
        chi4_peak=chi4_peak,
        xi_dyn=xi_dyn,
        absolute_mobile_rearrangement_fraction=facilitation,
        scaffold_overlap=scaffold_overlap,
        fs_final=float(dynamics.fs[-1]),
        msd_final=float(dynamics.msd[-1]),
    )


@dataclass(frozen=True)
class PinningResponse:
    """Pinning response over an ``f_p`` grid plus its dimensionless curves."""

    f_p_grid: tuple[float, ...]
    points: tuple[PinningPoint, ...]
    temperature: float
    sigma_mean: float

    def _first_defined(self, values: Sequence[float | None]) -> float | None:
        for value in values:
            if value is not None and np.isfinite(value) and value > 0.0:
                return float(value)
        return None

    def dimensionless(self) -> dict[str, list[float | None]]:
        """The dimensionless observables emitted for the published comparison.

        All curves are aligned to :attr:`f_p_grid`.  ``tau_alpha_ratio`` and the
        ``xi`` ratios are normalised to their smallest-``f_p`` defined value;
        ``xi_pts_over_sigma`` is in mean-diameter units; ``q_pin`` is already
        dimensionless in ``[0, 1]``.  ``absolute_mobile_rearrangement_fraction``
        is emitted as a reported diagnostic only (it is not a frozen prediction
        and is not compared to any published curve).
        """

        tau = [point.tau_alpha_representative for point in self.points]
        xi_pts = [point.xi_pts_representative for point in self.points]
        xi_dyn = [point.xi_dyn for point in self.points]

        tau_ref = self._first_defined(tau)
        xi_pts_ref = self._first_defined(xi_pts)
        xi_dyn_ref = self._first_defined(xi_dyn)

        def ratio(values: Sequence[float | None], reference: float | None) -> list[float | None]:
            if reference is None:
                return [None for _ in values]
            return [
                (float(v) / reference) if (v is not None and np.isfinite(v)) else None
                for v in values
            ]

        sigma = self.sigma_mean if self.sigma_mean > 0.0 else 1.0
        return {
            "f_p": [float(f) for f in self.f_p_grid],
            "tau_alpha_ratio": ratio(tau, tau_ref),
            "xi_pts_ratio": ratio(xi_pts, xi_pts_ref),
            "xi_pts_over_sigma": [
                (float(v) / sigma) if (v is not None and np.isfinite(v)) else None for v in xi_pts
            ],
            "xi_dyn_ratio": ratio(xi_dyn, xi_dyn_ref),
            "q_pin": [float(point.q_pin) for point in self.points],
            "absolute_mobile_rearrangement_fraction": [
                float(point.absolute_mobile_rearrangement_fraction) for point in self.points
            ],
        }

    def trends(self) -> dict[str, str]:
        """Classify the measured trend of every frozen dimensionless observable."""

        curves = self.dimensionless()
        return {name: classify_trend(curves[name]) for name in DECLARED_PREDICTIONS}

    def deviations(self) -> dict[str, dict[str, str]]:
        """Per-observable frozen-vs-measured trend, and whether they match."""

        measured = self.trends()
        report: dict[str, dict[str, str]] = {}
        for name, predicted in DECLARED_PREDICTIONS.items():
            got = measured.get(name, "undetermined")
            report[name] = {
                "predicted": predicted,
                "measured": got,
                "match": "yes" if got == predicted else "no",
            }
        return report

    def to_dict(self) -> dict[str, object]:
        return {
            "f_p_grid": list(self.f_p_grid),
            "temperature": self.temperature,
            "sigma_mean": self.sigma_mean,
            "points": [point.to_dict() for point in self.points],
            "dimensionless": self.dimensionless(),
            "trends": self.trends(),
            "deviations": self.deviations(),
        }


def pinning_response(
    system: ParticleSystem,
    f_p_grid: Sequence[float],
    *,
    temperature: float,
    seed: int,
    protocol: PinningProtocol | None = None,
) -> PinningResponse:
    """Measure the pinning response of ``system`` over a grid of pinned fractions.

    For every ``f_p`` a random-pinning mask is applied, the mobile particles are
    evolved under Bussi-NVT, and the dimensionless observables of the module
    docstring are measured.  All randomness is domain-separated from ``seed``
    and the ``f_p`` value, so the whole curve is reproducible per device.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    grid = tuple(float(f) for f in f_p_grid)
    if not grid:
        raise ValueError("f_p_grid must be non-empty")
    protocol = protocol or PinningProtocol()
    sigma_mean = float(system.diameters.detach().cpu().to(torch.float64).mean())

    points: list[PinningPoint] = []
    for f_p in grid:
        pinned = pin_system(system, f_p, _derive_seed(seed, f"pin_mask_{f_p:.6f}"))
        trajectory = simulate_pinned(
            pinned,
            temperature=temperature,
            protocol=protocol,
            seed=_derive_seed(seed, f"trajectory_{f_p:.6f}"),
        )
        points.append(_measure_point(trajectory, protocol))

    return PinningResponse(
        f_p_grid=grid,
        points=tuple(points),
        temperature=float(temperature),
        sigma_mean=sigma_mean,
    )
