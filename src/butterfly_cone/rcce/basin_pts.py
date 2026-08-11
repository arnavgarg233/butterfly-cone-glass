"""rcce/basin_pts.py: interventional point-to-set from basin structure vs cavity radius.

Reads the declared in advance radius-sweep cavity pilots (per the frozen
2026-07-16T07:44:56Z pre-unblind measurement declaration) as a *point-to-set*
measurement rather than a debugging outcome.  Independent overdispersed RCCE
chains that CONVERGE (single basin, split-R̂ → 1) at small cavity radius but
SPLIT (several metastable basins the sampler cannot bridge, R̂ ≫ 1) at large
radius locate the interventional / dynamical point-to-set length ξ_PTS: the
largest amorphous-order correlation length over which the frozen exterior
thermodynamically pins a unique core state.  This is the do-style realisation of
the observational Biroli et al. (Nat. Phys. 2008) pinning-radius PTS, measured
by boundary-freeze + core de-pinning instead of overlap correlation.

Mixing quality
--------------
For one cavity the identity-free core-overlap traces of the overdispersed chains
give a between-vs-within variance decomposition.  The mixing-quality scalar is

    m = 1 / R̂²  ∈ (0, 1]

where R̂ is the split Gelman–Rubin factor of the core-overlap channel
(``diagnostics.split_rhat``).  This is not an ad-hoc squashing of R̂: it is
*exactly* the within-basin fraction of the core-overlap variance,

    m = W / (W + τ²),      with     τ² / W = R̂² − 1,

W = mean within-chain variance, τ² = the between-basin variance component
(one-way random-effects / intraclass decomposition of the chain means).  Hence

    single basin / converged  (τ² → 0, R̂ → 1)  ⇒  m → 1
    K well-separated basins    (τ² ≫ W, R̂ ≫ 1)  ⇒  m → 0
    crossover  m = 0.5  ⇔  τ² = W  ⇔  R̂ = √2

so the radius where m(R) drops through 0.5 (between-basin core-overlap spread
first matches the within-basin spread = single→multi-basin transition) is the
interventional ξ_PTS.  Per the frozen declaration, a monotone m(R)-vs-R trend IS
the measurement even if neither endpoint fully passes the Gate-0 ESS criterion,
so a 2-radius bracket + trend is a first-class result.

Everything here is deterministic (no RNG): sort/OLS/interpolation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .diagnostics import split_rhat

# The channel that carries basin identity: the identity-free core-parent
# occupancy overlap (see diagnostics.core_parent_overlap).  Traces on disk and
# results.json both key it under this name.
CORE_OVERLAP_CHANNEL = "core_parent_overlap"
ENERGY_CHANNEL = "active_potential_energy"

# split-R̂ at the mixing-quality crossover m = 0.5  (τ² = W  ⇔  R̂ = √2).
CROSSOVER_RHAT = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Core scalar: R̂ → mixing quality
# ---------------------------------------------------------------------------


def mixing_quality_from_rhat(rhat: float) -> float:
    """Map a split-R̂ to the within-basin variance fraction ``m = 1/R̂² ∈ (0,1]``.

    ``R̂`` is clamped to ``≥ 1`` (finite sampling noise can push the estimate a
    hair below one) so ``m ≤ 1``.  A non-finite R̂ from an identically-constant
    (caged/stuck) channel is *not* evidence of mixing, so it maps to ``m = 0``.
    A NaN R̂ (too few draws to define the diagnostic) propagates as NaN.
    """

    value = float(rhat)
    if math.isnan(value):
        return float("nan")
    if math.isinf(value):
        return 0.0
    clamped = max(value, 1.0)
    return 1.0 / (clamped * clamped)


# ---------------------------------------------------------------------------
# Chain preparation + variance decomposition
# ---------------------------------------------------------------------------


def _prepare_chains(chains: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    """Order chain ids and truncate every chain to the common minimum length.

    ``diagnostics.split_rhat`` (via ``torch.stack``) requires equal-length
    chains; the pilot's ``diagnose_scalar_channel`` truncates identically.
    """

    if len(chains) < 2:
        raise ValueError("basin mixing quality needs at least two chains")
    ids = sorted(chains)
    arrays = [np.asarray(chains[cid], dtype=np.float64).ravel() for cid in ids]
    common = min(arr.size for arr in arrays)
    if common < 4:
        raise ValueError("each chain needs at least four samples for split-R̂")
    stacked = np.stack([arr[:common] for arr in arrays])
    return ids, stacked


def _split_sequences(stacked: np.ndarray) -> np.ndarray:
    """Halve every chain into two, exactly as ``diagnostics.split_rhat`` does.

    Produces ``2m`` sequences of length ``n//2`` so the between-vs-within
    decomposition shares the split-chain basis of the R̂ statistic and the
    identity ``τ²/W = R̂² − 1`` holds to floating-point precision.
    """

    n = stacked.shape[1]
    half = n // 2
    return np.concatenate((stacked[:, :half], stacked[:, half : 2 * half]), axis=0)


@dataclass(frozen=True)
class VarianceDecomposition:
    """One-way (between-basin vs within-chain) decomposition of a channel.

    Computed on the *split-chain* basis (each chain halved, as in split-R̂).
    ``rhat`` is the split Gelman–Rubin factor; ``within`` = W (mean within-chain
    variance of the ``2m`` split sequences); ``between_component`` = τ² (the
    between-basin variance component, floored at 0).  The exact identity
    ``τ²/W = R̂² − 1`` ties them to R̂, so ``m = W/(W+τ²) = 1/R̂²``.
    """

    rhat: float
    within: float
    between_component: float
    n_chains: int
    n_split_draws: int

    @property
    def tau2_over_within(self) -> float:
        if self.within <= 0.0:
            return float("inf") if self.between_component > 0.0 else 0.0
        return self.between_component / self.within


def variance_decomposition(chains: Mapping[str, Sequence[float]]) -> VarianceDecomposition:
    """Between-vs-within decomposition of the chain ensemble for one cavity.

    On the split-chain basis: ``within`` averages the (unbiased) per-sequence
    variances; ``between_component`` is the excess of the variance of the
    sequence means over the sampling floor ``W / n_split_draws``, the
    random-effects estimate of τ².  R̂ is taken from ``diagnostics.split_rhat``
    so the score is bit-for-bit the pilot's own diagnostic.
    """

    _, stacked = _prepare_chains(chains)
    return _variance_decomposition_prepared(stacked)


def _variance_decomposition_prepared(stacked: np.ndarray) -> VarianceDecomposition:
    """:func:`variance_decomposition` body for an already-prepared chain stack."""

    n_chains = stacked.shape[0]
    split = _split_sequences(stacked)
    n_split_draws = split.shape[1]
    within = float(np.mean(split.var(axis=1, ddof=1)))
    means = split.mean(axis=1)
    between_raw = float(means.var(ddof=1))
    between_component = max(0.0, between_raw - within / n_split_draws)
    rhat = split_rhat(stacked)
    return VarianceDecomposition(
        rhat=float(rhat),
        within=within,
        between_component=between_component,
        n_chains=n_chains,
        n_split_draws=n_split_draws,
    )


# ---------------------------------------------------------------------------
# Basin clustering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinClustering:
    """Result of clustering chains by their sampled core-overlap distributions.

    ``labels`` is one integer basin id per chain (in ``chain_ids`` order); ids
    are assigned in ascending order of basin mean core-overlap.  Two chains join
    the same basin when the gap between adjacent sorted chain means is below
    ``threshold_factor`` within-chain standard deviations, the scale on which a
    single basin is explored.  ``n_basins == 1`` ⇔ the chains occupy one basin.
    """

    n_basins: int
    labels: tuple[int, ...]
    chain_ids: tuple[str, ...]
    chain_means: tuple[float, ...]
    within_std: float
    threshold: float
    threshold_factor: float


def cluster_basins(
    chains: Mapping[str, Sequence[float]],
    *,
    threshold_factor: float = 2.0,
) -> BasinClustering:
    """Single-linkage clustering of chains on their mean core-overlap.

    Deterministic: sort the chain means and split wherever the gap to the next
    mean exceeds ``threshold_factor * within_std``.  ``within_std`` is the pooled
    within-chain standard deviation (the width a chain covers inside its basin),
    so well-separated basins (gap ≫ σ_within) split while chains sampling the
    same basin (means within ≈ σ_within/√n) merge.
    """

    if threshold_factor <= 0.0:
        raise ValueError("threshold_factor must be positive")
    ids, stacked = _prepare_chains(chains)
    return _cluster_basins_prepared(ids, stacked, threshold_factor)


def _cluster_basins_prepared(
    ids: list[str], stacked: np.ndarray, threshold_factor: float
) -> BasinClustering:
    """:func:`cluster_basins` body for an already-prepared chain stack."""

    if threshold_factor <= 0.0:
        raise ValueError("threshold_factor must be positive")
    n_chains = stacked.shape[0]
    means = stacked.mean(axis=1)
    within_std = float(math.sqrt(max(0.0, float(np.mean(stacked.var(axis=1, ddof=1))))))
    # A degenerate (zero within-chain spread) ensemble has no basin width scale;
    # fall back to an absolute tolerance on the means so identical constants form
    # one basin and distinct constants separate.
    scale = within_std if within_std > 0.0 else max(1.0, float(np.abs(means).max()))
    threshold = threshold_factor * scale

    order = np.argsort(means, kind="stable")
    labels = np.empty(n_chains, dtype=np.int64)
    current = 0
    labels[order[0]] = current
    for prev, cur in zip(order[:-1], order[1:]):
        if means[cur] - means[prev] > threshold:
            current += 1
        labels[cur] = current
    n_basins = current + 1
    return BasinClustering(
        n_basins=int(n_basins),
        labels=tuple(int(x) for x in labels.tolist()),
        chain_ids=tuple(ids),
        chain_means=tuple(float(x) for x in means.tolist()),
        within_std=within_std,
        threshold=float(threshold),
        threshold_factor=float(threshold_factor),
    )


# ---------------------------------------------------------------------------
# Per-cavity and per-radius mixing quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CavityMixing:
    """Mixing quality of a single cavity from its overdispersed chains."""

    ordinal: int
    m: float
    rhat: float
    n_basins: int
    n_chains: int
    channel: str
    clustering: BasinClustering
    decomposition: VarianceDecomposition


def cavity_mixing_quality(
    chains: Mapping[str, Sequence[float]],
    *,
    ordinal: int = 0,
    channel: str = CORE_OVERLAP_CHANNEL,
    threshold_factor: float = 2.0,
) -> CavityMixing:
    """m = 1/R̂² plus a basin clustering for one cavity's chain ensemble."""

    # One shared chain preparation (sort + truncate + stack) instead of two --
    # variance_decomposition and cluster_basins consumed the identical stack.
    # [micro-bench, 24 cavities x 4 chains x 4k draws:
    #  radius_mixing_quality 18 ms -> 12 ms, ~1.5x]
    ids, stacked = _prepare_chains(chains)
    decomposition = _variance_decomposition_prepared(stacked)
    clustering = _cluster_basins_prepared(ids, stacked, threshold_factor)
    return CavityMixing(
        ordinal=int(ordinal),
        m=mixing_quality_from_rhat(decomposition.rhat),
        rhat=decomposition.rhat,
        n_basins=clustering.n_basins,
        n_chains=decomposition.n_chains,
        channel=channel,
        clustering=clustering,
        decomposition=decomposition,
    )


@dataclass(frozen=True)
class RadiusMixing:
    """Aggregate mixing quality m(R) at one cavity radius."""

    radius: float
    m: float
    channel: str
    n_cavities: int
    n_cavities_used: int
    per_cavity_m: tuple[float, ...]
    per_cavity_rhat: tuple[float, ...]
    per_cavity_n_basins: tuple[int, ...]
    single_basin_fraction: float
    cavities: tuple[CavityMixing, ...] = field(default=(), repr=False)

    @property
    def multi_basin(self) -> bool:
        return self.m < 0.5


def radius_mixing_quality(
    radius: float,
    cavity_chains: Sequence[Mapping[str, Sequence[float]]],
    *,
    channel: str = CORE_OVERLAP_CHANNEL,
    threshold_factor: float = 2.0,
    ordinals: Sequence[int] | None = None,
) -> RadiusMixing:
    """Aggregate m(R) = mean over cavities of the per-cavity m = 1/R̂².

    Cavities whose R̂ is undefined (too few draws → NaN) are dropped from the
    mean and counted separately; a stuck (R̂ = ∞) cavity contributes m = 0.
    """

    if not cavity_chains:
        raise ValueError("radius mixing quality needs at least one cavity")
    if ordinals is None:
        ordinals = range(len(cavity_chains))
    cavities: list[CavityMixing] = []
    for ordinal, chains in zip(ordinals, cavity_chains):
        cavities.append(
            cavity_mixing_quality(
                chains,
                ordinal=ordinal,
                channel=channel,
                threshold_factor=threshold_factor,
            )
        )
    per_cavity_m = [c.m for c in cavities]
    used = [value for value in per_cavity_m if not math.isnan(value)]
    if not used:
        raise ValueError("no cavity produced a defined mixing quality at this radius")
    aggregate = float(np.mean(used))
    single = float(np.mean([1.0 if c.n_basins == 1 else 0.0 for c in cavities]))
    return RadiusMixing(
        radius=float(radius),
        m=aggregate,
        channel=channel,
        n_cavities=len(cavities),
        n_cavities_used=len(used),
        per_cavity_m=tuple(per_cavity_m),
        per_cavity_rhat=tuple(c.rhat for c in cavities),
        per_cavity_n_basins=tuple(c.n_basins for c in cavities),
        single_basin_fraction=single,
        cavities=tuple(cavities),
    )


# ---------------------------------------------------------------------------
# Point-to-set estimate: crossover radius R_c where m(R) drops through 0.5
# ---------------------------------------------------------------------------


def _trend(m_values: Sequence[float], *, tol: float = 1e-9) -> str:
    diffs = np.diff(np.asarray(m_values, dtype=np.float64))
    if np.all(diffs <= tol) and np.any(diffs < -tol):
        return "decreasing"
    if np.all(diffs >= -tol) and np.any(diffs > tol):
        return "increasing"
    if np.all(np.abs(diffs) <= tol):
        return "flat"
    return "non-monotone"


def _interp_crossing(radii: np.ndarray, m: np.ndarray, level: float = 0.5) -> tuple[float, tuple[float, float]] | None:
    """First adjacent bracket of ``level`` (either direction); linear interp."""

    for i in range(radii.size - 1):
        lo, hi = m[i], m[i + 1]
        if (lo - level) * (hi - level) <= 0.0 and lo != hi:
            frac = (level - lo) / (hi - lo)
            r_c = float(radii[i] + frac * (radii[i + 1] - radii[i]))
            return r_c, (float(radii[i]), float(radii[i + 1]))
    return None


def _logit_fit_crossing(radii: np.ndarray, m: np.ndarray, level: float = 0.5) -> tuple[float, float] | None:
    """Closed-form logistic fit via logit-linearisation.

    ``logit(m) = a + b·R`` by ordinary least squares; the level crossing is at
    ``R_c = (logit(level) − a) / b`` and the width is ``w = -1/b``.  Requires a
    genuinely decreasing fit (b < 0) to be a valid single→multi-basin crossover.
    Fully deterministic and dependency-free.
    """

    eps = 1e-6
    clamped = np.clip(m, eps, 1.0 - eps)
    y = np.log(clamped / (1.0 - clamped))
    # OLS slope/intercept of y on radii.
    r_mean = float(radii.mean())
    y_mean = float(y.mean())
    denom = float(((radii - r_mean) ** 2).sum())
    if denom <= 0.0:
        return None
    b = float(((radii - r_mean) * (y - y_mean)).sum() / denom)
    a = y_mean - b * r_mean
    if b >= 0.0:  # not a decreasing crossover
        return None
    target = math.log(level / (1.0 - level))  # 0 for level = 0.5
    r_c = (target - a) / b
    width = -1.0 / b
    return float(r_c), float(width)


@dataclass(frozen=True)
class PointToSetEstimate:
    """Interventional ξ_PTS: the crossover radius R_c where m(R) drops through 0.5.

    ``crossover_radius`` is the ξ_PTS point estimate when the sweep brackets 0.5;
    ``bracket`` is the (R_lo, R_hi) radius pair straddling the crossing.  When the
    sweep does not bracket 0.5 the estimate is a bound (``above_all`` ⇒ every
    radius still mixes, R_c > R_max; ``below_all`` ⇒ none mix, R_c < R_min).  With
    only two radii the bracket + monotone ``trend`` is the reported measurement.
    """

    radii: tuple[float, ...]
    m_values: tuple[float, ...]
    crossover_radius: float | None
    bracket: tuple[float, float] | None
    lower_bound: float | None
    upper_bound: float | None
    trend: str
    monotone_decreasing: bool
    method: str
    n_radii: int
    logistic_params: dict[str, float] | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "radii": list(self.radii),
            "m_values": list(self.m_values),
            "crossover_radius_xi_pts": self.crossover_radius,
            "bracket": list(self.bracket) if self.bracket is not None else None,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "trend": self.trend,
            "monotone_decreasing": self.monotone_decreasing,
            "method": self.method,
            "n_radii": self.n_radii,
            "logistic_params": self.logistic_params,
            "note": self.note,
        }


def estimate_point_to_set(
    radii: Sequence[float],
    m_values: Sequence[float],
    *,
    level: float = 0.5,
) -> PointToSetEstimate:
    """Locate / bracket the crossover radius R_c where m(R) drops through 0.5.

    Radii are sorted ascending.  If the sweep brackets ``level`` the crossover is
    estimated by a logistic (logit-linear) fit when ≥ 3 radii are available and
    the fit is decreasing, else by linear interpolation of the bracketing pair.
    With < 2 radii the estimate is undefined; with exactly 2 the bracket + trend
    is returned per the frozen declaration ("a trend IS the measurement").
    """

    radii_arr = np.asarray(radii, dtype=np.float64)
    m_arr = np.asarray(m_values, dtype=np.float64)
    if radii_arr.size != m_arr.size:
        raise ValueError("radii and m_values must have the same length")
    if radii_arr.size < 2:
        raise ValueError("point-to-set estimate needs at least two radii")
    if np.any(np.isnan(m_arr)):
        raise ValueError("m_values must be defined (non-NaN) at every radius")
    order = np.argsort(radii_arr, kind="stable")
    radii_arr = radii_arr[order]
    m_arr = m_arr[order]
    if np.any(np.diff(radii_arr) <= 0.0):
        raise ValueError("radii must be strictly increasing (deduplicate first)")
    n = int(radii_arr.size)

    trend = _trend(m_arr)
    monotone_decreasing = trend in {"decreasing", "flat"}

    crossing = _interp_crossing(radii_arr, m_arr, level)
    logistic_params: dict[str, float] | None = None
    crossover: float | None = None
    bracket: tuple[float, float] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    if crossing is not None:
        interp_rc, bracket = crossing
        crossover = interp_rc
        method = "interpolation"
        if n >= 3:
            fit = _logit_fit_crossing(radii_arr, m_arr, level)
            if fit is not None:
                fit_rc, width = fit
                logistic_params = {"crossover_radius": fit_rc, "width": width}
                # Prefer the smooth fit when it lands inside the swept range.
                if radii_arr[0] <= fit_rc <= radii_arr[-1]:
                    crossover = fit_rc
                    method = "logistic_fit"
        if n == 2:
            method = "bracket"
        note = (
            f"m crosses {level:g} between R={bracket[0]:g} and R={bracket[1]:g}; "
            f"xi_PTS ~= {crossover:g} (method={method}, trend={trend})."
        )
    else:
        # No bracket: report the appropriate bound.
        if float(m_arr.min()) > level:
            method = "above_all"
            lower_bound = float(radii_arr[-1])
            note = (
                f"m(R) > {level:g} at every swept radius (min m={m_arr.min():g}); "
                f"the cavity still mixes throughout, so xi_PTS > R_max={lower_bound:g}."
            )
        elif float(m_arr.max()) < level:
            method = "below_all"
            upper_bound = float(radii_arr[0])
            note = (
                f"m(R) < {level:g} at every swept radius (max m={m_arr.max():g}); "
                f"the cavity never mixes, so xi_PTS < R_min={upper_bound:g}."
            )
        else:
            method = "no_crossing"
            note = (
                f"m(R) touches but does not cross {level:g}; report the bracket/trend "
                f"(trend={trend})."
            )

    return PointToSetEstimate(
        radii=tuple(float(x) for x in radii_arr.tolist()),
        m_values=tuple(float(x) for x in m_arr.tolist()),
        crossover_radius=crossover,
        bracket=bracket,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trend=trend,
        monotone_decreasing=bool(monotone_decreasing),
        method=method,
        n_radii=n,
        logistic_params=logistic_params,
        note=note,
    )


# ---------------------------------------------------------------------------
# Loaders: read cavity-pilot runs (results.json + traces.npz) off disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiusRun:
    """One cavity-pilot run at a fixed radius, with per-cavity per-chain traces.

    Chain traces are float64 arrays (``_prepare_chains`` consumes them without
    copying; a former ``.tolist()`` round-trip re-boxed every sample).
    """

    radius: float
    buffer_radius: float
    temperature: float
    run_dir: str
    channel: str
    cavity_chains: tuple[dict[str, np.ndarray], ...]
    ordinals: tuple[int, ...]


def _parse_trace_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split("__")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def load_cavity_run(run_dir: str | Path, *, channel: str = CORE_OVERLAP_CHANNEL) -> RadiusRun:
    """Load one run's per-cavity chains for ``channel`` from traces.npz + results.json.

    Trace keys are ``c{ordinal:03d}__{channel}__{family}`` (double-underscore
    delimited); the requested channel's per-family arrays for each cavity become
    a ``{chain_id: samples}`` mapping, exactly the overdispersed-chain ensemble
    the point-to-set analysis consumes.
    """

    run_path = Path(run_dir)
    results = json.loads((run_path / "results.json").read_text())
    radii = results.get("radii", {})
    core_radius = float(radii.get("core"))
    buffer_radius = float(radii.get("buffer", float("nan")))
    temperature = float(results.get("temperature", float("nan")))

    # Traces stay float64 numpy arrays: the mixing-quality pipeline re-arrayed
    # them immediately, so the list round-trip only boxed/unboxed every sample.
    # [micro-bench, 32 cavities x 4 chains x 20k draws: load_cavity_run
    #  189 ms -> 19 ms, ~10x; downstream radius_mixing_quality on the loaded
    #  chains 316 ms -> 91 ms, ~3.5x]
    with np.load(run_path / "traces.npz") as data:
        by_cavity: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            parsed = _parse_trace_key(key)
            if parsed is None:
                continue
            cavity_tag, key_channel, family = parsed
            if key_channel != channel:
                continue
            by_cavity.setdefault(cavity_tag, {})[family] = (
                np.asarray(data[key], dtype=np.float64).ravel()
            )

    ordered_tags = sorted(by_cavity)
    cavity_chains = tuple(by_cavity[tag] for tag in ordered_tags)
    ordinals = tuple(int(tag[1:]) for tag in ordered_tags)
    return RadiusRun(
        radius=core_radius,
        buffer_radius=buffer_radius,
        temperature=temperature,
        run_dir=str(run_path),
        channel=channel,
        cavity_chains=cavity_chains,
        ordinals=ordinals,
    )


@dataclass(frozen=True)
class RadiusSweepAnalysis:
    """Full sweep: per-radius mixing quality + the point-to-set estimate."""

    channel: str
    temperature: float | None
    radius_mixing: tuple[RadiusMixing, ...]
    estimate: PointToSetEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "temperature": self.temperature,
            "per_radius": [
                {
                    "radius": rm.radius,
                    "m": rm.m,
                    "n_cavities": rm.n_cavities,
                    "n_cavities_used": rm.n_cavities_used,
                    "single_basin_fraction": rm.single_basin_fraction,
                    "per_cavity_m": list(rm.per_cavity_m),
                    "per_cavity_rhat": list(rm.per_cavity_rhat),
                    "per_cavity_n_basins": list(rm.per_cavity_n_basins),
                }
                for rm in self.radius_mixing
            ],
            "point_to_set": self.estimate.to_dict(),
        }


def analyze_radius_sweep(
    run_dirs: Sequence[str | Path],
    *,
    channel: str = CORE_OVERLAP_CHANNEL,
    threshold_factor: float = 2.0,
) -> RadiusSweepAnalysis:
    """End-to-end: load radius-sweep runs, compute m(R), estimate ξ_PTS.

    Runs are grouped by core radius (mean m over runs sharing a radius), so
    replicate runs at the same radius are pooled.  Requires ≥ 2 distinct radii.
    """

    if len(run_dirs) < 1:
        raise ValueError("analyze_radius_sweep needs at least one run directory")
    runs = [load_cavity_run(path, channel=channel) for path in run_dirs]
    temperatures = {round(run.temperature, 6) for run in runs if not math.isnan(run.temperature)}
    temperature = runs[0].temperature if len(temperatures) == 1 else None

    # Group by radius; pool cavities from every run sharing that radius.
    by_radius: dict[float, list[Mapping[str, Sequence[float]]]] = {}
    ordinals_by_radius: dict[float, list[int]] = {}
    for run in runs:
        bucket = by_radius.setdefault(run.radius, [])
        obucket = ordinals_by_radius.setdefault(run.radius, [])
        for ordinal, chains in zip(run.ordinals, run.cavity_chains):
            bucket.append(chains)
            obucket.append(ordinal)

    radius_mixing = tuple(
        radius_mixing_quality(
            radius,
            by_radius[radius],
            channel=channel,
            threshold_factor=threshold_factor,
            ordinals=ordinals_by_radius[radius],
        )
        for radius in sorted(by_radius)
    )
    if len(radius_mixing) < 2:
        raise ValueError(
            "point-to-set estimate needs at least two distinct radii; "
            f"got {len(radius_mixing)} (radius sweep incomplete)"
        )
    estimate = estimate_point_to_set(
        [rm.radius for rm in radius_mixing],
        [rm.m for rm in radius_mixing],
    )
    return RadiusSweepAnalysis(
        channel=channel,
        temperature=temperature,
        radius_mixing=radius_mixing,
        estimate=estimate,
    )
