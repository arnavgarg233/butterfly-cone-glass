"""Variance components for the causal-fraction estimand.

The input is a plain records table with one binary ``q`` outcome per
``(cavity_id, conditional_draw_id, branch_id)``.  The nesting is

``cavity -> conditional draw -> branch``

and the cavity is the exchangeable unit for uncertainty.  This module uses a
method-of-moments (MOM) nested ANOVA estimator, rather than REML, so it stays
NumPy-only and its finite-cell corrections are explicit.

For cell ``(c, d)`` let ``n_cd`` be its branch count, ``qbar_cd`` its branch
mean, and ``m_c`` the number of conditional draws in cavity ``c``.  The
within-cell mean square is

``s2_thermal = sum_cd sum_b (q_cdb - qbar_cd)^2 / sum_cd (n_cd - 1)``.

The local component follows the requested equal-cavity estimand.  For every
cavity with at least two draws, the ordinary sample variance of its cell
means is corrected as

``s2_local,c = s2(qbar_cd) - s2_thermal * mean_d(1 / n_cd)``.

Thus the naive draw-mean variance is inflated by the thermal noise divided by
the number of branches in each draw; the subtraction is the central
bias-correction in this estimator.  The reported local component is the
average of ``s2_local,c`` across eligible cavities, not a branch-weighted
average.

For the environmental component, the cavity mean is the equal-draw average
``qbar_c = mean_d(qbar_cd)``.  Its sample variance is corrected for the
finite number of draws and branches:

``s2_env = s2(qbar_c) - s2_local * mean_c(1 / m_c)
           - s2_thermal * mean_c(sum_d(1 / n_cd) / m_c**2)``.

These are standard unbalanced-design MOM corrections for the cell-mean
strata.  They assume a common random-effect variance within each stratum;
there is no REML fit or parametric distributional assumption.  Sampling
fluctuations can make a raw MOM component negative, so the public component
estimates use the nonnegative projection of the three raw estimates.  The
projected components form the reported total variance and therefore sum to
one after normalization.  ``observed_total_variance`` retains the ordinary
all-branch sample variance as a diagnostic; it is not used as the causal
budget denominator because it has finite-design weights that do not match
the equal-cavity/equal-draw estimand.

Bootstrap confidence intervals resample complete cavities with replacement.
They never resample branch rows independently.  Every randomized operation
takes an explicit seed and uses a private ``numpy.random.default_rng``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import numbers

import numpy as np


DEFAULT_Q_COL = "q"
DEFAULT_CAVITY_COL = "cavity_id"
DEFAULT_DRAW_COL = "conditional_draw_id"
DEFAULT_BRANCH_COL = "branch_id"

Records = Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class _CavitySummary:
    """Sufficient statistics for one cavity.

    Keeping this representation separate from the original rows lets the
    cavity bootstrap duplicate complete cavities without creating duplicate
    branch indices or repeatedly copying dictionaries.
    """

    cavity_id: object
    draw_means: tuple[float, ...]
    branch_counts: tuple[int, ...]
    within_sum_squares: float
    n_branches: int
    n_cells: int
    sum_q: int

    @property
    def n_draws(self) -> int:
        return len(self.draw_means)

    @property
    def draw_variance(self) -> float:
        if self.n_draws < 2:
            return float("nan")
        return float(np.var(np.asarray(self.draw_means, dtype=float), ddof=1))

    @property
    def mean_inverse_branches(self) -> float:
        return float(np.mean([1.0 / n for n in self.branch_counts]))

    @property
    def inverse_draw_count(self) -> float:
        return 1.0 / self.n_draws

    @property
    def branch_noise_coefficient_for_cavity_mean(self) -> float:
        return float(sum(1.0 / n for n in self.branch_counts) / (self.n_draws**2))

    @property
    def cavity_mean(self) -> float:
        return float(np.mean(np.asarray(self.draw_means, dtype=float)))

    @property
    def within_df(self) -> int:
        return self.n_branches - self.n_cells


@dataclass(frozen=True)
class VarianceComponents:
    """Nested variance-component estimate and the normalized causal budget.

    ``local_variance_raw`` and ``environmental_variance_raw`` are the
    unprojected MOM estimates.  The corresponding public component fields are
    clipped at zero for a valid variance budget.  ``total_variance`` is the
    sum of those projected components and is the denominator of ``f_local``.
    """

    n_cavities: int
    n_local_cavities: int
    n_draws: int
    n_cells: int
    n_branches: int
    thermal_df: int
    thermal_variance: float
    local_variance_raw: float
    local_variance: float
    local_variance_naive: float
    local_thermal_correction: float
    environmental_variance_raw: float
    environmental_variance: float
    total_variance: float
    observed_total_variance: float
    f_local: float
    f_env: float
    f_noise: float

    @property
    def var_local(self) -> float:
        """Short alias for the projected local-structural component."""
        return self.local_variance

    @property
    def var_environmental(self) -> float:
        """Long-form alias for the environmental component."""
        return self.environmental_variance

    @property
    def var_env(self) -> float:
        """Short alias for the environmental component."""
        return self.environmental_variance

    @property
    def var_thermal(self) -> float:
        """Short alias for the thermal component."""
        return self.thermal_variance

    @property
    def var_total(self) -> float:
        """Short alias for the variance-component budget denominator."""
        return self.total_variance

    @property
    def f_environmental(self) -> float:
        """Long-form alias for ``f_env``."""
        return self.f_env

    @property
    def f_thermal(self) -> float:
        """Alias for the thermal/noise fraction ``f_noise``."""
        return self.f_noise

    @property
    def component_sum(self) -> float:
        """Return the projected component sum (equal to ``total_variance``)."""
        return self.local_variance + self.environmental_variance + self.thermal_variance


@dataclass(frozen=True)
class BootstrapResult:
    """Cavity-bootstrap percentile intervals.

    The primary interval is the f-local interval exposed through ``point``,
    ``lo``, ``hi`` and ``boot_estimates`` to match the package's other
    bootstrap result objects.  Component intervals are available through the
    ``*_variance_ci`` properties.
    """

    estimate: VarianceComponents
    lo: float
    hi: float
    boot_estimates: tuple[float, ...]
    alpha: float
    seed: int
    local_variance_lo: float
    local_variance_hi: float
    environmental_variance_lo: float
    environmental_variance_hi: float
    thermal_variance_lo: float
    thermal_variance_hi: float

    @property
    def point(self) -> float:
        return self.estimate.f_local

    @property
    def f_local(self) -> float:
        return self.estimate.f_local

    @property
    def f_local_ci(self) -> tuple[float, float]:
        return self.lo, self.hi

    @property
    def local_variance_ci(self) -> tuple[float, float]:
        return self.local_variance_lo, self.local_variance_hi

    @property
    def environmental_variance_ci(self) -> tuple[float, float]:
        return self.environmental_variance_lo, self.environmental_variance_hi

    @property
    def thermal_variance_ci(self) -> tuple[float, float]:
        return self.thermal_variance_lo, self.thermal_variance_hi

    @property
    def n_boot(self) -> int:
        return len(self.boot_estimates)


def _coerce_binary(value: object) -> int:
    """Coerce a q value to 0/1 while rejecting silent endpoint corruption."""
    # Fast path for plain ints (the overwhelmingly common row type): `type` is
    # exact, so bools still take the isinstance path below.  Same result for
    # every input; skips two abc instance checks per row.
    # [micro-bench, 11k-row table: estimate_variance_components 23 ms -> 17 ms]
    if type(value) is int and (value == 0 or value == 1):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, numbers.Integral):
        integer = int(value)
        if integer in (0, 1):
            return integer
    if isinstance(value, numbers.Real):
        real = float(value)
        if np.isfinite(real) and real in (0.0, 1.0):
            return int(real)
    raise ValueError(f"expected a binary (0/1) q outcome, got {value!r}")


def _summarize_table(
    table: Records,
    q_col: str,
    cavity_col: str,
    draw_col: str,
    branch_col: str,
) -> tuple[_CavitySummary, ...]:
    """Validate records and collapse them into one summary per cavity."""
    rows = list(table)
    if not rows:
        raise ValueError("the branch table must not be empty")

    grouped: dict[object, dict[object, list[int]]] = {}
    seen_indices: set[tuple[object, object, object]] = set()
    required = (q_col, cavity_col, draw_col, branch_col)
    for row in rows:
        missing = [column for column in required if column not in row]
        if missing:
            raise KeyError(f"row missing required column(s) {missing}: {row!r}")
        cavity_id = row[cavity_col]
        draw_id = row[draw_col]
        branch_id = row[branch_col]
        try:
            index = (cavity_id, draw_id, branch_id)
            if index in seen_indices:
                raise ValueError(
                    "duplicate (cavity, conditional-draw, branch) index: "
                    f"{index!r}"
                )
            seen_indices.add(index)
        except TypeError as exc:
            raise ValueError("cavity, draw, and branch identifiers must be hashable") from exc

        grouped.setdefault(cavity_id, {}).setdefault(draw_id, []).append(_coerce_binary(row[q_col]))

    summaries: list[_CavitySummary] = []
    for cavity_id, draw_groups in grouped.items():
        draw_means: list[float] = []
        branch_counts: list[int] = []
        within_sum_squares = 0.0
        n_branches = 0
        sum_q = 0
        for outcomes in draw_groups.values():
            if not outcomes:
                raise ValueError("a conditional-draw cell must contain at least one branch")
            # Outcomes are validated 0/1 ints, so the cell statistics have exact
            # integer-arithmetic forms: the mean is ones/count (bit-identical to
            # np.mean of the float cast) and every residual is either (1 - mean)
            # or (-mean), giving the closed-form within-cell sum of squares
            # below (agrees with the former np.dot to f64 round-off).  Avoids
            # three numpy array constructions per cell.
            # [micro-bench, 11k-row table, 720 cells: 17 ms -> 12 ms]
            count = len(outcomes)
            ones = sum(outcomes)
            draw_mean = ones / count
            draw_means.append(draw_mean)
            branch_counts.append(count)
            within_sum_squares += ones * (1.0 - draw_mean) ** 2 + (count - ones) * draw_mean**2
            n_branches += count
            sum_q += ones

        summaries.append(
            _CavitySummary(
                cavity_id=cavity_id,
                draw_means=tuple(draw_means),
                branch_counts=tuple(branch_counts),
                within_sum_squares=within_sum_squares,
                n_branches=n_branches,
                n_cells=len(draw_means),
                sum_q=sum_q,
            )
        )

    if len(summaries) < 2:
        raise ValueError("at least two cavities are required for the environmental component")
    if not any(summary.n_draws >= 2 for summary in summaries):
        raise ValueError("at least one cavity must contain two conditional draws")
    if not any(summary.within_df > 0 for summary in summaries):
        raise ValueError("within-draw branch replication is required to estimate thermal variance")
    return tuple(summaries)


def _sample_variance(values: Sequence[float], label: str) -> float:
    if len(values) < 2:
        raise ValueError(f"at least two observations are required for {label}")
    return float(np.var(np.asarray(values, dtype=float), ddof=1))


def _fit_summaries(summaries: Sequence[_CavitySummary]) -> VarianceComponents:
    """Fit MOM components from cavity summaries, including bootstrap copies."""
    if len(summaries) < 2:
        raise ValueError("at least two cavities are required for the environmental component")

    n_cavities = len(summaries)
    n_cells = sum(summary.n_cells for summary in summaries)
    n_branches = sum(summary.n_branches for summary in summaries)
    thermal_df = sum(summary.within_df for summary in summaries)
    if thermal_df <= 0:
        raise ValueError("within-draw branch replication is required to estimate thermal variance")

    thermal_variance = max(
        0.0,
        float(sum(summary.within_sum_squares for summary in summaries) / thermal_df),
    )

    local_summaries = [summary for summary in summaries if summary.n_draws >= 2]
    if not local_summaries:
        raise ValueError("at least one cavity must contain two conditional draws")
    local_variance_naive = float(np.mean([summary.draw_variance for summary in local_summaries]))
    mean_inverse_branches = float(
        np.mean([summary.mean_inverse_branches for summary in local_summaries])
    )
    local_thermal_correction = thermal_variance * mean_inverse_branches
    local_variance_raw = local_variance_naive - local_thermal_correction

    cavity_means = [summary.cavity_mean for summary in summaries]
    cavity_mean_variance = _sample_variance(cavity_means, "between-cavity variance")
    mean_inverse_draws = float(np.mean([summary.inverse_draw_count for summary in summaries]))
    mean_branch_noise_coefficient = float(
        np.mean([summary.branch_noise_coefficient_for_cavity_mean for summary in summaries])
    )
    environmental_variance_raw = (
        cavity_mean_variance
        - local_variance_raw * mean_inverse_draws
        - thermal_variance * mean_branch_noise_coefficient
    )

    local_variance = max(0.0, local_variance_raw)
    environmental_variance = max(0.0, environmental_variance_raw)
    total_variance = local_variance + environmental_variance + thermal_variance
    if total_variance > 0.0:
        f_local = local_variance / total_variance
        f_env = environmental_variance / total_variance
        f_noise = thermal_variance / total_variance
    else:
        # An all-constant endpoint has no meaningful variance fraction.  The
        # deterministic convention is a zero budget rather than NaNs.
        f_local = f_env = f_noise = 0.0

    total_sum_q = sum(summary.sum_q for summary in summaries)
    if n_branches > 1:
        total_mean = total_sum_q / n_branches
        observed_total_variance = max(
            0.0,
            float((total_sum_q - n_branches * total_mean**2) / (n_branches - 1)),
        )
    else:
        observed_total_variance = 0.0

    return VarianceComponents(
        n_cavities=n_cavities,
        n_local_cavities=len(local_summaries),
        n_draws=sum(summary.n_draws for summary in summaries),
        n_cells=n_cells,
        n_branches=n_branches,
        thermal_df=thermal_df,
        thermal_variance=thermal_variance,
        local_variance_raw=local_variance_raw,
        local_variance=local_variance,
        local_variance_naive=local_variance_naive,
        local_thermal_correction=local_thermal_correction,
        environmental_variance_raw=environmental_variance_raw,
        environmental_variance=environmental_variance,
        total_variance=total_variance,
        observed_total_variance=observed_total_variance,
        f_local=f_local,
        f_env=f_env,
        f_noise=f_noise,
    )


def estimate_variance_components(
    table: Records,
    q_col: str = DEFAULT_Q_COL,
    cavity_col: str = DEFAULT_CAVITY_COL,
    draw_col: str = DEFAULT_DRAW_COL,
    branch_col: str = DEFAULT_BRANCH_COL,
) -> VarianceComponents:
    """Estimate thermal, local, and environmental variance components.

    Draws and branches may be unbalanced.  Cavities with only one draw are
    retained for the environmental term but cannot contribute to the explicit
    equal-cavity local average; ``n_local_cavities`` reports how many cavities
    supplied that term.
    """
    summaries = _summarize_table(table, q_col, cavity_col, draw_col, branch_col)
    return _fit_summaries(summaries)


def estimate(
    table: Records,
    q_col: str = DEFAULT_Q_COL,
    cavity_col: str = DEFAULT_CAVITY_COL,
    draw_col: str = DEFAULT_DRAW_COL,
    branch_col: str = DEFAULT_BRANCH_COL,
) -> VarianceComponents:
    """Short alias for :func:`estimate_variance_components`."""
    return estimate_variance_components(table, q_col, cavity_col, draw_col, branch_col)


def estimate_f_local(
    table: Records,
    q_col: str = DEFAULT_Q_COL,
    cavity_col: str = DEFAULT_CAVITY_COL,
    draw_col: str = DEFAULT_DRAW_COL,
    branch_col: str = DEFAULT_BRANCH_COL,
) -> float:
    """Return only the bias-corrected local causal fraction."""
    return estimate_variance_components(table, q_col, cavity_col, draw_col, branch_col).f_local


def _quantile_interval(values: Sequence[float], alpha: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True)
class _SummaryArrays:
    """Per-cavity sufficient statistics precomputed once for the bootstrap.

    Each field holds one value per cavity, computed through the exact same
    ``_CavitySummary`` properties that :func:`_fit_summaries` reads, so a
    bootstrap replicate reduces to fancy-indexed row gathers + axis reductions.
    """

    within_ss: np.ndarray
    within_df: np.ndarray
    n_draws: np.ndarray
    draw_variance: np.ndarray          # NaN for single-draw cavities
    mean_inverse_branches: np.ndarray
    inverse_draw_count: np.ndarray
    branch_noise_coefficient: np.ndarray
    cavity_mean: np.ndarray


def _summary_arrays(summaries: Sequence[_CavitySummary]) -> _SummaryArrays:
    return _SummaryArrays(
        within_ss=np.asarray([s.within_sum_squares for s in summaries], dtype=float),
        within_df=np.asarray([s.within_df for s in summaries], dtype=np.int64),
        n_draws=np.asarray([s.n_draws for s in summaries], dtype=np.int64),
        draw_variance=np.asarray(
            [s.draw_variance if s.n_draws >= 2 else float("nan") for s in summaries], dtype=float
        ),
        mean_inverse_branches=np.asarray([s.mean_inverse_branches for s in summaries], dtype=float),
        inverse_draw_count=np.asarray([s.inverse_draw_count for s in summaries], dtype=float),
        branch_noise_coefficient=np.asarray(
            [s.branch_noise_coefficient_for_cavity_mean for s in summaries], dtype=float
        ),
        cavity_mean=np.asarray([s.cavity_mean for s in summaries], dtype=float),
    )


def _bootstrap_components(
    arrays: _SummaryArrays, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`_fit_summaries` over ``(n_boot, n_cavities)`` index rows.

    Returns ``(f_local, local, environmental, thermal)`` replicate arrays.  Each
    row applies the identical MOM formulas (same elementwise operations and
    guard order as :func:`_fit_summaries`); only the summation tree of the
    row-wise means/sums differs, i.e. results agree to f64 round-off.
    """

    thermal_df = arrays.within_df[indices].sum(axis=1)
    eligible = arrays.n_draws[indices] >= 2
    n_eligible = eligible.sum(axis=1)
    bad = (thermal_df <= 0) | (n_eligible == 0)
    if bad.any():
        first = int(np.argmax(bad))
        if thermal_df[first] <= 0:
            raise ValueError("within-draw branch replication is required to estimate thermal variance")
        raise ValueError("at least one cavity must contain two conditional draws")

    thermal = np.maximum(0.0, arrays.within_ss[indices].sum(axis=1) / thermal_df)
    draw_variance = arrays.draw_variance[indices]
    local_naive = np.sum(np.where(eligible, draw_variance, 0.0), axis=1) / n_eligible
    mean_inverse_branches = (
        np.sum(np.where(eligible, arrays.mean_inverse_branches[indices], 0.0), axis=1) / n_eligible
    )
    local_raw = local_naive - thermal * mean_inverse_branches

    cavity_means = arrays.cavity_mean[indices]
    cavity_mean_variance = cavity_means.var(axis=1, ddof=1)
    mean_inverse_draws = arrays.inverse_draw_count[indices].mean(axis=1)
    mean_branch_noise = arrays.branch_noise_coefficient[indices].mean(axis=1)
    environmental_raw = (
        cavity_mean_variance - local_raw * mean_inverse_draws - thermal * mean_branch_noise
    )

    local = np.maximum(0.0, local_raw)
    environmental = np.maximum(0.0, environmental_raw)
    total = local + environmental + thermal
    f_local = np.divide(local, total, out=np.zeros_like(total), where=total > 0.0)
    return f_local, local, environmental, thermal


def bootstrap_cavities(
    table: Records,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    q_col: str = DEFAULT_Q_COL,
    cavity_col: str = DEFAULT_CAVITY_COL,
    draw_col: str = DEFAULT_DRAW_COL,
    branch_col: str = DEFAULT_BRANCH_COL,
) -> BootstrapResult:
    """Compute cavity-level percentile CIs for f-local and all components.

    A bootstrap replicate samples ``n_cavities`` entries from the original
    cavity-summary list with replacement.  The selected summaries, rather
    than individual branch rows, are passed to the exact same MOM estimator.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral) or n_boot < 2:
        raise ValueError("n_boot must be an integer at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")

    summaries = _summarize_table(table, q_col, cavity_col, draw_col, branch_col)
    point_estimate = _fit_summaries(summaries)
    rng = np.random.default_rng(int(seed))
    n_cavities = len(summaries)

    # Vectorized replicate loop: one (n_boot, n_cavities) index draw -- the PCG
    # stream is bit-identical to n_boot sequential size-n_cavities draws -- and
    # the per-replicate MOM fit becomes row-wise gathers/reductions instead of
    # rebuilding per-summary numpy scalars (draw_variance et al.) every time.
    # [micro-bench, 120 cavities x 6 draws x ~16 branches, n_boot=400:
    #  bootstrap_cavities 2.11 s -> 25 ms per call (~84x); the surviving ~23 ms
    #  is the shared table summarization + point fit, replicates are ~2 ms]
    indices = rng.integers(0, n_cavities, size=(int(n_boot), n_cavities))
    f_local_arr, local_arr, environmental_arr, thermal_arr = _bootstrap_components(
        _summary_arrays(summaries), indices
    )
    f_local_boot = [float(value) for value in f_local_arr]
    local_boot = [float(value) for value in local_arr]
    environmental_boot = [float(value) for value in environmental_arr]
    thermal_boot = [float(value) for value in thermal_arr]

    f_local_lo, f_local_hi = _quantile_interval(f_local_boot, alpha)
    local_lo, local_hi = _quantile_interval(local_boot, alpha)
    environmental_lo, environmental_hi = _quantile_interval(environmental_boot, alpha)
    thermal_lo, thermal_hi = _quantile_interval(thermal_boot, alpha)
    return BootstrapResult(
        estimate=point_estimate,
        lo=f_local_lo,
        hi=f_local_hi,
        boot_estimates=tuple(f_local_boot),
        alpha=float(alpha),
        seed=int(seed),
        local_variance_lo=local_lo,
        local_variance_hi=local_hi,
        environmental_variance_lo=environmental_lo,
        environmental_variance_hi=environmental_hi,
        thermal_variance_lo=thermal_lo,
        thermal_variance_hi=thermal_hi,
    )


def bootstrap_f_local(
    table: Records,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    q_col: str = DEFAULT_Q_COL,
    cavity_col: str = DEFAULT_CAVITY_COL,
    draw_col: str = DEFAULT_DRAW_COL,
    branch_col: str = DEFAULT_BRANCH_COL,
) -> BootstrapResult:
    """Short alias for :func:`bootstrap_cavities`."""
    return bootstrap_cavities(
        table,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        q_col=q_col,
        cavity_col=cavity_col,
        draw_col=draw_col,
        branch_col=branch_col,
    )


__all__ = [
    "BootstrapResult",
    "VarianceComponents",
    "bootstrap_cavities",
    "bootstrap_f_local",
    "estimate",
    "estimate_f_local",
    "estimate_variance_components",
]
