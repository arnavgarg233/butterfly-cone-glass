"""CI machinery (PLAN_v2.1 §21): pair-cluster bootstrap, Jeffreys intervals,
paired-difference normal approximation.

All randomized functions here take an explicit integer `seed` and build
their own `numpy.random.default_rng(seed)` -- deterministic given seed, and
independent of any global RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import PAIR_ID_COL, BranchTable, TableStatistic, beta_ppf, group_by, norm_ppf, with_override


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    alpha: float
    n_boot: int
    seed: int
    boot_estimates: tuple[float, ...]


def bootstrap_pairs(
    table: BranchTable,
    statistic: TableStatistic,
    *,
    pair_col: str = PAIR_ID_COL,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> BootstrapResult:
    """Nonparametric bootstrap CI resampling parent-cavity pairs.

    Why pairs, never branches within a pair: branches from the two
    counterfactual candidate states of one parent cavity share that parent's
    local environment, composition, and thermodynamic disturbance, so they
    are correlated -- only the pair is an independent draw. Each bootstrap
    replicate therefore draws a with-replacement sample of *pair ids* (not
    of rows), and every row belonging to a drawn pair is carried into the
    replicate as one atomic block; a pair drawn twice contributes two
    independent-looking copies of its full branch record, not two
    independently-resampled branches. `statistic` is any callable taking a
    branch table and returning a float (e.g. `estimands.estimate_ate(...,
    ...).pair_weighted` via `functools.partial`); it is never told which
    rows are "real" vs. resampled, so the exact same estimator code path
    computes the point estimate and every bootstrap replicate.
    """
    rows_by_pair = group_by(table, pair_col)
    pair_ids = sorted(rows_by_pair)
    n_pairs = len(pair_ids)
    if n_pairs == 0:
        raise ValueError("no pairs found in table")
    point = statistic(table)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n_pairs, size=n_pairs)
        resampled: list[object] = []
        for copy_index, idx in enumerate(draw):
            pid = pair_ids[idx]
            replicate_id = f"{pid}\0boot{copy_index}"
            for row in rows_by_pair[pid]:
                resampled.append(with_override(row, **{pair_col: replicate_id}))
        boots[b] = statistic(resampled)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        point=point,
        lo=float(lo),
        hi=float(hi),
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        boot_estimates=tuple(float(x) for x in boots),
    )


def independent_diff_ci(
    boot_a: BootstrapResult, boot_b: BootstrapResult, alpha: float = 0.05
) -> tuple[float, float, float]:
    """CI for A - B when A and B are bootstrapped over disjoint pair sets.

    Used when `estimands.ContrastResult.fully_matched` is False (comparator
    arms drawn on a different, only geometry-matched, pair sample): the two
    bootstrap distributions are independent, so their difference's
    percentile CI is built from the elementwise difference of the two
    (equal-length) replicate arrays. Requires `boot_a` and `boot_b` to share
    `n_boot` and, for a meaningful joint replicate stream, to have been
    drawn with different seeds (independent resampling).
    """
    if boot_a.n_boot != boot_b.n_boot:
        raise ValueError("bootstrap distributions must have equal n_boot to difference elementwise")
    diffs = np.asarray(boot_a.boot_estimates) - np.asarray(boot_b.boot_estimates)
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return boot_a.point - boot_b.point, float(lo), float(hi)


def jeffreys_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Jeffreys (Beta(1/2,1/2)-prior) equal-tailed CI for a single proportion.

    Posterior is Beta(k + 1/2, n - k + 1/2) (PLAN_v2.1 §39.2's own prior
    choice, reused here for consistency). Follows the standard
    boundary-adjusted convention (Brown, Cai & DasGupta 2001): the lower
    limit is clamped to 0 when k=0 and the upper limit to 1 when k=n, since
    the raw equal-tailed Beta quantile would otherwise report a spuriously
    positive lower (resp. sub-1 upper) bound at the observed boundary.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, a, b)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, a, b)
    return lo, hi


@dataclass(frozen=True)
class NormalApproxResult:
    point: float
    lo: float
    hi: float
    alpha: float
    n_pairs: int
    se: float


def paired_diff_normal_approx(deltas: list[float], alpha: float = 0.05) -> NormalApproxResult:
    """Paired-difference normal-approximation CI, as a cross-check on the bootstrap.

    Takes one already-collapsed number per pair (e.g. each pair's Delta q_c,
    or each pair's D_c under a fully-matched design) and treats that
    sequence as i.i.d. draws across pairs -- correct precisely because
    collapsing to one number per pair is what removes the within-pair
    correlation; the normal approximation must never be applied to raw
    per-branch outcomes.
    """
    arr = np.asarray(deltas, dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError("need at least 2 pairs for a normal-approximation CI")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    z = norm_ppf(1.0 - alpha / 2.0)
    return NormalApproxResult(point=mean, lo=mean - z * se, hi=mean + z * se, alpha=alpha, n_pairs=n, se=se)
