"""Variance reductions for shared-world branches and frozen pair covariates.

The CRN estimate is deliberately tied to :func:`butterfly_cone.stats.fate_anova.fate_anova`:
for a fate matrix ``Y[c, s, w]``, ``estimate_rho_w`` reports the corrected
noise-world component share

    ``rho_w_hat = V_noise_hat / V_total_hat``.

This is a method-of-moments share of outcome variance attributable to the
crossed noise-world factor, not a raw Pearson correlation of two selected
arms.  The bias correction and nonnegative projection are inherited from
``fate_anova`` so finite-world/cavity residual noise is not credited as CRN
signal.

The control-variate API assumes that ``covariate_pairs`` is a frozen,
pre-outcome per-pair covariate (for example, the declared in advance predicted
``Delta q-hat``).  There is no runtime way to verify when a caller measured or
frozen a value, so the function documents the requirement but cannot enforce
it.  The OLS coefficient is fitted once on the supplied pairs, and the paired
bootstrap is conditional on that fitted coefficient and the observed sample
mean of the covariate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import fate_anova, intervals, power_gate


def _validate_rho_w(rho_w: float) -> float:
    if isinstance(rho_w, (bool, np.bool_)):
        raise ValueError(f"rho_w must be a real number in [0, 1), got {rho_w!r}")
    value = float(rho_w)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"rho_w must be finite and in [0, 1), got {rho_w!r}")
    return value


def _validate_positive_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real positive number, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return result


def crn_effective_branches(n_branches: float, rho_w: float) -> float:
    """Return independent-branch equivalents under a shared-world CRN.

    If paired-world contrast variance is the independent-cells variance times
    ``1 - rho_w``, then ``n_branches`` physical branches have
    ``n_branches / (1 - rho_w)`` independent-branch equivalents.  The input is
    allowed to be real-valued because exact design algebra often starts from
    ``required_budget(...).n_branches_exact``; actual execution counts should
    still be integer-valued.
    """
    branches = _validate_positive_real(n_branches, "n_branches")
    rho = _validate_rho_w(rho_w)
    return branches / (1.0 - rho)


def _power_gate_mde_with_real_branches(
    n_pairs: int,
    n_branches: float,
    base_rate: float,
    power: float,
    alpha: float,
    n_arms: int,
) -> float:
    """Call ``power_gate.mde`` while preserving exact real branch algebra."""
    branches = _validate_positive_real(n_branches, "n_branches")
    if branches.is_integer():
        return power_gate.mde(
            n_pairs,
            int(branches),
            base_rate,
            power,
            alpha,
            n_arms,
        )
    # power_gate.mde intentionally accepts integer counts only.  Its closed
    # form scales as 1/sqrt(n_branches), so one validated integer evaluation
    # gives the exact continuous extension needed for budget calculations.
    one_branch_mde = power_gate.mde(n_pairs, 1, base_rate, power, alpha, n_arms)
    return float(one_branch_mde / np.sqrt(branches))


def crn_mde(
    n_pairs: int,
    n_branches: float,
    base_rate: float,
    rho_w: float,
    power: float = power_gate.DEFAULT_POWER,
    alpha: float = power_gate.DEFAULT_ALPHA,
    n_arms: int = power_gate.DEFAULT_N_ARMS,
) -> float:
    """Return the power-gate MDE after crediting shared-world correlation.

    This is exactly

    ``power_gate.mde(n_pairs, n_branches, ...) * sqrt(1 - rho_w)``.

    Equivalently it evaluates the independent-cell MDE at the effective count
    returned by :func:`crn_effective_branches`.  Integer and real-valued branch
    counts are both accepted here so the identity can be used before taking a
    campaign ceiling.
    """
    rho = _validate_rho_w(rho_w)
    independent_mde = _power_gate_mde_with_real_branches(
        n_pairs, n_branches, base_rate, power, alpha, n_arms
    )
    return float(independent_mde * np.sqrt(1.0 - rho))


def estimate_rho_w(Y: np.ndarray) -> float:
    """Estimate the CRN noise-world share from a fate matrix ``Y[c, s, w]``.

    The returned value is ``fate_anova(Y).noise_share``: the finite-``W``/``M``
    bias-corrected and nonnegative-projected noise-world variance component
    divided by the corresponding corrected total variance.  In the additive
    no-residual case this is the sample variance of the world effects divided
    by the sum of the structure and world-effect variances.  This estimator is
    appropriate when the same world index is reused across the paired arms;
    it does not claim that every pair of arm outcomes has equal Pearson
    correlation.
    """
    result = fate_anova.fate_anova(Y)
    return float(result.noise_share)


def _validate_pair_arrays(
    d_pairs: Sequence[float] | np.ndarray,
    covariate_pairs: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d_pairs, dtype=float)
    x = np.asarray(covariate_pairs, dtype=float)
    if d.ndim != 1 or x.ndim != 1:
        raise ValueError("d_pairs and covariate_pairs must both be one-dimensional")
    if d.size != x.size:
        raise ValueError("d_pairs and covariate_pairs must have the same length")
    if d.size < 2:
        raise ValueError("at least two pairs are required")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(x)):
        raise ValueError("d_pairs and covariate_pairs must contain only finite values")
    return d, x


def _validate_bootstrap_args(n_boot: int, alpha: float, seed: int) -> tuple[int, float, int]:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must be finite and in (0, 1), got {alpha!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    return int(n_boot), alpha_value, int(seed)


def _mean_adjusted_statistic(table: Sequence[Mapping[str, object]]) -> float:
    return float(np.mean([float(row["adjusted_contrast"]) for row in table]))


@dataclass(frozen=True)
class CVContrastResult:
    """Point, precision metadata, and pair-bootstrap CI for a CV contrast."""

    point: float
    lo: float
    hi: float
    variance_reduction_factor: float
    beta: float
    r: float
    adjusted_pairs: tuple[float, ...]
    boot_estimates: tuple[float, ...]
    alpha: float
    n_boot: int
    seed: int

    @property
    def adjusted_point(self) -> float:
        """Alias for ``point`` that makes the adjusted estimand explicit."""
        return self.point

    @property
    def ci(self) -> tuple[float, float]:
        """Percentile paired-bootstrap confidence interval."""
        return self.lo, self.hi

    @property
    def reduction_factor(self) -> float:
        """Alias for the effective-pairs factor ``1 / (1 - r**2)``."""
        return self.variance_reduction_factor

    @property
    def variance_reduction(self) -> float:
        """Alias for ``variance_reduction_factor``."""
        return self.variance_reduction_factor

    @property
    def factor(self) -> float:
        """Short alias for ``variance_reduction_factor``."""
        return self.variance_reduction_factor

    @property
    def r_squared(self) -> float:
        """Squared pair correlation used by the variance-gain formula."""
        return self.r * self.r

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Alias for the percentile paired-bootstrap interval."""
        return self.ci


def cv_adjusted_contrast(
    d_pairs: Sequence[float] | np.ndarray,
    covariate_pairs: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CVContrastResult:
    """Adjust paired contrasts with a frozen pre-outcome covariate.

    For pair ``i``, fit the intercept-plus-slope OLS regression of ``d_i`` on
    ``x_i`` and return the centered control-variate values

    ``d_adj_i = d_i - beta * (x_i - mean(x))``.

    The sample point remains ``mean(d)`` exactly in the estimand, while the
    residual pair scatter is reduced by ``1-r**2`` in the usual OLS model.  We
    report the corresponding effective-pairs / variance-reduction factor
    ``1/(1-r**2)``.  A constant covariate is treated as a null covariate with
    ``beta = r = 0``.

    ``covariate_pairs`` must be measured and frozen before the outcome in
    ``d_pairs`` is observed.  This function cannot enforce that temporal or
    ledger condition; passing an outcome-derived covariate would invalidate
    the claimed unbiasedness and is the caller's responsibility.  The paired
    bootstrap resamples the fixed adjusted values as pair-level atoms, using
    the same deterministic pattern as :func:`intervals.bootstrap_pairs`.
    """
    d, x = _validate_pair_arrays(d_pairs, covariate_pairs)
    n_boot_value, alpha_value, seed_value = _validate_bootstrap_args(n_boot, alpha, seed)

    d_mean = float(np.mean(d))
    x_centered = x - float(np.mean(x))
    d_centered = d - d_mean
    ss_x = float(np.dot(x_centered, x_centered))
    ss_d = float(np.dot(d_centered, d_centered))

    if ss_x <= 0.0:
        beta = 0.0
        r = 0.0
    else:
        covariance = float(np.dot(x_centered, d_centered))
        beta = covariance / ss_x
        if ss_d <= 0.0:
            r = 0.0
        else:
            r = float(covariance / np.sqrt(ss_x * ss_d))
            r = float(np.clip(r, -1.0, 1.0))

    adjusted = d - beta * x_centered
    factor_denominator = 1.0 - r * r
    variance_reduction_factor = (
        float(np.inf) if factor_denominator <= 0.0 else float(1.0 / factor_denominator)
    )

    # One row is one exchangeable pair.  `bootstrap_pairs` then supplies the
    # pair-id grouping and the explicit local RNG; there is no branch-level
    # resampling hidden in this array API.
    table = [
        {"pair_id": pair_index, "adjusted_contrast": float(value)}
        for pair_index, value in enumerate(adjusted)
    ]
    bootstrap = intervals.bootstrap_pairs(
        table,
        _mean_adjusted_statistic,
        n_boot=n_boot_value,
        alpha=alpha_value,
        seed=seed_value,
    )

    return CVContrastResult(
        point=d_mean,
        lo=bootstrap.lo,
        hi=bootstrap.hi,
        variance_reduction_factor=variance_reduction_factor,
        beta=float(beta),
        r=float(r),
        adjusted_pairs=tuple(float(value) for value in adjusted),
        boot_estimates=bootstrap.boot_estimates,
        alpha=bootstrap.alpha,
        n_boot=bootstrap.n_boot,
        seed=bootstrap.seed,
    )


__all__ = [
    "CVContrastResult",
    "crn_effective_branches",
    "crn_mde",
    "estimate_rho_w",
    "cv_adjusted_contrast",
]
