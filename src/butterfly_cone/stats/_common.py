"""Shared table conventions and small numeric primitives for `butterfly_cone.stats`.

## The exchangeable-unit convention

Every estimator in this package treats **the parent-cavity pair**, not the
individual branch, as the exchangeable unit. Branches drawn from the two
counterfactual candidate states of the same parent cavity are correlated
through that shared parent (shared local environment, shared composition,
shared thermodynamic disturbance); only the pair-level summary (one number
per pair) is an independent draw. Concretely: never resample, average, or
bootstrap over branch rows directly, always group by ``pair_id`` first and
let each pair contribute exactly one degree of freedom to any
variance/CI/resampling computation, even when a pair carries hundreds of
branches and another carries 64.

## Table schema (plain "records": no pandas dependency)

A *branch table* is a plain ``Sequence[Mapping[str, object]]``, a list of
dict rows, one row per simulated branch (the "dataframe of records" the task
spec asks for, without requiring pandas, which is not installed in the
target environment). Required columns for the paired-contrast machinery:

- ``pair_id`` (str), the parent-cavity pair identifier; the exchangeable unit.
- ``arm_family`` (str), caller-defined comparator family, e.g.
  ``"targeted"``, ``"random"``, ``"softness"``, ``"best_incumbent"``. Free
  text; this package does not hardcode which families exist.
- ``arm_sign`` (one of ``"+"``/``"-"``), the high-field vs low-field arm of
  a pair within a family.

Plus one or more *endpoint columns* holding a binary (0/1, or bool) outcome
per branch, the primary confirmation-branch event indicator, the downstream
H1 endpoint, or any alternative-definition variant column used for §21.5
replication (event-threshold variants, geometry variants, ...). Callers name
the endpoint column explicitly on every call; nothing in this package
auto-selects "the best" column.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import math

PAIR_ID_COL = "pair_id"
ARM_FAMILY_COL = "arm_family"
ARM_SIGN_COL = "arm_sign"
SIGNS: tuple[str, str] = ("+", "-")

BranchRow = Mapping[str, object]
BranchTable = Sequence[BranchRow]


def as01(value: object) -> int:
    """Validate and coerce a branch outcome to an int in {0, 1}.

    Raises ValueError on anything else (NaN, out-of-range, non-numeric) so
    malformed endpoint columns fail loudly rather than silently corrupting a
    proportion.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    raise ValueError(f"expected a binary (0/1) outcome, got {value!r}")


def require_columns(row: BranchRow, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in row]
    if missing:
        raise KeyError(f"row missing required column(s) {missing}: {row!r}")


def group_by(table: BranchTable, key_col: str) -> dict[object, list[BranchRow]]:
    groups: dict[object, list[BranchRow]] = {}
    for row in table:
        require_columns(row, (key_col,))
        groups.setdefault(row[key_col], []).append(row)
    return groups


def with_override(row: BranchRow, **overrides: object) -> Mapping[str, object]:
    """Return a view of `row` with `overrides` applied, without copying it.

    Used by the pair bootstrap to relabel a resampled pair copy's
    ``pair_id`` cheaply (``ChainMap`` avoids an O(columns) dict copy per row
    per bootstrap replicate).
    """
    return ChainMap(dict(overrides), row)


def stable_hash_int(*parts: str) -> int:
    """Deterministic, process-independent hash of `parts` as a big integer.

    Python's builtin ``hash()`` is salted per-process (``PYTHONHASHSEED``)
    and unsuitable for anything that must reproduce across runs/processes
    (cohort splits, permutation seeding, tie-breaks). This uses SHA-256 over
    NUL-joined parts, matching the domain-separated hashing convention used
    elsewhere in this project's harness.
    """
    message = b"\0".join(p.encode("utf-8") for p in parts)
    return int.from_bytes(hashlib.sha256(message).digest(), byteorder="big")


def rankdata_average(values: Sequence[float]) -> "list[float]":
    """1-indexed ranks with ties resolved by the average-rank convention.

    Pure-Python/no-scipy reimplementation of ``scipy.stats.rankdata(...,
    method="average")``, used by the rank-statistic AUC estimator.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Numpy-free special functions (no scipy in the target environment).
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 1e-14) -> float:
    """Continued-fraction evaluation for the incomplete beta function.

    Standard Numerical Recipes ``betacf`` (Lentz's algorithm). Used only by
    `betainc`/`beta_ppf` below.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b), no scipy required."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_front = a * math.log(x) + b * math.log(1.0 - x) - lbeta
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Quantile function (inverse CDF) of Beta(a, b) via bisection on `betainc`.

    `betainc(x, a, b)` is monotonically increasing in x on (0, 1), so
    bisection is robust even near the boundaries.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def norm_ppf(p: float) -> float:
    """Standard-normal quantile function (Acklam's rational approximation).

    Accurate to ~1.15e-9 absolute error; avoids a scipy dependency for the
    paired-difference normal-approximation cross-check.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


TableStatistic = Callable[[BranchTable], float]
