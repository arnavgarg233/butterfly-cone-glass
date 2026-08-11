"""§21.1 field-calibration diagnostics: reliability diagram, ECE/MCE,
isotonic recalibration, and rank-statistic AUC (no sklearn/scipy dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import rankdata_average


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_true: float


def _as_arrays(q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray") -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if q.shape != y.shape:
        raise ValueError("q_pred and y_true must have the same shape")
    if q.size == 0:
        raise ValueError("q_pred/y_true must be non-empty")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must be binary (0/1)")
    return q, y


def reliability_diagram(
    q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray", n_bins: int = 10
) -> list[ReliabilityBin]:
    """Binned predicted-field vs. empirical branch-frequency reliability data.

    Fixed-width bins over [0, 1]. Empty bins are included (count=0,
    mean_pred/mean_true=nan) so callers can see coverage gaps rather than
    have them silently vanish.
    """
    q, y = _as_arrays(q_pred, y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (q >= lo) & (q <= hi)
        else:
            mask = (q >= lo) & (q < hi)
        count = int(mask.sum())
        mean_pred = float(q[mask].mean()) if count else float("nan")
        mean_true = float(y[mask].mean()) if count else float("nan")
        bins.append(ReliabilityBin(lo=float(lo), hi=float(hi), count=count, mean_pred=mean_pred, mean_true=mean_true))
    return bins


def ece(q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray", n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |mean_pred - mean_true| over bins."""
    bins = reliability_diagram(q_pred, y_true, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("no data")
    return sum(b.count * abs(b.mean_pred - b.mean_true) for b in bins if b.count > 0) / total


def mce(q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray", n_bins: int = 10) -> float:
    """Maximum Calibration Error: worst-bin |mean_pred - mean_true|."""
    bins = reliability_diagram(q_pred, y_true, n_bins)
    non_empty = [abs(b.mean_pred - b.mean_true) for b in bins if b.count > 0]
    if not non_empty:
        raise ValueError("no data")
    return max(non_empty)


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A monotonic (nondecreasing) recalibration map fit by pool-adjacent-violators.

    `.predict(q_new)` linearly interpolates between fitted control points and
    clips to the fitted range's endpoints outside it (matching
    `sklearn.isotonic.IsotonicRegression`'s default `out_of_bounds="clip"`
    behaviour, reimplemented here without a sklearn dependency).
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def predict(self, q_new: "float | list[float] | np.ndarray") -> "float | np.ndarray":
        scalar_input = np.isscalar(q_new)
        result = np.interp(np.asarray(q_new, dtype=float), self.x, self.y)
        return float(result) if scalar_input else result


def isotonic_recalibrate(q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray") -> IsotonicCalibrator:
    """Fit a simple isotonic (monotonic, nondecreasing) recalibration of q_pred onto y_true.

    Implementation: sort by q_pred, run the pool-adjacent-violators
    algorithm (PAVA) on y_true in that order to obtain a nondecreasing
    fitted sequence, then collapse to unique x control points (averaging
    fitted values that share an x) for interpolation.
    """
    q, y = _as_arrays(q_pred, y_true)
    order = np.argsort(q, kind="mergesort")
    x_sorted = q[order]
    y_sorted = y[order]

    # Pool-adjacent-violators via a stack of (value, weight) blocks.
    values: list[float] = []
    weights: list[float] = []
    for v in y_sorted:
        values.append(float(v))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            values.pop()
            weights.pop()
            values[-1] = merged
            weights[-1] = w

    fitted = np.empty(len(y_sorted), dtype=float)
    pos = 0
    for v, w in zip(values, weights):
        block = int(round(w))
        fitted[pos : pos + block] = v
        pos += block

    # Collapse to unique x control points for a well-defined interpolant.
    unique_x: list[float] = []
    unique_y: list[float] = []
    i = 0
    n = len(x_sorted)
    while i < n:
        j = i
        while j + 1 < n and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        unique_x.append(float(x_sorted[i]))
        unique_y.append(float(fitted[i : j + 1].mean()))
        i = j + 1
    return IsotonicCalibrator(x=tuple(unique_x), y=tuple(unique_y))


def auc_rank(q_pred: "list[float] | np.ndarray", y_true: "list[float] | np.ndarray") -> float:
    """AUC via the Mann-Whitney U / rank-sum identity (no sklearn dependency).

    AUC = (sum of ranks among positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg),
    with ties resolved by average rank -- the standard equivalence between
    the Wilcoxon rank-sum statistic and the empirical AUC.
    """
    q, y = _as_arrays(q_pred, y_true)
    n_pos = int((y == 1.0).sum())
    n_neg = int((y == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires at least one positive and one negative example")
    ranks = np.asarray(rankdata_average(q.tolist()))
    rank_sum_pos = ranks[y == 1.0].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
