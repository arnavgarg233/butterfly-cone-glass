"""perturb/butterfly.py -- Gardner two-channel butterfly-cone analysis.

Turns the *bounded* Gardner divergence result into a clean positive by reading
the deterministic butterfly cone **before** Lyapunov saturation.  The old
analysis measured its axes at the final frame -- i.e. *after* the divergence had
already saturated to its plateau -- which collapses the exponential growth rate
toward zero and hides the ballistic front.  Everything here is built so the fits
see only the pre-saturation window.

Two channels feed this module, both produced by ``perturb.response``:

* **raw-displacement channel M** -- ``response.divergence_field`` (``(T, N)``),
  the matched-seed minimum-image branch divergence ``d_i(t)``;
* **cage-relative / structural channel S** -- ``response.cage_relative_divergence_field``
  (``(T, N)``), the drift-robust structural divergence.

Deliverables (all read from the *pre-saturation* window):

1. :func:`fit_lyapunov` -- fit ``D(t) = D0 * exp(2 lambda t)`` after
   auto-detecting the saturation onset (:func:`detect_saturation_onset`, the
   crux); the plateau ``D_sat`` falls out of the same detector.
2. :func:`butterfly_velocity` -- ballistic front speed ``v_b`` from the spatial
   front ``r_front(t)`` (the outer radius where local decorrelation crosses a
   threshold), fit ``r_front = v_b t + r0`` over the front-growth window.
3. :func:`channel_shielding` -- the structural "causal shielding time"
   ``t_shield`` where the *cage-relative* divergence crosses threshold, versus
   the raw-channel crossing; ``t_shield > t_raw`` means the timescale structure
   resists chaos.
4. :func:`intensive_check` -- ``lambda``, ``v_b``, ``t_shield`` must be
   N-independent (intensive).  This is a *consistency* test across the 1500/3000
   configs, **not** a finite-size-scaling fit.

Pure numpy float64, no RNG: every routine is deterministic and device
independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from butterfly_cone.events.trajectory import as_box, as_float_array, minimum_image
from butterfly_cone.perturb.response import total_divergence

# --- default detector / fit parameters (illustrative, all overridable) -------

SLOPE_FRAC_DEFAULT = 0.5        # onset where local slope drops below 0.5 * growth slope
SLOPE_WINDOW_DEFAULT = 5        # sliding-linear-fit width for the local slope
GROWTH_QUANTILE_DEFAULT = 0.9   # high quantile of local slopes := growth slope
MIN_LOG_RISE_DEFAULT = 1.0      # ln-units of rise required to call it "growth"
SHIELD_THRESHOLD_FRAC = 0.5     # crossing level as a fraction of the plateau
FRONT_THRESHOLD_FRAC = 0.1      # front level as a fraction of the field maximum
INTENSIVE_TOL_DEFAULT = 0.2     # relative spread below which quantities agree


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _as_series(values: Any) -> np.ndarray:
    """Reduce a divergence signal to a per-time total ``D(t)`` of shape ``(T,)``.

    A ``(T, N)`` per-particle field is summed over particles (reusing
    ``response.total_divergence``); a ``(T,)`` total is returned unchanged.
    """

    arr = as_float_array(values)
    if arr.ndim == 2:
        return total_divergence(arr)
    if arr.ndim == 1:
        return arr
    raise ValueError("divergence signal must be (T,) total or (T, N) field")


def _local_slope(t: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Per-sample local slope of ``y`` vs ``t`` from a centered linear fit.

    A sliding least-squares slope (width ``window``) is intrinsically smoothing,
    so a single noisy sample cannot fabricate a growth or a saturation knee.

    The slope of each window is the closed-form centered least-squares solution
    ``sum((t-t̄)(y-ȳ)) / sum((t-t̄)^2)`` -- the same line the former per-window
    ``np.polyfit(deg=1)`` solved (agreement to f64 round-off, ~1e-13 relative),
    with the interior windows evaluated in one strided batch instead of one
    Python-level SVD fit per sample.  A window with zero time spread keeps
    slope 0 exactly as before (``den > 0`` iff ``ptp > 0``).
    [micro-bench, 60 random series (n up to 200), window=5:
     detect_saturation_onset sweep 420 ms -> 8 ms, ~52x; fit_lyapunov on a
     T=400 series 14 ms -> 0.3 ms, ~45x]
    """

    n = y.size
    half = max(1, int(window) // 2)
    width = 2 * half + 1
    slopes = np.zeros(n)
    if n >= width:
        t_windows = np.lib.stride_tricks.sliding_window_view(t, width)
        y_windows = np.lib.stride_tricks.sliding_window_view(y, width)
        t_centered = t_windows - t_windows.mean(axis=1, keepdims=True)
        y_centered = y_windows - y_windows.mean(axis=1, keepdims=True)
        den = np.einsum("ij,ij->i", t_centered, t_centered)
        num = np.einsum("ij,ij->i", t_centered, y_centered)
        slopes[half:n - half] = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)
        edge_indices = list(range(half)) + list(range(n - half, n))
    else:
        edge_indices = list(range(n))
    for i in edge_indices:
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo >= 2:
            t_edge = t[lo:hi]
            t_c = t_edge - t_edge.mean()
            den_edge = float(t_c @ t_c)
            if den_edge > 0.0:
                y_edge = y[lo:hi]
                slopes[i] = float((t_c @ (y_edge - y_edge.mean())) / den_edge)
    return slopes


def _r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# The crux: saturation-onset detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaturationOnset:
    """Where a growing-then-saturating signal leaves its growth regime.

    The pre-saturation fit window is the half-open index range
    ``[start_index, index)`` into the (positive-filtered) samples; ``index`` is
    the first *saturated* sample.  ``growing`` is False for a signal with no
    resolvable exponential/ballistic growth (e.g. a pure plateau), in which case
    the window spans the whole series and any downstream rate collapses to 0.
    """

    index: int              # first saturated sample (exclusive end of fit window)
    time: float             # t at the onset
    start_index: int        # first sample of the fit window
    growth_slope: float     # reference growth slope of the signal
    growing: bool           # a growth regime was resolved
    saturated: bool         # a saturation knee was found before the series end
    method: str


def detect_saturation_onset(
    t: Any,
    y: Any,
    *,
    slope_frac: float = SLOPE_FRAC_DEFAULT,
    window: int = SLOPE_WINDOW_DEFAULT,
    growth_quantile: float = GROWTH_QUANTILE_DEFAULT,
    min_rise: float = MIN_LOG_RISE_DEFAULT,
) -> SaturationOnset:
    """Auto-detect the saturation onset where ``d y / dt -> 0``.

    ``y`` is the signal read on its natural growth scale: ``log D(t)`` for the
    exponential Lyapunov growth, or ``r_front(t)`` for the linear butterfly
    front.  The growth slope is a high quantile of the per-sample local slopes;
    the onset is the first sample (after growth begins) whose local slope has
    fallen below ``slope_frac`` of that growth slope.  This is deliberately
    biased to cut *slightly early* -- excluding a few late growth points is
    harmless, but including plateau points is exactly the bug this replaces.

    A signal that never rises by at least ``min_rise`` (in the units of ``y``)
    or whose growth slope is non-positive is reported ``growing=False``.
    """

    t = as_float_array(t)
    y = as_float_array(y)
    if t.shape != y.shape or y.ndim != 1:
        raise ValueError("t and y must be matching 1-D arrays")
    n = y.size
    method = (
        f"local-slope(window={window}) < {slope_frac} * "
        f"quantile_{growth_quantile}(slope); min_rise={min_rise}"
    )
    if n < 3:
        return SaturationOnset(
            index=n, time=float(t[-1]) if n else float("nan"), start_index=0,
            growth_slope=0.0, growing=False, saturated=False, method=method,
        )

    slopes = _local_slope(t, y, window)
    growth_slope = float(np.quantile(slopes, growth_quantile))
    total_rise = float(np.max(y) - np.min(y))
    # A truly flat signal has zero rise; guard against a numerical-noise slope
    # (~1e-16 from the local fit) masquerading as growth.
    growing = growth_slope > 0.0 and total_rise > 0.0 and total_rise >= min_rise
    if not growing:
        return SaturationOnset(
            index=n, time=float(t[-1]), start_index=0,
            growth_slope=max(growth_slope, 0.0), growing=False,
            saturated=False, method=method,
        )

    threshold = slope_frac * growth_slope
    grow_mask = slopes >= threshold
    start = int(np.argmax(grow_mask))  # first sample at established growth
    tail = np.where(~grow_mask[start:])[0]
    if tail.size == 0:
        return SaturationOnset(
            index=n, time=float(t[-1]), start_index=start,
            growth_slope=growth_slope, growing=True, saturated=False, method=method,
        )
    onset = start + int(tail[0])
    return SaturationOnset(
        index=onset, time=float(t[onset]), start_index=start,
        growth_slope=growth_slope, growing=True, saturated=True, method=method,
    )


# ---------------------------------------------------------------------------
# (1) Lyapunov exponent + saturation plateau
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LyapunovFit:
    lam: float              # Lyapunov exponent lambda
    slope: float            # fitted 2*lambda
    D0: float               # growth prefactor exp(intercept)
    D_sat: float            # saturation plateau
    onset: SaturationOnset
    n_fit: int              # samples used in the pre-saturation fit
    r2: float


def fit_lyapunov(
    t: Any,
    D: Any,
    *,
    slope_frac: float = SLOPE_FRAC_DEFAULT,
    window: int = SLOPE_WINDOW_DEFAULT,
    growth_quantile: float = GROWTH_QUANTILE_DEFAULT,
    min_log_rise: float = MIN_LOG_RISE_DEFAULT,
    divergence_power: float = 2.0,
) -> LyapunovFit:
    """Fit ``D(t) = D0 * exp(divergence_power * lambda t)`` on the pre-saturation window only.

    ``divergence_power`` states the power of the divergence signal ``D``:
    a SQUARED divergence ``|Δ|^2`` grows as ``exp(2 lambda t)`` (``divergence_power=2``),
    while a FIRST-POWER norm ``|Δ|`` (what ``response.divergence_field`` produces)
    grows as ``exp(lambda t)`` (``divergence_power=1``). The physical Lyapunov rate
    is ``lam = slope / divergence_power``; feeding a first-power norm with the
    default 2 would halve the reported rate.

    ``D`` is the total divergence ``D(t)`` (``(T,)``) or a per-particle field
    ``(T, N)`` that is summed to a total first.  The saturation onset is detected
    on ``log D`` and the exponential is fit strictly before it; ``D_sat`` is the
    median of the post-onset plateau.  A pure plateau (no resolvable growth)
    returns ``lambda = 0`` rather than a spurious rate.
    """

    t = as_float_array(t)
    D = _as_series(D)
    if t.shape != D.shape:
        raise ValueError("t and D must have the same length")

    pos = D > 0.0
    tt = t[pos]
    Dp = D[pos]
    if tt.size < 3:
        onset = SaturationOnset(
            index=tt.size, time=float("nan"), start_index=0, growth_slope=0.0,
            growing=False, saturated=False, method="insufficient positive samples",
        )
        d_sat = float(np.median(Dp)) if Dp.size else float("nan")
        return LyapunovFit(
            lam=0.0, slope=0.0, D0=d_sat, D_sat=d_sat, onset=onset, n_fit=0, r2=float("nan")
        )

    yy = np.log(Dp)
    onset = detect_saturation_onset(
        tt, yy, slope_frac=slope_frac, window=window,
        growth_quantile=growth_quantile, min_rise=min_log_rise,
    )
    if not onset.growing:
        d_sat = float(np.median(Dp))
        return LyapunovFit(
            lam=0.0, slope=0.0, D0=d_sat, D_sat=d_sat, onset=onset, n_fit=0, r2=float("nan")
        )

    lo, hi = onset.start_index, onset.index
    if hi - lo < 2:
        hi = min(lo + 2, tt.size)
    xt, yt = tt[lo:hi], yy[lo:hi]
    slope, intercept = (float(v) for v in np.polyfit(xt, yt, 1))
    if onset.saturated and onset.index < Dp.size:
        d_sat = float(np.median(Dp[onset.index:]))
    else:
        d_sat = float("nan")
    return LyapunovFit(
        lam=slope / divergence_power,
        slope=slope,
        D0=float(np.exp(intercept)),
        D_sat=d_sat,
        onset=onset,
        n_fit=int(hi - lo),
        r2=_r2(xt, yt, slope, intercept),
    )


# ---------------------------------------------------------------------------
# (2) Butterfly velocity from the spatial front
# ---------------------------------------------------------------------------


def front_position(
    t: Any,
    D_field: Any,
    positions: Any,
    center: Any,
    box: Any,
    *,
    threshold: float | None = None,
    threshold_frac: float = FRONT_THRESHOLD_FRAC,
    percentile: float = 100.0,
) -> np.ndarray:
    """Front radius ``r_front(t)``: outer distance-from-center of decorrelation.

    At each frame the particles whose divergence ``D_i(t)`` exceeds ``threshold``
    define the decorrelated region; ``r_front`` is the ``percentile`` (default
    outermost) minimum-image distance of those particles from ``center``.  When
    ``threshold`` is None it is ``threshold_frac`` of the whole-field maximum.
    Frames with no decorrelated particle contribute ``r_front = 0``.
    """

    field = as_float_array(D_field)
    if field.ndim != 2:
        raise ValueError("D_field must be a (T, N) per-particle field")
    positions = as_float_array(positions)
    center = as_float_array(center)
    box = as_box(box)
    if field.shape[1] != positions.shape[0]:
        raise ValueError("D_field and positions disagree on N")

    disp = minimum_image(positions - center[None, :], box)
    radius = np.linalg.norm(disp, axis=1)
    thr = threshold_frac * float(np.max(field)) if threshold is None else float(threshold)

    # One (T, N) threshold pass; the default percentile == 100 is exactly the
    # per-frame max of the decorrelated radii (numpy's percentile at q=100
    # returns the largest sorted element bit for bit), so it vectorizes as a
    # masked row max; q == 0 is symmetric.  Other percentiles keep the exact
    # np.percentile call but only visit frames with any decorrelated particle.
    # [micro-bench, T=200, N=2000: front_position 9.2 ms -> 0.6 ms, ~15x]
    r_front = np.zeros(field.shape[0])
    above = field >= thr
    any_above = above.any(axis=1)
    if percentile == 100.0:
        masked = np.where(above, radius[None, :], -np.inf)
        r_front[any_above] = masked.max(axis=1)[any_above]
    elif percentile == 0.0:
        masked = np.where(above, radius[None, :], np.inf)
        r_front[any_above] = masked.min(axis=1)[any_above]
    else:
        for ti in np.nonzero(any_above)[0]:
            r_front[ti] = float(np.percentile(radius[above[ti]], percentile))
    return r_front


@dataclass(frozen=True)
class ButterflyVelocity:
    v_b: float              # front speed (slope of r_front vs t)
    r0: float               # fitted front intercept
    onset: SaturationOnset
    n_fit: int
    r2: float
    r_front: np.ndarray     # the r_front(t) series that was fit


def fit_butterfly_velocity(
    t: Any,
    r_front: Any,
    *,
    slope_frac: float = SLOPE_FRAC_DEFAULT,
    window: int = SLOPE_WINDOW_DEFAULT,
    growth_quantile: float = GROWTH_QUANTILE_DEFAULT,
    min_rise: float = 0.0,
    through_origin: bool = False,
) -> ButterflyVelocity:
    """Fit ``r_front = v_b t + r0`` over the ballistic (pre-saturation) window.

    The front stops advancing once it fills the box; that plateau is detected by
    the same slope-drop onset (applied directly to ``r_front``, which grows
    linearly) so the speed is read only where the cone is still opening.  A
    stationary front returns ``v_b = 0``.
    """

    t = as_float_array(t)
    r = as_float_array(r_front)
    if t.shape != r.shape:
        raise ValueError("t and r_front must have the same length")

    onset = detect_saturation_onset(
        t, r, slope_frac=slope_frac, window=window,
        growth_quantile=growth_quantile, min_rise=min_rise,
    )
    if not onset.growing:
        return ButterflyVelocity(
            v_b=0.0, r0=float(np.median(r)), onset=onset, n_fit=0, r2=float("nan"), r_front=r
        )

    lo, hi = onset.start_index, onset.index
    if hi - lo < 2:
        hi = min(lo + 2, r.size)
    xt, yt = t[lo:hi], r[lo:hi]
    if through_origin:
        denom = float(np.sum(xt * xt))
        slope = float(np.sum(xt * yt) / denom) if denom > 0.0 else float("nan")
        intercept = 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(xt, yt, 1))
    return ButterflyVelocity(
        v_b=slope,
        r0=intercept,
        onset=onset,
        n_fit=int(hi - lo),
        r2=_r2(xt, yt, slope, intercept),
        r_front=r,
    )


def butterfly_velocity(
    t: Any,
    D_field: Any,
    positions: Any,
    center: Any,
    box: Any,
    *,
    threshold: float | None = None,
    threshold_frac: float = FRONT_THRESHOLD_FRAC,
    percentile: float = 100.0,
    slope_frac: float = SLOPE_FRAC_DEFAULT,
    window: int = SLOPE_WINDOW_DEFAULT,
    growth_quantile: float = GROWTH_QUANTILE_DEFAULT,
    min_rise: float = 0.0,
    through_origin: bool = False,
) -> ButterflyVelocity:
    """Convenience: build ``r_front(t)`` from a field, then fit ``v_b``."""

    r_front = front_position(
        t, D_field, positions, center, box,
        threshold=threshold, threshold_frac=threshold_frac, percentile=percentile,
    )
    return fit_butterfly_velocity(
        t, r_front, slope_frac=slope_frac, window=window,
        growth_quantile=growth_quantile, min_rise=min_rise, through_origin=through_origin,
    )


# ---------------------------------------------------------------------------
# (4) Structural channel: causal shielding time
# ---------------------------------------------------------------------------


def crossing_time(t: Any, signal: Any, threshold: float) -> float:
    """First (linearly interpolated) time at which ``signal`` reaches ``threshold``.

    Returns NaN if the signal never crosses.
    """

    t = as_float_array(t)
    s = as_float_array(signal)
    above = s >= threshold
    if not np.any(above):
        return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return float(t[0])
    s0, s1 = float(s[idx - 1]), float(s[idx])
    t0, t1 = float(t[idx - 1]), float(t[idx])
    if s1 == s0:
        return t1
    frac = (float(threshold) - s0) / (s1 - s0)
    return t0 + frac * (t1 - t0)


@dataclass(frozen=True)
class ChannelShielding:
    t_shield: float             # cage-relative (structural) crossing time
    t_onset_raw: float          # raw-displacement crossing time
    shielded: bool              # t_shield > t_onset_raw
    lag: float                  # t_shield - t_onset_raw
    threshold_frac: float
    raw_threshold: float
    structural_threshold: float


def channel_shielding(
    t: Any,
    M_total: Any,
    S_total: Any,
    *,
    threshold_frac: float = SHIELD_THRESHOLD_FRAC,
) -> ChannelShielding:
    """Causal shielding time from the raw vs cage-relative channels.

    Each channel is normalised by its own plateau (its maximum) and crossed at
    ``threshold_frac`` of it, so the two are compared at the same *relative*
    decorrelation level.  ``t_shield`` (the structural / cage-relative crossing)
    exceeding the raw crossing means the timescale structure resists chaos.
    """

    M = _as_series(M_total)
    S = _as_series(S_total)
    m_thr = threshold_frac * float(np.max(M)) if M.size else float("nan")
    s_thr = threshold_frac * float(np.max(S)) if S.size else float("nan")
    t_raw = crossing_time(t, M, m_thr)
    t_shield = crossing_time(t, S, s_thr)
    shielded = bool(np.isfinite(t_shield) and np.isfinite(t_raw) and t_shield > t_raw)
    lag = float(t_shield - t_raw) if np.isfinite(t_shield) and np.isfinite(t_raw) else float("nan")
    return ChannelShielding(
        t_shield=t_shield,
        t_onset_raw=t_raw,
        shielded=shielded,
        lag=lag,
        threshold_frac=threshold_frac,
        raw_threshold=m_thr,
        structural_threshold=s_thr,
    )


# ---------------------------------------------------------------------------
# (5) Intensive-quantity consistency check (NOT an FSS fit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntensiveCheck:
    values: dict[int, float]    # N -> quantity
    mean: float
    rel_spread: float           # (max - min) / |mean| over finite values
    consistent: bool            # rel_spread <= tol
    tol: float
    n_used: int


def intensive_check(
    values_by_N: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    tol: float = INTENSIVE_TOL_DEFAULT,
) -> IntensiveCheck:
    """Check that an intensive quantity is N-independent across configs.

    This is a *consistency* test -- ``lambda``, ``v_b`` and ``t_shield`` should
    not depend on system size -- and deliberately **not** a finite-size-scaling
    fit: it compares the values directly via their relative spread
    ``(max - min) / |mean|`` and flags agreement when that is within ``tol``.
    """

    items = dict(values_by_N) if isinstance(values_by_N, Mapping) else dict(values_by_N)
    values = {int(n): float(v) for n, v in items.items()}
    finite = np.array([v for v in values.values() if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return IntensiveCheck(
            values=values, mean=float("nan"), rel_spread=float("nan"),
            consistent=False, tol=tol, n_used=0,
        )
    mean = float(finite.mean())
    spread = float(finite.max() - finite.min())
    rel_spread = spread / abs(mean) if mean != 0.0 else float("inf")
    consistent = bool(np.isfinite(rel_spread) and rel_spread <= tol)
    return IntensiveCheck(
        values=values, mean=mean, rel_spread=rel_spread,
        consistent=consistent, tol=tol, n_used=int(finite.size),
    )


# ---------------------------------------------------------------------------
# Aggregator: the full two-channel butterfly-cone report for one config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ButterflyReport:
    lyapunov: LyapunovFit
    velocity: ButterflyVelocity | None
    shielding: ChannelShielding | None
    D_sat: float


def analyze_config(
    t: Any,
    M_field: Any,
    *,
    S_field: Any = None,
    positions: Any = None,
    center: Any = None,
    box: Any = None,
    front_threshold: float | None = None,
    front_threshold_frac: float = FRONT_THRESHOLD_FRAC,
    front_percentile: float = 100.0,
    shield_threshold_frac: float = SHIELD_THRESHOLD_FRAC,
    **fit_kwargs: Any,
) -> ButterflyReport:
    """Run the full butterfly-cone analysis for a single config.

    Always fits the Lyapunov growth + plateau from the raw channel ``M_field``.
    The front speed is computed only when ``positions``/``center``/``box`` are
    supplied; the shielding time only when a cage-relative ``S_field`` is given.

    The optional-analysis knobs are all overridable so a robustness sweep can
    thread them through one call (all defaults reproduce the pre-sweep behaviour):

    * ``front_percentile`` -- the front radius percentile (``front_position``);
      100 = outermost decorrelated particle (the default), 90/95 trim the tail.
    * ``front_threshold_frac`` / ``front_threshold`` -- the front decorrelation
      level (fraction of the field maximum, or an absolute level).
    * the saturation-onset fit knobs in ``fit_kwargs`` (``slope_frac``,
      ``window``, ``growth_quantile``) drive BOTH the Lyapunov ``lambda`` fit and
      the ballistic ``v_b`` fit, so a fit-window sweep moves the two together;
      ``min_log_rise``/``divergence_power`` remain Lyapunov-only.
    """

    # M_field is a FIRST-POWER divergence norm (response.divergence_field), so it
    # grows as exp(lambda t): divergence_power=1 unless the caller overrides.
    lyap = fit_lyapunov(t, M_field, divergence_power=fit_kwargs.pop("divergence_power", 1.0), **fit_kwargs)
    velocity: ButterflyVelocity | None = None
    if positions is not None and center is not None and box is not None:
        # Reuse the same onset-fit knobs for the ballistic front so a fit-window
        # sweep moves lambda and v_b together (unset -> the module defaults that
        # butterfly_velocity already used, so the default path is unchanged).
        velocity = butterfly_velocity(
            t, M_field, positions, center, box,
            threshold=front_threshold, threshold_frac=front_threshold_frac,
            percentile=front_percentile,
            slope_frac=fit_kwargs.get("slope_frac", SLOPE_FRAC_DEFAULT),
            window=fit_kwargs.get("window", SLOPE_WINDOW_DEFAULT),
            growth_quantile=fit_kwargs.get("growth_quantile", GROWTH_QUANTILE_DEFAULT),
        )
    shielding: ChannelShielding | None = None
    if S_field is not None:
        shielding = channel_shielding(t, M_field, S_field, threshold_frac=shield_threshold_frac)
    return ButterflyReport(
        lyapunov=lyap, velocity=velocity, shielding=shielding, D_sat=lyap.D_sat
    )
