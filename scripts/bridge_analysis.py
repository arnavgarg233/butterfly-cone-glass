#!/usr/bin/env python
"""scripts/bridge_analysis.py -- THE chaos-entropy bridge result.

Pure re-analysis (no MD) of the ``bridge-Tladder`` butterfly-cone campaign under
``runs/gardner/``.  The campaign froze a single glass former down a temperature
ladder (5 temperatures) and, at each temperature, measured the deterministic
butterfly cone from matched-seed perturbation ensembles.  Independently, the
configurational entropy ``s_c(T)`` of the *same* system was measured by the
Frenkel--Ladd / thermodynamic-integration ladder in ``runs/sc_energy_ladder/``.

THE QUESTION -- the potential unification of the paper: does *chaos* (the cone
observables lambda, v_b, D_sat/N) track *configurational entropy* s_c across T?
The co-freeze prediction is that chaos and entropy freeze together, so on cooling
(s_c decreasing) every chaos observable DECREASES with it -- a monotone, positive
lambda(s_c) relationship.  The alternative is decoupling: chaos survives (or even
grows) while the entropy collapses toward the Kauzmann point.  Either way is a
finding; this script reports it honestly.

Pipeline
--------
1. Reuse the SAME cone estimators as ``scripts/gardner_r0.py`` /
   ``src/butterfly_cone/perturb/butterfly.py`` (imported read-only): ``collect_fields``
   loads every ``bridge-Tladder`` ensemble and integrates the two divergence
   channels once; ``result_from_fields`` reads lambda / v_b / D_sat from the
   pre-saturation window.
2. Keep only the linear-response kicks (deltas 0.01, 0.03; the violator
   delta 0.1 is excluded from the campaign and defensively re-excluded here).
3. Pool the 6 perturbation ensembles of each config into one
   (lambda, v_b, D_sat/N) triple per temperature.
4. Correlate each observable against s_c(T) across the 5 temperatures:
   Spearman rho + p (n=5; a perfect monotone gives p ~= 0.0083) and a linear
   fit ``observable = a * s_c + b``.  Flag monotonicity and any kink near the
   mode-coupling crossover T_MCT = 0.108.
5. Write ``runs/bridge_analysis/bridge.png`` (chaos observables vs s_c) and
   ``runs/bridge_analysis/bridge_analysis.json``.

The aggregation + correlation logic (steps 3--4) are pure functions so the tests
exercise them on synthetic data plus a real-data smoke.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gardner_r0 as gr0  # noqa: E402  (read-only reuse of the cone estimators)

# ---------------------------------------------------------------------------
# Landed constants: config -> temperature.  s_c(T) is read from sc_curve.json.
# ---------------------------------------------------------------------------

#: The bridge campaign config index -> temperature (from
#: runs/gardner/bridge-Tladder/config.yaml `values.configs`).
CONFIG_TEMPERATURE: dict[int, float] = {
    0: 0.150,
    1: 0.130,
    2: 0.108,
    3: 0.090,
    4: 0.075,
}

#: Mode-coupling crossover temperature of the ladder (campaign `temperature`).
T_MCT = 0.108

#: The linear-response kicks retained; the violator kick is excluded.
VIOLATOR_DELTA = 0.1

DEFAULT_RUN_DIR = _ROOT / "runs" / "gardner" / "bridge-Tladder"
DEFAULT_SC_CURVE = _ROOT / "runs" / "sc_energy_ladder" / "sc_curve.json"
DEFAULT_OUT_DIR = _ROOT / "runs" / "bridge_analysis"


# ---------------------------------------------------------------------------
# s_c(T) from the Frenkel--Ladd / TI ladder
# ---------------------------------------------------------------------------


def sc_by_temperature(sc_curve_path: Path) -> dict[float, float]:
    """Mean configurational entropy per particle ``s_c`` at each temperature.

    ``sc_curve.json`` carries one record per (temperature, replica); the
    per-temperature ``s_c`` is the mean of ``s_configurational`` over replicas
    -- the same convention the ladder itself reports.
    """

    data = json.loads(Path(sc_curve_path).read_text())
    by_t: dict[float, list[float]] = {}
    for rec in data.get("records", []):
        t = round(float(rec["temperature"]), 4)
        by_t.setdefault(t, []).append(float(rec["s_configurational"]))
    return {t: float(np.mean(vs)) for t, vs in by_t.items()}


def sc_for_config(config: int, sc_map: Mapping[float, float]) -> float:
    """s_c for a config index via its temperature (nearest-temperature match)."""

    temp = CONFIG_TEMPERATURE[config]
    if temp in sc_map:
        return sc_map[temp]
    # tolerant lookup (float keys)
    best = min(sc_map, key=lambda k: abs(k - temp))
    if abs(best - temp) > 1e-3:
        raise KeyError(f"no s_c near T={temp} in curve (closest {best})")
    return sc_map[best]


# ---------------------------------------------------------------------------
# Pure aggregation: pool a config's ensembles into one observable triple
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.array([float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def _mean_std_sem(values: Sequence[float]) -> tuple[float, float, float, int]:
    """(mean, sample-std, standard-error-of-mean, n) over finite values."""

    arr = _finite(values)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, sem, n


@dataclass(frozen=True)
class ConfigAggregate:
    """One temperature's pooled chaos observables."""

    config: int
    temperature: float
    s_c: float
    N: int
    n_ensembles: int
    n_resolved: int
    frac_resolved: float
    lam_mean: float
    lam_std: float
    lam_sem: float
    lam_used: str            # "resolved" | "all-growing" | "all"
    v_b_mean: float
    v_b_std: float
    v_b_sem: float
    d_sat_mean: float        # raw plateau (extensive)
    d_sat_std: float
    d_sat_per_N_mean: float  # D_sat / N (intensive)
    d_sat_per_N_std: float
    d_sat_per_N_sem: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def aggregate_config(
    config: int,
    temperature: float,
    s_c: float,
    results: Sequence[Any],
) -> ConfigAggregate:
    """Pool a config's ensemble results into one observable triple.

    ``results`` is any sequence of objects exposing ``lam``, ``v_b``, ``D_sat``,
    ``N`` and ``resolved`` (e.g. :class:`gardner_r0.EnsembleResult`).  lambda is
    pooled over the *resolved* ensembles (clean log-linear cone, r2 >= floor); if
    none resolve we fall back to all growing ensembles (``lam > 0``), and finally
    to all ensembles -- always flagging which set was used.  v_b and D_sat pool
    over every ensemble (they need no resolution gate).  D_sat/N is the intensive
    plateau.
    """

    results = list(results)
    n_ens = len(results)
    N = int(results[0].N) if results else 0

    resolved = [r for r in results if bool(getattr(r, "resolved", False))]
    n_resolved = len(resolved)
    growing = [r for r in results if np.isfinite(r.lam) and r.lam > 0.0]
    if resolved:
        lam_src, lam_used = resolved, "resolved"
    elif growing:
        lam_src, lam_used = growing, "all-growing"
    else:
        lam_src, lam_used = results, "all"

    lam_mean, lam_std, lam_sem, _ = _mean_std_sem([r.lam for r in lam_src])
    v_b_mean, v_b_std, v_b_sem, _ = _mean_std_sem([r.v_b for r in results])
    d_sat_mean, d_sat_std, _, _ = _mean_std_sem([r.D_sat for r in results])
    per_n = [r.D_sat / r.N for r in results if r.N]
    d_sat_per_N_mean, d_sat_per_N_std, d_sat_per_N_sem, _ = _mean_std_sem(per_n)

    return ConfigAggregate(
        config=config, temperature=temperature, s_c=s_c, N=N,
        n_ensembles=n_ens, n_resolved=n_resolved,
        frac_resolved=(n_resolved / n_ens) if n_ens else float("nan"),
        lam_mean=lam_mean, lam_std=lam_std, lam_sem=lam_sem, lam_used=lam_used,
        v_b_mean=v_b_mean, v_b_std=v_b_std, v_b_sem=v_b_sem,
        d_sat_mean=d_sat_mean, d_sat_std=d_sat_std,
        d_sat_per_N_mean=d_sat_per_N_mean, d_sat_per_N_std=d_sat_per_N_std,
        d_sat_per_N_sem=d_sat_per_N_sem,
    )


# ---------------------------------------------------------------------------
# Pure correlation: chaos observable vs s_c across temperature
# ---------------------------------------------------------------------------


def exact_spearman_p(x: Sequence[float], y: Sequence[float], rho: float) -> float:
    """Exact one-sided permutation p-value for Spearman rho (small n).

    For n=5 the asymptotic t-approximation scipy returns is meaningless (it
    reports p ~ 1e-24 for a perfect ordering); the honest statement for so few
    points is the exact permutation test.  We enumerate all ``n!`` relabelings of
    ``y``, recompute rho, and report the fraction at least as extreme as the
    observed rho *in its observed direction* (one-sided) -- so a perfect n=5
    monotone gives 1/120 ~= 0.0083, matching the advance declaration.  Falls back to the
    asymptotic value for n too large to enumerate cheaply (n > 8).
    """

    from itertools import permutations

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2 or not np.isfinite(rho) or rho == 0.0:
        return float("nan")
    if n > 8:
        _, p = stats.spearmanr(x, y)
        return float(p)
    count = 0
    total = 0
    for perm in permutations(range(n)):
        r, _ = stats.spearmanr(x, y[list(perm)])
        total += 1
        if rho > 0 and r >= rho - 1e-12:
            count += 1
        elif rho < 0 and r <= rho + 1e-12:
            count += 1
    return count / total


@dataclass(frozen=True)
class Correlation:
    name: str
    n: int
    x: list[float]           # s_c
    y: list[float]           # observable
    spearman_rho: float
    spearman_p: float        # exact one-sided permutation p (small n)
    spearman_p_asymptotic: float
    pearson_r: float
    slope: float             # linear fit y = slope * s_c + intercept
    intercept: float
    fit_r2: float
    monotone: bool           # |rho| == 1 (strictly monotone ordering)
    direction: str           # "increasing" | "decreasing" | "flat" | "n/a"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def correlate(name: str, s_c: Sequence[float], observable: Sequence[float]) -> Correlation:
    """Spearman rho+p and a linear fit of ``observable`` against ``s_c``.

    Points with a non-finite value on either axis are dropped in lockstep.  With
    a perfect monotone ordering of n=5 points Spearman gives rho=+/-1 at
    p ~= 0.0083.  ``monotone`` is True iff |rho| == 1 (the co-freeze prediction
    is a positive, monotone-increasing lambda(s_c)).
    """

    x = np.asarray(s_c, dtype=float)
    y = np.asarray(observable, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 3:
        return Correlation(
            name=name, n=n, x=list(map(float, x)), y=list(map(float, y)),
            spearman_rho=float("nan"), spearman_p=float("nan"),
            spearman_p_asymptotic=float("nan"),
            pearson_r=float("nan"), slope=float("nan"), intercept=float("nan"),
            fit_r2=float("nan"), monotone=False, direction="n/a",
        )

    rho, p_asym = stats.spearmanr(x, y)
    rho = float(rho)
    p = exact_spearman_p(x, y, rho)
    pear = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    monotone = bool(np.isfinite(rho) and abs(abs(rho) - 1.0) < 1e-9)
    if not np.isfinite(rho) or rho == 0.0:
        direction = "flat"
    elif rho > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return Correlation(
        name=name, n=n, x=list(map(float, x)), y=list(map(float, y)),
        spearman_rho=rho, spearman_p=float(p), spearman_p_asymptotic=float(p_asym),
        pearson_r=pear, slope=slope, intercept=intercept, fit_r2=fit_r2,
        monotone=monotone, direction=direction,
    )


def detect_kink(aggs: Sequence[ConfigAggregate], attr: str, near_T: float = T_MCT) -> dict[str, Any]:
    """Flag a slope-sign change (kink) in ``attr`` vs s_c, and locate it.

    The configs are ordered by s_c (cooling = decreasing s_c).  A kink is a sign
    change of consecutive first differences dy = y[i+1]-y[i]; the routine reports
    whether the sequence is monotone, the index/temperature of any turning point,
    and whether the nearest turning point sits at the config closest to
    ``near_T`` (the MCT crossover).
    """

    ordered = sorted(aggs, key=lambda a: a.s_c)
    ys = np.array([getattr(a, attr) for a in ordered], dtype=float)
    temps = [a.temperature for a in ordered]
    diffs = np.diff(ys)
    signs = np.sign(diffs)
    nz = signs[signs != 0]
    monotone = bool(nz.size == 0 or np.all(nz == nz[0]))
    turning: list[dict[str, Any]] = []
    for i in range(1, len(diffs)):
        if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
            # turning point at interior node i (between diffs i-1 and i)
            turning.append({"index": i, "temperature": temps[i], "s_c": ordered[i].s_c,
                            "value": float(ys[i])})
    mct_config = min(CONFIG_TEMPERATURE, key=lambda c: abs(CONFIG_TEMPERATURE[c] - near_T))
    mct_temp = CONFIG_TEMPERATURE[mct_config]
    kink_at_mct = any(abs(t["temperature"] - mct_temp) < 1e-6 for t in turning)
    return {
        "attr": attr, "monotone": monotone,
        "turning_points": turning, "kink_at_T_MCT": kink_at_mct,
        "T_MCT": mct_temp,
    }


# ---------------------------------------------------------------------------
# Heavy path: load the cone fields and build per-config aggregates
# ---------------------------------------------------------------------------


def build_aggregates(
    run_dir: Path,
    sc_map: Mapping[float, float],
    *,
    max_branches: int | None = None,
    r2_resolved: float = gr0.R2_RESOLVED_DEFAULT,
) -> tuple[list[ConfigAggregate], dict[str, Any]]:
    """Load every bridge ensemble, filter deltas, and pool per config.

    Returns ``(aggregates, meta)`` where ``meta`` carries the per-ensemble table
    and the delta bookkeeping so the JSON record is self-contained.
    """

    cache = gr0.collect_fields(run_dir, max_branches=max_branches)

    kept: dict[int, list[Any]] = {}
    per_ensemble: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for fld in cache.fields:
        delta = fld.delta
        if delta is not None and abs(delta - VIOLATOR_DELTA) < 1e-9:
            excluded.append({"label": fld.label, "delta": delta, "reason": "violator delta"})
            continue
        res = gr0.result_from_fields(fld, r2_resolved=r2_resolved)
        kept.setdefault(fld.config, []).append(res)
        per_ensemble.append({
            "config": fld.config, "label": fld.label, "delta": delta,
            "N": res.N, "n_branches": res.n_branches,
            "lam": res.lam, "lam_r2": res.lam_r2, "resolved": res.resolved,
            "growing": res.growing, "saturated": res.saturated,
            "v_b": res.v_b, "D_sat": res.D_sat, "onset_time": res.onset_time,
        })

    aggregates: list[ConfigAggregate] = []
    for cfg in sorted(kept):
        temp = CONFIG_TEMPERATURE[cfg]
        s_c = sc_for_config(cfg, sc_map)
        aggregates.append(aggregate_config(cfg, temp, s_c, kept[cfg]))

    meta = {
        "run_dir": str(run_dir),
        "n_ensembles_analyzed": len(per_ensemble),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "r2_resolved": r2_resolved,
        "per_ensemble": sorted(per_ensemble, key=lambda d: (d["config"], d["label"])),
        "skipped_load": cache.skipped,
    }
    return aggregates, meta


def bridge_report(aggregates: Sequence[ConfigAggregate]) -> dict[str, Any]:
    """The correlations + kink diagnostics for the three chaos observables."""

    aggs = sorted(aggregates, key=lambda a: a.config)
    s_c = [a.s_c for a in aggs]
    correlations = {
        "lambda": correlate("lambda", s_c, [a.lam_mean for a in aggs]),
        "v_b": correlate("v_b", s_c, [a.v_b_mean for a in aggs]),
        "D_sat_per_N": correlate("D_sat_per_N", s_c, [a.d_sat_per_N_mean for a in aggs]),
    }
    kinks = {
        "lambda": detect_kink(aggs, "lam_mean"),
        "v_b": detect_kink(aggs, "v_b_mean"),
        "D_sat_per_N": detect_kink(aggs, "d_sat_per_N_mean"),
    }

    # Verdict: chaos is controlled by entropy iff all three observables track
    # s_c monotonically in the SAME (positive) direction -- the co-freeze
    # signature.  Otherwise report decoupling.
    lam_c = correlations["lambda"]
    all_pos_monotone = all(
        c.monotone and c.direction == "increasing" for c in correlations.values()
    )
    lambda_tracks = bool(lam_c.monotone and lam_c.direction == "increasing")
    if all_pos_monotone:
        verdict = "chaos controlled by entropy (co-freeze): all observables increase monotonically with s_c"
    elif lambda_tracks:
        verdict = "partial: lambda co-freezes with s_c, but not all observables track monotonically"
    else:
        verdict = "decoupled: chaos does NOT track configurational entropy monotonically"

    return {
        "correlations": {k: v.to_dict() for k, v in correlations.items()},
        "kinks": kinks,
        "co_freeze_all_observables": all_pos_monotone,
        "lambda_tracks_entropy": lambda_tracks,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(aggregates: Sequence[ConfigAggregate], report: dict[str, Any], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggs = sorted(aggregates, key=lambda a: a.s_c)
    s_c = np.array([a.s_c for a in aggs])
    temps = [a.temperature for a in aggs]
    sc_mct = CONFIG_TEMPERATURE[min(CONFIG_TEMPERATURE, key=lambda c: abs(CONFIG_TEMPERATURE[c] - T_MCT))]
    sc_mct = next(a.s_c for a in aggs if abs(a.temperature - T_MCT) < 1e-6)

    panels = [
        ("lambda", "lam_mean", "lam_sem", r"$\lambda$  (Lyapunov rate) [1/t.u.]"),
        ("v_b", "v_b_mean", "v_b_sem", r"$v_b$  (butterfly velocity) [$\sigma$/t.u.]"),
        ("D_sat_per_N", "d_sat_per_N_mean", "d_sat_per_N_sem", r"$D_{\rm sat}/N$  (plateau) [$\sigma$]"),
    ]
    # Colorblind-safe sequential-by-temperature colouring (viridis).
    cmap = plt.get_cmap("viridis")
    tmin, tmax = min(temps), max(temps)
    colors = [cmap(0.1 + 0.8 * (t - tmin) / (tmax - tmin)) for t in temps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, mattr, sattr, ylabel) in zip(axes, panels):
        y = np.array([getattr(a, mattr) for a in aggs])
        yerr = np.array([getattr(a, sattr) for a in aggs])
        corr = report["correlations"][name]

        # MCT crossover marker
        ax.axvline(sc_mct, color="0.6", ls=":", lw=1.2, zorder=0)
        ax.text(sc_mct, 0.98, r"  $T_{\rm MCT}$", transform=ax.get_xaxis_transform(),
                va="top", ha="left", color="0.4", fontsize=9)

        # linear fit line
        if np.isfinite(corr["slope"]):
            xs = np.linspace(s_c.min(), s_c.max(), 50)
            ax.plot(xs, corr["slope"] * xs + corr["intercept"], "-",
                    color="0.35", lw=1.4, zorder=1,
                    label=f"fit: {corr['slope']:.3g}·s_c+{corr['intercept']:.3g}")

        ax.errorbar(s_c, y, yerr=yerr, fmt="none", ecolor="0.5", elinewidth=1,
                    capsize=3, zorder=2)
        ax.scatter(s_c, y, c=colors, s=90, edgecolor="k", linewidth=0.6, zorder=3)
        for a, yi in zip(aggs, y):
            ax.annotate(f"T={a.temperature:g}", (a.s_c, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="0.25")

        ax.set_xlabel(r"configurational entropy  $s_c(T)$")
        ax.set_ylabel(ylabel)
        rho, p = corr["spearman_rho"], corr["spearman_p"]
        ax.set_title(f"{ylabel.split('  ')[0]}  vs  $s_c$\n"
                     rf"Spearman $\rho$={rho:+.3f}, p={p:.4f}", fontsize=10)
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)

    verdict = report["verdict"]
    fig.suptitle("Chaos--entropy bridge: butterfly-cone observables vs configurational entropy across T\n"
                 f"cooling: s_c 2.73 -> 1.68  |  {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    run_dir: Path = DEFAULT_RUN_DIR,
    sc_curve: Path = DEFAULT_SC_CURVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_branches: int | None = None,
) -> dict[str, Any]:
    sc_map = sc_by_temperature(sc_curve)
    aggregates, meta = build_aggregates(run_dir, sc_map, max_branches=max_branches)
    report = bridge_report(aggregates)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "bridge.png"
    make_figure(aggregates, report, out_png)

    record = {
        "meta": meta,
        "config_temperature": CONFIG_TEMPERATURE,
        "s_c_by_temperature": {str(k): v for k, v in sc_map.items()},
        "per_config": [a.to_dict() for a in sorted(aggregates, key=lambda a: a.config)],
        **report,
        "figure": str(out_png),
    }
    (out_dir / "bridge_analysis.json").write_text(json.dumps(record, indent=2))
    return record


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{nd}f}"


def print_report(record: dict[str, Any]) -> None:
    print("=" * 78)
    print("CHAOS-ENTROPY BRIDGE  (bridge-Tladder cone campaign, linear-response deltas)")
    print("=" * 78)
    print(f"{'T':>7} {'s_c':>7} {'lambda':>10} {'v_b':>10} {'D_sat/N':>10} {'res':>6}")
    for a in record["per_config"]:
        print(f"{a['temperature']:>7g} {a['s_c']:>7.3f} "
              f"{_fmt(a['lam_mean']):>10} {_fmt(a['v_b_mean']):>10} "
              f"{_fmt(a['d_sat_per_N_mean']):>10} "
              f"{a['n_resolved']}/{a['n_ensembles']:>1}")
    print("-" * 78)
    for name in ("lambda", "v_b", "D_sat_per_N"):
        c = record["correlations"][name]
        k = record["kinks"][name]
        print(f"{name:>12}: Spearman rho={c['spearman_rho']:+.4f}  p={c['spearman_p']:.4f}  "
              f"fit slope={_fmt(c['slope'],3)}  monotone={c['monotone']} ({c['direction']})  "
              f"kink@MCT={k['kink_at_T_MCT']}")
    print("-" * 78)
    print("VERDICT:", record["verdict"])
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--sc-curve", type=Path, default=DEFAULT_SC_CURVE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-branches", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    record = run(args.run_dir, args.sc_curve, args.out_dir, max_branches=args.max_branches)
    print_report(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
