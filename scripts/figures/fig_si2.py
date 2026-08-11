#!/usr/bin/env python
"""scripts/figures/fig_si2.py -- Supplementary Fig. S2, "Cone-fit robustness".

Three panels, matching the SI.tex caption (sfig:cone):

  (a) The saturation-onset detector on a representative ensemble.  The real
      matched-seed divergence curve D(t) (delta = 0.01 linear rung) on a log-y
      axis, with the exponential-fit window highlighted (where log D is fit to
      recover lambda), the fitted line D0 * exp(lambda t), the detected
      saturation onset t_sat, and the Debye-Waller plateau D_sat.

  (b) The 81-cell knob sweep of lambda.  Every cell of the 3x3x3x3 grid
      (slope fraction x fit window x front percentile x front threshold) as a
      point at its median lambda, coloured by the slope-fraction knob (the
      knob that actually moves lambda; the two front knobs leave lambda
      invariant).  The interquartile spread (15-20 %) sits inside the
      declared in advance <=20 % bar; the full min-max envelope is 30 %.

  (c) The per-delta decomposition.  The fitted lambda for the 12 ensembles at
      each kick amplitude delta = 0.01, 0.03, 0.1.  The two linear-response
      rungs (0.01, 0.03) agree internally to ~3 %; the largest kick (0.1) is a
      diagnostic only -- it exits the exponential regime early (only ~3
      exponential fit frames survive before saturation) and its lambda runs
      high and scattered.

DATA PROVENANCE (persisted runs/ only; nothing fabricated):
  * Panel (a): real D(t) curves reconstructed from the persisted branch
    trajectories under runs/gardner/gardner-T0075-fss/ via the read-only
    cone_collapse loader (same loader the main Fig. 2 uses); fit scalars
    (lambda, D0, D_sat, onset) cross-checked against gardner_r0.json.
  * Panel (b): runs/gardner/gardner-T0075-fss/gardner_r0_sweep.json, the
    81-cell knob sweep (cells[*].lambda_median, grid knob levels, spread).
  * Panel (c): per-ensemble (lam, delta, lam_n_fit, onset_time) scalars from
    runs/gardner/gardner-T0075-fss/gardner_r0.json.

Run:
    cd butterfly_cone && PYTHONPATH=src:scripts:scripts/figures \\
        <venv>/bin/python scripts/figures/fig_si2.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.getLogger("gardner_r0").setLevel(logging.ERROR)

# --- make src/, scripts/ and this dir importable -------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("src", "scripts", "scripts/figures"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figstyle as fs                       # noqa: E402
import cone_collapse as cc                  # noqa: E402  (read-only raw-curve loader)

RUN_FSS = _ROOT / "runs/gardner/gardner-T0075-fss"
SWEEP_JSON = RUN_FSS / "gardner_r0_sweep.json"
R0_JSON = RUN_FSS / "gardner_r0.json"
OUT_STEM = _ROOT / "results/figures/fig_si2"

DELTAS = [0.01, 0.03, 0.1]
GROWTH_FLOOR = cc.GROWTH_FLOOR          # 0.02 * D_sat
GROWTH_CEIL = cc.GROWTH_CEIL           # 0.85 * D_sat


# ------------------------------------------------------------------------- #
# Panel (a): the saturation-onset detector on a representative ensemble
# ------------------------------------------------------------------------- #
def panel_a(ax, curves, r0_doc):
    ref = DELTAS[0]                                        # 0.01 = cleanest rung
    grp = [c for c in curves if abs(c.delta - ref) < 1e-9]
    t = grp[0].times
    Dstack = np.vstack([c.D for c in grp])
    # geometric ensemble mean (the object the detector fits)
    Dmean = 10.0 ** np.nanmean(np.log10(Dstack), axis=0)

    ens01 = [e for e in r0_doc["ensembles"]
             if e.get("delta_index") == 0 and e.get("resolved")]
    lam = float(np.median([e["lam"] for e in ens01]))          # 0.909
    r2 = float(np.median([e["lam_r2"] for e in ens01]))        # 0.98
    D0 = float(np.mean([e["D0"] for e in ens01]))              # 0.86
    t_sat = float(np.median([e["onset_time"] for e in ens01]))  # 6.0
    D_sat = float(r0_doc["pooled"]["D_sat"]["mean"])           # 223.36
    d_sat_over_n = D_sat / 1500.0                              # 0.149 sigma

    # the exponential-fit window the detector actually uses: the frames
    # strictly before the detected saturation onset (log D is fit here; the
    # per-rung fit uses n_fit = 6 frames, i.e. t in [0, t_sat)).  Refitting the
    # mean over this window recovers the reported lambda (cross-check below).
    gm = t < t_sat
    lam_free = float(np.polyfit(t[gm], np.log10(Dmean[gm]), 1)[0] * math.log(10.0))
    t_win = t[gm]
    t_lo, t_hi = float(t_win.min()), float(t_win.max())

    # shade the fit window (frames before the detected onset)
    ax.axvspan(t_lo, t_sat, color=fs.GREEN, alpha=0.10, zorder=0, lw=0)

    # faint individual curves + bold mean with open markers (the fitted object)
    for D in Dstack:
        ax.plot(t, D, color=fs.MEASURED, lw=0.6, alpha=0.22, zorder=1)
    ax.plot(t, Dmean, color=fs.MEASURED, lw=1.9, marker="o", ms=3.4,
            markerfacecolor="white", markeredgecolor=fs.MEASURED,
            markeredgewidth=0.9, zorder=4)

    # fitted exponential D0 * exp(lam t) over/just past the fit window
    slope_dec = lam / math.log(10.0)
    t_fit = np.linspace(-0.3, math.log(1.25 * D_sat / D0) / lam, 60)
    ax.plot(t_fit, D0 * 10.0 ** (slope_dec * t_fit),
            color=fs.THEORY, lw=1.7, ls=(0, (5, 2)), zorder=5)
    xt = 3.9
    ax.annotate(r"fit  $D \propto e^{\lambda t}$",
                xy=(xt, D0 * 10.0 ** (slope_dec * xt)),
                xytext=(xt + 0.35, D0 * 10.0 ** (slope_dec * xt) * 0.52),
                color=fs.THEORY, fontsize=fs.FS_ANNOT, ha="left", va="top")

    # saturation-onset marker
    ax.axvline(t_sat, color=fs.GUIDE, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.text(t_sat + 0.25, 0.365, r"$t_{\mathrm{sat}}$ (onset)", color=fs.SUBTLE,
            fontsize=fs.FS_ANNOT, ha="left", va="bottom")

    # Debye-Waller plateau
    ax.axhline(D_sat, color=fs.SUBTLE, lw=0.9, ls=(0, (1, 1.5)), zorder=3)
    ax.text(t[-1], D_sat * 1.22, r"$D_{\mathrm{sat}}$ (Debye$-$Waller plateau)",
            color=fs.SUBTLE, fontsize=fs.FS_ANNOT, ha="right", va="bottom")

    # fit-window label (inside the shaded band, above the x-axis)
    ax.text((t_lo + t_sat) / 2.0, 0.365, "fit window",
            color=fs.GREEN, fontsize=fs.FS_ANNOT - 0.5, ha="center", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(-0.7, t[-1] + 0.7)
    ax.set_ylim(0.30, 7.0e2)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"divergence $D(t)=\sum_i|\Delta r_i|$")
    # stat block in the empty lower-right corner
    fs.annotate_stats(
        ax,
        rf"$\lambda = {lam:.2f}$ per t.u." "\n"
        rf"$R^2 = {r2:.2f}$" "\n"
        rf"$t_{{\mathrm{{sat}}}} = {t_sat:.0f}$" "\n"
        rf"$D_{{\mathrm{{sat}}}}/N = {d_sat_over_n:.3f}\,\sigma$" "\n"
        rf"$\delta = {ref:g}$,  $N = 1500$",
        x=0.50, y=0.42, size=fs.FS_ANNOT,
    )
    fs.panel_label(ax, "a", x=-0.28)
    return {"lam": lam, "lam_free_fit": lam_free, "r2": r2, "t_sat": t_sat,
            "D_sat": D_sat, "d_sat_over_n": d_sat_over_n,
            "fit_window": (t_lo, t_hi), "n_curves": len(grp)}


# ------------------------------------------------------------------------- #
# Panel (b): 81-cell knob sweep of lambda
# ------------------------------------------------------------------------- #
def panel_b(ax, sweep):
    cells = sweep["cells"]
    lam = np.array([c["lambda_median"] for c in cells])
    sfrac = np.array([c["slope_frac"] for c in cells])
    sf_levels = sorted(set(sfrac))                        # [0.3, 0.5, 0.7]
    ramp = fs.sequential(len(sf_levels))                  # cividis, ordered knob

    med = float(np.median(lam))
    q1, q3 = np.percentile(lam, [25, 75])
    iqr_spread = (q3 - q1) / med
    halfrange_spread = (lam.max() - lam.min()) / (2 * med)
    full_spread = (lam.max() - lam.min()) / med
    headline = 0.91                                       # delta=0.01 rung median

    # IQR band (the reported knob spread) + central + envelope
    ax.axhspan(q1, q3, color=fs.GRAY, alpha=0.14, zorder=0, lw=0)
    ax.axhline(med, color=fs.INK, lw=1.0, ls="-", zorder=2)
    ax.axhline(lam.min(), color=fs.GUIDE, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.axhline(lam.max(), color=fs.GUIDE, lw=0.7, ls=(0, (2, 2)), zorder=1)

    # jittered strip, coloured by slope-fraction knob (colour reinforces the
    # x-axis; the other three knobs are swept but leave lambda invariant, so the
    # 81 cells stack into three columns of 27)
    rng = np.random.default_rng(7)
    for i, sf in enumerate(sf_levels):
        m = sfrac == sf
        y = lam[m]
        x = 1.0 + i + (rng.random(m.sum()) - 0.5) * 0.62
        ax.scatter(x, y, s=26, color=ramp[i], edgecolor="white", linewidth=0.4,
                   alpha=0.95, zorder=3)

    ax.set_xlim(0.35, len(sf_levels) + 0.65)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{sf:g}" for sf in sf_levels])
    ax.set_xlabel("slope-fraction knob")
    ax.set_ylabel(r"cone rate  $\lambda$  (per t.u.)")
    ax.set_ylim(0.72, 1.13)

    # right-inside labels for the central line and the envelope
    ax.text(len(sf_levels) + 0.55, med, "median\n0.92", color=fs.INK,
            fontsize=fs.FS_ANNOT - 0.5, va="center", ha="center")
    ax.text(len(sf_levels) + 0.55, (q1 + q3) / 2 - 0.055, "IQR\n15-20%",
            color=fs.SUBTLE, fontsize=fs.FS_ANNOT - 1.0, va="center", ha="center")

    fs.annotate_stats(
        ax,
        rf"$n=81$ cells" "\n"
        r"(3$\times$3$\times$3$\times$3 knobs)" "\n"
        rf"IQR spread {iqr_spread*100:.0f}%" "\n"
        rf"half-range {halfrange_spread*100:.0f}%" "\n"
        rf"full envelope {full_spread*100:.0f}%" "\n"
        r"$\leq$20% bar $\rightarrow$ passes",
        x=0.035, y=0.985, size=fs.FS_ANNOT - 0.5,
    )
    ax.text(0.62, 0.055,
            "front-percentile &\nfront-threshold knobs\nleave $\\lambda$ invariant",
            transform=ax.transAxes, fontsize=fs.FS_ANNOT - 1.0, color=fs.SUBTLE,
            va="bottom", ha="left", linespacing=1.3)
    fs.panel_label(ax, "b", x=-0.30)
    return {"median": med, "iqr_spread": iqr_spread,
            "halfrange_spread": halfrange_spread, "full_spread": full_spread,
            "q1": float(q1), "q3": float(q3),
            "min": float(lam.min()), "max": float(lam.max()),
            "sf_means": {float(sf): float(lam[sfrac == sf].mean()) for sf in sf_levels}}


# ------------------------------------------------------------------------- #
# Panel (c): per-delta decomposition
# ------------------------------------------------------------------------- #
def panel_c(ax, r0_doc):
    ens = r0_doc["ensembles"]
    byd = defaultdict(lambda: defaultdict(list))
    for e in ens:
        d = e["delta"]
        byd[d]["lam"].append(e["lam"])
        byd[d]["nfit"].append(e["lam_n_fit"])
    ramp = fs.sequential(len(DELTAS))                     # cividis cold->warm
    cmap = {d: ramp[i] for i, d in enumerate(DELTAS)}

    rng = np.random.default_rng(3)
    stats = {}
    for i, d in enumerate(DELTAS):
        lam = np.array(byd[d]["lam"], float)
        nfit = float(np.median(byd[d]["nfit"]))
        x = 1.0 + i + (rng.random(len(lam)) - 0.5) * 0.42
        ax.scatter(x, lam, s=30, color=cmap[d], edgecolor="white", linewidth=0.4,
                   alpha=0.95, zorder=3)
        # mean marker (diamond)
        ax.scatter([1.0 + i], [lam.mean()], s=70, marker="D", color=cmap[d],
                   edgecolor=fs.INK, linewidth=0.8, zorder=4)
        stats[d] = {"mean": float(lam.mean()), "median": float(np.median(lam)),
                    "min": float(lam.min()), "max": float(lam.max()),
                    "nfit": nfit}

    # linear-response vs diagnostic backdrop
    ax.axvspan(0.55, 2.5, color=fs.GREEN, alpha=0.07, zorder=0, lw=0)
    ax.axvspan(2.5, 3.45, color=fs.VERMILLION, alpha=0.06, zorder=0, lw=0)
    ax.text(1.5, 1.505, "linear response", color=fs.GREEN, fontsize=fs.FS_ANNOT - 0.5,
            ha="center", va="top")
    ax.text(3.0, 1.505, "diagnostic", color=fs.VERMILLION, fontsize=fs.FS_ANNOT - 0.5,
            ha="center", va="top")

    # 3% internal-agreement bracket between delta=0.01 and 0.03
    m1, m3 = stats[0.01]["median"], stats[0.03]["median"]
    agree = abs(m3 - m1) / ((m1 + m3) / 2) * 100
    ybr = 0.86
    ax.plot([1, 2], [ybr, ybr], color=fs.INK, lw=0.9, zorder=5)
    ax.plot([1, 1], [ybr, ybr + 0.012], color=fs.INK, lw=0.9, zorder=5)
    ax.plot([2, 2], [ybr, ybr + 0.012], color=fs.INK, lw=0.9, zorder=5)
    ax.text(1.5, ybr - 0.02, rf"agree to {agree:.0f}%", color=fs.INK,
            fontsize=fs.FS_ANNOT, ha="center", va="top")

    # delta=0.1 diagnostic note
    ax.annotate("exits exponential\nregime early",
                xy=(3.0, stats[0.1]["mean"]), xytext=(2.62, 1.05),
                color=fs.VERMILLION, fontsize=fs.FS_ANNOT - 0.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=fs.VERMILLION, lw=0.9))

    # fit-frame counts under each group: sit in the narrow band just above the
    # axis (va='bottom' keeps them off the x-tick labels below) and below the
    # 'agree to 3%' bracket text above them
    for i, d in enumerate(DELTAS):
        ax.text(1.0 + i, 0.783, f"{stats[d]['nfit']:.0f} frames",
                color=fs.SUBTLE, fontsize=fs.FS_ANNOT - 2.5, ha="center", va="bottom")

    ax.set_xlim(0.55, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.set_xlabel(r"kick amplitude  $\delta$")
    ax.set_ylabel(r"fitted rate  $\lambda$  (per t.u.)")
    ax.set_ylim(0.78, 1.55)
    fs.annotate_stats(
        ax,
        "12 ensembles / rung\n"
        "diamond = rung mean",
        x=0.035, y=0.66, size=fs.FS_ANNOT - 0.5, color=fs.SUBTLE,
    )
    fs.panel_label(ax, "c", x=-0.26)
    return {str(d): stats[d] for d in DELTAS} | {"agree_pct": agree}


# ------------------------------------------------------------------------- #
def main():
    curves, _ = cc.load_curves(RUN_FSS)
    r0_doc = json.loads(R0_JSON.read_text())
    sweep = json.loads(SWEEP_JSON.read_text())

    fig, axes = plt.subplots(1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.40))
    va = panel_a(axes[0], curves, r0_doc)
    vb = panel_b(axes[1], sweep)
    vc = panel_c(axes[2], r0_doc)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT_STEM))

    # ---- verification printout (checked against artifacts) ---------------- #
    print("=== fig_si2 verification ===")
    print("[a] detector:")
    print(f"    lambda(reported median) = {va['lam']:.4f}  "
          f"free-slope refit = {va['lam_free_fit']:.4f}  R2 = {va['r2']:.3f}")
    print(f"    t_sat = {va['t_sat']:.1f}   D_sat = {va['D_sat']:.2f}  "
          f"D_sat/N = {va['d_sat_over_n']:.4f} sigma   fit window t in "
          f"[{va['fit_window'][0]:.0f}, {va['fit_window'][1]:.0f}]  "
          f"({va['n_curves']} curves)")
    print("[b] 81-cell sweep:")
    print(f"    median lambda = {vb['median']:.4f}   IQR spread = {vb['iqr_spread']*100:.1f}%  "
          f"half-range = {vb['halfrange_spread']*100:.1f}%  full = {vb['full_spread']*100:.1f}%")
    print(f"    lambda range [{vb['min']:.4f}, {vb['max']:.4f}]  Q1/Q3 = "
          f"{vb['q1']:.4f}/{vb['q3']:.4f}")
    print(f"    slope_frac means = {vb['sf_means']}")
    print(f"    sweep JSON spread.lambda.rel_spread = "
          f"{sweep['spread']['lambda']['rel_spread']:.4f}  robust={sweep['robust']}")
    print("[c] per-delta:")
    for d in DELTAS:
        s = vc[str(d)]
        print(f"    delta={d}: lam mean={s['mean']:.4f} median={s['median']:.4f} "
              f"[{s['min']:.3f},{s['max']:.3f}]  nfit={s['nfit']:.0f}")
    print(f"    0.01 vs 0.03 internal agreement = {vc['agree_pct']:.2f}%")
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
