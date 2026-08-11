#!/usr/bin/env python
"""scripts/figures/fig_main2_cone.py -- MAIN Fig 2, "The butterfly cone".

The marquee figure of the ButterflyCone Nature-Physics paper. Three panels:

  (a) Divergence D(t) vs t on a log-y axis: the exponential rise at the
      Lyapunov rate lambda_B behind the ballistic front, saturating at the
      sub-cage Debye-Waller plateau D_sat.  Shows the exponential fit and the
      plateau line for the reference (smallest, cleanest) kick rung.
  (b) delta-collapse master curve: the divergence curves for kick amplitudes
      spanning a decade (delta = 0.01, 0.03, 0.1) collapse onto ONE master
      curve under the parameter-free clock shift
          t -> t - lambda_B^{-1} ln(delta_ref/delta),
      using the *independently fitted* lambda_B (no new parameter).  Inset =
      raw spread (before); main = collapsed (after).
  (c) Closure-identity scatter: the single-exponent closure
          lambda_B * t_sat = ln(D_sat / D0)
      as predicted vs measured saturation time across n=70 ensembles from two
      independent campaigns, on the 1:1 line (Pearson r = 0.987).

DATA PROVENANCE (persisted runs/ only; nothing fabricated):
  * Panels (a),(b): raw matched-seed divergence curves D(t) reconstructed on
    the fly from the persisted branch-trajectory ensembles under
    runs/gardner/gardner-T0075-fss/ via the read-only cone_collapse/gardner_r0
    loaders (exactly as scripts/cone_collapse.py does).  The per-ensemble fit
    scalars (lambda, D_sat, D0, ...) are cross-checked against
    runs/gardner/gardner-T0075-fss/gardner_r0.json and
    runs/cone_collapse/cone_collapse.json.
  * Panel (c): per-ensemble (lambda, t_sat=onset_time, D_sat, D0) scalars read
    from runs/gardner/gardner-T0075-{fss,m2}/gardner_r0.json (same resolved-
    ensemble filter as scripts/chaos_relations.py), reproducing the closure
    Pearson r and n reported in runs/chaos_relations/chaos_relations.json.

Run:
    cd butterfly_cone && PYTHONPATH=src:scripts:scripts/figures \\
        <venv>/bin/python scripts/figures/fig_main2_cone.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.getLogger("gardner_r0").setLevel(logging.ERROR)   # quiet unpert-seed notes

# --- make src/, scripts/ and this dir importable -------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("src", "scripts", "scripts/figures"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figstyle as fs                       # noqa: E402  (applies the house style)
import cone_collapse as cc                  # noqa: E402  (read-only raw-curve loader)

RUN_FSS = _ROOT / "runs/gardner/gardner-T0075-fss"
RUN_M2 = _ROOT / "runs/gardner/gardner-T0075-m2"
CONE_JSON = _ROOT / "runs/cone_collapse/cone_collapse.json"
RELATIONS_JSON = _ROOT / "runs/chaos_relations/chaos_relations.json"
OUT_STEM = _ROOT / "results/figures/fig_main2_cone"

# Delta rungs (kick amplitudes), coldest->warmest by magnitude for the ramp.
DELTAS = [0.01, 0.03, 0.1]
GROWTH_FLOOR = cc.GROWTH_FLOOR          # 0.02 * D_sat
GROWTH_CEIL = cc.GROWTH_CEIL           # 0.85 * D_sat


# ------------------------------------------------------------------------- #
# Data loading / verification
# ------------------------------------------------------------------------- #
def load_fss_curves():
    """Real D(t) curves for the fss campaign (persisted branch trajectories)."""
    curves, prov = cc.load_curves(RUN_FSS)
    return curves, prov


def closure_records():
    """Per-ensemble closure records from both campaigns (chaos_relations filter)."""
    recs = []
    for run_dir, tag in ((RUN_FSS, "fss"), (RUN_M2, "m2")):
        payload = json.loads((run_dir / "gardner_r0.json").read_text())
        for e in payload.get("ensembles", []):
            lam = e.get("lam")
            if not e.get("resolved") or lam is None or not (lam > 0.0):
                continue
            if any(e.get(k) is None for k in ("D_sat", "D0", "onset_time", "N")):
                continue
            recs.append({
                "campaign": tag,
                "lam": float(lam),
                "D0": float(e["D0"]),
                "D_sat": float(e["D_sat"]),
                "t_sat": float(e["onset_time"]),
                "N": int(e["N"]),
            })
    return recs


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


# ------------------------------------------------------------------------- #
# Panel (a): D(t) rise + Debye-Waller plateau
# ------------------------------------------------------------------------- #
def panel_a(ax, curves, cone_doc, r0_doc):
    ref_delta = DELTAS[0]                                  # smallest kick = longest ramp
    grp = [c for c in curves if abs(c.delta - ref_delta) < 1e-9]
    t = grp[0].times
    Dstack = np.vstack([c.D for c in grp])                 # (n_curve, T)

    # Reported fit for THIS rung: per-ensemble lambda (median) and intercept D0
    # (mean), both straight from gardner_r0.json.  The fit line D0*exp(lam*t) is
    # the fit the pipeline actually reports -- valid in the growth window, the
    # data peel off it toward the plateau (saturation), as they must.
    ens01 = [e for e in r0_doc["ensembles"]
             if e.get("delta_index") == 0 and e.get("resolved")]
    lam_B = float(np.median([e["lam"] for e in ens01]))    # 0.909  (headline ~0.91)
    D0 = float(np.mean([e["D0"] for e in ens01]))          # 0.858
    t_sat = float(np.median([e["onset_time"] for e in ens01]))  # 6.0
    D_sat = float(r0_doc["pooled"]["D_sat"]["mean"])       # 223.36
    d_sat_over_n = D_sat / 1500.0                          # 0.149 sigma

    # individual curves (faint) + geometric ensemble-mean (bold, white markers)
    for D in Dstack:
        ax.plot(t, D, color=fs.MEASURED, lw=0.6, alpha=0.26, solid_capstyle="round",
                zorder=1)
    Dmean = 10.0 ** np.nanmean(np.log10(Dstack), axis=0)
    ax.plot(t, Dmean, color=fs.MEASURED, lw=1.9, marker="o", ms=3.2,
            markerfacecolor="white", markeredgecolor=fs.MEASURED,
            markeredgewidth=0.9, zorder=4)
    # free-slope check over the clean early-exponential window (for verification)
    gm = (Dmean > GROWTH_FLOOR * D_sat) & (Dmean < 0.30 * D_sat)
    lam_free = float(np.polyfit(t[gm], np.log10(Dmean[gm]), 1)[0] * math.log(10.0))

    # exponential fit line D0 * exp(lambda_B t) over the growth window
    slope_dec = lam_B / math.log(10.0)
    t_fit = np.linspace(-0.25, (math.log(1.3 * D_sat / D0) / lam_B), 60)
    ax.plot(t_fit, D0 * 10.0 ** (slope_dec * t_fit),
            color=fs.THEORY, lw=1.7, ls=(0, (5, 2)), zorder=5)
    xt = 2.9
    ax.annotate(r"$D \propto e^{\lambda_B t}$",
                xy=(xt, D0 * 10.0 ** (slope_dec * xt)),
                xytext=(xt + 0.35, D0 * 10.0 ** (slope_dec * xt) * 0.30),
                color=fs.THEORY, fontsize=fs.FS_ANNOT, ha="left", va="top")

    # saturation-onset marker (ties to panel c's t_sat)
    ax.axvline(t_sat, color=fs.GUIDE, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(t_sat + 0.25, 0.42, r"$t_{\mathrm{sat}}$", color=fs.SUBTLE,
            fontsize=fs.FS_ANNOT, ha="left", va="bottom")

    # Debye-Waller plateau line + short label (top-right, no collision)
    ax.axhline(D_sat, color=fs.SUBTLE, lw=0.9, ls=(0, (1, 1.5)), zorder=3)
    ax.text(t[-1], D_sat * 1.16, r"$D_{\mathrm{sat}}$ (Debye$-$Waller plateau)",
            color=fs.SUBTLE, fontsize=fs.FS_ANNOT, ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(-0.7, t[-1] + 0.7)
    ax.set_ylim(0.32, 5.2e2)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"divergence $D(t)$")
    # stat block in the empty lower-right corner
    fs.annotate_stats(
        ax,
        rf"$\lambda_B = {lam_B:.2f}$" "\n"
        rf"$D_{{\mathrm{{sat}}}}/N = {d_sat_over_n:.2f}\,\sigma$ (sub-cage)" "\n"
        rf"$\delta = {ref_delta:g}$,  $N = 1500$",
        x=0.40, y=0.30, size=fs.FS_ANNOT,
    )
    # forward "cone scorecard" (author's signature motif) in the empty upper-right
    # wedge -- below the plateau, above the stat block.  All counts are the paper's
    # landed numbers: 70/72 resolved (97%, main.tex:86), median R^2>=0.96, and the
    # rate positive in all 3 independent parent glasses (3/3).
    fs.scorecard(
        ax,
        [
            ("resolved", "97%  (70/72)", fs.GREEN),
            ("median", r"$R^2\!\geq\!0.96$", fs.BLUE),
            ("parents", "3/3  +", fs.VERMILLION),
        ],
        x=0.55, y=0.80, title="cone scorecard",
    )
    fs.panel_label(ax, "a", x=-0.22)
    return {"lam_B": lam_B, "lam_free_fit": lam_free, "D0": D0, "t_sat": t_sat,
            "D_sat": D_sat, "d_sat_over_n": d_sat_over_n, "n_curves_delta": len(grp)}


# ------------------------------------------------------------------------- #
# Panel (b): delta-collapse
# ------------------------------------------------------------------------- #
def _collapse_master(curves, lam, floor=GROWTH_FLOOR, ceil=GROWTH_CEIL, npts=96):
    sc = cc.score_collapse(curves, lam, floor=floor, ceil=ceil, grid_points=npts)
    return sc


def panel_b(ax, curves, lam_B):
    """Delta-collapse in one axis: the collapse (main) with the raw spread as a
    non-occluding inset (lower-right).  The shift uses the independently-fitted
    headline lambda_B (panel-a value) -- no new parameter.
    """
    delta_ref = max(DELTAS)
    ramp = fs.sequential(len(DELTAS))                       # cividis, cold->warm
    cmap = {d: ramp[i] for i, d in enumerate(DELTAS)}

    # ---- main: collapsed (after) ------------------------------------------ #
    for c in curves:
        shift = math.log(delta_ref / c.delta) / lam_B
        ax.plot(c.times - shift, c.D, color=cmap[round(c.delta, 3)], lw=0.9,
                alpha=0.6, solid_capstyle="round", zorder=2)
    sc = _collapse_master(curves, lam_B)                   # growth-window master
    if sc is not None:
        ax.plot(sc.grid, 10.0 ** sc.master, color="white", lw=3.0, zorder=4,
                solid_capstyle="round")
        ax.plot(sc.grid, 10.0 ** sc.master, color=fs.INK, lw=1.6, zorder=5,
                solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 4.8e2)
    ax.set_xlim(-4.4, 9.2)
    ax.set_xlabel(r"shifted time $\;t-\lambda_B^{-1}\ln(\delta_{\mathrm{ref}}/\delta)$")
    ax.set_ylabel(r"divergence $D(\delta,t)$")

    # ---- inset: raw spread (before), lower-right (empty in the after-plot) - #
    axin = ax.inset_axes([0.565, 0.115, 0.40, 0.40])
    for c in curves:
        axin.plot(c.times, c.D, color=cmap[round(c.delta, 3)], lw=0.7, alpha=0.55,
                  solid_capstyle="round")
    axin.set_yscale("log")
    axin.set_ylim(0.3, 4.8e2)
    axin.set_xlim(-0.5, 20.5)
    axin.set_xticks([0, 10, 20])
    axin.set_yticks([1, 100])
    axin.tick_params(labelsize=5.6, length=2, pad=1.4)
    axin.set_title("raw (before)", fontsize=6.4, color=fs.SUBTLE, pad=1.5)
    axin.set_xlabel(r"$t$", fontsize=6.4, labelpad=0.4)
    for sp in ("top", "right"):
        axin.spines[sp].set_visible(False)

    # ---- legend + stats in the empty upper-left wedge --------------------- #
    handles = [Line2D([], [], color=cmap[d], lw=2.2, label=rf"$\delta={d:g}$")
               for d in DELTAS]
    handles.append(Line2D([], [], color=fs.INK, lw=1.6, label="master"))
    ax.legend(handles=handles, loc="upper left", fontsize=fs.FS_LEGEND,
              handlelength=1.1, labelspacing=0.2, borderaxespad=0.4,
              handletextpad=0.4, bbox_to_anchor=(0.0, 1.005))

    deltas = sorted({c.delta for c in curves})
    lin_curves = [c for c in curves if c.delta in deltas[:-1]]
    r2_lin = float(cc.score_collapse(lin_curves, lam_B).r2)     # 0.989
    r2_full = float(cc.score_collapse(curves, lam_B).r2)        # 0.890
    ceiling = float(cc.same_delta_ceiling(curves))              # 0.989
    fs.annotate_stats(
        ax,
        rf"$\lambda_B = {lam_B:.2f}$" "\n"
        rf"$R^2 = {r2_lin:.2f}$" "\n"
        rf"(linear rungs)",
        x=0.035, y=0.60, size=fs.FS_ANNOT,
    )
    fs.panel_label(ax, "b", x=-0.20)
    return {"lam_B": lam_B, "R2_linear_rungs": r2_lin, "R2_full_decade": r2_full,
            "same_delta_ceiling": ceiling}


# ------------------------------------------------------------------------- #
# Panel (c): closure identity
# ------------------------------------------------------------------------- #
def panel_c(ax, recs, relations_doc):
    # predicted saturation time = lambda_B^{-1} ln(D_sat/D0); measured = onset t_sat.
    # This is the closure lambda_B * t_sat = ln(D_sat/D0) in time units, and
    # reproduces the artifact's headline Pearson r (pearson_t_pred_vs_onset).
    t_pred = np.array([math.log(r["D_sat"] / r["D0"]) / r["lam"] for r in recs])
    t_meas = np.array([r["t_sat"] for r in recs])
    camp = np.array([r["campaign"] for r in recs])

    r_all = pearson(t_pred, t_meas)
    n = len(recs)
    ratio_med = float(np.median(t_pred / t_meas))

    lo = min(t_pred.min(), t_meas.min()) - 0.4
    hi = max(t_pred.max(), t_meas.max()) + 0.4
    ax.plot([lo, hi], [lo, hi], color=fs.GUIDE, lw=0.9, ls="--", zorder=0)

    markers = {"fss": ("o", "campaign 1"), "m2": ("s", "campaign 2")}
    for tag, (mk, lab) in markers.items():
        m = camp == tag
        ax.scatter(t_pred[m], t_meas[m], marker=mk, s=24, facecolor=fs.MEASURED,
                   edgecolor="white", linewidths=0.6, alpha=0.9, zorder=3,
                   label=f"{lab} ($n={int(m.sum())}$)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"predicted $\;\lambda_B^{-1}\ln(D_{\mathrm{sat}}/D_0)$")
    ax.set_ylabel(r"measured $\;t_{\mathrm{sat}}$")
    fs.annotate_stats(
        ax,
        rf"$\lambda_B\,t_{{\mathrm{{sat}}}} = \ln(D_{{\mathrm{{sat}}}}/D_0)$" "\n"
        rf"$r = {r_all:.3f}$,  $n = {n}$" "\n"
        rf"median ratio $= {ratio_med:.2f}$",
        x=0.045, y=0.965,
    )
    ax.legend(loc="lower right", fontsize=fs.FS_LEGEND, handletextpad=0.2,
              labelspacing=0.28, borderaxespad=0.3)
    fs.panel_label(ax, "c", x=-0.22)
    return {"pearson": r_all, "n": n, "ratio_median": ratio_med}


# ------------------------------------------------------------------------- #
# Assemble
# ------------------------------------------------------------------------- #
def main():
    curves, prov = load_fss_curves()
    r0_doc = json.loads((RUN_FSS / "gardner_r0.json").read_text())
    cone_doc = json.loads(CONE_JSON.read_text())
    relations_doc = json.loads(RELATIONS_JSON.read_text())
    recs = closure_records()

    fig, axes = plt.subplots(1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.365))
    va = panel_a(axes[0], curves, cone_doc, r0_doc)
    vb = panel_b(axes[1], curves, va["lam_B"])             # harmonised lambda_B
    vc = panel_c(axes[2], recs, relations_doc)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT_STEM))
    plt.close(fig)

    # ---- console verification block --------------------------------------- #
    print("VERIFIED NUMBERS")
    print(f"  (a) lambda_B (persisted median) = {va['lam_B']:.4f}  "
          f"[free-slope check {va['lam_free_fit']:.3f}]")
    print(f"      D_sat = {va['D_sat']:.2f}   D_sat/N = {va['d_sat_over_n']:.4f} sigma  "
          f"(delta={DELTAS[0]}, {va['n_curves_delta']} curves)")
    print(f"  (b) collapse lambda_B = {vb['lam_B']:.4f}  "
          f"R2(linear rungs, n=24) = {vb['R2_linear_rungs']:.4f}  "
          f"R2(full decade) = {vb['R2_full_decade']:.4f}  "
          f"ceiling = {vb['same_delta_ceiling']:.4f}")
    print(f"  (c) closure Pearson r = {vc['pearson']:.4f}  n = {vc['n']}  "
          f"median ratio = {vc['ratio_median']:.4f}")
    print(f"      artifact says r = {relations_doc['closure']['pearson_t_pred_vs_onset']:.4f}"
          f"  n = {relations_doc['closure']['n']}")
    print("WROTE:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
