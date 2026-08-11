#!/usr/bin/env python3
"""Fig 3 - The Debye-Waller ceiling (ButterflyCone "butterfly cone", Nature Physics target).

Three panels, all data pulled ONLY from persisted runs/ artifacts (keys verified):

  (a) The identity on the 1:1 line: measured plateau D_sat/N vs the Gaussian
      prediction c*.u_DW, per configuration. The dashed y=x is the Gaussian
      ceiling (c* = 1.303); points sit just below it, most closely at the deepest
      T, where the empirical prefactor is 1.247 (4.3% below Gaussian). An inset
      note records the unkicked-pairwise cross-check (0.1487 ~ 0.1489).
  (b) The ceiling approached on cooling: the ratio c(T) = (D_sat/N)/u_DW rises
      monotonically toward the Gaussian target 1.303 as T falls; the better-
      equilibrated deep analysis (fss, T=0.075) lands at 1.247.
  (c) The sign-flipped bridge leg: c(T) vs configurational entropy s_c, with a
      linear fit and Spearman rho(c, s_c) = -1.000 (the entropy legs for lambda
      and D_sat correlate +1 with s_c; the DW ratio flips sign).

Sources
-------
  runs/dw_identity/dw_identity.json          (single deep-T identity, 2 configs)
  runs/dw_identity_ladder/dw_identity_ladder.json  (5-rung T ladder + bridge leg)

Run
---
  cd butterfly_cone && PYTHONPATH=src:scripts/figures \
    python \
    scripts/figures/fig_main3_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402  (applies the shared style on import)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figures/fig_main3_ceiling"


# --------------------------------------------------------------------------- #
# Load + verify data
# --------------------------------------------------------------------------- #
def load():
    dw = json.loads((ROOT / "runs/dw_identity/dw_identity.json").read_text())
    ld = json.loads(
        (ROOT / "runs/dw_identity_ladder/dw_identity_ladder.json").read_text()
    )
    return dw, ld


def main():
    dw, ld = load()

    idn = dw["identity"]
    c_gauss = idn["c"]                       # 1.302940  (Gaussian chi_3 constant)
    c_emp = idn["empirical_c"]               # 1.247344
    u_dw_agg = idn["u_DW"]                   # 0.119381
    meas_agg = idn["measured_D_sat_over_N"]  # 0.148910  (landed)
    pred_agg = idn["predicted_D_sat_over_N"] # 0.155547  (c_gauss * u_DW)
    rel_err = idn["rel_error"]               # 0.04267
    pairwise = dw["pairwise_divergence_per_particle"]  # 0.148725 (unkicked)
    landed = dw["landed_D_sat_over_N"]                 # 0.148910

    # per-config identity points (the deep T=0.075 analysis, "fss")
    fss_u = np.array([p["u_DW"] for p in dw["per_config"]])
    fss_meas = np.array(
        [p["pairwise_divergence_per_particle_plateau"] for p in dw["per_config"]]
    )
    fss_pred = c_gauss * fss_u

    # 5-rung temperature ladder
    rungs = ld["per_temperature"]
    T = np.array([r["temperature"] for r in rungs])
    s_c = np.array([r["s_c"] for r in rungs])
    u_dw = np.array([r["u_DW"] for r in rungs])
    d_over_n = np.array([r["d_sat_over_n"] for r in rungs])
    d_sem = np.array([r["d_sat_over_n_sem"] for r in rungs])
    c_lad = np.array([r["c"] for r in rungs])
    pred_lad = c_gauss * u_dw

    corr = ld["correlation"]
    rho = corr["spearman_rho"]        # -1.0
    pear = corr["pearson_r"]          # -0.982
    r2 = corr["fit_r2"]               # 0.965
    slope = corr["slope"]             # -0.236
    intercept = corr["intercept"]     # 1.615
    sp_p = corr["spearman_p"]         # 0.008333
    deepest_gap = ld["trend"]["deepest_distance_to_target"]  # 0.0863
    deepest_c = ld["trend"]["deepest_value"]                 # 1.2167

    # --- verification prints (fail loud if the artifact drifts) ------------- #
    assert abs(c_gauss - 1.302940) < 1e-5
    assert abs(c_emp - 1.247344) < 1e-5
    assert abs(rel_err - 0.042670) < 1e-5
    assert abs(pairwise - 0.148725) < 1e-5 and abs(landed - 0.148910) < 1e-5
    assert abs(rho - (-1.0)) < 1e-9
    assert abs(np.median(c_lad / (d_over_n / u_dw)) - 1.0) < 1e-9  # c == (D/N)/u_DW
    print(f"[verify] Gaussian c* = {c_gauss:.6f}  empirical c = {c_emp:.6f}  "
          f"agree {rel_err*100:.1f}%")
    print(f"[verify] unkicked pairwise {pairwise:.6f} ~ landed {landed:.6f}  "
          f"(|d|={abs(pairwise-landed):.2e})")
    _cold_to_hot = c_lad[np.argsort(T)]
    print(f"[verify] ladder c(T) cold->hot: "
          f"{', '.join(f'{v:.4f}' for v in _cold_to_hot)}  "
          f"target {c_gauss:.4f}  Spearman rho(c,s_c)={rho:+.3f} (p={sp_p:.4f})")

    # --------------------------------------------------------------------- #
    # Temperature -> colour  (cividis, cold = navy per the style guide)
    # --------------------------------------------------------------------- #
    order = np.argsort(T)                 # ascending T (cold -> hot)
    T_sorted = T[order]
    cols_sorted = fs.sequential(len(T_sorted))     # navy -> khaki
    col_of_T = {t: cols_sorted[i] for i, t in enumerate(T_sorted)}
    lad_colors = [col_of_T[t] for t in T]
    cold_color = cols_sorted[0]
    # discrete colourbar boundaries at midpoints between the sorted T values
    mids = (T_sorted[:-1] + T_sorted[1:]) / 2
    bounds = np.concatenate([
        [T_sorted[0] - (mids[0] - T_sorted[0])],
        mids,
        [T_sorted[-1] + (T_sorted[-1] - mids[-1])],
    ])
    cmap = ListedColormap(cols_sorted)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # --------------------------------------------------------------------- #
    # Figure: 3 square panels in a row + a slim shared T colourbar
    # --------------------------------------------------------------------- #
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.37), sharey=False
    )
    for ax in (axa, axb, axc):
        ax.set_box_aspect(1.0)          # identical square boxes -> aligned labels

    # ===================================================================== #
    # (a) The identity on the 1:1 line
    # ===================================================================== #
    all_pred = np.concatenate([pred_lad, fss_pred])
    all_meas = np.concatenate([d_over_n, fss_meas])
    lo = 0.90 * float(min(all_pred.min(), all_meas.min()))
    hi = 1.06 * float(all_pred.max())
    # Gaussian ceiling: y = x
    axa.plot([lo, hi], [lo, hi], color=fs.GUIDE, lw=1.1, ls="--", zorder=0)
    # deep-T empirical line: measured = (c_emp/c_gauss) * predicted
    axa.plot([lo, hi], [lo * c_emp / c_gauss, hi * c_emp / c_gauss],
             color=fs.SUBTLE, lw=0.9, ls=":", zorder=0)
    # ladder rungs, coloured by T, with SEM error bars
    for xi, yi, ei, ci in zip(pred_lad, d_over_n, d_sem, lad_colors):
        axa.errorbar(xi, yi, yerr=ei, fmt="o", ms=6.5, mfc=ci, mec="white",
                     mew=0.6, ecolor=fs.SUBTLE, elinewidth=0.8, capsize=1.6,
                     zorder=3)
    # deep-T "fss" identity configs (the headline analysis point)
    axa.scatter(fss_pred, fss_meas, marker="*", s=170, facecolor=fs.VERMILLION,
                edgecolor="white", linewidth=0.6, zorder=4)
    axa.set_xlim(lo, hi)
    axa.set_ylim(lo, hi)
    axa.set_xlabel(r"predicted  $c^{*}\,u_{\mathrm{DW}}$   (Gaussian)")
    axa.set_ylabel(r"measured  $D_{\mathrm{sat}}/N$")
    # inline line labels
    axa.text(hi, hi, " 1:1", color=fs.SUBTLE, fontsize=fs.FS_ANNOT - 0.5,
             va="center", ha="left", rotation=45, rotation_mode="anchor")
    axa.text(0.905 * hi, 0.905 * hi * c_emp / c_gauss, "empirical ",
             color=fs.SUBTLE, fontsize=fs.FS_ANNOT - 1.5, va="top", ha="right",
             rotation=41, rotation_mode="anchor")
    fs.annotate_stats(
        axa,
        r"$D_{\mathrm{sat}}/N = c\,u_{\mathrm{DW}}$" + "\n"
        + r"Gaussian $c^{*}=1.303$" + "\n"
        + r"deep-$T$ empirical $1.247$" + "\n"
        + r"agree $4.3\%$",
        x=0.05, y=0.965,
    )
    axa.text(0.97, 0.045,
             "unkicked pairwise\n"
             r"$0.1487 \approx 0.1489$",
             transform=axa.transAxes, fontsize=fs.FS_ANNOT - 1.0,
             color=fs.SUBTLE, va="bottom", ha="right", linespacing=1.3)
    fs.panel_label(axa, "a", x=-0.26)

    # ===================================================================== #
    # (b) The ceiling approached on cooling: c(T) vs T
    # ===================================================================== #
    axb.axhline(c_gauss, color=fs.VERMILLION_MUTED, lw=1.1, ls="--", zorder=1)
    axb.text(0.157, c_gauss + 0.010,
             r"Gaussian ceiling  $c^{*}=1.303$", color=fs.VERMILLION_MUTED,
             fontsize=fs.FS_ANNOT - 0.5, va="bottom", ha="right")
    axb.plot(T[order], c_lad[order], "-", color=fs.INK, lw=1.1, zorder=2)
    for ti, ci_, col in zip(T, c_lad, lad_colors):
        axb.plot(ti, ci_, "o", ms=6.8, mfc=col, mec="white", mew=0.6, zorder=3)
    # deeper, better-equilibrated fss estimate at T = 0.075
    axb.plot(0.075, c_emp, "*", ms=14, mfc=fs.VERMILLION, mec="white",
             mew=0.6, zorder=4)
    axb.annotate("deep analysis (fss)  1.247",
                 xy=(0.0758, c_emp + 0.004), xytext=(0.089, 1.275),
                 fontsize=fs.FS_ANNOT - 0.5, color=fs.VERMILLION,
                 va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", color=fs.VERMILLION, lw=0.6))
    # cooling cue (up-left along the trend), in the empty right wedge
    axb.annotate("", xy=(0.114, 1.215), xytext=(0.138, 1.120),
                 arrowprops=dict(arrowstyle="->", color=fs.SUBTLE, lw=1.1))
    axb.text(0.140, 1.107, "cooling", color=fs.SUBTLE,
             fontsize=fs.FS_ANNOT, va="top", ha="right")
    axb.set_xlim(0.068, 0.158)
    axb.set_ylim(0.96, 1.35)
    axb.set_xlabel(r"temperature  $T$")
    axb.set_ylabel(r"$c(T)=(D_{\mathrm{sat}}/N)\,/\,u_{\mathrm{DW}}$")
    fs.annotate_stats(
        axb,
        rf"monotone; deepest $= {deepest_c:.3f}$" + "\n"
        + rf"gap to $c^{{*}}$ $= {deepest_gap:.3f}$",
        x=0.05, y=0.215, color=fs.SUBTLE,
    )
    fs.panel_label(axb, "b", x=-0.24)

    # ===================================================================== #
    # (c) The sign-flipped bridge leg: c vs s_c
    # ===================================================================== #
    xs = np.linspace(s_c.min() * 0.955, s_c.max() * 1.045, 50)
    axc.plot(xs, intercept + slope * xs, "-", color=fs.INK, lw=1.0, zorder=1)
    axc.axhline(c_gauss, color=fs.VERMILLION_MUTED, lw=0.9, ls="--", zorder=1)
    axc.text(s_c.min() * 0.96, c_gauss + 0.010, r"Gaussian ceiling  $c^{*}=1.303$",
             color=fs.VERMILLION_MUTED, fontsize=fs.FS_ANNOT - 0.5,
             va="bottom", ha="left")
    for si, ci_, col in zip(s_c, c_lad, lad_colors):
        axc.plot(si, ci_, "o", ms=6.8, mfc=col, mec="white", mew=0.6, zorder=3)
    axc.set_xlim(s_c.min() * 0.955, s_c.max() * 1.045)
    axc.set_ylim(0.96, 1.35)
    axc.set_xlabel(r"configurational entropy  $s_c$")
    axc.set_ylabel(r"ratio  $c(T)$")
    # right-aligned stat block in the empty upper-right wedge (above the trend,
    # below the ceiling) -- the only region the anti-diagonal trend leaves clear.
    axc.text(
        0.965, 0.845,
        r"sign-flipped bridge leg" + "\n"
        + rf"$\rho={rho:+.2f}$,  $p={sp_p:.4f}$" + "\n"
        + rf"$r={pear:.3f}$,  $R^{{2}}={r2:.3f}$" + "\n"
        + r"(vs $+1$ for $\lambda,\,D_{\mathrm{sat}}$)",
        transform=axc.transAxes, fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE,
        va="top", ha="right", linespacing=1.4,
    )
    fs.panel_label(axc, "c", x=-0.20)

    # --------------------------------------------------------------------- #
    # shared T colourbar (spans all three panels, right edge)
    # --------------------------------------------------------------------- #
    cb = fig.colorbar(sm, ax=(axa, axb, axc), location="right",
                      fraction=0.020, pad=0.012, aspect=22, shrink=0.92,
                      ticks=T_sorted)
    cb.set_label(r"temperature  $T$", fontsize=fs.FS_LABEL)
    cb.ax.set_yticklabels([f"{t:.3f}" for t in T_sorted])
    cb.ax.tick_params(labelsize=fs.FS_TICK - 0.5)
    cb.outline.set_linewidth(0.6)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    print("[saved]", *[str(p) for p in paths])


if __name__ == "__main__":
    main()
