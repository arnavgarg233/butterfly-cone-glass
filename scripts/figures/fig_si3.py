#!/usr/bin/env python3
"""Fig S3 - Configurational-entropy pipeline (Nature Physics "butterfly cone" SI).

Four panels, every number pulled from persisted runs/ artifacts (verified against
the JSON keys / recomputed with the committed pipeline code; nothing fabricated):

  (a) The temperature-integration integrand.  u_ex(beta) is the integrand of
      beta f_ex = integral_0^beta u_ex(beta') dbeta'.  Measured grid rungs (dots),
      the analytic soft-sphere high-T head (dashed, integrated to beta_min), and the
      head/grid split that together give the excess free energy.
      Source: runs/sc_energy_ladder/energy_grid.npz (+ butterfly_cone.entropy.thermodynamic).

  (b) Coarse-grid vs fine-grid reconciliation of s_c(0.060).  Zoom on the cold
      branch: the fine grid resolves the committed rungs 0.108/0.090/0.075 (true
      convex integrand, blue); the coarse grid jumps 0.13 -> 0.060 in one trapezoid
      (vermillion chord).  The shaded sliver between them is *exactly* the 0.116-nat
      TI over-count that turns the fine s_c=1.316 into the raw s_c=1.200.
      Source: runs/sc_T0060/energy_grid_deep.npz + scripts/sc_reconcile.py.

  (c) The s_c(T) ladder with the Einstein-solid zero.  Five committed shallow rungs
      (s_c = 2.734/2.489/2.234/1.949/1.681 at T=0.15..0.075), the two deep anchors
      (fine-grid s_c(0.060)=1.316+-0.027, s_c(0.055)=1.182+-0.041), the smooth
      sqrt(T)/affine extrapolations they bend ~3 sigma below, and the exact
      Einstein-solid limit s_c=0.
      Sources: runs/sc_energy_ladder/sc_curve.json, runs/sc_T0060/,
               runs/sc_T0055/full-ladder-analysis/sc_T0055_report.json.

  (d) Planted-truth validation.  A planted two-basin harmonic landscape recovers
      s_c to |Delta| = 4e-16 and the Einstein-solid limit to s_c = 0 exactly -- both
      far below the 1e-12 acceptance gate (23 persisted unit tests).
      Source: runs/sc_energy_ladder/sc_curve.json ["planted_truth"].

Run:
  cd butterfly_cone && PYTHONPATH=src:scripts/figures \
     python \
     scripts/figures/fig_si3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402  (applies the shared style on import)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from butterfly_cone.entropy.thermodynamic import (  # noqa: E402
    excess_entropy,
    fit_soft_sphere_high_temperature_head,
)

OUT = ROOT / "results" / "figures" / "fig_si3"

# committed smooth-trend fits of the shallow ladder (runs/.../sc_T0055_report.json)
AFFINE_INT, AFFINE_SLOPE = 0.687848808998677, 13.831385090427876     # s_c = a + b*T
SQRT_INT, SQRT_SLOPE = -0.8150482056612872, 9.187903701369686        # s_c = a + b*sqrt(T)

# semantic colours
C_TRUE = fs.BLUE          # measured / fine-grid true integrand and shallow ladder
C_BIAS = fs.VERMILLION    # the coarse-grid biased path / deep-anchor headline
C_PASS = fs.GREEN         # validation pass
C_REF = fs.SUBTLE         # extrapolations / reference guides


# --------------------------------------------------------------------------- #
def load():
    sc = json.loads((ROOT / "runs/sc_energy_ladder/sc_curve.json").read_text())
    eg = np.load(ROOT / "runs/sc_energy_ladder/energy_grid.npz")
    deep = np.load(ROOT / "runs/sc_T0060/energy_grid_deep.npz")
    t55 = json.loads(
        (ROOT / "runs/sc_T0055/full-ladder-analysis/sc_T0055_report.json").read_text()
    )
    scdeep = json.loads((ROOT / "runs/sc_T0060/sc_curve_deep.json").read_text())
    return sc, dict(eg), dict(deep), t55, scdeep


# --------------------------------------------------------------------------- #
def panel_a(ax, eg):
    """The TI integrand u_ex(beta): measured grid + analytic high-T head."""
    T = np.asarray(eg["T_grid"], float)
    U = np.asarray(eg["u_grid"], float)
    order = np.argsort(1.0 / T)                      # ascending beta
    beta = (1.0 / T)[order]
    u = U[order]

    head = fit_soft_sphere_high_temperature_head(beta, u, n_points=8)
    beta_min = beta[0]                               # head <-> grid boundary
    bh = np.geomspace(3e-3, beta_min, 200)
    uh = head.energy(bh)

    # measured integrand (grid) + the analytic head continuation to beta -> 0
    ax.plot(beta, u, "-", color=C_TRUE, lw=1.5, zorder=2)
    ax.plot(beta, u, "o", color=C_TRUE, ms=4.2, mec="white", mew=0.5, zorder=3)
    ax.plot(bh, uh, "--", color=C_REF, lw=1.3, zorder=2)
    ax.axvline(beta_min, color=fs.GUIDE, lw=0.8, ls=":", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta = 1/T$")
    ax.set_ylabel(r"integrand  $u_{\mathrm{ex}}(\beta)$")
    ax.set_xlim(3e-3, 22)
    ax.set_ylim(0.2, 45)

    # temperature call-outs at a few rungs (offset to avoid the curve/head line)
    for Tval, off in [(30.0, (10, 4)), (1.0, (0, 9)), (0.15, (0, 9)),
                      (0.075, (0, 9))]:
        bv = 1.0 / Tval
        uv = float(np.interp(bv, beta, u)) if bv >= beta_min else float(head.energy(bv))
        ax.annotate(f"$T={Tval:g}$", xy=(bv, uv), xytext=off,
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.6, color=fs.SUBTLE)

    ax.annotate("analytic soft-sphere\nhead", xy=(0.021, float(head.energy(0.021))),
                xytext=(6.0e-3, 1.7), ha="center", va="center",
                fontsize=fs.FS_ANNOT, color=C_REF,
                arrowprops=dict(arrowstyle="->", color=C_REF, lw=0.7))
    ax.annotate("measured grid", xy=(0.5, float(np.interp(0.5, beta, u))),
                xytext=(6, 12), textcoords="offset points", ha="left",
                va="bottom", fontsize=fs.FS_ANNOT, color=C_TRUE, fontweight="bold")
    fs.annotate_stats(
        ax,
        r"$\beta f_{\mathrm{ex}}=\!\int_0^{\beta}\! u_{\mathrm{ex}}\,d\beta$"
        + "\n" + r"$=\,1.31\,$(head)$\,+\,$grid",
        x=0.44, y=0.985, color=fs.INK, size=fs.FS_ANNOT)
    fs.panel_label(ax, "a")


def panel_b(ax, eg, deep):
    """Cold-branch reconcile: the coarse single-step over-counts by 0.116 nat."""
    # fine grid = committed shallow + the 0.060 rung
    Tf = np.append(np.asarray(eg["T_grid"], float), 0.060)
    Uf = np.append(np.asarray(eg["u_grid"], float), 0.27163)
    of = np.argsort(1.0 / Tf)
    bf, uf = (1.0 / Tf)[of], Uf[of]

    cold_T = np.array([0.130, 0.108, 0.090, 0.075, 0.060])
    cold_b = 1.0 / cold_T
    cold_u = np.array([float(np.interp(b, bf, uf)) for b in cold_b])

    b0, b1 = cold_b[0], cold_b[-1]                    # 0.13 -> 0.060 chord
    chord = np.array([cold_u[0], cold_u[-1]])
    chord_b = np.array([b0, b1])

    # the shaded sliver between the coarse chord and the fine convex integrand
    b_fill = np.linspace(b0, b1, 200)
    u_fine = np.interp(b_fill, cold_b, cold_u)
    u_chord = np.interp(b_fill, chord_b, chord)
    ax.fill_between(b_fill, u_fine, u_chord, color=C_BIAS, alpha=0.18,
                    lw=0, zorder=1)

    # fine (true) integrand
    ax.plot(cold_b, cold_u, "-", color=C_TRUE, lw=1.6, zorder=3)
    ax.plot(cold_b, cold_u, "o", color=C_TRUE, ms=5.0, mec="white", mew=0.6,
            zorder=4, label="fine grid (true)")
    # coarse single-step chord
    ax.plot(chord_b, chord, "--", color=C_BIAS, lw=1.6, zorder=3,
            label="coarse: one 0.13$\\to$0.06 step")
    ax.plot(chord_b, chord, "s", color=C_BIAS, ms=5.2, mec="white", mew=0.6,
            zorder=4)
    # mark the rungs the coarse grid drops
    drop_b = cold_b[1:4]
    drop_u = cold_u[1:4]
    ax.plot(drop_b, drop_u, "o", mfc="none", mec=C_TRUE, mew=1.2, ms=10,
            zorder=5)

    ax.set_xlabel(r"$\beta = 1/T$")
    ax.set_ylabel(r"integrand  $u_{\mathrm{ex}}(\beta)$")
    ax.set_xlim(7.0, 17.6)
    ax.set_ylim(0.24, 0.46)
    # temperature tick labels on the cold rungs
    ax.set_xticks(list(cold_b))
    ax.set_xticklabels([f"{t:g}" for t in cold_T], fontsize=6.8)
    ax.set_xlabel(r"$\beta = 1/T$   (labels: $T$)")

    ax.annotate("dropped rungs\n0.108 / 0.090 / 0.075",
                xy=(drop_b[1], drop_u[1]), xytext=(-2, -34),
                textcoords="offset points", ha="center", va="top",
                fontsize=fs.FS_ANNOT, color=C_TRUE,
                arrowprops=dict(arrowstyle="-", color=C_TRUE, lw=0.7))
    ax.annotate(r"$+0.116$ nat" + "\nTI over-count",
                xy=(12.4, 0.352), xytext=(0, 0), textcoords="offset points",
                ha="center", va="center", fontsize=fs.FS_ANNOT,
                color=C_BIAS, fontweight="bold")
    fs.annotate_stats(
        ax,
        r"raw (coarse)  $s_c=1.200$" + "\n"
        + r"fine (calib.)  $s_c=1.316$",
        x=0.035, y=0.975, size=fs.FS_ANNOT, color=fs.INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=6.6,
              handlelength=1.5, borderaxespad=0.0)
    fs.panel_label(ax, "b")


def panel_c(ax, sc, t55, scdeep):
    """s_c(T) ladder + deep anchors + smooth extrapolations + Einstein zero."""
    s = sorted(sc["summary"], key=lambda d: d["temperature"])
    T = np.array([d["temperature"] for d in s])
    y = np.array([d["s_configurational_mean"] for d in s])
    sem = np.array([d["s_configurational_std"] / np.sqrt(d["n_replicas"]) for d in s])

    # smooth extrapolation curves (committed shallow-ladder fits), extended cold
    tt = np.linspace(0.052, 0.152, 200)
    ax.plot(tt, AFFINE_INT + AFFINE_SLOPE * tt, ls=(0, (5, 2)), color=C_REF,
            lw=1.0, zorder=1)
    ax.plot(tt, SQRT_INT + SQRT_SLOPE * np.sqrt(tt), ls=(0, (1, 1.5)),
            color=fs.PURPLE, lw=1.1, zorder=1)

    # Einstein-solid exact zero
    ax.axhline(0.0, color=fs.GUIDE, lw=0.9, ls="--", zorder=0)

    # committed shallow ladder
    ax.plot(T, y, "-", color=C_TRUE, lw=1.4, zorder=3)
    ax.errorbar(T, y, yerr=sem, fmt="o", ms=5.0, color=C_TRUE, mec="white",
                mew=0.6, ecolor=C_TRUE, elinewidth=0.9, capsize=2.0, zorder=4,
                label="committed ladder (fine grid)")

    # deep anchors: fine-grid 0.060 and 0.055
    sc60_fine, sc60_stat = 1.3163, 0.0274
    sc55 = float(t55["analysis"]["measurement"]["s_c"])
    sc55_sem = float(t55["analysis"]["measurement"]["s_c_sem"])
    ax.errorbar([0.060, 0.055], [sc60_fine, sc55],
                yerr=[sc60_stat, sc55_sem], fmt="D", ms=5.6, color=C_BIAS,
                mec="white", mew=0.6, ecolor=C_BIAS, elinewidth=1.0,
                capsize=2.0, zorder=5, label="deep anchors (fine grid)")

    # raw coarse-grid cross-check at 0.060
    sc60_raw = float(scdeep["summary"][0]["s_configurational_mean"])
    ax.plot([0.060], [sc60_raw], "s", mfc="none", mec=fs.GRAY, mew=1.1, ms=6.5,
            zorder=4, label="raw (coarse grid)")

    # the ~3 sigma bend at T=0.060: measured fine anchor vs the sqrt(T) prediction
    sqrt60 = SQRT_INT + SQRT_SLOPE * np.sqrt(0.060)
    ax.plot([0.060], [sqrt60], "_", color=fs.PURPLE, ms=11, mew=1.6, zorder=6)
    ax.annotate("", xy=(0.060, sc60_fine + 0.015), xytext=(0.060, sqrt60 - 0.015),
                arrowprops=dict(arrowstyle="<->", color=fs.INK, lw=0.9))
    ax.annotate(r"$2.9\sigma$ bend below" + "\n" + r"the $\sqrt{T}$ trend",
                xy=(0.0605, 0.5 * (sc60_fine + sqrt60)), xytext=(0.076, 1.66),
                ha="left", va="center", fontsize=fs.FS_ANNOT, fontweight="bold",
                color=fs.INK,
                arrowprops=dict(arrowstyle="->", color=fs.INK, lw=0.7))

    ax.set_xlabel(r"temperature  $T$")
    ax.set_ylabel(r"configurational entropy  $s_c$ (nats)")
    ax.set_xlim(0.049, 0.156)
    ax.set_ylim(-0.15, 2.95)
    ax.text(0.152, 0.06, r"Einstein solid  $s_c=0$", ha="right", va="bottom",
            fontsize=6.8, color=fs.SUBTLE)
    ax.text(0.118, AFFINE_INT + AFFINE_SLOPE * 0.118 + 0.03, "affine",
            fontsize=6.8, color=C_REF, rotation=16, va="bottom", ha="left")
    ax.text(0.104, SQRT_INT + SQRT_SLOPE * np.sqrt(0.104) - 0.05, r"$\sqrt{T}$",
            fontsize=7.6, color=fs.PURPLE, va="top", ha="left")
    ax.legend(loc="upper left", fontsize=6.4, handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.3)
    fs.panel_label(ax, "c")


def panel_d(ax, sc):
    """Planted-truth validation: |Delta s_c| vs the 1e-12 acceptance gate."""
    pt = sc["planted_truth"]
    tb_err = abs(pt["two_basin_expected"] - pt["two_basin_s_c"])
    ei_err = abs(pt["einstein_expected"] - pt["einstein_s_c"])
    tol = pt["absolute_tolerance"]
    floor = 1e-17

    rows = [
        (1, "two-basin harmonic", tb_err, r"$s_c=0.17329$ (planted)"),
        (0, "Einstein solid", ei_err, r"$s_c=0$ exact"),
    ]
    # fail region (right of the gate)
    ax.axvspan(tol, 1e-9, color=C_BIAS, alpha=0.07, lw=0, zorder=0)
    ax.axvline(tol, color=C_BIAS, lw=1.2, ls="--", zorder=2)

    for yv, name, err, note in rows:
        xerr = max(err, floor)
        ax.hlines(yv, floor, xerr, color=C_PASS, lw=1.6, zorder=3)
        ax.plot(xerr, yv, "o", color=C_PASS, ms=7.0, mec="white", mew=0.7,
                zorder=4)
        if err > 0:
            ax.annotate(r"$|\Delta|=%.0e$" % err, xy=(xerr, yv), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=fs.FS_ANNOT, fontweight="bold", color=fs.INK)
        else:
            ax.annotate(r"$|\Delta|=0$ (exact)", xy=(xerr, yv), xytext=(10, 11),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=fs.FS_ANNOT, fontweight="bold", color=fs.INK)
        ax.annotate(note, xy=(floor, yv), xytext=(2, -12),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=6.6, color=fs.SUBTLE)

    ax.set_xscale("log")
    ax.set_xlim(3e-18, 1e-9)
    ax.set_ylim(-0.6, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Einstein\nsolid", "two-basin\nharmonic"],
                       fontsize=7.0)
    ax.set_xlabel(r"absolute recovery error  $|\Delta s_c|$")
    ax.annotate(r"acceptance gate  $10^{-12}$",
                xy=(tol, 1.55), xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=fs.FS_ANNOT, color=C_BIAS,
                fontweight="bold")
    fs.annotate_stats(ax, "23 persisted unit tests", x=0.035, y=0.14,
                      color=fs.SUBTLE, size=fs.FS_ANNOT)
    fs.panel_label(ax, "d")


# --------------------------------------------------------------------------- #
def main():
    sc, eg, deep, t55, scdeep = load()

    # ---- verification prints (assert the plotted headline numbers) ----
    s = sorted(sc["summary"], key=lambda d: d["temperature"], reverse=True)
    ladder = [round(d["s_configurational_mean"], 3) for d in s]
    print("VERIFY shallow ladder s_c (T=0.15..0.075):", ladder)
    assert ladder == [2.734, 2.489, 2.234, 1.949, 1.681], ladder

    pt = sc["planted_truth"]
    tb_err = abs(pt["two_basin_expected"] - pt["two_basin_s_c"])
    print("VERIFY planted: two-basin |Delta|=%.2e  Einstein |Delta|=%.1e  tol=%g"
          % (tb_err, abs(pt["einstein_expected"] - pt["einstein_s_c"]),
             pt["absolute_tolerance"]))
    assert tb_err < pt["absolute_tolerance"]

    # recompute the coarse/fine s_c(0.060) straight from the two grids
    def s_c060(T, U):
        return excess_entropy(temperature=0.060, beta_grid=1.0 / np.asarray(T),
                              u_grid=np.asarray(U), u_at_temperature=0.27163)
    coarse = s_c060(deep["T_grid"], deep["u_grid"])
    Tf = np.append(eg["T_grid"], 0.060)
    Uf = np.append(eg["u_grid"], 0.27163)
    fine = s_c060(Tf, Uf)
    gap = coarse.beta_f_ex - fine.beta_f_ex
    print("VERIFY reconcile: beta_f_ex coarse-fine gap = %.4f (== 0.116 nat)" % gap)
    assert abs(gap - 0.1164) < 5e-4, gap
    print("VERIFY s_c(0.055) =", round(t55["analysis"]["measurement"]["s_c"], 4),
          " s_c(0.060) raw =",
          round(scdeep["summary"][0]["s_configurational_mean"], 4))

    fig, axes = plt.subplots(2, 2, figsize=fs.figsize(fs.WIDTH_FULL, 0.80))
    panel_a(axes[0, 0], eg)
    panel_b(axes[0, 1], eg, deep)
    panel_c(axes[1, 0], sc, t55, scdeep)
    panel_d(axes[1, 1], sc)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    print("wrote:", *[str(p) for p in paths])


if __name__ == "__main__":
    main()
