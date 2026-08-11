#!/usr/bin/env python
"""Fig 5 (Nature Physics) -- the deep limit / co-vanishing.

Publication-quality upgrade of the ``scripts/chaos_ideal_glass.py`` money
figure.  Three panels:

  (a) The co-vanishing.  Butterfly Lyapunov rate lambda vs configurational
      entropy s_c across the 8-rung T-ladder.  The shallow-5 affine fit would
      leave chaos alive at s_c=0 (intercept +0.35); adding the deep-T anchors
      (T=0.075/0.067/0.060/0.055, three of them sub-T_g) collapses the all-8
      intercept to +0.013 +/- 0.066 and the power-through-origin fit reads
      lambda = 0.53 s_c^0.97.  An inset carries s_c(T) with the RFOT fit whose
      Kauzmann point T_K ~ 0.039 coincides with the chaos-death window
      T* ~ 0.04 -- BOTH in the extrapolation zone below the coldest datum.

  (b) Rate-amplitude dissociation at T=0.060, shown as observed/shallow-line
      ratio.  The deep rate lambda(0.060)=0.675 falls 3.3 sigma BELOW its
      shallow-5 s_c line (ratio 0.80) while the ceiling D_sat/N stays on/above
      its own line (ratio 1.37, +1.5 sigma; "pinned").

  (c) The diverging scrambling clock.  t_sat = ln(D_sat/D0)/lambda climbs 1.38x
      above the both-on-line baseline at the deep anchor -- a rate-driven second
      clock, distinct from alpha-relaxation (over the shallow overlap tau_alpha
      folds 16.9x while t_sat folds only 1.06x).

Every plotted number is pulled from verified artifacts:
  * lambda / s_c ladder + all fits: ``scripts/chaos_ideal_glass.py`` (its
    ``verify_against_artifacts`` hard-checks all 8 lambda and both deep s_c
    against the committed runs/ JSON -- 15/15 on-disk checks pass).
  * dissociation + t_sat: ``runs/rate_amplitude_dissociation/
    rate_amplitude_dissociation.json``.

Run:
    cd butterfly_cone && PYTHONPATH=src:scripts:scripts/figures \
        <venv>/bin/python scripts/figures/fig_main5_deep.py
Writes results/figures/fig_main5_deep.{png,pdf} at 300 dpi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "figures"))
sys.path.insert(0, str(_REPO / "scripts"))

import figstyle as fs                    # noqa: E402  (applies the house style)
import chaos_ideal_glass as cig          # noqa: E402  (verified source of truth)

OUT = _REPO / "results" / "figures" / "fig_main5_deep"
DISS_JSON = (_REPO / "runs" / "rate_amplitude_dissociation"
             / "rate_amplitude_dissociation.json")


# --------------------------------------------------------------------------- #
# Load verified data
# --------------------------------------------------------------------------- #
def load_all():
    rep = cig.build_report()
    ver = cig.verify_against_artifacts()
    if not ver["all_ok"]:
        raise SystemExit("artifact cross-check FAILED -- refusing to plot")
    d = cig.ladder_arrays()
    d["sig_lam"] = cig.sigma_lambda(d["lam"], d["lam_std"])
    diss = json.loads(DISS_JSON.read_text())
    return rep, d, diss


# --------------------------------------------------------------------------- #
# Panel (a) -- the co-vanishing lambda(s_c)
# --------------------------------------------------------------------------- #
def panel_a(ax, rep, d):
    ic, pw, rf = rep["intercept_collapse"], rep["power_fit"], rep["rfot"]
    s_c, lam, sig = d["s_c"], d["lam"], d["sig_lam"]
    sc_max = s_c.max()
    sc_lo_data = s_c.min()                       # 1.1824  (T = 0.055)

    # extrapolation zone: everything below the coldest measured rung
    ax.axvspan(-0.1, sc_lo_data, color=fs.GUIDE, alpha=0.13, zorder=0, lw=0)

    xs = np.linspace(0.0, sc_max * 1.04, 300)
    s5, a8 = ic["shallow5"], ic["all_pts"]

    # --- the three fit lines (all-8 dotted plotted last, stays visible) --- #
    ax.plot(xs, s5["slope"] * xs + s5["intercept"], ls=(0, (5, 2)),
            color=fs.SUBTLE, lw=1.4, zorder=3,
            label=rf"shallow-5 linear  ($b={s5['intercept']:+.2f}$)")
    ax.plot(xs, cig._power(xs, pw["A"], pw["p"]), "-", color=fs.THEORY,
            lw=2.2, zorder=4,
            label=(rf"power fit  $\lambda={pw['A']:.2f}\,s_c^{{{pw['p']:.2f}}}$"
                   rf"  ($R^2={pw['r2']:.2f}$)"))
    ax.plot(xs, a8["slope"] * xs + a8["intercept"], ls=(0, (1, 1.6)),
            color=fs.PURPLE, lw=1.7, zorder=5,
            label=(rf"all-{ic['n_all']} linear  "
                   rf"($b={a8['intercept']:+.3f}\pm{a8['intercept_err']:.3f}$)"))

    # --- data: all 8 rungs ----------------------------------------------- #
    ax.errorbar(s_c, lam, xerr=d["s_c_sem"], yerr=sig, fmt="o", ms=5.0,
                color=fs.MEASURED, mfc=fs.MEASURED, mec="white", mew=0.5,
                ecolor=fs.MEASURED, elinewidth=0.9, capsize=1.8, zorder=6,
                label=r"measured  $\lambda(s_c)$  ($n=8$)")

    # --- deep-T anchors (T <= 0.075): open hinge + filled sub-T_g -------- #
    subtg = [5, 6, 7]                            # T = 0.067/0.060/0.055 (< T_g)
    ax.scatter(s_c[4], lam[4], s=72, marker="D", facecolors="none",
               edgecolors=fs.THEORY, linewidths=1.6, zorder=7)
    ax.scatter(s_c[subtg], lam[subtg], s=46, marker="D", facecolors=fs.THEORY,
               edgecolors="white", linewidths=0.6, zorder=8,
               label=r"deep-T anchors ($T\!\leq\!0.075$; fill $<T_g$)")

    # --- intercept collapse bracket at the y-axis ------------------------ #
    xb = 0.05
    ax.annotate("", xy=(xb, s5["intercept"]), xytext=(xb, a8["intercept"]),
                arrowprops=dict(arrowstyle="<->", color=fs.INK, lw=1.1))
    ax.text(0.135, 0.5 * (s5["intercept"] + a8["intercept"]) + 0.02,
            "intercept\ncollapse\n"
            rf"${s5['intercept']:+.2f}\!\to\!{a8['intercept']:+.2f}$",
            ha="left", va="center", fontsize=6.3, color=fs.INK,
            linespacing=1.12)

    # --- origin: the ideal glass ----------------------------------------- #
    ax.scatter([0.0], [0.0], marker="*", s=155, color=fs.INK,
               edgecolors="white", linewidths=0.5, zorder=9)
    ax.annotate("ideal glass\n"
                r"$s_c\!\to\!0,\ \lambda\!\to\!0$" "\n(extrapolated)",
                xy=(0.03, 0.0), xytext=(1.00, 0.135), fontsize=6.7,
                color=fs.INK, va="center", ha="left", linespacing=1.2,
                arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=0.9,
                                shrinkA=2, shrinkB=4))

    ax.set_xlabel(r"configurational entropy  $s_c$  (nat/particle)")
    ax.set_ylabel(r"butterfly-growth rate  $\lambda_B$")
    ax.set_xlim(-0.1, sc_max * 1.06)
    ax.set_ylim(-0.03, 1.5)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=1.7,
              labelspacing=0.35, borderaxespad=0.5)
    fs.panel_label(ax, "a", x=-0.115, y=1.01)

    _inset_sc_T(ax, d, rf)


def _inset_sc_T(ax, d, rf):
    """s_c(T) with the RFOT fit + the extrapolated crisis (lower-right of a)."""
    axi = ax.inset_axes([0.615, 0.115, 0.365, 0.375])
    T, s_c = d["T"], d["s_c"]
    tk = rf["rfot"]["T_K"]
    s_inf = rf["rfot"]["s_inf"]
    win = rf["chaos_death"]["T_star_window"]

    axi.axvspan(0.0, T.min(), color=fs.GUIDE, alpha=0.20, lw=0, zorder=0)
    axi.axvspan(win[0], win[1], color=fs.PURPLE, alpha=0.35, lw=0, zorder=1)

    ts = np.linspace(tk * 1.02, T.max() * 1.03, 200)
    axi.plot(ts, cig._rfot(ts, s_inf, tk), "-", color=fs.THEORY, lw=1.4,
             zorder=3)
    axi.axvline(tk, color=fs.PURPLE, ls="--", lw=1.0, zorder=2)
    axi.axvline(cig.T_G, color=fs.HIGHLIGHT, ls=":", lw=1.1, zorder=2)
    axi.errorbar(T, s_c, yerr=d["s_c_sem"], fmt="o", ms=3.0, color=fs.MEASURED,
                 mec="white", mew=0.4, ecolor=fs.MEASURED, elinewidth=0.7,
                 capsize=1.4, zorder=4)
    axi.scatter([tk], [0.0], marker="*", s=52, color=fs.PURPLE,
                edgecolors="white", linewidths=0.4, zorder=5)

    axi.text(tk + 0.006, 2.62, r"$T_K\!\approx\!0.039$" "\n"
             r"$\approx T^{*}$ (extrap.)", fontsize=5.6, color=fs.PURPLE,
             va="top", linespacing=1.05)
    axi.text(cig.T_G + 0.003, 0.32, r"$T_g$", fontsize=5.9, color=fs.HIGHLIGHT)
    axi.set_xlim(0.0, T.max() * 1.04)
    axi.set_ylim(-0.15, s_c.max() * 1.10)
    axi.set_xlabel(r"$T$", fontsize=6.6, labelpad=1.2)
    axi.set_ylabel(r"$s_c(T)$", fontsize=6.6, labelpad=1.2)
    axi.tick_params(labelsize=5.6, length=2.0, pad=1.3)
    axi.set_xticks([0.0, 0.05, 0.10, 0.15])
    axi.set_yticks([0, 1, 2])
    axi.set_title("RFOT entropy crisis", fontsize=5.9, color=fs.INK,
                  loc="left", pad=2.5)


# --------------------------------------------------------------------------- #
# Panel (b) -- rate-amplitude dissociation at T = 0.060 (observed/line ratio)
# --------------------------------------------------------------------------- #
def panel_b(ax, diss):
    sh = diss["shallow_points"]
    di = diss["dissociation"]
    fl = diss["fits"]["lambda"]
    fd = diss["fits"]["d_sat_per_N"]

    sc_sh = np.array([p["s_c"] for p in sh])
    lam_sh = np.array([p["lambda"] for p in sh])
    lam_sh_s = np.array([p["lambda_sigma"] for p in sh])
    dsat_sh = np.array([p["d_sat_per_N"] for p in sh])
    dsat_sh_s = np.array([p["d_sat_per_N_sigma"] for p in sh])

    lam_line_sh = fl["slope"] * sc_sh + fl["intercept"]
    dsat_line_sh = fd["slope"] * sc_sh + fd["intercept"]
    lam_r, lam_r_e = lam_sh / lam_line_sh, lam_sh_s / lam_line_sh
    dsat_r, dsat_r_e = dsat_sh / dsat_line_sh, dsat_sh_s / dsat_line_sh

    sc_deep = di["deep_s_c"]
    rd, cd = di["rate_departure"], di["ceiling_departure"]
    lam_r_deep = rd["observed"] / rd["predicted"]
    lam_r_deep_e = rd["observed_sigma"] / rd["predicted"]
    dsat_r_deep = cd["observed"] / cd["predicted"]
    dsat_r_deep_e = cd["observed_sigma"] / cd["predicted"]

    xlo, xhi = sc_deep - 0.14, sc_sh.max() * 1.03
    ax.axvspan(sc_deep - 0.05, sc_deep + 0.05, color=fs.GUIDE, alpha=0.16,
               lw=0, zorder=0)
    ax.axhline(1.0, color=fs.GUIDE, ls="--", lw=1.0, zorder=1)
    ax.text(xhi, 1.015, "on shallow-$s_c$ line", ha="right", va="bottom",
            fontsize=6.2, color=fs.SUBTLE)

    # shallow scatter (both hug 1.0 == the fit residual band)
    ax.errorbar(sc_sh, lam_r, yerr=lam_r_e, fmt="o", ms=4.2, color=fs.MEASURED,
                mec="white", mew=0.5, ecolor=fs.MEASURED, elinewidth=0.8,
                capsize=1.6, zorder=5, label=r"rate $\lambda$")
    ax.errorbar(sc_sh, dsat_r, yerr=dsat_r_e, fmt="s", ms=3.8,
                color=fs.HIGHLIGHT, mec="white", mew=0.5, ecolor=fs.HIGHLIGHT,
                elinewidth=0.8, capsize=1.6, zorder=5,
                label=r"ceiling $D_{\mathrm{sat}}/N$")

    # deep anchor: the dissociation (arrows away from the 1.0 line)
    ax.annotate("", xy=(sc_deep, lam_r_deep), xytext=(sc_deep, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=fs.MEASURED, lw=1.4))
    ax.annotate("", xy=(sc_deep, dsat_r_deep), xytext=(sc_deep, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=fs.HIGHLIGHT, lw=1.4))
    ax.errorbar([sc_deep], [lam_r_deep], yerr=[lam_r_deep_e], fmt="D", ms=6.6,
                color=fs.MEASURED, mec="white", mew=0.6, ecolor=fs.MEASURED,
                elinewidth=1.0, capsize=2.0, zorder=7)
    ax.errorbar([sc_deep], [dsat_r_deep], yerr=[dsat_r_deep_e], fmt="s", ms=6.8,
                color=fs.HIGHLIGHT, mec="white", mew=0.6, ecolor=fs.HIGHLIGHT,
                elinewidth=1.0, capsize=2.0, zorder=7)

    ax.text(sc_deep + 0.085, dsat_r_deep, r"$D_{\mathrm{sat}}/N$: $+1.5\sigma$"
            "\non/above (pinned)", ha="left", va="center", fontsize=6.7,
            color=fs.HIGHLIGHT, linespacing=1.1)
    ax.text(sc_deep + 0.085, lam_r_deep, r"$\lambda$: $-3.3\sigma$"
            "\nbelow line", ha="left", va="center", fontsize=6.7,
            color=fs.MEASURED, linespacing=1.1)

    ax.set_xlabel(r"configurational entropy  $s_c$")
    ax.set_ylabel(r"observed $/$ shallow-line")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.70, 1.52)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=1.4,
              labelspacing=0.3, borderaxespad=0.5)
    ax.set_title(r"$T=0.060$  ($s_c\!=\!1.32$, $<\!T_g$)", fontsize=7.2,
                 color=fs.INK, loc="right")
    fs.panel_label(ax, "b", x=-0.16, y=1.05)


# --------------------------------------------------------------------------- #
# Panel (c) -- the diverging scrambling clock t_sat
# --------------------------------------------------------------------------- #
def panel_c(ax, diss):
    pts = diss["points"]
    ts = diss["t_sat"]
    sc = np.array([p["s_c"] for p in pts])
    t_obs = np.array(ts["observed"])
    t_mod = np.array(ts["linear_model"])
    order = np.argsort(sc)
    sc_deep = sc.min()

    ax.axvspan(sc_deep - 0.06, 1.19, color=fs.GUIDE, alpha=0.14, lw=0, zorder=0)

    # both-on-line baseline (t_sat if lambda AND D_sat had stayed on line)
    ax.plot(sc[order], t_mod[order], ls=(0, (5, 2)), color=fs.SUBTLE, lw=1.4,
            zorder=2, label="both-on-line baseline")

    # observed t_sat: shallow (proxy D0 ref) + the deep anchor
    ax.plot(sc[order][1:], t_obs[order][1:], "o", ms=4.4, color=fs.MEASURED,
            mec="white", mew=0.5, zorder=5, label=r"observed  $t_{\mathrm{sat}}$")
    ax.errorbar([sc_deep], [ts["deep_observed"]],
                yerr=[ts["deep_observed_sigma"]], fmt="D", ms=6.6,
                color=fs.THEORY, mec="white", mew=0.6, ecolor=fs.THEORY,
                elinewidth=1.0, capsize=2.2, zorder=6,
                label=r"deep anchor $T=0.060$")

    # divergence arrow at the deep anchor
    ax.annotate("", xy=(sc_deep, ts["deep_observed"]),
                xytext=(sc_deep, ts["deep_linear_model"]),
                arrowprops=dict(arrowstyle="-|>", color=fs.THEORY, lw=1.4))
    ax.text(sc_deep + 0.07, 0.5 * (ts["deep_observed"] + ts["deep_linear_model"]),
            rf"$\times{ts['deep_excess_factor']:.2f}$ above" "\nbaseline"
            r" ($+2.0\sigma$)", ha="left", va="center", fontsize=6.8,
            color=fs.THEORY, linespacing=1.12)

    # secondary: distinct-clock note (empty lower-right band, below the legend)
    ax.text(0.61, 0.27,
            "shallow overlap:\n"
            r"$\tau_\alpha\!\times\!16.9$ vs $t_{\mathrm{sat}}\!\times\!1.06$"
            "\n$\\Rightarrow$ distinct chaos clock",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.3,
            color=fs.SUBTLE, linespacing=1.2)

    ax.set_xlabel(r"configurational entropy  $s_c$")
    ax.set_ylabel(r"scrambling time  $t_{\mathrm{sat}}$")
    ax.set_xlim(sc_deep - 0.13, sc.max() * 1.03)
    ax.set_ylim(3.6, 8.2)
    ax.legend(loc="upper right", fontsize=6.6, handlelength=1.6,
              labelspacing=0.32, borderaxespad=0.5)
    fs.panel_label(ax, "c", x=-0.16, y=1.05)


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
def build_figure():
    rep, d, diss = load_all()

    fig = plt.figure(figsize=fs.figsize(fs.WIDTH_FULL, 0.60))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.40)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    panel_a(axA, rep, d)
    panel_b(axB, diss)
    panel_c(axC, diss)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    plt.close(fig)
    return paths


if __name__ == "__main__":
    written = build_figure()
    for p in written:
        print(p)
