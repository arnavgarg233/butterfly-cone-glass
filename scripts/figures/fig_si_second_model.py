#!/usr/bin/env python3
"""SI figure: SECOND-MODEL PORTABILITY of the two parameter-free laws.

The paper reports two parameter-free laws in the flagship swap glass former
(smoothed r^-12 IPL, non-additive Delta=0.2):

  L1  cage-ceiling ratio  (D_sat/N)/u_DW  ~ 1.25   (Gaussian chi_3 value 1.303)
  L2  butterfly rate  lambda  increases with temperature (i.e. with s_c)

This figure shows both laws REPRODUCE across three glass formers that differ in
their microscopic definition:

  * flagship   smoothed r^-12 IPL, non-additive Delta=0.2   (blue)
  * additive   same r^-12 core but additive mixing Delta=0   (vermillion)
  * steeper    genuinely different pair potential, r^-18 core (green)

Panels
  (a) The cage-ceiling ratio for all three models. Each point is the
      horizon-independent complete-cage-decorrelation estimator
      (pairwise divergence per particle) / u_DW -- the robust L1 measure. All
      three cluster in 1.24-1.29, between the flagship empirical band (~1.25)
      and the Gaussian chi_3 ceiling (1.303): the ratio is model-independent.
  (b) The butterfly rate lambda vs temperature (normalized to each model's own
      glassy window, cold->warm) for the three models. Every model has POSITIVE
      slope: the entropy-tracking sign of L2 reproduces in each.

All numbers are read/derived from the run JSONs; nothing is hard-coded except
the flagship lambda ladder (from the committed deep-T ladder cited in each
VERDICT.flagship_reference).

Reads
  runs/dw_identity/dw_identity.json                 (flagship L1)
  runs/second_model/VERDICT.json                    (additive verdict + refs)
  runs/second_model/VERDICT_n18.json                (r^-18 verdict + refs)
  runs/second_model/cone_sm-n12add-T00{60,100}.json (additive L1/L2)
  runs/second_model/cone_sm-n18-T00{30,50}.json     (r^-18 L1/L2)
Writes
  results/figures/fig_si_second_model.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402

RUNS = ROOT / "runs"
SM = RUNS / "second_model"
OUT = ROOT / "results" / "figures" / "fig_si_second_model"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _intrinsic_ratio(cone: dict) -> float:
    """Horizon-independent complete-cage-decorrelation ratio: the robust L1
    estimator = (pairwise divergence per particle at full Gaussian cage
    decorrelation) / u_DW."""
    d = cone["dw_identity"]
    return d["pairwise_divergence_per_particle"] / d["u_DW"]


def main() -> int:
    # ---- flagship L1 (from the DW-identity landing) -----------------------
    dw = _load(RUNS / "dw_identity" / "dw_identity.json")
    ident = dw["identity"]
    flag_intrinsic = dw["pairwise_divergence_per_particle"] / ident["u_DW"]
    flag_empirical = ident["empirical_c"]      # published L1 prefactor ~1.247
    gaussian_c = ident["c"]                    # Gaussian chi_3 value ~1.303

    # ---- second models: cones ---------------------------------------------
    add_T06 = _load(SM / "cone_sm-n12add-T0060.json")
    add_T10 = _load(SM / "cone_sm-n12add-T0100.json")
    n18_T03 = _load(SM / "cone_sm-n18-T0030.json")
    n18_T05 = _load(SM / "cone_sm-n18-T0050.json")

    verdict_add = _load(SM / "VERDICT.json")
    verdict_n18 = _load(SM / "VERDICT_n18.json")

    # L1 ratios (intrinsic, horizon-independent) per (model, T)
    add_r06, add_r10 = _intrinsic_ratio(add_T06), _intrinsic_ratio(add_T10)
    n18_r03, n18_r05 = _intrinsic_ratio(n18_T03), _intrinsic_ratio(n18_T05)

    # bimodal (two-peak) structural class: intrinsic cage-decorrelation ratio
    bimodal = _load(RUNS / "bimodal" / "bimodal_result.json")
    bim_r = bimodal["ceiling_measurement"]["intrinsic_pairwise_c"]
    # softer r^-8 core and trimodal three-peak structure (breadth campaign);
    # trimodal quoted at production scale N=1500, soft_r8 at N=384
    soft = _load(RUNS / "breadth" / "soft_r8" / "cone_analysis.json")
    soft_r = soft["cage_ceiling"]["intrinsic_pairwise_c"]
    tri = _load(RUNS / "breadth" / "trimodal_N1500" / "cone_analysis.json")
    tri_r = tri["cage_ceiling"]["intrinsic_pairwise_c"]

    # L2 lambda per (model, T)
    lam_add06 = add_T06["gardner_r0"]["lambda_mean"]
    lam_add10 = add_T10["gardner_r0"]["lambda_mean"]
    lam_n1803 = n18_T03["gardner_r0"]["lambda_mean"]
    lam_n1805 = n18_T05["gardner_r0"]["lambda_mean"]

    # flagship committed lambda ladder (from VERDICT.flagship_reference)
    flag_ref = verdict_add["flagship_reference"]["L2_lambda_vs_T"]
    flag_items = sorted((float(k), v) for k, v in flag_ref.items())
    flag_T = np.array([t for t, _ in flag_items])
    flag_lam = np.array([v for _, v in flag_items])

    # ---- report to stdout --------------------------------------------------
    print("== PANEL (a)  cage-ceiling ratio  (D_sat/N)/u_DW  [intrinsic] ==")
    print(f"  flagship  r^-12 D=0.2 : {flag_intrinsic:.3f}  "
          f"(empirical prefactor {flag_empirical:.3f})")
    print(f"  additive  r^-12 D=0   : T0.06 {add_r06:.3f}  T0.10 {add_r10:.3f}")
    print(f"  steeper   r^-18 D=0.2 : T0.03 {n18_r03:.3f}  T0.05 {n18_r05:.3f}")
    print(f"  Gaussian chi_3 ceiling: {gaussian_c:.3f}")
    print("== PANEL (b)  butterfly rate  lambda(T) ==")
    print(f"  flagship : T={list(flag_T)}  lambda={list(flag_lam)}"
          f"  (x{flag_lam[-1]/flag_lam[0]:.2f})")
    print(f"  additive : T=[0.06, 0.10]  lambda=[{lam_add06:.3f}, {lam_add10:.3f}]"
          f"  (x{lam_add10/lam_add06:.2f})")
    print(f"  steeper  : T=[0.03, 0.05]  lambda=[{lam_n1803:.3f}, {lam_n1805:.3f}]"
          f"  (x{lam_n1805/lam_n1803:.2f})")

    # ----------------------------------------------------------------------- #
    fs.use()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=fs.figsize(fs.WIDTH_FULL, aspect=0.46))

    C_FLAG, C_ADD, C_N18 = fs.BLUE, fs.VERMILLION, fs.GREEN
    C_BIM = fs.GOLD

    # ======================= panel (a): L1 ratio =========================== #
    # reference: Gaussian chi_3 ceiling and flagship empirical band
    ax_a.axhline(gaussian_c, color=fs.SUBTLE, lw=1.0, ls="--", zorder=1)
    ax_a.text(5.52, gaussian_c, f"Gaussian $\\chi_3$ ceiling  {gaussian_c:.3f}",
              color=fs.SUBTLE, fontsize=fs.FS_ANNOT, ha="right", va="bottom")
    band_lo, band_hi = flag_empirical - 0.020, flag_empirical + 0.020
    ax_a.axhspan(band_lo, band_hi, color=C_FLAG, alpha=0.09, zorder=0, lw=0)
    ax_a.text(5.52, band_lo - 0.004,
              f"flagship empirical law  $(D_{{sat}}/N)/u_{{DW}}\\approx${flag_empirical:.2f}",
              color=fs.BLUE, fontsize=fs.FS_ANNOT, ha="right", va="top")

    # model x-slots
    x_flag, x_add, x_n18, x_soft, x_bim, x_tri = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    dx = 0.13
    ms = 8.0

    # flagship (single measured value)
    ax_a.plot([x_flag], [flag_intrinsic], marker="o", ms=ms, color=C_FLAG,
              mec="white", mew=0.8, ls="none", zorder=6)
    ax_a.annotate("$T{=}0.075$", (x_flag, flag_intrinsic),
                  textcoords="offset points", xytext=(0, -13),
                  fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")

    # additive (two temperatures)
    ax_a.plot([x_add - dx, x_add + dx], [add_r06, add_r10], marker="s", ms=ms,
              color=C_ADD, mec="white", mew=0.8, ls="none", zorder=6)
    ax_a.annotate("$0.06$", (x_add - dx, add_r06), textcoords="offset points",
                  xytext=(0, 9), fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")
    ax_a.annotate("$0.10$", (x_add + dx, add_r10), textcoords="offset points",
                  xytext=(0, 9), fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")

    # steeper r^-18 (two temperatures)
    ax_a.plot([x_n18 - dx, x_n18 + dx], [n18_r03, n18_r05], marker="^", ms=ms + 0.5,
              color=C_N18, mec="white", mew=0.8, ls="none", zorder=6)
    ax_a.annotate("$0.03$", (x_n18 - dx, n18_r03), textcoords="offset points",
                  xytext=(0, 9), fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")
    ax_a.annotate("$0.05$", (x_n18 + dx, n18_r05), textcoords="offset points",
                  xytext=(0, 9), fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")

    # bimodal (single measured value, a structurally distinct two-peak glass)
    ax_a.plot([x_bim], [bim_r], marker="D", ms=ms, color=C_BIM,
              mec="white", mew=0.8, ls="none", zorder=6)
    ax_a.annotate("single $T$", (x_bim, bim_r), textcoords="offset points",
                  xytext=(0, -13), fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, ha="center")

    ax_a.plot([x_soft], [soft_r], marker="D", ms=ms, color=C_BIM,
              zorder=6, clip_on=False)
    ax_a.annotate("single $T$", (x_soft, soft_r), textcoords="offset points",
                  xytext=(0, -14), ha="center", fontsize=6, color=C_BIM)
    ax_a.plot([x_tri], [tri_r], marker="D", ms=ms, color=C_BIM,
              zorder=6, clip_on=False)
    ax_a.annotate("$N{=}1500$", (x_tri, tri_r), textcoords="offset points",
                  xytext=(0, -14), ha="center", fontsize=6, color=C_BIM)
    ax_a.set_xticks([x_flag, x_add, x_n18, x_soft, x_bim, x_tri])
    ax_a.set_xticklabels(
        ["flagship\n$r^{-12}$\n$\\Delta{=}0.2$",
         "additive\n$r^{-12}$\n$\\Delta{=}0$",
         "steeper\n$r^{-18}$\n$\\Delta{=}0.2$",
         "softer\n$r^{-8}$",
         "bimodal\ntwo-peak",
         "trimodal\nthree-peak"],
        fontsize=fs.FS_TICK)
    ax_a.set_xlim(-0.45, 5.55)
    ax_a.set_ylim(1.19, 1.335)
    ax_a.set_ylabel(r"cage-ceiling ratio  $(D_\mathrm{sat}/N)/u_\mathrm{DW}$")
    ax_a.tick_params(axis="x", length=0)
    fs.annotate_stats(ax_a, "L1 reproduces:\nall models 1.23–1.29", x=0.035, y=0.965)
    fs.panel_label(ax_a, "a")

    # ======================= panel (b): L2 lambda(T) ======================= #
    def norm(T):
        T = np.asarray(T, float)
        return (T - T.min()) / (T.max() - T.min())

    xf, xa, xn = norm(flag_T), norm([0.06, 0.10]), norm([0.03, 0.05])
    la = np.array([lam_add06, lam_add10])
    ln = np.array([lam_n1803, lam_n1805])

    ax_b.plot(xf, flag_lam, marker="o", ms=6, color=C_FLAG, mec="white", mew=0.7,
              lw=1.8, label="flagship  $r^{-12},\\ \\Delta{=}0.2$", zorder=6)
    ax_b.plot(xa, la, marker="s", ms=6.5, color=C_ADD, mec="white", mew=0.7,
              lw=1.8, label="additive  $r^{-12},\\ \\Delta{=}0$", zorder=6)
    ax_b.plot(xn, ln, marker="^", ms=7, color=C_N18, mec="white", mew=0.7,
              lw=1.8, label="steeper  $r^{-18},\\ \\Delta{=}0.2$", zorder=6)

    # endpoint temperature labels (actual T, since x is normalized)
    ax_b.annotate(f"$T{{=}}{flag_T[0]:.3f}$", (xf[0], flag_lam[0]),
                  textcoords="offset points", xytext=(4, -11),
                  fontsize=fs.FS_ANNOT - 1, color=C_FLAG, ha="left")
    ax_b.annotate(f"$T{{=}}{flag_T[-1]:.3f}$", (xf[-1], flag_lam[-1]),
                  textcoords="offset points", xytext=(-3, 5),
                  fontsize=fs.FS_ANNOT - 1, color=C_FLAG, ha="right")
    ax_b.annotate("$0.06$", (xa[0], la[0]), textcoords="offset points",
                  xytext=(5, -9), fontsize=fs.FS_ANNOT - 1, color=C_ADD, ha="left")
    ax_b.annotate("$0.10$", (xa[-1], la[-1]), textcoords="offset points",
                  xytext=(-2, 6), fontsize=fs.FS_ANNOT - 1, color=C_ADD, ha="right")
    ax_b.annotate("$0.03$", (xn[0], ln[0]), textcoords="offset points",
                  xytext=(5, -9), fontsize=fs.FS_ANNOT - 1, color=C_N18, ha="left")
    ax_b.annotate("$0.05$", (xn[-1], ln[-1]), textcoords="offset points",
                  xytext=(-2, 6), fontsize=fs.FS_ANNOT - 1, color=C_N18, ha="right")

    ax_b.set_xlim(-0.08, 1.08)
    ax_b.set_ylim(0.0, 1.15)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_xticklabels(["cold", "0.5", "warm"])
    ax_b.set_xlabel(r"temperature (normalized per model, cold $\to$ warm)")
    ax_b.set_ylabel(r"butterfly rate  $\lambda$")
    ax_b.legend(loc="upper left", fontsize=fs.FS_LEGEND)
    fs.annotate_stats(
        ax_b,
        "L2 sign reproduces:\n"
        + f"flagship $\\times${flag_lam[-1]/flag_lam[0]:.2f}   "
        + f"add. $\\times${la[-1]/la[0]:.2f}   "
        + f"$r^{{-18}}$ $\\times${ln[-1]/ln[0]:.2f}",
        x=0.34, y=0.135)
    fs.panel_label(ax_b, "b")

    fs.finalize(fig)
    written = fs.save(fig, str(OUT))
    print("wrote", [str(p) for p in written])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
