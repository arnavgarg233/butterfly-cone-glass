#!/usr/bin/env python3
"""SI figure: RANDOM PINNING at FIXED TEMPERATURE decouples entropy from T.

The temperature ladder confounds configurational entropy s_c with temperature T:
cooling lowers both.  Random pinning breaks that link -- it lowers s_c at a FIXED
bath temperature (T = 0.075) by freezing a fraction f_p of particles in place.
This figure shows that the butterfly (chaos) rate lambda and the cage-saturation
plateau D_sat both fall as s_c falls under pinning, tracking the SAME entropy law
that governs the temperature ladder -- so the ladder's lambda(s_c) trend is driven
by entropy, not by temperature per se.

Panels
  (a) Butterfly rate lambda vs configurational entropy s_c.  Each point is one
      pinning fraction f_p (colour = f_p, deepest/most-pinned navy -> unpinned
      khaki); error bars are the across-ensemble std of lambda.  The dashed line is
      the temperature-ladder law lambda = 0.53 * s_c^0.97 evaluated over the same
      s_c range.  At FIXED T, pinning walks the system down this same curve: the
      rate falls because the entropy falls.
  (b) Both parameter-free observables normalised to the unpinned (f_p = 0) state,
      vs f_p: the chaos rate lambda and the cage plateau D_sat/mobile fall together,
      slightly STEEPER than the entropy-only ladder reference (1 - f_p)^0.97.  Inset:
      the front speed v_b RISES with f_p while lambda falls -- the two rigidities
      (chaotic vs elastic) split.

All numbers are read from the persisted run JSON; nothing is hard-coded except the
committed ladder-law constants (0.53, 0.97) named in the run's reference_ladder.

Reads
  runs/pinning_fixedT/pinning_fixedT.json
Writes
  results/figures/fig_si_pinning.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs  # noqa: E402  (applies the shared style on import)
import matplotlib.pyplot as plt  # noqa: E402

SRC = ROOT / "runs" / "pinning_fixedT" / "pinning_fixedT.json"
OUT = ROOT / "results" / "figures" / "fig_si_pinning"

# ladder-law constants (from run reference_ladder: "lambda = 0.53 * s_c^0.97")
LADDER_A, LADDER_B = 0.53, 0.97

# semantic observable colours (consistent with the main two-rigidities figure)
C_LAM = fs.BLUE          # lambda  -- butterfly / chaos rate (measured primary)
C_DSAT = fs.GREEN_MUTED  # D_sat   -- cage-saturation plateau (secondary entropic)
C_LADDER = fs.VERMILLION  # entropy-ladder law / prediction (dashed theory line)
C_VB = fs.GOLD           # v_b     -- elastic front speed (the odd-one-out)


def load() -> dict:
    return json.loads(SRC.read_text())


def main() -> int:
    data = load()
    an = data["analysis"]
    per = {row["f_p"]: row for row in data["per_f_p"]}
    T = data["temperature"]

    f_p = np.array(an["f_p"], float)
    s_c = np.array(an["s_c_estimate"], float)
    lam = np.array(an["lambda_mean"], float)
    dsat = np.array(an["D_sat_per_mobile_mean"], float)
    lam_ratio = np.array(an["lambda_ratio_to_fp0"], float)
    dsat_ratio = np.array(an["D_sat_per_mobile_ratio_to_fp0"], float)
    ladder_ratio = np.array(an["ladder_predicted_ratio_(1-fp)^0.97"], float)

    lam_std = np.array([per[f]["lambda_std"] for f in an["f_p"]], float)
    vb = np.array([per[f]["v_b_mean"] for f in an["f_p"]], float)

    # ladder law evaluated at the measured s_c points (for the report)
    ladder_lam_at_pts = LADDER_A * s_c ** LADDER_B

    # ---- verification prints (assert plotted numbers against the JSON) -------
    print(f"== fig_si_pinning  (pinning at FIXED T = {T}) ==")
    print("f_p            :", list(f_p))
    print("s_c_estimate   :", [round(v, 4) for v in s_c])
    print("lambda_mean    :", [round(v, 4) for v in lam])
    print("lambda_std     :", [round(v, 4) for v in lam_std])
    print("D_sat/mobile   :", [round(v, 4) for v in dsat])
    print("lambda_ratio   :", [round(v, 4) for v in lam_ratio])
    print("D_sat_ratio    :", [round(v, 4) for v in dsat_ratio])
    print("ladder_ratio   :", [round(v, 4) for v in ladder_ratio])
    print("v_b_mean       :", [round(v, 4) for v in vb])
    print(f"ladder law     : lambda = {LADDER_A} * s_c^{LADDER_B}")
    print("ladder @ pts   :", [round(v, 4) for v in ladder_lam_at_pts])
    # sanity: monotone falls
    assert np.all(np.diff(lam) < 0), "lambda not monotone in f_p order"
    assert np.all(np.diff(dsat) < 0), "D_sat not monotone in f_p order"
    assert np.all(np.diff(s_c) < 0), "s_c not monotone in f_p order"
    assert np.all(np.diff(vb) > 0), "v_b should rise with f_p"

    # ---------------------------------------------------------------------- #
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=fs.figsize(fs.WIDTH_FULL, aspect=0.50))

    # colour by f_p: deepest / most-pinned = navy, unpinned = khaki (cividis).
    # sequential() index 0 = deepest; f_p descending maps largest f_p -> navy.
    seq = fs.sequential(len(f_p), lo=0.06, hi=0.80)
    order_deep_to_shallow = np.argsort(-f_p)          # f_p 0.2, 0.1, 0.05, 0.0
    col_by_idx = {}
    for c, idx in zip(seq, order_deep_to_shallow):
        col_by_idx[int(idx)] = c

    # ======================= panel (a): lambda(s_c) ======================== #
    # temperature-ladder law across the same s_c range (dashed theory line)
    s_grid = np.linspace(s_c.min() - 0.025, s_c.max() + 0.025, 200)
    lam_grid = LADDER_A * s_grid ** LADDER_B
    ax_a.plot(s_grid, lam_grid, color=C_LADDER, lw=1.5, ls="--", zorder=2)
    ax_a.annotate(
        "temperature-ladder law\n$\\lambda = 0.53\\,s_c^{\\,0.97}$",
        xy=(s_grid[40], lam_grid[40]), xytext=(1.352, 0.615),
        fontsize=fs.FS_ANNOT, color=C_LADDER, va="bottom", ha="left",
        arrowprops=dict(arrowstyle="-", color=C_LADDER, lw=0.7,
                        shrinkA=2, shrinkB=2))

    # faint connector to guide the eye through the falling measured points
    ax_a.plot(s_c, lam, color=fs.GUIDE, lw=1.0, ls="-", zorder=1, alpha=0.9)

    # measured points, coloured by f_p, with across-ensemble error bars
    i_leftmost = int(np.argmax(f_p))                  # most-pinned == lowest s_c
    for i in range(len(f_p)):
        col = col_by_idx[i]
        ax_a.errorbar(s_c[i], lam[i], yerr=lam_std[i], fmt="o", ms=7.0,
                      color=col, mec="white", mew=0.8, ecolor=fs.INK,
                      elinewidth=1.0, capsize=2.4, zorder=5)
        # f_p label BESIDE the marker (horizontal offset) so it clears the
        # vertical error bar; leftmost point labelled to the right, others left.
        if i == i_leftmost:
            dx, ha = 11, "left"
        else:
            dx, ha = -11, "right"
        ax_a.annotate(f"$f_p = {f_p[i]:.2f}$", (s_c[i], lam[i]),
                      textcoords="offset points", xytext=(dx, 2),
                      fontsize=fs.FS_ANNOT - 0.5, color=fs.INK,
                      ha=ha, va="center")

    ax_a.set_xlabel(r"configurational entropy  $s_c$")
    ax_a.set_ylabel(r"butterfly rate  $\lambda$")
    ax_a.set_title(r"pinning lowers $s_c$ at fixed $T$: rate $\lambda$ falls")
    ax_a.set_xlim(1.30, 1.735)
    ax_a.set_ylim(0.58, 0.955)
    ax_a.annotate(r"$\leftarrow$ more pinning", xy=(0.035, 0.90),
                  xycoords="axes fraction", fontsize=fs.FS_ANNOT,
                  color=fs.SUBTLE, ha="left", va="center")
    fs.panel_label(ax_a, "a")

    # ==================== panel (b): normalised ratios ===================== #
    ax_b.axhline(1.0, color=fs.GUIDE, lw=0.8, ls=":", zorder=0)

    # entropy-only ladder reference (1 - f_p)^0.97, smooth curve
    fp_grid = np.linspace(0.0, f_p.max(), 200)
    ax_b.plot(fp_grid, (1.0 - fp_grid) ** LADDER_B, color=C_LADDER, lw=1.5,
              ls="--", zorder=2, label=r"$(1-f_p)^{0.97}$ ladder")

    ax_b.plot(f_p, lam_ratio, marker="o", ms=6.5, color=C_LAM, mec="white",
              mew=0.7, lw=1.7, zorder=5, label=r"chaos rate  $\lambda$")
    ax_b.plot(f_p, dsat_ratio, marker="s", ms=6.0, color=C_DSAT, mec="white",
              mew=0.7, lw=1.7, zorder=4, label=r"saturation  $D_{\mathrm{sat}}$")

    # steeper-than-ladder callout at the deepest pinning
    ax_b.annotate(
        f"$\\times{lam_ratio[-1]:.2f}$ vs ladder $\\times{ladder_ratio[-1]:.2f}$\n"
        r"(chaos & cage soften together,"
        "\n"
        r"steeper than entropy alone)",
        xy=(f_p[-1], lam_ratio[-1]), xytext=(f_p[-1] - 0.006, 0.705),
        fontsize=fs.FS_ANNOT - 0.5, color=fs.INK, ha="right", va="bottom")

    ax_b.set_xlabel(r"pinning fraction  $f_p$")
    ax_b.set_ylabel(r"observable / unpinned ($f_p{=}0$) value")
    ax_b.set_title(r"$\lambda$ and $D_{\mathrm{sat}}$ fall together under pinning")
    ax_b.set_xlim(-0.012, 0.212)
    ax_b.set_ylim(0.66, 1.035)
    ax_b.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax_b.legend(loc="lower left", fontsize=fs.FS_LEGEND, bbox_to_anchor=(0.0, 0.0))
    fs.panel_label(ax_b, "b")

    # inset: front speed v_b RISES with f_p (elastic rigidity, the odd-one-out)
    axin = ax_b.inset_axes([0.575, 0.60, 0.385, 0.355])
    axin.plot(f_p, vb, marker="D", ms=4.6, color=C_VB, mec="white", mew=0.6,
              lw=1.5, zorder=3)
    axin.set_xlim(-0.012, 0.212)
    axin.set_ylim(2.0, 3.9)
    axin.set_xticks([0.0, 0.1, 0.2])
    axin.set_yticks([2.5, 3.0, 3.5])
    axin.tick_params(labelsize=fs.FS_TICK - 2.0, length=2.2, pad=1.5)
    axin.set_title(r"front speed  $v_b\ \uparrow$", fontsize=fs.FS_ANNOT - 0.5,
                   loc="left", pad=2.0, color=C_VB)
    axin.set_xlabel(r"$f_p$", fontsize=fs.FS_ANNOT - 1.5, labelpad=1.0)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    print("wrote:", *[str(p) for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
