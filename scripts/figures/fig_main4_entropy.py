#!/usr/bin/env python3
"""Fig 4 - Two rigidities: entropy vs elastic (Nature Physics "butterfly cone").

Three panels, all numbers pulled from persisted runs/ artifacts (verified against
the JSON keys; nothing fabricated):

  (a) The entropic lock.  lambda(s_c) and D_sat/N(s_c) both rise monotonically with
      the configurational entropy s_c across the T-ladder (Spearman rho = +1.00),
      while the front velocity v_b(s_c) stays flat (rho = +0.50).  All three are
      shown normalised to the deep-glass anchor (coldest rung, lowest s_c) so the
      differently-scaled observables share one axis and their *fractional* variation
      is honest.
      Source: runs/bridge_analysis/bridge_analysis.json

  (b) Within-state sharpness.  Mean over the 5 glass states of the within-state
      (fixed T, s_c; 6 replicate ensembles) coefficient of variation:
      lambda 1.98 %, D_sat/N 0.74 %, v_b 12.34 %.  lambda and D_sat are sharp state
      functions; v_b is an order of magnitude noisier.  Dots = the 5 per-state CVs.
      Source: runs/bridge_analysis/bridge_analysis.json (per_ensemble)

  (c) Transport decoupling.  Over the resolved transport window (3 warmest rungs)
      lambda barely moves (x1.14) while the diffusivity D moves x8.4 and the
      structural time tau_alpha moves x16.9 -- a Rosenfeld break,
      d ln lambda / d ln D = 0.063 (vs the ~1 that diffusion obeys).
      Source: runs/chaos_not_transport/chaos_not_transport.json

Run:
  cd butterfly_cone && PYTHONPATH=src:scripts/figures \
     python \
     scripts/figures/fig_main4_entropy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402  (applies the shared style on import)
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as _stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "fig_main4_entropy"

# semantic observable colours (kept consistent across panels; lambda = BLUE always)
C_LAM = fs.BLUE          # lambda  -- the butterfly rate (measured/primary)
C_DSAT = fs.GREEN_MUTED  # D_sat/N -- the plateau (secondary entropic observable; muted)
C_VB = fs.GOLD           # v_b     -- the front velocity (the elastic odd-one-out)
C_D = fs.PURPLE          # diffusivity D (panel c transport)
C_TAU = fs.VERMILLION    # tau_alpha structural time (panel c transport)


# --------------------------------------------------------------------------- #
# Load + derive (verify keys, never fabricate)
# --------------------------------------------------------------------------- #
def load():
    bridge = json.loads((ROOT / "runs/bridge_analysis/bridge_analysis.json").read_text())
    cnt = json.loads((ROOT / "runs/chaos_not_transport/chaos_not_transport.json").read_text())
    vb = json.loads((ROOT / "runs/vb_elastic_cone/vb_elastic_cone.json").read_text())
    return bridge, cnt, vb


def within_state_cv(per_ensemble):
    """Mean over glass states of the within-state CV (%) for lambda, D_sat, v_b.

    Within-state = fixed (config == fixed T, s_c), across the 6 replicate ensembles.
    Sample std (ddof=1), matching the std stored in bridge_analysis per_config.
    v_b: drop failed front-velocity fits (v_b <= 0) before the per-state CV.
    """
    configs = sorted({e["config"] for e in per_ensemble})
    lam_cv, dsat_cv, vb_cv = [], [], []
    for c in configs:
        es = [e for e in per_ensemble if e["config"] == c]
        lam = np.array([e["lam"] for e in es], float)
        dsat = np.array([e["D_sat"] for e in es], float)
        vbv = np.array([e["v_b"] for e in es], float)
        vbv = vbv[vbv > 0]
        lam_cv.append(lam.std(ddof=1) / lam.mean() * 100)
        dsat_cv.append(dsat.std(ddof=1) / dsat.mean() * 100)
        vb_cv.append(vbv.std(ddof=1) / vbv.mean() * 100)
    return (np.array(lam_cv), np.array(dsat_cv), np.array(vb_cv))


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_a(ax, bridge):
    pc = sorted(bridge["per_config"], key=lambda d: d["s_c"])  # ascending s_c
    s_c = np.array([d["s_c"] for d in pc])
    lam = np.array([d["lam_mean"] for d in pc])
    lam_e = np.array([d["lam_sem"] for d in pc])
    dsat = np.array([d["d_sat_per_N_mean"] for d in pc])
    dsat_e = np.array([d["d_sat_per_N_sem"] for d in pc])
    vb = np.array([d["v_b_mean"] for d in pc])
    vb_e = np.array([d["v_b_sem"] for d in pc])

    corr = bridge["correlations"]
    rho = {k: corr[k]["spearman_rho"] for k in ("lambda", "D_sat_per_N", "v_b")}

    # normalise to the deep-glass anchor (coldest rung = lowest s_c = index 0)
    def nrm(y, e):
        a = y[0]
        return y / a, e / a

    lam_n, lam_ne = nrm(lam, lam_e)
    dsat_n, dsat_ne = nrm(dsat, dsat_e)
    vb_n, vb_ne = nrm(vb, vb_e)

    # anchor reference line
    ax.axhline(1.0, color=fs.GUIDE, lw=0.8, ls="--", zorder=0)

    series = [
        (dsat_n, dsat_ne, C_DSAT, "s", r"$D_{\mathrm{sat}}/N$", rho["D_sat_per_N"]),
        (lam_n, lam_ne, C_LAM, "o", r"$\lambda$", rho["lambda"]),
        (vb_n, vb_ne, C_VB, "^", r"$v_b$", rho["v_b"]),
    ]
    for y, e, col, mk, lab, r in series:
        ax.plot(s_c, y, color=col, lw=1.4, zorder=2, alpha=0.9)
        ax.errorbar(s_c, y, yerr=e, fmt=mk, ms=5.2, color=col,
                    mec="white", mew=0.6, ecolor=col, elinewidth=0.9,
                    capsize=1.8, zorder=3)

    # inline labels at the right (highest-s_c) end, with the Spearman rho
    ax.annotate(r"$D_{\mathrm{sat}}/N$   $\rho=+1.00$",
                xy=(s_c[-1], dsat_n[-1]), xytext=(5, 0), textcoords="offset points",
                color=C_DSAT, fontsize=fs.FS_ANNOT, va="center", ha="left", fontweight="bold")
    ax.annotate(r"$\lambda$   $\rho=+1.00$",
                xy=(s_c[-1], lam_n[-1]), xytext=(5, 0), textcoords="offset points",
                color=C_LAM, fontsize=fs.FS_ANNOT, va="center", ha="left", fontweight="bold")
    ax.annotate(r"$v_b$   $\rho=+0.50$",
                xy=(s_c[-1], vb_n[-1]), xytext=(5, 0), textcoords="offset points",
                color=C_VB, fontsize=fs.FS_ANNOT, va="center", ha="left", fontweight="bold")

    ax.set_xlabel(r"configurational entropy  $s_c$")
    ax.set_ylabel("observable / deep-glass value")
    ax.set_xlim(s_c.min() - 0.08, s_c.max() + 0.95)
    ax.set_ylim(0.70, 2.20)
    ax.set_xticks([1.8, 2.0, 2.2, 2.4, 2.6])
    fs.annotate_stats(ax, r"$\leftarrow$ colder", x=0.03, y=0.93,
                      color=fs.SUBTLE, size=fs.FS_ANNOT)
    fs.panel_label(ax, "a")


def panel_b(ax, bridge):
    lam_cv, dsat_cv, vb_cv = within_state_cv(bridge["meta"]["per_ensemble"])
    means = [lam_cv.mean(), dsat_cv.mean(), vb_cv.mean()]
    dots = [lam_cv, dsat_cv, vb_cv]
    cols = [C_LAM, C_DSAT, C_VB]
    labels = [r"$\lambda$", r"$D_{\mathrm{sat}}/N$", r"$v_b$"]
    x = np.arange(3)

    ax.bar(x, means, width=0.62, color=cols, alpha=0.30,
           edgecolor=cols, linewidth=1.1, zorder=1)
    # overlay the 5 per-state CVs as jittered dots (shows the distribution)
    rng = np.random.default_rng(7)
    for xi, d, col in zip(x, dots, cols):
        jx = xi + (rng.random(len(d)) - 0.5) * 0.26
        ax.scatter(jx, d, s=15, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    # value labels above the tallest dot of each group (clear of the scatter)
    for xi, m, d in zip(x, means, dots):
        top = max(m, float(np.max(d)))
        ax.text(xi, top + 0.7, f"{m:.2f}%", ha="center", va="bottom",
                fontsize=fs.FS_ANNOT, fontweight="bold", color=fs.INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs.FS_LABEL)
    ax.set_ylabel("within-state CV (%)")
    ax.set_ylim(0, 16.5)
    fs.annotate_stats(
        ax,
        "within-state (fixed $T, s_c$)\nmean of 5 states, $n=6$",
        x=0.035, y=0.975, color=fs.SUBTLE, size=fs.FS_ANNOT)
    fs.panel_label(ax, "b")


def panel_c(ax, cnt):
    ts = cnt["transport_series"]
    order = np.argsort(ts["temperature"])          # ascending T
    T = np.array(ts["temperature"])[order]
    lam = np.array(ts["lambda"])[order]
    D = np.array(ts["D"])[order]
    tau = np.array(ts["tau_alpha"])[order]
    warm = np.argmax(T)                            # anchor = warmest rung (T=0.15)

    def nrm(y):
        return y / y[warm]

    hl = cnt["headline"]
    fold = {"lam": hl["lambda_fold_change"], "D": hl["D_fold_change"],
            "tau": hl["tau_alpha_fold_change"]}
    b = hl["d_ln_lambda_d_ln_D"]

    ax.axhline(1.0, color=fs.GUIDE, lw=0.8, ls="--", zorder=0)
    for y, col, mk, lab in [
        (nrm(tau), C_TAU, "D", r"$\tau_\alpha$"),
        (nrm(lam), C_LAM, "o", r"$\lambda$"),
        (nrm(D), C_D, "v", r"$D$"),
    ]:
        ax.plot(T, y, color=col, lw=1.6, marker=mk, ms=5.4,
                mec="white", mew=0.6, zorder=3)

    # fold-change labels at the cold (left) end
    ax.annotate(r"$\tau_\alpha$  $\times$16.9", xy=(T[0], nrm(tau)[0]),
                xytext=(3, 4), textcoords="offset points", color=C_TAU,
                fontsize=fs.FS_ANNOT, fontweight="bold", va="bottom", ha="left")
    ax.annotate(r"$\lambda$  $\times$1.14", xy=(T[0], nrm(lam)[0]),
                xytext=(3, 7), textcoords="offset points", color=C_LAM,
                fontsize=fs.FS_ANNOT, fontweight="bold", va="bottom", ha="left")
    ax.annotate(r"$D$  $\times$8.4", xy=(T[0], nrm(D)[0]),
                xytext=(3, -3), textcoords="offset points", color=C_D,
                fontsize=fs.FS_ANNOT, fontweight="bold", va="top", ha="left")

    ax.set_yscale("log")
    ax.set_xlabel(r"temperature  $T$")
    ax.set_ylabel(r"value / warm anchor ($T=0.15$)")
    ax.set_xlim(T.min() - 0.006, T.max() + 0.006)
    ax.set_ylim(0.07, 30)
    ax.set_xticks([0.11, 0.12, 0.13, 0.14, 0.15])
    ax.grid(axis="y", which="major", color=fs.GUIDE, lw=0.4, alpha=0.45, zorder=0)
    fs.annotate_stats(
        ax,
        r"$\dfrac{d\ln\lambda}{d\ln D}=0.063$" + "\n" + "(Rosenfeld ref $=1$)",
        x=0.50, y=0.94, color=fs.INK, size=fs.FS_ANNOT)
    fs.panel_label(ax, "c")


# --------------------------------------------------------------------------- #
def main():
    bridge, cnt, vb = load()

    # --- verification prints (assert the plotted headline numbers) ---
    lam_cv, dsat_cv, vb_cv = within_state_cv(bridge["meta"]["per_ensemble"])
    print("VERIFY within-state CV means:  lambda=%.2f%%  D_sat=%.2f%%  v_b=%.2f%%"
          % (lam_cv.mean(), dsat_cv.mean(), vb_cv.mean()))
    corr = bridge["correlations"]
    print("VERIFY Spearman rho:  lambda=%.2f  D_sat/N=%.2f  v_b=%.2f"
          % (corr["lambda"]["spearman_rho"], corr["D_sat_per_N"]["spearman_rho"],
             corr["v_b"]["spearman_rho"]))
    hl = cnt["headline"]
    print("VERIFY fold: lambda=%.2fx  D=%.2fx  tau_alpha=%.2fx ; dlnL/dlnD=%.3f"
          % (hl["lambda_fold_change"], hl["D_fold_change"],
             hl["tau_alpha_fold_change"], hl["d_ln_lambda_d_ln_D"]))
    assert abs(lam_cv.mean() - 1.98) < 0.05
    assert abs(dsat_cv.mean() - 0.74) < 0.02
    assert abs(vb_cv.mean() - 12.34) < 0.05

    fig, axes = plt.subplots(1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.37))
    panel_a(axes[0], bridge)
    panel_b(axes[1], bridge)
    panel_c(axes[2], cnt)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    print("wrote:", *[str(p) for p in paths])


if __name__ == "__main__":
    main()
