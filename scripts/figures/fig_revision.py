#!/usr/bin/env python3
"""Figures added in the JNCS revision.

figure4_deep_rev  (a) measured lambda_B, D_sat/N and s_c against T, no
                  extrapolation; (b) lambda_B against s_c on the cooling ladder
                  and for parents of different annealing T_f run at T = 0.075.
figS1_fitwindows  every native revision run: D(t), the detected fit window, the
                  fitted line and its residuals, at both fit cadences.

Inputs (shipped): results/revision/bridge_front_corrected.json,
results/revision/deep_T00xx_gardner_r0.json, results/revision/summary.json.
figS2 (not called by main) also needs the unshipped runs/revision/*.npz.
Outputs go to results/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO / "results/revision"
FIGS = REPO / "results/figures"
DEEP = {0.067: 1.5023, 0.060: 1.3164, 0.055: 1.1824}   # s_c of the deep states


def ladder():
    d = json.loads((RESULTS / "bridge_front_corrected.json").read_text())
    rows = [(r["temperature"], r["lam_mean"], r["lam_sem"], r["d_sat_per_N_mean"], r["s_c"], False)
            for r in d["per_config"]]
    for t, sc in DEEP.items():
        g = json.loads((RESULTS / f"deep_T{int(round(t * 1000)):04d}_gardner_r0.json").read_text())
        lam = np.array(g["across_config"]["lambda"]["values"])
        dsat = np.array(g["across_config"]["D_sat"]["values"]) / g["ensembles"][0]["N"]
        rows.append((t, lam.mean(), lam.std(ddof=1) / np.sqrt(lam.size), dsat.mean(), sc, True))
    return sorted(rows, key=lambda r: -r[0])


def fig4(summary):
    fs.use()
    rows = ladder()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    T = np.array([r[0] for r in rows])
    ref = rows[0]
    for idx, lab, col, mk in ((1, r"$\lambda_{{\rm B}}$", fs.BLUE, "o"),
                              (3, r"$D_{{\rm sat}}/N$", fs.GREEN, "s"),
                              (4, r"$s_c$", fs.VERMILLION, "^")):
        y = np.array([r[idx] for r in rows]) / ref[idx]
        a.plot(T, y, "-", color=col, lw=1.0, alpha=0.6)
        a.plot(T, y, mk, color=col, ms=4.5, label=lab)
    a.axvline(0.072, color=fs.GUIDE, ls=":", lw=0.8)
    a.text(0.0725, 0.30, r"$T_g$", color=fs.SUBTLE, fontsize=fs.FS_ANNOT)
    a.set_xlabel(r"temperature $T$")
    a.set_ylabel(r"value / value at $T=0.150$")
    a.set_ylim(0.25, 1.05)
    a.legend(frameon=False, loc="lower right", fontsize=fs.FS_LEGEND)
    fs.panel_label(a, "a")

    def by_tf(cond):
        out = {}
        for r in summary:
            if cond(r):
                out.setdefault(round(r["parent_T"], 3), []).append(r["fits"]["abs"]["aggregate"]["cad1"]["lambda_mean"])
        sc_of = {round(x[0], 3): x[4] for x in rows}
        k = sorted(out)
        return (np.array([sc_of[t] for t in k]), np.array([np.mean(out[t]) for t in k]),
                np.array([np.std(out[t], ddof=1) / np.sqrt(len(out[t])) for t in k]))
    x, y, e = by_tf(lambda r: abs(r["run_T"] - r["parent_T"]) < 1e-9)
    b.errorbar(x, y, yerr=e, fmt="o", color=fs.BLUE, ms=5, label=r"run at $T=T_{\rm f}$")
    x, y, e = by_tf(lambda r: abs(r["run_T"] - 0.075) < 1e-9)
    b.errorbar(x, y, yerr=e, fmt="s", color=fs.VERMILLION, ms=5, label=r"run at $T=0.075$")
    b.set_xlabel(r"configurational entropy $s_c(T_{\rm f})$ (nat/particle)")
    b.set_ylabel(r"growth rate $\lambda_{{\rm B}}$ (1/t.u.)")
    b.legend(frameon=False, loc="upper left", fontsize=fs.FS_LEGEND)
    fs.panel_label(b, "b")
    fs.finalize(fig)
    fig.savefig(FIGS / "figure4_deep_rev.pdf")
    fig.savefig(FIGS / "figure4_deep_rev.png", dpi=200)


def figS1(summary):
    fs.use()
    native = sorted([r for r in summary if r["run_T"] == r["parent_T"] and r["rep"] == 0], key=lambda r: -r["run_T"])
    fig, axes = plt.subplots(2, len(native), figsize=(1.6 * len(native) + 0.6, 3.6),
                             gridspec_kw={"height_ratios": [2.2, 1]}, sharex="col")
    t_all = None
    for j, r in enumerate(native):
        t = np.array(r["times"])
        top, bot = axes[0, j], axes[1, j]
        for k, site in enumerate(r["fits"]["abs"]["per_site"]):
            d = np.array(site["D_t"])
            top.semilogy(t, d, color=fs.GRAY, lw=0.7, alpha=0.8)
            for lab, col in (("cad1", fs.BLUE), ("cad025", fs.VERMILLION)):
                f = site[lab]
                if "t_start" not in f:
                    continue
                ft = np.array(f["fit_times"])
                if k == 0:
                    top.axvspan(f["t_start"], f["t_end"], color=col, alpha=0.10, lw=0)
                yfit = np.log(d[np.searchsorted(t, ft)]) - np.array(f["residuals"])
                top.semilogy(ft, np.exp(yfit), color=col, lw=1.0)
                bot.plot(ft, f["residuals"], "o-" if lab == "cad1" else ".-", color=col, ms=2.5, lw=0.6)
        a1 = r["fits"]["abs"]["aggregate"]
        top.set_title(rf"$T={r['run_T']:.3f}$" + "\n" +
                      r"$\lambda_B=$" + f"{a1['cad1']['lambda_mean']:.3f}" + r"$\pm$" + f"{a1['cad1']['lambda_sem']:.3f}",
                      fontsize=fs.FS_ANNOT)
        bot.axhline(0, color=fs.GUIDE, lw=0.6)
        bot.set_xlabel(r"$t$ (t.u.)")
        top.set_xlim(0, 14)
        if j == 0:
            top.set_ylabel(r"$D(t)$")
            bot.set_ylabel("residual\nof $\\ln D$")
    fs.finalize(fig)
    fig.savefig(FIGS / "figS1_fitwindows.pdf")
    fig.savefig(FIGS / "figS1_fitwindows.png", dpi=200)


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig4(summary)
    figS1(summary)


if __name__ == "__main__":
    main()


def figS2():
    """Front radius r_front(t) and the fitted line at both cadences, one parent per temperature."""
    from butterfly_cone.perturb.butterfly import butterfly_velocity
    fs.use()
    temps = [0.150, 0.130, 0.108, 0.090, 0.075]
    fig, axes = plt.subplots(1, len(temps), figsize=(1.6 * len(temps) + 0.6, 2.2), sharey=True)
    for ax, T in zip(axes, temps):
        z = np.load(REPO / f"runs/revision/native_T{T:.3f}_r0.npz")
        t, pos, box = z["times"], z["positions"], float(z["box"])
        for k in range(3):
            field = z[f"field_{k}_abs"]
            for step, col, mk in ((4, fs.BLUE, "o"), (1, fs.VERMILLION, ".")):
                v = butterfly_velocity(t[::step], field[::step], pos, z["centres"][k], box)
                tt = t[::step]
                ax.plot(tt, v.r_front, mk, color=col, ms=2.5 if step == 1 else 3.5, alpha=0.6)
                if v.v_b > 0 and v.n_fit >= 2:
                    i0 = v.onset.start_index
                    xs = tt[i0:i0 + v.n_fit]
                    ax.plot(xs, v.v_b * xs + v.r0, "-", color=col, lw=1.0)
        ax.axhline(0.5 * box, color=fs.GUIDE, ls="--", lw=0.7)
        ax.set_xlim(0, 8)
        ax.set_title(rf"$T={T:.3f}$", fontsize=fs.FS_ANNOT)
        ax.set_xlabel(r"$t$ (t.u.)")
    axes[0].set_ylabel(r"front radius $r_{\rm front}/\sigma$")
    fs.finalize(fig)
    fig.savefig(FIGS / "figS2_fronts.pdf")
    fig.savefig(FIGS / "figS2_fronts.png", dpi=200)
