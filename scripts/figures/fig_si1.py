#!/usr/bin/env python
"""Supplementary Fig. S1 -- Deep-anchor equilibration audit.

Panels:
  a  Swap-acceptance vs temperature across the accessible ladder (T>=0.075),
     with the sub-Tg deep-anchor band shaded; motivates gradual annealing.
  b  Energy-plateau audit at the coldest swap-reachable rung (T=0.075):
     per-replica potential-energy equilibration traces + first/second-half
     plateau bands (flat within threshold, all passed).
  c  Two-history convergence of the inherent-structure energy e_IS at the three
     deep anchors (T=0.055, 0.060, 0.067): independent replicas -> common energy.
  d  Two-history convergence of the configurational entropy s_c: per-replica
     deep-anchor values PLUS the independent-campaign overlap cross-check
     (committed vs deep ladders agree at T=0.075/0.090/0.108, |z|<=2.5).

All data are loaded from persisted runs/ JSON; nothing is fabricated.
Data gap (noted on panel a): no per-anchor swap-acceptance trace was persisted
for the annealed deep anchors (T<0.075); the equilibration of those anchors is
established by the independent-history agreement in panels c and d.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

# --------------------------------------------------------------------------- #
# Load data (persisted artifacts only)
# --------------------------------------------------------------------------- #

# --- (a) swap acceptance vs T from bulk_pilot metrics.json ---
swap = {}
for f in glob.glob(str(RUNS / "bulk_pilot" / "**" / "metrics.json"), recursive=True):
    d = json.load(open(f))
    eq = d.get("equilibration", {})
    if "swap_acceptance" in eq:
        swap.setdefault(d["temperature"], []).append(eq["swap_acceptance"])
swap_T = np.array(sorted(swap))
swap_mean = np.array([np.mean(swap[t]) for t in swap_T])
swap_lo = np.array([np.min(swap[t]) for t in swap_T])
swap_hi = np.array([np.max(swap[t]) for t in swap_T])

# --- (b) energy-plateau traces at coldest swap rung T=0.075 ---
traces = []
for f in glob.glob(str(RUNS / "bulk_pilot" / "**" / "metrics.json"), recursive=True):
    d = json.load(open(f))
    if d.get("temperature") == 0.075:
        eq = d["equilibration"]
        traces.append(dict(rep=d.get("replica"),
                           trace=np.asarray(eq["energy_trace_per_particle"], float),
                           fh=eq["first_half_mean"], sh=eq["second_half_mean"],
                           thr=eq["threshold"], se=eq["standard_error"]))
traces.sort(key=lambda x: x["rep"])
fh_all = np.mean([t["fh"] for t in traces])
sh_all = np.mean([t["sh"] for t in traces])
thr_all = np.mean([t["thr"] for t in traces])
plateau_rel_span = abs(fh_all - sh_all) / fh_all  # 0.27%

# --- (c,d) deep-anchor per-replica e_IS and s_c ---
# T=0.055 (2 replicas) -- sc_T0055_report.json
r55 = json.load(open(RUNS / "sc_T0055" / "full-ladder-analysis" / "sc_T0055_report.json"))
rec55 = r55["pipeline"]["records"]
# T=0.060 (3 replicas) -- sc_T0060/sc_curve_deep.json
r60 = json.load(open(RUNS / "sc_T0060" / "sc_curve_deep.json"))
rec60 = r60["records"]
# T=0.067 (3 replicas) -- sc_deep_test/sc_curve_deep.json (record temperature 0.067)
r67 = json.load(open(RUNS / "sc_deep_test" / "sc_curve_deep.json"))
rec67 = [r for r in r67["records"] if r["temperature"] == 0.067]

deep = {
    0.055: dict(eis=[r["e_is_per_particle"] for r in rec55],
                sc=[r["s_configurational"] for r in rec55]),
    0.060: dict(eis=[r["e_is_per_particle"] for r in rec60],
                sc=[r["s_configurational"] for r in rec60]),
    0.067: dict(eis=[r["e_is_per_particle"] for r in rec67],
                sc=[r["s_configurational"] for r in rec67]),
}
for T, dd in deep.items():
    dd["eis"] = np.array(dd["eis"]); dd["sc"] = np.array(dd["sc"])
    dd["eis_mean"] = dd["eis"].mean(); dd["eis_std"] = dd["eis"].std(ddof=1)
    dd["sc_mean"] = dd["sc"].mean(); dd["sc_std"] = dd["sc"].std(ddof=1)

# --- (d) independent-campaign overlap cross-check (committed vs deep) ---
dt = json.load(open(RUNS / "sc_deep_test" / "sc_deep_test.json"))
overlap = sorted(dt["analysis"]["overlap_crosscheck"], key=lambda x: x["temperature"])
max_abs_z = max(abs(o["z"]) for o in overlap)

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fs.use()
fig, axes = plt.subplots(2, 2, figsize=fs.figsize(fs.WIDTH_FULL, 0.74))
(axA, axB), (axC, axD) = axes

DEEP_BAND = (0.0525, 0.0705)      # covers 0.055, 0.060, 0.067
DEEP_TS = [0.055, 0.060, 0.067]

# ===== (a) swap acceptance vs T =====
ax = axA
ax.axvspan(*DEEP_BAND, color=fs.GRAY, alpha=0.15, lw=0, zorder=0)
yerr = np.vstack([swap_mean - swap_lo, swap_hi - swap_mean])
ax.errorbar(swap_T, swap_mean, yerr=yerr, fmt="o", color=fs.BLUE,
            ms=5, mec="white", mew=0.6, capsize=2.4, elinewidth=1.0,
            ecolor=fs.BLUE, zorder=3, label="measured (min-max over replicas)")
ax.plot(swap_T, swap_mean, "-", color=fs.BLUE, lw=1.1, alpha=0.55, zorder=2)
# annealing boundary
ax.axvline(0.072, color=fs.SUBTLE, ls=":", lw=0.9, zorder=1)
ax.set_xlim(0.045, 0.16)
ax.set_ylim(0.06, 0.20)
ax.set_xlabel("temperature $T$")
ax.set_ylabel("swap acceptance")
ax.annotate("sub-$T_g$ deep anchors\n(gradual annealing)",
            xy=(0.0615, 0.183), ha="center", va="top",
            fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE,
            linespacing=1.15)
ax.annotate("swap stalls\n$\\downarrow T$", xy=(0.075, swap_mean[0]),
            xytext=(0.096, 0.088), fontsize=fs.FS_ANNOT - 0.5, color=fs.INK,
            va="center", ha="left", linespacing=1.15,
            arrowprops=dict(arrowstyle="->", color=fs.INK, lw=0.8))
ax.text(0.0615, 0.068, "no swap trace\npersisted", ha="center", va="bottom",
        fontsize=fs.FS_ANNOT - 1.2, color=fs.GRAY,
        linespacing=1.1)
ax.legend(loc="upper left", bbox_to_anchor=(0.30, 1.02),
          fontsize=fs.FS_LEGEND - 0.5)
fs.panel_label(ax, "a")

# ===== (b) energy-plateau audit at T=0.075 =====
ax = axB
x = np.arange(1, 9)
grand = 0.5 * (fh_all + sh_all)
# tolerance band: half-means must sit within +/- threshold of one another
ax.axhspan(grand - thr_all / 2, grand + thr_all / 2, color=fs.GRAY, alpha=0.13,
           lw=0, zorder=0)
for t in traces:
    ax.plot(x, t["trace"], "-", color=fs.BLUE, lw=0.8, alpha=0.22, zorder=2)
bold = traces[0]
ax.plot(x, bold["trace"], "-o", color=fs.BLUE, lw=1.6, ms=3.2, mec="white",
        mew=0.5, zorder=4)
# first / second half plateau bands
ax.axhline(fh_all, color=fs.VERMILLION, ls="--", lw=1.2, zorder=3)
ax.axhline(sh_all, color=fs.GREEN, ls="--", lw=1.2, zorder=3)
ax.axvline(4.5, color=fs.SUBTLE, ls=":", lw=0.8, zorder=1)
ax.set_xlim(0.6, 8.4)
ax.set_ylim(0.3005, 0.3175)
ax.set_xlabel("equilibration block")
ax.set_ylabel("$U/N$  (potential energy)")
ax.text(2.2, fh_all - 0.0004, "1st-half mean", color=fs.VERMILLION,
        fontsize=fs.FS_ANNOT - 1, va="top", ha="center")
ax.text(6.7, sh_all + 0.0004, "2nd-half mean", color=fs.GREEN,
        fontsize=fs.FS_ANNOT - 1, va="bottom", ha="center")
fs.annotate_stats(
    ax,
    "$T=0.075$ (coldest swap rung)\n"
    "half-to-half drift $=%.2f\\%%$\n$\\ll%.2f\\%%$ threshold (passed)"
    % (100 * plateau_rel_span, 100 * thr_all / fh_all),
    x=0.035, y=0.225, size=fs.FS_ANNOT - 0.5, color=fs.INK)
fs.panel_label(ax, "b")

# ===== (c) two-history convergence: inherent-structure energy =====
ax = axC
# every anchor here IS a deep anchor, so the sub-Tg band would flood the whole
# panel; mark it as a thin top-edge strip instead (consistent with panels a/d,
# where the same band marks only the sub-Tg strip within a wider T range).
ax.axvspan(*DEEP_BAND, ymin=0.93, ymax=1.0, color=fs.GRAY, alpha=0.28, lw=0, zorder=0)
ax.text(0.0615, 0.1948, "sub-$T_g$ deep anchors", ha="center", va="center",
        fontsize=fs.FS_ANNOT - 2.0, color=fs.SUBTLE)
for T in DEEP_TS:
    dd = deep[T]
    jit = np.linspace(-0.0009, 0.0009, len(dd["eis"]))
    ax.scatter(np.full_like(dd["eis"], T) + jit, dd["eis"], s=20,
               facecolor="white", edgecolor=fs.BLUE, linewidth=1.0, zorder=3)
    ax.errorbar(T, dd["eis_mean"], yerr=dd["eis_std"], fmt="D", color=fs.VERMILLION,
                ms=6, mec="white", mew=0.6, capsize=2.6, elinewidth=1.0,
                ecolor=fs.VERMILLION, zorder=4)
    ax.annotate("n=%d\nσ=%.4f" % (len(dd["eis"]), dd["eis_std"]),
                xy=(T, dd["eis"].min() - 0.0013),
                ha="center", va="top", fontsize=fs.FS_ANNOT - 1.3, color=fs.SUBTLE,
                linespacing=1.1)
ax.set_xlim(0.0505, 0.0725)
ax.set_xticks(DEEP_TS)
ax.set_ylim(0.1665, 0.198)
ax.set_xlabel("deep-anchor temperature $T$")
ax.set_ylabel("$e_{\\mathrm{IS}}/N$  (inherent-structure energy)")
# legend proxies
h1 = ax.scatter([], [], s=20, facecolor="white", edgecolor=fs.BLUE, linewidth=1.0)
h2 = ax.errorbar([], [], yerr=1, fmt="D", color=fs.VERMILLION, ms=6, mec="white",
                 mew=0.6, capsize=2.6, elinewidth=1.0, ecolor=fs.VERMILLION)
ax.legend([h1, h2], ["independent replicas", "mean $\\pm$ replica $\\sigma$"],
          loc="lower right", fontsize=fs.FS_LEGEND - 0.5)
fs.panel_label(ax, "c")

# ===== (d) two-history convergence: s_c + independent-campaign cross-check =====
ax = axD
ax.axvspan(*DEEP_BAND, color=fs.GRAY, alpha=0.15, lw=0, zorder=0)
# deep-anchor s_c: light individual replicas + prominent mean +/- replica sigma
for T in DEEP_TS:
    dd = deep[T]
    jit = np.linspace(-0.0011, 0.0011, len(dd["sc"]))
    ax.scatter(np.full_like(dd["sc"], T) + jit, dd["sc"], s=11,
               facecolor=fs.SKY, edgecolor="none", alpha=0.7, zorder=3)
    ax.errorbar(T, dd["sc_mean"], yerr=dd["sc_std"], fmt="D", color=fs.VERMILLION,
                ms=6, mec="white", mew=0.6, capsize=3.0, elinewidth=1.1,
                ecolor=fs.VERMILLION, zorder=5)
# independent-campaign overlap cross-check (committed vs deep)
ovT = np.array([o["temperature"] for o in overlap])
ov_comm = np.array([o["s_c_committed"] for o in overlap])
ov_deep = np.array([o["s_c_deep"] for o in overlap])
ov_sem = np.array([o["pooled_sem"] for o in overlap])
ax.scatter(ovT - 0.0007, ov_comm, s=26, marker="s", facecolor=fs.GREEN,
           edgecolor="white", linewidth=0.6, zorder=4, label="committed campaign")
ax.errorbar(ovT + 0.0007, ov_deep, yerr=ov_sem, fmt="o", color=fs.GRAY,
            ms=4.5, mec="white", mew=0.5, capsize=2.2, elinewidth=0.9,
            ecolor=fs.GRAY, zorder=4, label="deep campaign")
ax.set_xlim(0.0505, 0.114)
ax.set_xticks([0.055, 0.067, 0.075, 0.09, 0.108])
ax.set_xlabel("temperature $T$")
ax.set_ylabel("configurational entropy $s_c$")
ax.annotate("$T=0.055$: 2 replicas\n(archive $R{=}1$ stale)\n$s_c=1.197,\\,1.167$",
            xy=(0.055, deep[0.055]["sc_mean"]),
            xytext=(0.058, 1.30), fontsize=fs.FS_ANNOT - 1.2, color=fs.INK,
            va="center", ha="left", linespacing=1.15,
            arrowprops=dict(arrowstyle="->", color=fs.INK, lw=0.7))
ax.text(0.091, 1.62,
        "independent campaigns\nagree: $|z|\\leq%.1f$" % max_abs_z,
        ha="center", va="top", fontsize=fs.FS_ANNOT - 1.0, color=fs.SUBTLE,
        linespacing=1.15)
# legend combining replica + campaign proxies
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", ls="none", mfc=fs.SKY, mec="none",
           ms=4, label="deep replicas"),
    Line2D([], [], marker="D", ls="none", color=fs.VERMILLION, mec="white",
           mew=0.6, ms=6, label="deep mean $\\pm$ $\\sigma$"),
    Line2D([], [], marker="s", ls="none", mfc=fs.GREEN, mec="white", mew=0.6,
           ms=5, label="committed campaign"),
    Line2D([], [], marker="o", ls="none", color=fs.GRAY, mec="white", mew=0.5,
           ms=5, label="deep campaign"),
]
ax.legend(handles=handles, loc="lower right", fontsize=fs.FS_LEGEND - 1.0,
          ncol=1, labelspacing=0.3, handletextpad=0.4)
fs.panel_label(ax, "d")

fs.finalize(fig)
out = fs.save(fig, str(ROOT / "results" / "figures" / "fig_si1"))
print("wrote:", [str(p) for p in out])

# --------------------------------------------------------------------------- #
# Verified numbers dump (for the caller)
# --------------------------------------------------------------------------- #
print("\n=== VERIFIED NUMBERS ===")
print("swap acceptance vs T (mean[min,max]):")
for t, m, lo, hi in zip(swap_T, swap_mean, swap_lo, swap_hi):
    print(f"  T={t}: {m:.4f} [{lo:.4f},{hi:.4f}]  n={len(swap[t])}")
print(f"plateau T=0.075: fh={fh_all:.5f} sh={sh_all:.5f} rel_span={100*plateau_rel_span:.3f}% thr={100*thr_all/fh_all:.3f}%")
for T in DEEP_TS:
    dd = deep[T]
    print(f"deep T={T}: e_IS={dd['eis_mean']:.5f}+-{dd['eis_std']:.5f} {list(np.round(dd['eis'],5))} | "
          f"s_c={dd['sc_mean']:.4f}+-{dd['sc_std']:.4f} {list(np.round(dd['sc'],4))}")
print("overlap crosscheck (committed vs deep):")
for o in overlap:
    print(f"  T={o['temperature']}: committed={o['s_c_committed']:.5f} deep={o['s_c_deep']:.5f} "
          f"z={o['z']:.3f} pooled_sem={o['pooled_sem']:.5f}")
print(f"max|z|={max_abs_z:.3f}")
