"""fig_si4 -- Interventional anchors (pinning do-operator + point-to-set instrument).

Panels
------
(a) Frozen random-pinning postdiction: dimensionless trends vs pin fraction f_p
    (tau_alpha ratio and pinning overlap q_pin increasing = the two frozen matches;
     dynamic-length ratio xi_dyn shown as the single-seed non-robust channel).
    A frozen-vs-measured trend table below it lists all five channels with the
    honest 2/5 count and the misses shown.
(b) Interventional point-to-set: mixing quality m(R)=1/Rhat^2 vs cavity radius R
    on the T=0.13 ergodicity positive-control center, with the n_basins=1 control
    point highlighted and the 8-center ANOVA crossover xi_PTS=0.95 (CI) overlaid.
(c) The cavity non-mixing split-Rhat vs R (log scale): Rhat~1 while a single basin
    is guaranteed, then diverges as distinct metastable cores appear.

Data (persisted runs/ JSON only):
  runs/pinning_postdiction/pinning_postdiction.json
  runs/xi_pts_centers/results.json
  runs/ergodicity_positive_control/grid-T013-R{0p5,0p7,1p0,1p5,2p0}/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import figstyle as fs

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
pin = json.loads((RUNS / "pinning_postdiction" / "pinning_postdiction.json").read_text())
xic = json.loads((RUNS / "xi_pts_centers" / "results.json").read_text())

dim = pin["response"]["dimensionless"]
f_p = np.array(dim["f_p"], dtype=float)
tau_ratio = np.array(dim["tau_alpha_ratio"], dtype=float)
q_pin = np.array(dim["q_pin"], dtype=float)
xi_dyn = np.array(dim["xi_dyn_ratio"], dtype=float)

# ergodicity positive-control m(R) grid at T=0.13
ERGO = RUNS / "ergodicity_positive_control"
grid_dirs = {
    0.5: "grid-T013-R0p5",
    0.7: "grid-T013-R0p7",
    1.0: "grid-T013-R1p0",
    1.5: "grid-T013-R1p5",
    2.0: "grid-T013-R2p0",
}
R_grid, m_grid, rhat_grid, nbasin_grid = [], [], [], []
for R in sorted(grid_dirs):
    mj = json.loads((ERGO / grid_dirs[R] / "metrics.json").read_text())
    R_grid.append(R)
    m_grid.append(mj["mixing"]["m"])
    rh = mj["mixing"]["split_rhat"]
    rhat_grid.append(np.nan if rh is None else rh)
    nbasin_grid.append(mj["mixing"]["n_basins"])
R_grid = np.array(R_grid)
m_grid = np.array(m_grid)
rhat_grid = np.array(rhat_grid)

# 8-center point-to-set crossover
xi_pts = xic["crossover_location_ci"]["mean"]
xi_lo = xic["crossover_location_ci"]["ci_low"]
xi_hi = xic["crossover_location_ci"]["ci_high"]
xi_n = xic["crossover_location_ci"]["n"]
per_center_x = np.array(xic["crossover_location_ci"]["values"], dtype=float)  # 6 finite crossovers
n_split = xic["interpretation"]["n_centers_split_at_large_R_p_lt_0p05"]       # 7 of 8
p_cons = xic["combined_significance_conservative"]["combined_p"]

# control point (T=0.13, R=0.5): m, split-Rhat, n_basins
m_ctrl = m_grid[0]
rhat_ctrl = rhat_grid[0]

print(f"[data] f_p={f_p.tolist()}")
print(f"[data] tau_alpha_ratio {tau_ratio[0]:.3f} -> {tau_ratio[-1]:.3f}")
print(f"[data] q_pin {q_pin[0]:.3f} -> {q_pin[-1]:.3f}")
print(f"[data] xi_dyn_ratio {xi_dyn.min():.3f}..{xi_dyn.max():.3f}")
print(f"[data] m(R) grid R={R_grid.tolist()} m={[round(x,4) for x in m_grid]}")
print(f"[data] split_rhat={[None if np.isnan(x) else round(x,4) for x in rhat_grid]} nbasins={nbasin_grid}")
print(f"[data] xi_PTS={xi_pts:.4f} CI[{xi_lo:.4f},{xi_hi:.4f}] n={xi_n}; split {n_split}/8; p_cons={p_cons:.2e}")

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
C_TAU = fs.BLUE       # tau_alpha ratio  (frozen match)
C_Q = fs.GREEN        # q_pin overlap    (frozen match)
C_DYN = fs.GOLD       # xi_dyn ratio     (single-seed, not bootstrap-robust)
C_M = fs.BLUE         # m(R) mixing curve
C_CTRL = fs.GREEN     # ergodicity control point
C_XI = fs.VERMILLION  # xi_PTS crossover
GOOD = fs.GREEN
BAD = fs.VERMILLION

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=False)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
              hspace=0.10, wspace=0.30, left=0.085, right=0.915, top=0.95, bottom=0.09)

ax_a = fig.add_subplot(gs[0, 0])   # pinning trends
ax_t = fig.add_subplot(gs[1, 0])   # trend table
ax_b = fig.add_subplot(gs[0, 1])   # m(R) crossover
ax_c = fig.add_subplot(gs[1, 1])   # split-Rhat

# ----------------------------- (a) pinning trends -------------------------- #
# left axis: dimensionless ratios (tau_alpha, xi_dyn), both start at 1
l1, = ax_a.plot(f_p, tau_ratio, "-o", color=C_TAU, ms=4.5, lw=1.7,
                label=r"$\tau_\alpha$ ratio  (match)", zorder=5)
l3, = ax_a.plot(f_p, xi_dyn, "--s", color=C_DYN, ms=4.0, lw=1.4, mfc="white",
                mec=C_DYN, label=r"$\xi_{\mathrm{dyn}}$ ratio  (single-seed)", zorder=4)
ax_a.set_ylabel(r"ratio  ($\tau_\alpha,\ \xi_{\mathrm{dyn}}$)")
ax_a.set_xlabel(r"pin fraction  $f_p$")
ax_a.set_ylim(0.7, 2.7)
ax_a.set_xlim(-0.015, 0.315)

# right axis: pinning overlap q_pin
ax_a2 = ax_a.twinx()
ax_a2.spines["top"].set_visible(False)
ax_a2.spines["right"].set_visible(True)
ax_a2.spines["right"].set_color(C_Q)
l2, = ax_a2.plot(f_p, q_pin, "-^", color=C_Q, ms=4.8, lw=1.7,
                 label=r"$q_{\mathrm{pin}}$ overlap  (match)", zorder=5)
ax_a2.set_ylabel(r"pinning overlap  $q_{\mathrm{pin}}$", color=C_Q)
ax_a2.tick_params(axis="y", colors=C_Q)
ax_a2.set_ylim(-0.005, 0.345)

ax_a.legend(handles=[l1, l2, l3], loc="upper left", fontsize=6.6,
            handlelength=1.5, borderaxespad=0.2, labelspacing=0.3)
fs.panel_label(ax_a, "a", x=-0.155, y=1.05)

# ------------------------- (a-table) frozen vs measured -------------------- #
ax_t.axis("off")
ax_t.set_xlim(0, 1)
ax_t.set_ylim(0, 1)

cols_x = [0.0, 0.46, 0.72, 0.95]  # channel(left), frozen, measured, match (centered)
header = ["channel", "frozen", "meas.", "match"]
rows = [
    (r"$\tau_\alpha$ ratio",        r"$\uparrow$",   r"$\uparrow$",           "yes"),
    (r"$q_{\mathrm{pin}}$ overlap", r"$\uparrow$",   r"$\uparrow$",           "yes"),
    (r"$\xi_{\mathrm{dyn}}$ length", "~",            "~*",                    "boot"),
    (r"$\xi_{\mathrm{PTS}}$ ratio",  r"$\uparrow$",  "~",                     "no"),
    (r"event fraction",             r"$\downarrow$", r"$-$",                  "no"),
]
DEJA = {"fontfamily": "DejaVu Sans"}
y_top = 0.88
dy = 0.135
# inline heading (non-bold, house style) + header row
ax_t.text(0.0, 1.05, "frozen vs measured trends", ha="left", va="bottom",
          fontsize=fs.FS_ANNOT, color=fs.SUBTLE)
for x, h in zip(cols_x, header):
    ha = "left" if x == 0.0 else "center"
    ax_t.text(x, y_top, h, ha=ha, va="center", fontsize=6.8,
              fontweight="bold", color=fs.INK)
ax_t.plot([0.0, 1.0], [y_top - 0.055, y_top - 0.055], color=fs.INK, lw=0.8)

sym = {"yes": ("✓", GOOD), "no": ("✗", BAD), "boot": ("✗", fs.SUBTLE)}
for i, (chan, declared, meas, verdict) in enumerate(rows):
    y = y_top - 0.10 - (i + 1) * dy
    ax_t.text(cols_x[0], y, chan, ha="left", va="center", fontsize=6.9, color=fs.INK)
    ax_t.text(cols_x[1], y, declared, ha="center", va="center", fontsize=7.4, color=fs.INK)
    ax_t.text(cols_x[2], y, meas, ha="center", va="center", fontsize=7.4, color=fs.INK)
    mk, col = sym[verdict]
    ax_t.text(cols_x[3], y, mk, ha="center", va="center", fontsize=9.0,
              fontweight="bold", color=col, **DEJA)
# footnotes on separate rows (no collision)
y_f1 = y_top - 0.10 - (len(rows) + 1) * dy
ax_t.text(0.0, y_f1, r"* dynamic-length peak fails 8-seed bootstrap",
          ha="left", va="center", fontsize=6.0, color=fs.SUBTLE)
ax_t.text(0.0, y_f1 - 0.115, r"honest 2/5 matches", ha="left", va="center",
          fontsize=7.4, color=fs.INK, fontweight="bold")
ax_t.text(1.0, y_f1 - 0.115, r"$p=0.37$", ha="right", va="center",
          fontsize=7.4, color=fs.SUBTLE)
fs.panel_label(ax_t, "b", x=-0.155, y=1.05)

# --------------------------- (c) m(R) crossover ---------------------------- #
# xi_PTS band (8-center ANOVA)
ax_b.axvspan(xi_lo, xi_hi, color=C_XI, alpha=0.12, lw=0, zorder=0)
ax_b.axvline(xi_pts, color=C_XI, lw=1.3, ls="-", zorder=1)
# m=0.5 crossover threshold
ax_b.axhline(0.5, color=fs.GUIDE, lw=0.9, ls=":", zorder=1)
ax_b.text(2.06, 0.52, r"$m=0.5$", ha="right", va="bottom",
          fontsize=6.5, color=fs.SUBTLE)

# m(R) curve
ax_b.plot(R_grid, m_grid, "-o", color=C_M, ms=5.0, lw=1.7, zorder=4,
          label=r"$m(R)$  (control center)")
# highlight ergodicity control point (R=0.5, n_basins=1)
ax_b.plot([R_grid[0]], [m_grid[0]], marker="*", ms=13, color=C_CTRL,
          mec=fs.INK, mew=0.5, zorder=6, ls="none")
# per-center crossover ticks along m=0.5 (the 8-center distribution)
ax_b.plot(per_center_x, np.full_like(per_center_x, 0.5), "|", color=C_XI,
          ms=8, mew=1.2, zorder=5)

ax_b.set_xlabel(r"cavity radius  $R$  ($\sigma$)")
ax_b.set_ylabel(r"mixing quality  $m(R)$")
ax_b.set_xlim(0.35, 2.12)
ax_b.set_ylim(-0.04, 1.10)

ax_b.annotate(
    "ergodicity\ncontrol\n" + rf"$n_{{\mathrm{{basins}}}}=1$" + "\n" + rf"$m={m_ctrl:.3f}$",
    xy=(R_grid[0], m_grid[0]), xytext=(0.43, 0.36),
    fontsize=6.3, color=C_CTRL, va="center", ha="left", linespacing=1.25,
    arrowprops=dict(arrowstyle="-", color=C_CTRL, lw=0.7))
ax_b.annotate(
    rf"$\xi_{{\mathrm{{PTS}}}}={xi_pts:.2f}\,\sigma$" + "\n"
    + rf"[{xi_lo:.2f}, {xi_hi:.2f}], {n_split}/8 split",
    xy=(xi_pts, 0.5), xytext=(1.18, 0.30),
    fontsize=6.3, color=C_XI, va="center", ha="left", linespacing=1.15,
    arrowprops=dict(arrowstyle="-", color=C_XI, lw=0.7))
fs.panel_label(ax_b, "c", x=-0.16, y=1.06)

# --------------------------- (d) split-Rhat -------------------------------- #
finite = ~np.isnan(rhat_grid)
ax_c.plot(R_grid[finite], rhat_grid[finite], "-o", color=C_M, ms=5.0, lw=1.7, zorder=4)
# control limit
ax_c.axhline(1.05, color=fs.GUIDE, lw=0.9, ls=":", zorder=1)
ax_c.text(2.06, 1.09, r"$\hat{R}=1.05$ limit", ha="right", va="bottom",
          fontsize=6.3, color=fs.SUBTLE)
# highlight control point
ax_c.plot([R_grid[0]], [rhat_grid[0]], marker="*", ms=13, color=C_CTRL,
          mec=fs.INK, mew=0.5, zorder=6, ls="none")
# R=1.5 split-Rhat is non-finite (degenerate zero within-chain variance): off-scale
ax_c.plot([1.5], [34.0], marker="^", ms=7, color=BAD, mfc="white", mec=BAD,
          mew=1.2, zorder=6, ls="none")
ax_c.annotate("non-finite\n" + r"($R=1.5$)", xy=(1.5, 34.0), xytext=(1.24, 34.0),
              ha="right", va="center", fontsize=6.0, color=BAD, linespacing=1.05)
ax_c.set_yscale("log")
ax_c.set_xlabel(r"cavity radius  $R$  ($\sigma$)")
ax_c.set_ylabel(r"split-$\hat{R}$")
ax_c.set_xlim(0.35, 2.12)
ax_c.set_ylim(0.9, 55)
ax_c.annotate(rf"1 basin, $\hat{{R}}={rhat_ctrl:.3f}$", xy=(R_grid[0], rhat_grid[0]),
              xytext=(0.62, 1.9), fontsize=6.3, color=C_CTRL, va="center", ha="left",
              arrowprops=dict(arrowstyle="-", color=C_CTRL, lw=0.7))
ax_c.text(1.98, 30, "multi-basin\n(split)", ha="right", va="top",
          fontsize=6.3, color=BAD, linespacing=1.1)
fs.panel_label(ax_c, "d", x=-0.16, y=1.05)

# --------------------------------------------------------------------------- #
out = ROOT / "results" / "figures" / "fig_si4"
paths = fs.save(fig, str(out))
print("[saved]", [str(p) for p in paths])
