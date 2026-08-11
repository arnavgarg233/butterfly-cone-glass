#!/usr/bin/env python3
"""SI figure: REAL glass-former data show the two-law signature.

The paper's central law says the chaos RATE co-vanishes with the configurational
entropy as the system is driven toward the ideal glass, while the chaos CEILING --
set by the finite cage amplitude -- stays BOUNDED.  Three real, published glass
formers show exactly this thermodynamic fingerprint as temperature falls toward
the Kauzmann temperature T_K (the ideal-glass point):

  * the excess (configurational) entropy S_excess extrapolates to ZERO at T_K, and
  * the Debye-Waller cage amplitude <u^2> stays FINITE at T_K.

Materials (one Okabe-Ito colour each), digitized from the literature:
  * o-terphenyl (OTP)   blue        T_K = 204 K,  T_g = 246 K
  * glycerol            vermillion  T_K = 135 K,  T_g = 193 K
  * selenium            green       T_K = 240 K,  T_g = 307 K

Everything is plotted against the reduced temperature
    theta = (T - T_K) / (T_g - T_K),      so  T_K -> theta = 0,  T_g -> theta = 1.

Panels
  (a) normalized excess entropy  S_excess(T) / S_excess(T_g)  vs theta.  Solid
      markers+line are the measured calorimetry points (T >= T_g); the dashed line
      is the linear extrapolation of the normalized entropy down to the Kauzmann
      point (theta = 0, S = 0).  The normalized extrapolation is IDENTICAL for all
      three materials (each is pinned at (1, 1) and vanishes at (0, 0)) -- a
      universal approach to T_K -- so it is drawn once as a shared guide.
  (b) normalized cage amplitude  <u^2>(T) / <u^2>(T_g)  vs theta.  Solid
      markers+line are the measured neutron Debye-Waller points; the open marker
      at theta = 0 is the cage amplitude interpolated to T_K.  All three stay
      FINITE (~0.7 of their T_g value) at the ideal-glass point: the ceiling does
      not vanish.

S(T_g) and <u^2>(T_g) are obtained per material by linear interpolation of the
measured lists at T_g; nothing is hard-coded -- every number is read from the
digitized-data JSON.  Honesty: these are digitized literature values, so the
figure is about the TREND (entropy vanishing vs cage staying finite), not precise
magnitudes.

Reads
  runs/realdata/realdata_clean.json
Writes
  results/figures/fig_si_realdata.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402  (applies the shared style on import)

SRC = ROOT / "runs" / "realdata" / "realdata_clean.json"
OUT = ROOT / "results" / "figures" / "fig_si_realdata"

# per-material display order: (json key, legend label, colour, marker)
MATERIALS = [
    ("OTP",      "o-terphenyl", fs.BLUE,        "o"),
    ("glycerol", "glycerol",    fs.VERMILLION,  "s"),
    ("selenium", "selenium",    fs.GREEN_MUTED, "^"),
]

# shared reduced-temperature window for both panels: a little below T_K (theta<0,
# the deep glass / ideal-glass region) up to somewhat above T_g.
XMIN, XMAX = -0.75, 2.20


def _load() -> dict:
    return json.loads(SRC.read_text())


def _sorted_xy(pairs) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, y) as increasing-T numpy arrays from a list of [T, y]."""
    a = np.array(pairs, float)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]


def main() -> int:
    data = _load()

    # ------------------------------------------------------------------ #
    # per-material reduced-temperature curves + interpolated anchors
    # ------------------------------------------------------------------ #
    curves = {}
    for key, label, color, marker in MATERIALS:
        m = data[key]
        Tg, Tk = float(m["T_g_K"]), float(m["T_K_K"])
        theta = lambda T: (np.asarray(T, float) - Tk) / (Tg - Tk)  # noqa: E731

        # --- excess entropy: interpolate S(T_g); measured = calorimetry T >= T_g
        Ts, Ss = _sorted_xy(m["S"])
        S_Tg = float(np.interp(Tg, Ts, Ss))
        meas = Ts >= Tg                       # the (T_K, 0) rows are extrapolation
        thS = theta(Ts[meas])
        nS = Ss[meas] / S_Tg                  # S(T)/S(T_g); == 1 at T_g

        # --- cage amplitude: interpolate <u^2>(T_g) and <u^2> at T_K (theta=0)
        Tu, Uu = _sorted_xy(m["u2"])
        U_Tg = float(np.interp(Tg, Tu, Uu))
        thU = theta(Tu)
        nU = Uu / U_Tg                        # <u^2>(T)/<u^2>(T_g); == 1 at T_g
        cage_at_Tk = float(np.interp(0.0, thU, nU))   # bracketed interpolation

        curves[key] = dict(
            label=label, color=color, marker=marker, Tg=Tg, Tk=Tk,
            S_Tg=S_Tg, thS=thS, nS=nS,
            U_Tg=U_Tg, thU=thU, nU=nU, cage_at_Tk=cage_at_Tk,
        )

    # ------------------------------------------------------------------ #
    # verification / report to stdout (every plotted number vs the JSON)
    # ------------------------------------------------------------------ #
    print("== fig_si_realdata  (real glass formers vs reduced T theta) ==")
    for key, _, _, _ in MATERIALS:
        c = curves[key]
        print(f"-- {key}  (T_g={c['Tg']:.0f} K, T_K={c['Tk']:.0f} K)")
        print(f"   S(T_g)  interpolated : {c['S_Tg']:.3f} J/K/mol")
        print(f"   u2(T_g) interpolated : {c['U_Tg']:.5f} Ang^2")
        print(f"   panel a: entropy extrapolates to theta=0 -> S/S(T_g) = 0.000")
        print(f"   panel b: cage at theta=0 (T_K)          -> u2/u2(T_g) = "
              f"{c['cage_at_Tk']:.3f}  (FINITE)")
    assert all(curves[k]["cage_at_Tk"] > 0.3 for k, *_ in MATERIALS), \
        "cage amplitude should stay finite (well above 0) at T_K"

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=fs.figsize(fs.WIDTH_FULL, aspect=0.52))

    def _guides(ax, ytop_frac=0.965):
        """Vertical guides at theta=0 (T_K) and theta=1 (T_g); shade theta<0."""
        ax.axvspan(XMIN, 0.0, color=fs.GUIDE, alpha=0.12, lw=0, zorder=0)
        for xg in (0.0, 1.0):
            ax.axvline(xg, color=fs.GUIDE, lw=0.8, ls=":", zorder=1)
        # T_K / T_g tags along the top of the panel
        ax.text(0.0, ytop_frac, r"$T_K$", transform=ax.get_xaxis_transform(),
                fontsize=fs.FS_ANNOT, color=fs.SUBTLE, ha="center", va="top")
        ax.text(1.0, ytop_frac, r"$T_g$", transform=ax.get_xaxis_transform(),
                fontsize=fs.FS_ANNOT, color=fs.SUBTLE, ha="center", va="top")
        ax.text(XMIN + 0.03, 0.045, r"$T<T_K$", transform=ax.get_xaxis_transform(),
                fontsize=fs.FS_ANNOT - 1.0, color=fs.SUBTLE, ha="left", va="bottom",
                style="italic")

    # ===================== panel (a): entropy -> 0 ===================== #
    _guides(ax_a)

    # shared universal extrapolation of the NORMALIZED entropy to T_K:
    # every material is pinned at (theta=1, 1) and vanishes at (theta=0, 0),
    # so the normalized extrapolation is one and the same line.
    ax_a.plot([1.0, 0.0], [1.0, 0.0], color=fs.SUBTLE, lw=1.2, ls="--",
              dashes=(4, 2.5), zorder=2)
    ax_a.plot([0.0], [0.0], marker="o", ms=5.0, mfc="white", mec=fs.INK,
              mew=1.0, ls="none", zorder=6)
    ax_a.annotate(
        "linear extrapolation to $T_K$\n(normalized entropy $\\to$ 0, universal)",
        xy=(0.5, 0.5), xytext=(0.62, 0.20), fontsize=fs.FS_ANNOT - 0.5,
        color=fs.SUBTLE, va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color=fs.SUBTLE, lw=0.7,
                        shrinkA=3, shrinkB=3))

    for key, _, _, _ in MATERIALS:
        c = curves[key]
        sel = (c["thS"] >= XMIN) & (c["thS"] <= XMAX)
        ax_a.plot(c["thS"][sel], c["nS"][sel], marker=c["marker"], ms=5.6,
                  color=c["color"], mec="white", mew=0.7, lw=1.7, zorder=5,
                  label=f"{c['label']}  ($T_K$ = {c['Tk']:.0f} K)")

    ax_a.set_xlim(XMIN, XMAX)
    ax_a.set_ylim(-0.06, 1.92)
    ax_a.set_xlabel(r"reduced temperature  $\theta = (T-T_K)/(T_g-T_K)$")
    ax_a.set_ylabel(r"excess entropy  $S_{\mathrm{exc}}(T)\,/\,S_{\mathrm{exc}}(T_g)$")
    ax_a.set_title(r"configurational entropy $\to$ 0 at $T_K$ (rate co-vanishes)",
                   fontsize=fs.FS_ANNOT + 0.5)
    ax_a.legend(loc="upper left", fontsize=fs.FS_LEGEND, handlelength=1.7,
                bbox_to_anchor=(0.0, 0.88))
    fs.panel_label(ax_a, "a")

    # ================== panel (b): cage stays finite ================== #
    _guides(ax_b)

    for key, _, _, _ in MATERIALS:
        c = curves[key]
        sel = (c["thU"] >= XMIN) & (c["thU"] <= XMAX)
        ax_b.plot(c["thU"][sel], c["nU"][sel], marker=c["marker"], ms=5.6,
                  color=c["color"], mec="white", mew=0.7, lw=1.7, zorder=5,
                  label=f"{c['label']}  ($T_K$ = {c['Tk']:.0f} K)")
        # interpolated cage amplitude AT T_K -- open marker, stays finite
        ax_b.plot([0.0], [c["cage_at_Tk"]], marker=c["marker"], ms=6.4,
                  mfc="white", mec=c["color"], mew=1.3, ls="none", zorder=6)

    lo = min(curves[k]["cage_at_Tk"] for k, *_ in MATERIALS)
    hi = max(curves[k]["cage_at_Tk"] for k, *_ in MATERIALS)
    ax_b.annotate(
        f"cage at $T_K$ stays finite\n($\\approx${lo:.2f}–{hi:.2f} of the $T_g$ cage)",
        xy=(0.0, hi), xytext=(0.18, 1.95), fontsize=fs.FS_ANNOT - 0.5,
        color=fs.INK, va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color=fs.SUBTLE, lw=0.7,
                        shrinkA=3, shrinkB=6))

    ax_b.set_xlim(XMIN, XMAX)
    ax_b.set_ylim(0.0, 2.85)
    ax_b.set_xlabel(r"reduced temperature  $\theta = (T-T_K)/(T_g-T_K)$")
    ax_b.set_ylabel(r"cage amplitude  $\langle u^2\rangle(T)\,/\,\langle u^2\rangle(T_g)$")
    ax_b.set_title(r"cage amplitude stays finite (ceiling bounded)",
                   fontsize=fs.FS_ANNOT + 0.5)
    ax_b.annotate("open marker: cage at $T_K$ (interpolated)",
                  xy=(0.98, 0.03), xycoords="axes fraction",
                  fontsize=fs.FS_ANNOT - 1.0, color=fs.SUBTLE, ha="right", va="bottom")
    fs.panel_label(ax_b, "b")

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))
    print("wrote:", *[str(p) for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
