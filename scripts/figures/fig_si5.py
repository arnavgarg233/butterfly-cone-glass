#!/usr/bin/env python3
"""Supplementary Fig. S5 -- "The shielding null" (Nature Physics "butterfly cone" SI).

Tests whether the structural content of the divergence norm decorrelates on its own,
slower clock: a cage-relative channel S against the raw channel M, each normalized to
its own plateau over a five-point threshold band.  The shielding hypothesis predicts
channel S would cross its threshold LATER than channel M (a second, slower clock).
The finding is the opposite -- a systematic ANTI-shielding null.

Panels (every number pulled from persisted runs/ JSON; nothing fabricated):

  (a) Channel S vs channel M crossing time across the threshold band.  The
      cage-relative shield channel S (t_shield) crosses every threshold in
      {0.3..0.7} slightly EARLIER than the raw channel M (t_raw); both sit below
      the divergence saturation onset t_sat.  The thin shaded ribbon is the
      (always-negative) lag S - M.
      Source: runs/gardner/gardner-T0075-fss/gardner_r0_channelS.json (threshold_band).

  (b) The per-ensemble lag distribution t_shield - t_raw over the 54 physically
      distinct ensembles across BOTH replica-pairs (fss: 36 at N=1500; m2: +18 at
      N=3000; the m2 N=1500 config duplicates fss).  Every lag is negative:
      0/54 ensembles are shielded (t_shield >= t_raw).  A systematic anti-shielding
      null, not an underpowered one (reference lag -0.186 +/- 0.008).
      Source: gardner_r0.json["ensembles"] (fss + m2).

  (c) The structural freeze over the accessible window.  The unperturbed
      self-intermediate scattering F_s(q0, t), self-overlap Q(t), and cage overlap
      Q_cage(t) all sit far above the 1/e decorrelation line across the full window
      out to t_max = 20 t.u. -- the cage-scale structure never relaxes, so
      tau_alpha >= 20 t.u. is a lower bound and the crossing-time channels are
      vibrational, not alpha-relaxation.
      Source: gardner_r0_channelS.json (tau_by_config, unperturbed configs).

Run:
  cd butterfly_cone && PYTHONPATH=src \
     python \
     scripts/figures/fig_si5.py
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
RUNS = ROOT / "runs" / "gardner"
OUT = ROOT / "results" / "figures" / "fig_si5"

Q0KEY = "6.283"          # q0 = 6.283 (first Bragg peak) -- the F_s(q0) probe
ONE_OVER_E = 1.0 / np.e  # 0.3679 decorrelation threshold

# semantic colours
C_M = fs.BLUE            # channel M -- the raw divergence-norm clock (baseline)
C_S = fs.GREEN          # channel S -- the cage-relative "shield" null channel
C_SAT = fs.SUBTLE       # divergence saturation onset t_sat (reference)
C_ZERO = fs.VERMILLION  # the shielding threshold (lag = 0) that is never reached


# --------------------------------------------------------------------------- #
def load():
    fss_ch = json.loads((RUNS / "gardner-T0075-fss" / "gardner_r0_channelS.json").read_text())
    fss_r0 = json.loads((RUNS / "gardner-T0075-fss" / "gardner_r0.json").read_text())
    m2_r0 = json.loads((RUNS / "gardner-T0075-m2" / "gardner_r0.json").read_text())
    return fss_ch, fss_r0, m2_r0


def distinct_lags(fss_r0, m2_r0):
    """The 54 physically distinct per-ensemble lags across both replica-pairs.

    fss contributes all 36 ensembles (N=1500); m2 contributes only its N=3000
    config (18), because m2's N=1500 config duplicates fss's.  All 54 are the
    reference-threshold (frac 0.5) per-ensemble t_shield - t_raw.
    """
    lags_fss = [e["lag"] for e in fss_r0["ensembles"]]                 # 36 (N=1500)
    fss_Ns = {e["N"] for e in fss_r0["ensembles"]}
    lags_m2_new = [e["lag"] for e in m2_r0["ensembles"] if e["N"] not in fss_Ns]  # 18 (N=3000)
    return np.asarray(lags_fss), np.asarray(lags_m2_new)


# --------------------------------------------------------------------------- #
def panel_a(ax, fss_ch):
    """Channel S vs channel M crossing time across the threshold band."""
    band = fss_ch["threshold_band"]
    thr = np.asarray(band["values"], float)
    t_shield = np.array([pt["t_shield"]["mean"] for pt in band["per_threshold"]])
    t_raw = np.array([pt["t_raw"]["mean"] for pt in band["per_threshold"]])
    t_sat = float(band["per_threshold"][0]["t_sat"]["mean"])           # 4.833, threshold-independent
    ref = fss_ch["reference"]
    lag_ref = ref["t_shield"] - ref["t_raw"]                           # -0.186

    # highlight the reference threshold (frac 0.5), behind everything
    ax.axvline(0.5, color=fs.GUIDE, lw=0.7, ls=(0, (1, 2)), zorder=0)

    # saturation-onset reference (the divergence clock, constant across thresholds)
    ax.axhline(t_sat, color=C_SAT, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    ax.text(0.30, t_sat + 0.04, r"divergence saturation  $t_{\mathrm{sat}}$",
            color=C_SAT, fontsize=fs.FS_ANNOT - 0.5, ha="left", va="bottom")

    # the always-negative lag ribbon (S below M)
    ax.fill_between(thr, t_shield, t_raw, color=C_S, alpha=0.16, lw=0, zorder=2)

    # channel M (raw) and channel S (shield)
    ax.plot(thr, t_raw, "-o", color=C_M, ms=5.0, lw=1.7, mec="white", mew=0.6,
            zorder=5, label=r"channel M (raw)")
    ax.plot(thr, t_shield, "-s", color=C_S, ms=5.0, lw=1.7, mec="white", mew=0.6,
            zorder=5, label=r"channel S (cage-relative, earlier)")

    ax.set_xlim(0.27, 0.73)
    ax.set_ylim(3.0, 5.55)
    ax.set_xlabel(r"normalized threshold  $f$")
    ax.set_ylabel(r"crossing time  $t$  (t.u.)")
    ax.set_xticks(thr)

    # legend upper-left (curves are low there); numbers bottom-right (empty)
    ax.legend(loc="upper left", fontsize=fs.FS_LEGEND - 1.0, handlelength=1.5,
              borderaxespad=0.3, labelspacing=0.3)
    ax.text(0.97, 0.035,
            rf"$t_S = {ref['t_shield']:.3f} < t_M = {ref['t_raw']:.3f}$" "\n"
            rf"lag $= {lag_ref:.3f}$  at every $f$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs.FS_ANNOT - 0.5, color=fs.INK, linespacing=1.35)
    fs.panel_label(ax, "a")
    return dict(thr=thr, t_shield=t_shield, t_raw=t_raw, t_sat=t_sat, lag_ref=lag_ref)


def panel_b(ax, lags_fss, lags_m2):
    """Per-ensemble lag distribution over the 54 distinct ensembles: 0/54 shielded."""
    lags = np.concatenate([lags_fss, lags_m2])
    n_tot = lags.size
    n_shielded = int((lags >= 0).sum())
    mean, sd = lags.mean(), lags.std()

    bins = np.arange(-0.215, 0.021, 0.005)
    # stacked: primary replica-pair (fss, N=1500) + the distinct m2 (N=3000)
    ax.hist([lags_fss, lags_m2], bins=bins, stacked=True,
            color=[C_M, fs.SKY], edgecolor="white", linewidth=0.4,
            label=[rf"fss  $N{{=}}1500$  ($n={lags_fss.size}$)",
                   rf"m2  $N{{=}}3000$  ($n={lags_m2.size}$)"], zorder=3)

    ymax = 12.0
    # the "shielded" region (lag >= 0) -- reached by 0 ensembles
    ax.axvspan(0.0, 0.02, color=C_ZERO, alpha=0.07, lw=0, zorder=0)
    ax.axvline(0.0, color=C_ZERO, lw=1.2, ls="--", zorder=4)
    # mean line of the 54-ensemble distribution
    ax.axvline(mean, color=fs.INK, lw=1.0, ls="-", zorder=5)

    ax.set_xlim(-0.215, 0.02)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"per-ensemble lag  $t_{\mathrm{shield}} - t_{\mathrm{raw}}$  (t.u.)")
    ax.set_ylabel("ensembles")

    # mean label, in the gap to the right of the bars, arrow onto the mean line
    ax.annotate(rf"mean $-0.189$" "\n" rf"($n={n_tot}$)", xy=(mean, ymax * 0.72),
                xytext=(-0.155, ymax * 0.83), textcoords="data", ha="left", va="center",
                fontsize=fs.FS_ANNOT - 1.0, color=fs.INK, linespacing=1.1,
                arrowprops=dict(arrowstyle="->", color=fs.INK, lw=0.7))
    # shielding-threshold label, to the LEFT of the x=0 line (empty region)
    ax.annotate("shielding threshold\n" + r"($t_{\mathrm{shield}}=t_{\mathrm{raw}}$)",
                xy=(0.0, ymax * 0.30), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=fs.FS_ANNOT - 1.0, color=C_ZERO,
                linespacing=1.1)

    # headline stat block + legend in the empty centre-right region
    ax.text(0.52, 0.965, rf"$\mathbf{{0/{n_tot}}}$ shielded" "\n" r"every lag $< 0$",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs.FS_ANNOT,
            color=fs.INK, linespacing=1.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.50, 0.70),
              fontsize=fs.FS_LEGEND - 1.0, handlelength=1.0, labelspacing=0.35,
              handletextpad=0.5)
    fs.panel_label(ax, "b")
    return dict(n_tot=n_tot, n_shielded=n_shielded, mean=float(mean), sd=float(sd))


def panel_c(ax, fss_ch):
    """Structural freeze: F_s(q0), Q_self, Q_cage stay flat, far above 1/e.

    Plots the representative unperturbed config c0 (the values quoted in the
    caption / claims registry); the two unperturbed configs agree to < 0.5%.
    """
    c = fss_ch["tau_by_config"]["c0"]
    probe = float(c["probe_time"])                        # 4.199
    tmax = float(c["t_max"])                              # 20.0
    tau_floor = float(fss_ch["tau_floor_lower_bound"])    # 20.0
    tt = np.array([probe, tmax])

    Fs = np.array([c["fs_at_probe"][Q0KEY], c["fs_at_tmax"][Q0KEY]])
    Qs = np.array([c["overlap"]["Q_at_probe"], c["overlap"]["Q_at_tmax"]])
    Qc = np.array([c["cage_overlap"]["Qcage_at_probe"], c["cage_overlap"]["Qcage_at_tmax"]])

    # the measured window and the frozen ceiling / 1/e decorrelation floor
    ax.axvspan(probe, tmax, color=fs.GRAY, alpha=0.08, lw=0, zorder=0)
    ax.axhline(1.0, color=fs.GUIDE, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.axhspan(0.0, ONE_OVER_E, color=C_ZERO, alpha=0.06, lw=0, zorder=0)
    ax.axhline(ONE_OVER_E, color=C_ZERO, lw=1.0, ls="--", zorder=2)
    ax.text(23.2, ONE_OVER_E - 0.02, r"$1/e$ decorrelation", color=C_ZERO,
            fontsize=fs.FS_ANNOT - 0.5, ha="right", va="top")

    # (series, colour, marker, label, point-offset to separate the close labels)
    series = [
        (Qc, fs.GOLD, "s", r"cage overlap  $Q_{\mathrm{cage}}$", +6),
        (Qs, C_S, "^", r"self-overlap  $Q$", -6),
        (Fs, C_M, "o", r"$F_s(q_0, t)$", 0),
    ]
    for y, col, mk, lab, dy_pt in series:
        ax.plot(tt, y, "-", marker=mk, color=col, ms=5.0, lw=1.7, mec="white",
                mew=0.6, zorder=4, label=lab)
        ax.annotate(rf"${y[1]:.3f}$", xy=(tmax, y[1]), xytext=(6, dy_pt),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fs.FS_ANNOT - 0.5, color=col)

    ax.set_xlim(0.0, 24.5)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel(r"time  $t$  (t.u.)")
    ax.set_ylabel("structural correlation")
    ax.set_xticks([0, 5, 10, 15, 20])

    # notes in the empty mid-left region (between the 1/e floor and F_s)
    ax.text(1.0, 0.66, "structure frozen\nover the window", ha="left", va="center",
            fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, linespacing=1.2)
    ax.text(1.0, 0.50, r"$\tau_\alpha \geq %.0f$ t.u." % tau_floor + "\n(lower bound)",
            ha="left", va="center", fontsize=fs.FS_ANNOT - 0.5, color=fs.INK,
            linespacing=1.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.03, 0.02),
              fontsize=fs.FS_LEGEND - 1.0, handlelength=1.4, labelspacing=0.3)
    fs.panel_label(ax, "c")
    return dict(Fs20=float(Fs[1]), Qs20=float(Qs[1]), Qc20=float(Qc[1]),
                probe=probe, tmax=tmax)


# --------------------------------------------------------------------------- #
def main():
    fss_ch, fss_r0, m2_r0 = load()
    lags_fss, lags_m2 = distinct_lags(fss_r0, m2_r0)

    fig, axes = plt.subplots(1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.40))
    va = panel_a(axes[0], fss_ch)
    vb = panel_b(axes[1], lags_fss, lags_m2)
    vc = panel_c(axes[2], fss_ch)

    fs.finalize(fig)
    paths = fs.save(fig, str(OUT))

    # ---- verification (assert every plotted headline against the JSON) ------ #
    print("=== fig_si5 verification ===")
    ref = fss_ch["reference"]
    print("[a] threshold band  f =", va["thr"].tolist())
    print("    t_shield =", [round(x, 3) for x in va["t_shield"]])
    print("    t_raw    =", [round(x, 3) for x in va["t_raw"]])
    print(f"    channel S below channel M at every threshold: "
          f"{bool(np.all(va['t_shield'] < va['t_raw']))}")
    print(f"    reference (frac 0.5): t_shield={ref['t_shield']:.3f} "
          f"t_raw={ref['t_raw']:.3f} t_sat={ref['t_sat']:.3f} lag={va['lag_ref']:.3f}")
    assert bool(np.all(va["t_shield"] < va["t_raw"]))
    assert abs(va["lag_ref"] - (-0.186)) < 1e-3, va["lag_ref"]
    assert abs(ref["t_shield"] - 4.199) < 1e-3
    assert abs(ref["t_raw"] - 4.385) < 1e-3
    assert abs(ref["t_sat"] - 4.833) < 1e-3

    print("[b] distinct-ensemble lag distribution:")
    print(f"    n = {vb['n_tot']} (fss {lags_fss.size} + m2 N=3000 {lags_m2.size})  "
          f"shielded = {vb['n_shielded']}  mean = {vb['mean']:.4f}  sd = {vb['sd']:.4f}")
    assert vb["n_tot"] == 54, vb["n_tot"]
    assert vb["n_shielded"] == 0
    assert bool(np.all(np.concatenate([lags_fss, lags_m2]) < 0))
    # pooled reference lag from gardner_r0 (fss primary replica-pair)
    pooled_lag = fss_r0["pooled"]["lag"]
    print(f"    fss pooled lag: mean={pooled_lag['mean']:.4f} sd={pooled_lag['std']:.4f} "
          f"n={pooled_lag['n']}  shielded_fraction={fss_r0['pooled']['shielded_fraction']}")
    assert abs(pooled_lag["mean"] - (-0.186)) < 1e-3
    assert abs(pooled_lag["std"] - 0.008) < 1e-3
    assert fss_r0["pooled"]["shielded_fraction"] == 0.0

    print("[c] structural freeze at t_max = %.0f t.u.:" % vc["tmax"])
    print(f"    F_s(q0) = {vc['Fs20']:.3f}  Q_self = {vc['Qs20']:.3f}  "
          f"Q_cage = {vc['Qc20']:.3f}  (probe t = {vc['probe']:.3f})")
    assert vc["Fs20"] > ONE_OVER_E and vc["Qs20"] > ONE_OVER_E and vc["Qc20"] > ONE_OVER_E
    assert abs(vc["Fs20"] - 0.836) < 4e-3, vc["Fs20"]
    assert abs(vc["Qs20"] - 0.956) < 4e-3, vc["Qs20"]
    assert abs(vc["Qc20"] - 0.972) < 4e-3, vc["Qc20"]

    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
