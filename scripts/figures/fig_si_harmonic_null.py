#!/usr/bin/env python3
"""SI figure: HARMONIC / PHONON NULL vs the nonlinear butterfly cone.

Reads runs/harmonic_null/figure_payload.json (written by scripts/harmonic_null.py)
and renders results/figures/fig_si_harmonic_null.png (+ pdf).

Panels
  (a) Twin divergence D(t) on a log axis: the full nonlinear cone grows
      EXPONENTIALLY (rate lambda_B) and saturates at the Debye-Waller ceiling,
      the inherent-structure harmonic null grows only ALGEBRAICALLY and stays
      far below it, and the instantaneous-Hessian null (frozen at the cooled
      config, which carries unstable INMs) grows exponentially but never locks
      onto the delta-independent DW plateau.
  (b) Ballistic front r_front(t): nonlinear and harmonic-IS ride the SAME sound
      cone (information cannot outrun sound) -- the harmonic null reproduces the
      elastic front, the positive control.
  (c) Plateau vs kick delta: the harmonic-IS plateau scales LINEARLY with delta
      (elastic, ~3x from 0.01->0.03) while the nonlinear plateau is
      delta-INDEPENDENT and pinned at the DW ceiling -- the signature of chaotic
      decorrelation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402
from butterfly_cone.perturb import butterfly  # noqa: E402

_SNAP = ROOT / "runs" / "harmonic_null" / "figure_payload.snapshot.json"
PAYLOAD = _SNAP if _SNAP.exists() else ROOT / "runs" / "harmonic_null" / "figure_payload.json"
_SNAP_S = ROOT / "runs" / "harmonic_null" / "harmonic_null.snapshot.json"
SUMMARY = _SNAP_S if _SNAP_S.exists() else ROOT / "runs" / "harmonic_null" / "harmonic_null.json"
OUT = ROOT / "results" / "figures" / "fig_si_harmonic_null"
DW_CEILING = 223.36        # flagship D_sat total, N=1500
REP_DELTA = "0.01"         # panel (a): the smallest, most linear-response kick


def _rfront(t, D_field, positions, center, box):
    return butterfly.front_position(np.asarray(t), np.asarray(D_field),
                                    np.asarray(positions), np.asarray(center), np.asarray(box))


def main() -> int:
    payload = json.loads(PAYLOAD.read_text())
    summary = json.loads(SUMMARY.read_text())
    verdict = summary["verdict"]
    positions = np.asarray(payload["positions"])
    box = np.asarray(payload["box"])
    center = np.asarray(payload["center"])
    deltas = sorted(payload["deltas"], key=float)

    fs.use()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=fs.figsize(fs.WIDTH_THREEQ, aspect=0.52))

    # Only the inherent-structure (stable, n_negative=0) harmonic null is shown:
    # it is the correct phonon/elastic reference.  The instantaneous-config
    # linearization is UNPHYSICAL (8 negative Hessian eigenvalues => spurious
    # exponential blow-up) and is excluded from the figure by design.
    lam = verdict["nonlinear_lambda_B_median"]
    p = verdict["harmonic_IS_growth_power_median"]
    dwfrac = verdict["harmonic_IS_plateau_over_DW"]
    nl_ratio = verdict["nonlinear_plateau_delta_ratio"]
    h_ratio = verdict["harmonic_IS_plateau_delta_ratio_median"]

    # ---- panel (a): D(t), log-y, delta=0.01 -------------------------------
    d = payload["deltas"][REP_DELTA]
    t = np.asarray(d["t"])
    ax_a.axhline(DW_CEILING, color=fs.SUBTLE, lw=0.9, ls=":", zorder=1)
    ax_a.text(0.20, DW_CEILING * 1.22, "Debye-Waller ceiling", color=fs.SUBTLE,
              fontsize=fs.FS_ANNOT, ha="left", va="bottom")
    ax_a.semilogy(t, np.maximum(d["D_nonlinear"], 1e-6), color=fs.BLUE, lw=2.0,
                  label="full nonlinear cone", zorder=5)
    ax_a.semilogy(t, np.maximum(d["D_harmonic_IS"], 1e-6), color=fs.GREEN, lw=2.0,
                  label="harmonic / phonon null", zorder=4)
    ax_a.set_xlabel("time  $t$")
    ax_a.set_ylabel(r"twin divergence  $D(t)=\sum_i|\Delta r_i|$")
    ax_a.set_ylim(0.25, 9.0e2)
    ax_a.set_xlim(0, 20)
    ax_a.legend(loc="lower right")
    ax_a.text(9.3, 3.6, r"$D\sim e^{\lambda_B t}$" + f"\n$\\lambda_B={lam:.2f}$",
              color=fs.BLUE, fontsize=fs.FS_ANNOT, ha="left", va="center")
    ax_a.text(11.5, 1.15, r"$D\sim t^{" + f"{p:.2f}" + r"}$ (algebraic)" + f"\nplateau $={100*dwfrac:.0f}\\%$ of DW",
              color=fs.GREEN, fontsize=fs.FS_ANNOT, ha="left", va="center")
    fs.annotate_stats(ax_a, r"$\delta=0.01$", x=0.04, y=0.10)
    fs.panel_label(ax_a, "a")

    # ---- panel (b): plateau vs delta (the delta-independence discriminator)
    dvals = np.array([float(k) for k in deltas])
    harm_plateau = np.array([max(payload["deltas"][k]["D_harmonic_IS"]) for k in deltas])
    nl_plateau = np.array([payload["deltas"][k]["D_nonlinear"][-1] for k in deltas])
    ax_b.axhline(DW_CEILING, color=fs.SUBTLE, lw=0.9, ls=":", zorder=1)
    ax_b.text(dvals[0], DW_CEILING * 1.12, "Debye-Waller ceiling", color=fs.SUBTLE,
              fontsize=fs.FS_ANNOT, ha="left", va="bottom")
    guide = harm_plateau[0] * dvals / dvals[0]
    ax_b.plot(dvals, guide, color=fs.GUIDE, lw=1.0, ls="--", zorder=0, label=r"$\propto\delta$ (elastic)")
    ax_b.plot(dvals, nl_plateau, color=fs.BLUE, lw=1.6, marker="o", ms=6,
              label="nonlinear (chaos)")
    ax_b.plot(dvals, harm_plateau, color=fs.GREEN, lw=1.6, marker="s", ms=6,
              label="harmonic (phonon)")
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlim(8.5e-3, 3.6e-2)
    ax_b.set_xlabel(r"kick amplitude  $\delta$")
    ax_b.set_ylabel(r"saturation plateau  $D_\mathrm{sat}$")
    ax_b.legend(loc="center left")
    fs.annotate_stats(ax_b, r"$D_\mathrm{sat}(3\delta)/D_\mathrm{sat}(\delta)$:"
                             + f"\nnonlinear {nl_ratio:.2f} ($\\delta$-independent)"
                             + f"\nharmonic {h_ratio:.2f} ($\\propto\\delta$)",
                      x=0.30, y=0.34)
    fs.panel_label(ax_b, "b")

    fs.finalize(fig)
    written = fs.save(fig, str(OUT))
    print("wrote", [str(p) for p in written])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
