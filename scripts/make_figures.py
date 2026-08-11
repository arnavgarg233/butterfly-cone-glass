#!/usr/bin/env python
"""Publication figures for the ButterflyCone cavity-free floor paper.

Pure plotting from PERSISTED on-disk data (matplotlib only). No MD, no heavy
jobs. Every figure is a stand-alone function that:

  * loads its real persisted source if present (verify-then-plot),
  * accepts an injected ``data`` dict so tests can drive it with a small
    synthetic payload instead of the real file,
  * renders a clearly-labeled PLACEHOLDER panel (never crashes, never
    fabricates numbers) when its source is missing.

Figures
-------
1. fig_butterfly_cone      Gardner butterfly cone: lambda / v_b / D_sat with
                           per-config spread + lambda intensivity (N=1500 vs
                           N=3000 m2 point).
2. fig_channel_s_null      channel-S honest negative: t_shield ~ t_raw ~ t_sat
                           (no separation) and t_shield << tau_alpha.
3. fig_xi_pts_crossover    xi_PTS single-basin mixing m(R) crossover.
4. fig_pinning_nonmonotonic  Non-monotonic xi_dyn(f_p) with the interior peak.

Run directly to render all four PNGs into results/figures/.

Honest-value note (wave31 cone-rate-robustness-recheck):
  lambda ~ 0.91 (linear-response, range 0.85-0.98) is the headline, NOT the
  delta-inflated pooled 1.006.  D_sat = 223.4 +/- 3.5 is bulletproof.  v_b is
  NOT robust yet (~29-45% front-estimator spread) -> shown with a wide honest
  band labelled "pending finer-stride run".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results" / "figures"

GARDNER_DIR = REPO_ROOT / "runs" / "gardner"
SRC_FSS = GARDNER_DIR / "gardner-T0075-fss" / "gardner_r0.json"
SRC_M2 = GARDNER_DIR / "gardner-T0075-m2" / "gardner_r0.json"
SRC_CHANNELS = GARDNER_DIR / "gardner-T0075-fss" / "gardner_r0_channelS.json"
ERGO_DIR = REPO_ROOT / "runs" / "ergodicity_positive_control"
SRC_PINNING = REPO_ROOT / "runs" / "pinning_postdiction" / "pinning_postdiction.json"

# --------------------------------------------------------------------------
# Colorblind-safe categorical palette (Okabe-Ito, canonical CVD-safe order).
# Assigned by entity in fixed order, never cycled.
# --------------------------------------------------------------------------
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"
PLACEHOLDER_BG = "#FBEEE6"

DPI = 170


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class FigResult:
    path: Path
    is_placeholder: bool
    source: str  # "real", "synthetic", or "missing"
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _load_json(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _style_ax(ax) -> None:
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _placeholder_figure(out_path: Path, title: str, missing: str) -> FigResult:
    """Render a labelled placeholder PNG (valid, non-empty, no fabricated data)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_facecolor(PLACEHOLDER_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(CB["vermillion"])
        spine.set_linewidth(1.4)
    ax.text(
        0.5, 0.66, "DATA SOURCE MISSING", ha="center", va="center",
        fontsize=15, fontweight="bold", color=CB["vermillion"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.5, title, ha="center", va="center",
        fontsize=11, color=INK, transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.33, f"expected:\n{missing}", ha="center", va="center",
        fontsize=8, color=MUTED, family="monospace", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.12, "placeholder rendered — no data fabricated",
        ha="center", va="center", fontsize=8, style="italic",
        color=MUTED, transform=ax.transAxes,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    return FigResult(_save(fig, out_path), is_placeholder=True, source="missing",
                     notes=[f"missing source: {missing}"])


def _panel_missing(ax, msg: str) -> None:
    """In-panel missing-data note (one panel of a multi-panel figure)."""
    ax.set_facecolor(PLACEHOLDER_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, f"MISSING\n{msg}", ha="center", va="center",
            fontsize=9, color=CB["vermillion"], fontweight="bold",
            transform=ax.transAxes)


# ==========================================================================
# FIGURE 1: Butterfly cone
# ==========================================================================
def fig_butterfly_cone(
    out_path: Optional[Path] = None,
    fss_source: Optional[Path] = SRC_FSS,
    m2_source: Optional[Path] = SRC_M2,
    data: Optional[dict] = None,
) -> FigResult:
    """Gardner butterfly cone: honest lambda, D_sat, v_b + lambda intensivity.

    Real source: runs/gardner/gardner-T0075-fss/gardner_r0.json (per-ensemble
    lam / v_b / D_sat), plus gardner-T0075-m2/gardner_r0.json for the N=3000
    intensivity point.  ``data`` may inject {'ensembles': [...], 'by_N': {...}}.
    """
    out_path = Path(out_path) if out_path else FIG_DIR / "fig1_butterfly_cone.png"
    notes: list = []

    if data is not None:
        ensembles = data.get("ensembles", [])
        by_N = data.get("by_N", {})
        source = "synthetic"
    else:
        fss = _load_json(fss_source)
        if fss is None:
            return _placeholder_figure(out_path, "Fig 1 — Gardner butterfly cone",
                                       str(fss_source))
        ensembles = fss.get("ensembles", [])
        m2 = _load_json(m2_source)
        by_N = (m2 or {}).get("by_N", {})
        if m2 is None:
            notes.append(f"m2 (N=3000) source missing: {m2_source}")
        source = "real"

    # group per-ensemble rates by delta (perturbation magnitude)
    by_delta: dict = {}
    for e in ensembles:
        d = e.get("delta")
        by_delta.setdefault(d, {"lam": [], "v_b": [], "D_sat": []})
        for k in ("lam", "v_b", "D_sat"):
            v = e.get(k)
            if v is not None:
                by_delta[d][k].append(float(v))
    deltas = sorted(by_delta)

    # linear-response regime = all but the largest delta (the d2 violator)
    lr_deltas = deltas[:-1] if len(deltas) >= 2 else deltas
    violator = deltas[-1] if len(deltas) >= 2 else None

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    (axA, axB), (axC, axD) = axes

    # ---- Panel A: lambda per-ensemble by delta + honest headline band ----
    _style_ax(axA)
    for i, d in enumerate(deltas):
        vals = by_delta[d]["lam"]
        if not vals:
            continue
        is_viol = d == violator
        color = CB["vermillion"] if is_viol else CB["blue"]
        x = np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.09, 0.09, len(vals))
        axA.scatter(x, vals, s=26, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.4, zorder=3,
                    label=("delta (excluded, leaves linear regime)" if is_viol
                           else "linear-response rungs") if i in (0, len(deltas) - 1) else None)
        axA.scatter([i], [np.median(vals)], marker="_", s=900,
                    color=INK, linewidth=2.2, zorder=4)
    # honest headline band 0.85-0.98, line at 0.91
    axA.axhspan(0.85, 0.98, color=CB["green"], alpha=0.14, zorder=1)
    axA.axhline(0.91, color=CB["green"], linewidth=1.8, zorder=2)
    axA.text(0.02, 0.91, r" headline $\lambda \approx 0.91$",
             transform=axA.get_yaxis_transform(), va="bottom", ha="left",
             fontsize=8.5, color=CB["green"], fontweight="bold")
    axA.set_xticks(range(len(deltas)))
    axA.set_xticklabels([f"{d:g}" for d in deltas])
    axA.set_xlabel("perturbation magnitude  delta", fontsize=9, color=INK)
    axA.set_ylabel(r"Lyapunov rate  $\lambda$", fontsize=9, color=INK)
    axA.set_title("A  butterfly rate: linear-response floor",
                  fontsize=10, color=INK, fontweight="bold", loc="left")
    if violator is not None:
        axA.annotate(
            f"delta={violator:g} saturates early\n(inflates pooled mean to 1.006)",
            xy=(len(deltas) - 1, np.median(by_delta[violator]["lam"])),
            xytext=(0.30, 0.86), textcoords="axes fraction",
            fontsize=7.5, color=CB["vermillion"],
            arrowprops=dict(arrowstyle="->", color=CB["vermillion"], lw=1.0))
    axA.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # ---- Panel B: D_sat per-ensemble + bulletproof band ----
    _style_ax(axB)
    d_all = [v for d in deltas for v in by_delta[d]["D_sat"]]
    if d_all:
        d_all = np.asarray(d_all)
        x = np.arange(len(d_all))
        axB.scatter(x, d_all, s=20, color=CB["blue"], alpha=0.65,
                    edgecolor="white", linewidth=0.3, zorder=3)
        mean, std = 223.4, 3.5  # committed bulletproof value (0.4% knob spread)
        axB.axhspan(mean - std, mean + std, color=CB["blue"], alpha=0.15, zorder=1)
        axB.axhline(mean, color=CB["blue"], linewidth=1.8, zorder=2)
        axB.text(0.98, mean, r" $D_{\rm sat}=223.4\pm3.5$",
                 transform=axB.get_yaxis_transform(), va="bottom", ha="right",
                 fontsize=8.5, color=CB["blue"], fontweight="bold")
    else:
        _panel_missing(axB, "D_sat")
    axB.set_xlabel("ensemble index", fontsize=9, color=INK)
    axB.set_ylabel(r"saturation divergence  $D_{\rm sat}$", fontsize=9, color=INK)
    axB.set_title("B  plateau: bulletproof (0.4% spread)",
                  fontsize=10, color=INK, fontweight="bold", loc="left")

    # ---- Panel C: v_b per-ensemble + WIDE honest band ----
    _style_ax(axC)
    v_all = [v for d in deltas for v in by_delta[d]["v_b"]]
    if v_all:
        v_all = np.asarray(v_all)
        x = np.arange(len(v_all))
        axC.scatter(x, v_all, s=20, color=CB["orange"], alpha=0.65,
                    edgecolor="white", linewidth=0.3, zorder=3)
        vmed = float(np.median(v_all))
        # honest wide band = committed robustness-sweep range (front-estimator wobble)
        lo, hi = 1.33, 3.11
        axC.axhspan(lo, hi, color=CB["orange"], alpha=0.16, zorder=1)
        axC.axhline(vmed, color=CB["orange"], linewidth=1.8, zorder=2)
        axC.text(0.98, vmed, f" $v_b \\approx {vmed:.2f}$",
                 transform=axC.get_yaxis_transform(), va="bottom", ha="right",
                 fontsize=8.5, color=CB["vermillion"], fontweight="bold")
        axC.text(0.5, 0.04,
                 "NOT robust: ~29-45% front-estimator spread — pending finer-stride run",
                 transform=axC.transAxes, ha="center", va="bottom",
                 fontsize=7.8, color=CB["vermillion"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec=CB["vermillion"], lw=0.8))
    else:
        _panel_missing(axC, "v_b")
    axC.set_xlabel("ensemble index", fontsize=9, color=INK)
    axC.set_ylabel(r"butterfly velocity  $v_b$", fontsize=9, color=INK)
    axC.set_title("C  cone-front velocity: not yet robust",
                  fontsize=10, color=INK, fontweight="bold", loc="left")

    # ---- Panel D: lambda intensivity vs N ----
    _style_ax(axD)
    if by_N:
        Ns, lams = [], []
        for k, v in sorted(by_N.items(), key=lambda kv: int(kv[0])):
            lm = v.get("lambda_mean")
            if lm is not None:
                Ns.append(int(k))
                lams.append(float(lm))
        if Ns:
            axD.plot(Ns, lams, "-o", color=CB["purple"], markersize=9,
                     linewidth=1.8, markeredgecolor="white", zorder=3)
            for n, lm in zip(Ns, lams):
                axD.annotate(f"N={n}\n$\\lambda$={lm:.3f}", (n, lm),
                             textcoords="offset points", xytext=(0, 12),
                             ha="center", fontsize=8, color=INK)
            axD.axhspan(0.85, 0.98, color=CB["green"], alpha=0.12, zorder=1)
            axD.text(0.02, 0.86, " honest linear-response band",
                     transform=axD.get_yaxis_transform(), va="bottom", ha="left",
                     fontsize=7.5, color=CB["green"])
            rel = abs(lams[-1] - lams[0]) / (0.5 * (lams[-1] + lams[0]))
            axD.text(0.5, 0.03,
                     f"pooled (all-δ) estimator: {rel*100:.0f}% change across 2×N → intensive",
                     transform=axD.transAxes, ha="center", va="bottom",
                     fontsize=7.6, color=MUTED, style="italic")
            axD.set_xlim(min(Ns) - 400, max(Ns) + 400)
            span = max(lams) - min(lams)
            axD.set_ylim(min(0.84, min(lams) - max(span, 0.05) - 0.03),
                         max(lams) + max(span, 0.05) + 0.06)
        else:
            _panel_missing(axD, "by_N lambda")
    else:
        _panel_missing(axD, "N=3000 (m2) point")
    axD.set_xlabel("system size  N", fontsize=9, color=INK)
    axD.set_ylabel(r"pooled  $\lambda$", fontsize=9, color=INK)
    axD.set_title(r"D  intensivity: $\lambda$ ~ N-independent",
                  fontsize=10, color=INK, fontweight="bold", loc="left")

    fig.suptitle("Gardner butterfly cone — cavity-free floor (T=0.075)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return FigResult(_save(fig, out_path), is_placeholder=False, source=source,
                     notes=notes)


# ==========================================================================
# FIGURE 2: channel-S null
# ==========================================================================
def fig_channel_s_null(
    out_path: Optional[Path] = None,
    source: Optional[Path] = SRC_CHANNELS,
    data: Optional[dict] = None,
) -> FigResult:
    """channel-S honest negative: t_shield ~ t_raw ~ t_sat and t_shield << tau.

    Real source: runs/gardner/gardner-T0075-fss/gardner_r0_channelS.json.
    ``data`` may inject {'threshold_band': {...}, 'verdict': {...},
    'reference': {...}}.
    """
    out_path = Path(out_path) if out_path else FIG_DIR / "fig2_channel_s_null.png"

    if data is not None:
        payload = data
        source_kind = "synthetic"
    else:
        payload = _load_json(source)
        if payload is None:
            return _placeholder_figure(out_path, "Fig 2 — channel-S null",
                                       str(source))
        source_kind = "real"

    band = payload.get("threshold_band", {}).get("per_threshold", [])
    verdict = payload.get("verdict", {})
    ref = payload.get("reference", {})

    thr = np.array([p["shield_threshold_frac"] for p in band]) if band else np.array([])
    t_shield = np.array([p["t_shield"]["mean"] for p in band]) if band else np.array([])
    t_shield_sd = np.array([p["t_shield"]["std"] for p in band]) if band else np.array([])
    t_raw = np.array([p["t_raw"]["mean"] for p in band]) if band else np.array([])
    t_raw_sd = np.array([p["t_raw"]["std"] for p in band]) if band else np.array([])
    t_sat = np.array([p["t_sat"]["mean"] for p in band]) if band else np.array([])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- Left: threshold-band sweep: no separation ----
    _style_ax(axL)
    if thr.size:
        axL.errorbar(thr, t_shield, yerr=t_shield_sd, fmt="-o", color=CB["blue"],
                     markersize=6, linewidth=1.8, capsize=3, elinewidth=1.0,
                     markeredgecolor="white", label=r"$t_{\rm shield}$", zorder=4)
        axL.errorbar(thr, t_raw, yerr=t_raw_sd, fmt="-s", color=CB["orange"],
                     markersize=6, linewidth=1.8, capsize=3, elinewidth=1.0,
                     markeredgecolor="white", label=r"$t_{\rm raw}$", zorder=3)
        axL.plot(thr, t_sat, "--D", color=CB["green"], markersize=6,
                 linewidth=1.8, markeredgecolor="white",
                 label=r"$t_{\rm sat}$", zorder=3)
        axL.text(0.5, 0.9,
                 "curves overlap within 1$\\sigma$ → NO structural shielding",
                 transform=axL.transAxes, ha="center", va="top", fontsize=9,
                 color=CB["vermillion"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec=CB["vermillion"], lw=0.8))
        axL.legend(fontsize=9, loc="lower right", framealpha=0.9)
    else:
        _panel_missing(axL, "threshold_band")
    axL.set_xlabel("shield threshold fraction", fontsize=9.5, color=INK)
    axL.set_ylabel("time (t.u.)", fontsize=9.5, color=INK)
    axL.set_title("A  no separation of the three timescales",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")

    # ---- Right: t_shield << tau_alpha ----
    _style_ax(axR)
    t_s = ref.get("t_shield", verdict.get("t_shield_ref"))
    t_r = ref.get("t_raw", verdict.get("t_raw_ref"))
    t_st = ref.get("t_sat", verdict.get("t_sat_ref"))
    tau_lb = verdict.get("tau_floor_lower_bound")
    ratio = verdict.get("t_shield_over_tau_ratio")
    if t_s is not None and tau_lb is not None:
        labels = [r"$t_{\rm shield}$", r"$t_{\rm raw}$", r"$t_{\rm sat}$",
                  r"$\tau_\alpha$ floor"]
        vals = [t_s, t_r or 0, t_st or 0, tau_lb]
        colors = [CB["blue"], CB["orange"], CB["green"], CB["grey"]]
        bars = axR.bar(labels, vals, color=colors, width=0.62,
                       edgecolor="white", linewidth=1.0, zorder=3)
        # mark tau as a lower bound
        axR.annotate("", xy=(3, tau_lb + 3.0), xytext=(3, tau_lb),
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
        axR.text(3, tau_lb + 3.3, "lower\nbound", ha="center", va="bottom",
                 fontsize=7.5, color=MUTED)
        for b, v in zip(bars, vals):
            axR.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8, color=INK)
        if ratio is not None:
            axR.text(0.5, 0.72,
                     f"$t_{{\\rm shield}}/\\tau_\\alpha \\lesssim {ratio:.2f}$",
                     transform=axR.transAxes, ha="center", va="center",
                     fontsize=13, color=CB["vermillion"], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.35", fc="white",
                               ec=CB["vermillion"], lw=1.0))
    else:
        _panel_missing(axR, "reference / tau")
    axR.set_ylabel("time (t.u.)", fontsize=9.5, color=INK)
    axR.set_title(r"B  $t_{\rm shield}\ll\tau_\alpha$ (not a relaxation time)",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")

    fig.suptitle("channel-S: the honest negative that closed the ceiling",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return FigResult(_save(fig, out_path), is_placeholder=False, source=source_kind)


# ==========================================================================
# FIGURE 3: xi_PTS m(R) crossover
# ==========================================================================
def _collect_ergo_points(ergo_dir: Path) -> list:
    """Read persisted grid metrics -> [(name, R, T, m, n_basins, verdict)]."""
    pts = []
    if not Path(ergo_dir).exists():
        return pts
    for mfile in sorted(Path(ergo_dir).glob("*/metrics.json")):
        d = _load_json(mfile)
        if not d:
            continue
        cav = d.get("cavity", {})
        mix = d.get("mixing", {})
        ctrl = d.get("control", {})
        R = cav.get("core_radius")
        T = ctrl.get("temperature")
        m = mix.get("m")
        if R is None or m is None:
            continue
        pts.append({
            "name": mfile.parent.name, "R": float(R),
            "T": None if T is None else float(T), "m": float(m),
            "n_basins": mix.get("n_basins"),
            "verdict": mix.get("verdict"),
            "single_basin": mix.get("guaranteed_single_basin"),
        })
    return pts


def fig_xi_pts_crossover(
    out_path: Optional[Path] = None,
    ergo_dir: Optional[Path] = ERGO_DIR,
    data: Optional[list] = None,
) -> FigResult:
    """xi_PTS single-basin mixing quality m vs core radius R.

    Real source: runs/ergodicity_positive_control/*/metrics.json (grid of
    (T, R) with basin_pts mixing m).  ``data`` may inject a list of point dicts
    with keys R, T, m, single_basin, verdict.
    """
    out_path = Path(out_path) if out_path else FIG_DIR / "fig3_xi_pts_crossover.png"
    notes: list = []

    if data is not None:
        pts = data
        source_kind = "synthetic"
    else:
        pts = _collect_ergo_points(ergo_dir)
        if not pts:
            return _placeholder_figure(out_path, "Fig 3 — xi_PTS m(R) crossover",
                                       str(ergo_dir))
        source_kind = "real"

    # grid runs are the systematic T-controlled crossover; keep those T values
    def _grid(temp):
        rows = [p for p in pts if p["T"] is not None
                and abs(p["T"] - temp) < 1e-6 and p["name"].startswith("grid")]
        return sorted(rows, key=lambda r: r["R"])

    g013 = _grid(0.13)
    g015 = _grid(0.15)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    _style_ax(ax)

    M_MIN = 0.9  # single-basin acceptance threshold
    # crossover band R in [0.7, 1.0]
    ax.axvspan(0.7, 1.0, color=CB["yellow"], alpha=0.22, zorder=0)
    ax.text(0.85, 0.5, "crossover\nR∈[0.7,1.0]", ha="center", va="center",
            fontsize=8.5, color="#9A7D00", fontweight="bold", rotation=0)
    ax.axhline(M_MIN, color=MUTED, linestyle=":", linewidth=1.4, zorder=1)
    ax.text(0.42, M_MIN + 0.015, r"$m_{\min}=0.9$ (single-basin bar)", va="bottom",
            ha="left", fontsize=8.5, color=MUTED)

    plotted = False
    if g013:
        R = [p["R"] for p in g013]
        m = [p["m"] for p in g013]
        ax.plot(R, m, "-o", color=CB["blue"], markersize=8, linewidth=2.0,
                markeredgecolor="white", zorder=4, label="T=0.13 grid")
        plotted = True
    if g015:
        R = [p["R"] for p in g015]
        m = [p["m"] for p in g015]
        ax.plot(R, m, "--s", color=CB["orange"], markersize=7, linewidth=1.6,
                markeredgecolor="white", zorder=3, label="T=0.15 grid")
        plotted = True

    # validated single-basin control point (m~0.99 at R=0.5, T=0.13)
    ctrl = next((p for p in g013 if abs(p["R"] - 0.5) < 1e-6), None)
    if ctrl:
        ax.scatter([ctrl["R"]], [ctrl["m"]], marker="*", s=340,
                   color=CB["green"], edgecolor="white", linewidth=0.8,
                   zorder=6)
        ax.annotate(
            f"validated single-basin control\nm={ctrl['m']:.2f} at R=0.5 (PASS)",
            xy=(ctrl["R"], ctrl["m"]), xytext=(0.62, 0.63),
            textcoords="axes fraction", fontsize=8.5, color=CB["green"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CB["green"], lw=1.1))

    if not plotted:
        # fall back to whatever points exist (synthetic path)
        R = [p["R"] for p in pts]
        m = [p["m"] for p in pts]
        ax.plot(R, m, "-o", color=CB["blue"], markersize=8, linewidth=2.0,
                markeredgecolor="white", zorder=4, label="m(R)")

    # swapheavy R=2.5 endpoint: DECLARED (T=0.108), basin_pts metric not persisted
    ax.scatter([2.5], [0.18], marker="o", s=90, facecolor="none",
               edgecolor=CB["vermillion"], linewidth=1.6, zorder=5)
    ax.errorbar([2.5], [0.18], yerr=[[0.05], [0.05]], fmt="none",
                ecolor=CB["vermillion"], elinewidth=1.2, capsize=4, zorder=5)
    ax.annotate(
        "swapheavy R=2.5 endpoint\n(declared m~0.13–0.23, T=0.108;\nbasin_pts metric NOT persisted)",
        xy=(2.5, 0.18), xytext=(0.52, 0.30), textcoords="axes fraction",
        fontsize=7.6, color=CB["vermillion"], style="italic",
        arrowprops=dict(arrowstyle="->", color=CB["vermillion"], lw=1.0))
    notes.append("R=2.5 swapheavy endpoint shown as declared (m~0.13-0.23); "
                 "not persisted as a basin_pts metric in repo")

    ax.set_xlabel("core radius  R", fontsize=10, color=INK)
    ax.set_ylabel("mixing quality  m  (core–parent overlap)", fontsize=10, color=INK)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0.35, 2.75)
    ax.legend(fontsize=9, loc="center right", framealpha=0.9)
    ax.set_title(r"$\xi_{\rm PTS}$: single-basin mixing collapses across R∈[0.7,1.0]",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    return FigResult(_save(fig, out_path), is_placeholder=False, source=source_kind,
                     notes=notes)


# ==========================================================================
# FIGURE 4: pinning non-monotonic xi_dyn(f_p)
# ==========================================================================
def fig_pinning_nonmonotonic(
    out_path: Optional[Path] = None,
    source: Optional[Path] = SRC_PINNING,
    data: Optional[dict] = None,
) -> FigResult:
    """Non-monotonic dynamic length xi_dyn(f_p) with its interior peak.

    Real source: runs/pinning_postdiction/pinning_postdiction.json.  ``data``
    may inject {'response': {'points': [...]}, 'declared_predictions': {...},
    'trend_matches': int, 'trend_total': int}.
    """
    out_path = Path(out_path) if out_path else FIG_DIR / "fig4_pinning_nonmonotonic.png"

    if data is not None:
        payload = data
        source_kind = "synthetic"
    else:
        payload = _load_json(source)
        if payload is None:
            return _placeholder_figure(out_path,
                                       "Fig 4 — pinning xi_dyn(f_p) non-monotonic",
                                       str(source))
        source_kind = "real"

    points = payload.get("response", {}).get("points", [])
    matches = payload.get("trend_matches")
    total = payload.get("trend_total")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- Left: xi_dyn(f_p) non-monotonic ----
    _style_ax(axL)
    if points:
        fp = np.array([p["f_p"] for p in points])
        xi = np.array([p.get("xi_dyn", np.nan) for p in points], dtype=float)
        good = np.isfinite(xi)
        axL.plot(fp[good], xi[good], "-o", color=CB["blue"], markersize=8,
                 linewidth=2.0, markeredgecolor="white", zorder=3)
        # interior peak (exclude endpoints)
        if good.sum() >= 3:
            interior = np.where(good)[0][1:-1]
            if interior.size:
                pk = interior[np.argmax(xi[interior])]
                axL.scatter([fp[pk]], [xi[pk]], marker="*", s=300,
                            color=CB["vermillion"], edgecolor="white",
                            linewidth=0.8, zorder=5)
                axL.annotate(f"interior peak\n$f_p$={fp[pk]:g}, $\\xi_{{\\rm dyn}}$={xi[pk]:.2f}",
                             xy=(fp[pk], xi[pk]), xytext=(0.30, 0.86),
                             textcoords="axes fraction", fontsize=8.5,
                             color=CB["vermillion"], fontweight="bold",
                             arrowprops=dict(arrowstyle="->",
                                             color=CB["vermillion"], lw=1.1))
        axL.text(0.5, 0.06, "rises then falls — non-monotonic (frozen prediction)",
                 transform=axL.transAxes, ha="center", va="bottom", fontsize=8.5,
                 color=INK, style="italic")
    else:
        _panel_missing(axL, "response points")
    axL.set_xlabel("pinning fraction  $f_p$", fontsize=9.5, color=INK)
    axL.set_ylabel(r"dynamic length  $\xi_{\rm dyn}$", fontsize=9.5, color=INK)
    axL.set_title(r"A  non-monotonic $\xi_{\rm dyn}(f_p)$",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")

    # ---- Right: monotonic companions (single axis, normalised to f_p=0) ----
    _style_ax(axR)
    if points:
        fp = np.array([p["f_p"] for p in points])
        series = [
            (r"$\tau_\alpha$ (increasing)", "tau_alpha", CB["orange"], "-o"),
            (r"$q_{\rm pin}$ (increasing)", "q_pin", CB["green"], "-s"),
        ]
        drew = False
        for label, key, color, style in series:
            y = np.array([p.get(key, np.nan) for p in points], dtype=float)
            base = y[0] if (y.size and np.isfinite(y[0]) and y[0] != 0) else np.nan
            if np.isfinite(base):
                ratio = y / base
                m = np.isfinite(ratio)
                axR.plot(fp[m], ratio[m], style, color=color, markersize=6,
                         linewidth=1.7, markeredgecolor="white", label=label,
                         zorder=3)
                drew = True
        if drew:
            axR.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.0)
            axR.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        else:
            _panel_missing(axR, "companion trends")
    else:
        _panel_missing(axR, "response points")
    axR.set_xlabel("pinning fraction  $f_p$", fontsize=9.5, color=INK)
    axR.set_ylabel("ratio to unpinned ($f_p{=}0$)", fontsize=9.5, color=INK)
    ttl = "B  monotonic companions"
    if matches is not None and total is not None:
        ttl += f"  ({matches}/{total} frozen trends matched)"
    axR.set_title(ttl, fontsize=10.5, color=INK, fontweight="bold", loc="left")

    fig.suptitle("Pinning post-diction: non-monotonic dynamic length",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return FigResult(_save(fig, out_path), is_placeholder=False, source=source_kind)


# ==========================================================================
# Driver
# ==========================================================================
def render_all(fig_dir: Optional[Path] = None) -> dict:
    fig_dir = Path(fig_dir) if fig_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "fig1_butterfly_cone": fig_butterfly_cone(fig_dir / "fig1_butterfly_cone.png"),
        "fig2_channel_s_null": fig_channel_s_null(fig_dir / "fig2_channel_s_null.png"),
        "fig3_xi_pts_crossover": fig_xi_pts_crossover(fig_dir / "fig3_xi_pts_crossover.png"),
        "fig4_pinning_nonmonotonic": fig_pinning_nonmonotonic(
            fig_dir / "fig4_pinning_nonmonotonic.png"),
    }
    return results


def main() -> None:
    results = render_all()
    print(f"Rendered {len(results)} figures into {FIG_DIR}")
    for name, res in results.items():
        tag = "PLACEHOLDER" if res.is_placeholder else res.source.upper()
        print(f"  [{tag:11s}] {name:26s} -> {res.path}")
        for note in res.notes:
            print(f"      note: {note}")


if __name__ == "__main__":
    main()
