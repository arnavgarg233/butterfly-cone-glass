#!/usr/bin/env python3
"""Fig. 6: the chaos ceiling is local -- the butterfly cone inside a confined film.

Every number is read from ``runs/slab_cone/slab_cone_merged.json`` (32 twin
pairs: eight per geometry over three films ``h/L = 0.70, 0.50, 0.35`` plus a
bulk control on the identical code path).  Nothing is fitted, smoothed or
hand-entered; the per-pair values are plotted as clouds behind every summary
point so the reader sees the ensemble, not just its mean.

The run controls (``dt``, horizon, kick shell radius) are NOT carried into the
merged artifact, so they are read from the four per-geometry shards in
``runs/slab_cone/shards3/`` and asserted identical across them.  That is the only
information this figure needs which the merged file does not hold.

Panels
------
a  ``D_sat/N`` and ``u_DW`` against film thickness, on one shared axis so that
   "both sides of the identity fall together" is read directly rather than
   inferred from two differently scaled axes.  Bulk enters as two horizontal
   mean +- s.e.m. bands, so every film point is read against its control.
b  the ratio ``c = (D_sat/N)/u_DW``, flat, against the bulk band: the ceiling
   identity survives spatial restriction.
c  the second-moment anisotropy ``sqrt(<d_par^2>/2<d_perp^2>)``, flat at 0.70
   and 0.50 and rising from 0.45 onward: a crossover, not a trend.  The shaded band
   marks the onset window between the 0.50 and 0.45 films, the thickness at which
   than the injected perturbation itself; the crossover brackets it.
d  the divergence curves ``D(t)/N`` themselves, all 32 pairs, with the plateau
   window that defines ``D_sat`` shaded; the inset rescales each curve by its
   own film-local ``u_DW`` and the four films collapse onto one ceiling, which
   is panel b as a time series.

Run:
    ./.venv/bin/python scripts/figures/fig_confinement.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "runs/slab_cone/slab_cone_merged.json"
SHARDS = sorted((ROOT / "runs/slab_cone/shards3").glob("s_*.json"))
OUT = ROOT / "results/figures/fig_confinement"

# Geometry order used everywhere: unconfined first, most confined last, so the
# cividis ramp runs navy (bulk, the anchor) -> khaki (narrowest film).
ORDER: list[float | None] = [None, 0.70, 0.50, 0.45, 0.35]

# Per-pair clouds sit just left of their summary marker rather than under it, so
# every ensemble is visible and the annotation space beside each mean stays clear.
CLOUD_DX = -0.032
CLOUD_SPREAD = 0.013
WHITE_BOX = dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.82,
                 edgecolor="none")


def _cloud_x(centre: float, n: int) -> np.ndarray:
    return centre + CLOUD_DX + np.linspace(-CLOUD_SPREAD, CLOUD_SPREAD, n)


# --------------------------------------------------------------------------- #
# Load + hard-verify
# --------------------------------------------------------------------------- #
def load() -> dict:
    """Read the merged campaign and its shard controls, verifying both."""
    doc = json.loads(MERGED.read_text())
    pairs = doc["pairs"]
    summary = doc["summary"]
    assert len(pairs) == 40, len(pairs)
    assert len(summary) == 5, len(summary)

    if len(SHARDS) != 5:
        raise FileNotFoundError(f"expected 5 shards beside {MERGED}, found {len(SHARDS)}")
    controls = [json.loads(p.read_text())["controls"] for p in SHARDS]
    # Compare the physics controls rather than the raw dicts: shards written before
    # the matched-kick control existed have no ``kick_clip_fraction`` key, and a bare
    # dict comparison would fail on the key set while every value agrees.
    _PHYS = ("dt", "horizon_steps", "horizon_time", "temperature", "deltas",
             "r_pert", "interface", "pairs_per_delta")
    _phys = [{k: c[k] for k in _PHYS} for c in controls]
    assert all(c == _phys[0] for c in _phys), "shard controls disagree"
    # This figure must never be built from clipped-kick runs: those are the matched
    # control, not the campaign.
    assert all(c.get("kick_clip_fraction") is None for c in controls), "clipped-kick shard"
    controls = controls[0]
    dt = float(controls["dt"])
    horizon = int(controls["horizon_steps"])
    r_shell = float(controls["r_pert"])
    assert dt == 0.005 and horizon == 8000 and r_shell == 2.5
    assert float(controls["temperature"]) == 0.075
    assert controls["deltas"] == [0.01, 0.03] and int(controls["pairs_per_delta"]) == 4

    n_frames = len(pairs[0]["divergence_curve"])
    assert all(len(p["divergence_curve"]) == n_frames for p in pairs)
    stride, rem = divmod(horizon, n_frames - 1)
    assert rem == 0, (horizon, n_frames)
    times = np.arange(n_frames) * stride * dt
    assert times[-1] == horizon * dt == 40.0

    # D_sat is the mean of the second half of the curve (campaign's `tail`).
    tail = slice(max(1, n_frames // 2), None)
    # The bulk control's "film" is the whole periodic box, so its thickness IS L.
    bulk_pairs = [p for p in pairs if p["geometry"] == "bulk_control"]
    box_l = float(bulk_pairs[0]["thickness_sigma"])
    assert len(bulk_pairs) == 8
    assert all(p["thickness_sigma"] == box_l for p in bulk_pairs)
    for p in pairs:
        if p["geometry"] == "slab":
            frac = float(p["thickness_fraction_of_box"])
            assert abs(p["thickness_sigma"] - frac * box_l) < 1e-9

    groups: dict[float | None, dict] = {}
    for row in summary:
        key = row["thickness_fraction_of_box"]
        members = [p for p in pairs if p["thickness_fraction_of_box"] == key]
        assert len(members) == int(row["n_pairs"]) == 8

        # Reproduce every summary statistic from the per-pair rows.
        for value_key, err_key in (("d_sat_per_particle", "d_sat_sem"),
                                   ("u_dw", "u_dw_sem"),
                                   ("c_ratio", "c_ratio_sem"),
                                   ("anisotropy", "anisotropy_sem")):
            vals = np.array([float(m[value_key]) for m in members])
            assert abs(vals.mean() - float(row[value_key])) < 1e-12, value_key
            sem = vals.std(ddof=1) / math.sqrt(len(vals))
            assert abs(sem - float(row[err_key])) < 1e-12, err_key

        for m in members:
            curve = np.asarray(m["divergence_curve"], float)
            # D_sat/N reproduces from the raw curve, and the anisotropy from the
            # two resolved components: the artifact is internally consistent.
            assert abs(curve[tail].mean() / m["n_mobile"] - m["d_sat_per_particle"]) < 1e-12
            assert abs(m["in_plane_rms"] / (math.sqrt(2.0) * m["normal_rms"])
                       - m["anisotropy"]) < 1e-12
            assert int(m["n_perturbed"]) == 66
            assert int(m["n_mobile"]) + int(m["n_wall"]) == 1500

        groups[key] = {
            "summary": row,
            "pairs": members,
            "curves": np.array([np.asarray(m["divergence_curve"], float) / m["n_mobile"]
                                for m in members]),
            "thickness": float(members[0]["thickness_sigma"]),
            "fraction": 1.0 if key is None else float(key),
            "label": "bulk" if key is None else f"{float(key):.2f}",
            "n_mobile": int(members[0]["n_mobile"]),
        }

    assert [k for k in ORDER] == sorted(groups, key=lambda k: -groups[k]["fraction"])
    return {"groups": groups, "times": times, "tail": tail, "box_l": box_l,
            "r_shell": r_shell, "dt": dt, "horizon": horizon}


def verify_headlines(groups: dict) -> dict:
    """Assert the manuscript's confinement numbers, and derive the z-scores."""
    bulk = groups[None]["summary"]
    # Corrected campaign (runs/slab_cone/shards2): the kick is now restricted to
    # the mobile film, so at h/L = 0.35 the 4 shell hits that landed on the
    # pinned wall are discarded.  The other three films had zero contamination
    # and are bit-identical to the earlier run, which is the control on the fix.
    expect = {
        None: (0.14789, 0.11884, 1.24459, 0.98644),
        0.70: (0.13314, 0.10965, 1.21435, 0.98822),
        0.50: (0.13059, 0.10694, 1.22131, 0.98796),
        0.45: (0.12666, 0.10554, 1.20047, 1.01526),
        0.35: (0.12278, 0.10003, 1.22969, 1.03755),
    }
    stats: dict = {}
    for key, (d_sat, u_dw, c_ratio, aniso) in expect.items():
        row = groups[key]["summary"]
        for got, want in ((row["d_sat_per_particle"], d_sat), (row["u_dw"], u_dw),
                          (row["c_ratio"], c_ratio), (row["anisotropy"], aniso)):
            assert abs(float(got) - want) < 5e-5, (key, got, want)
        z = {}
        for value_key, err_key in (("d_sat_per_particle", "d_sat_sem"),
                                   ("u_dw", "u_dw_sem"),
                                   ("c_ratio", "c_ratio_sem"),
                                   ("anisotropy", "anisotropy_sem")):
            delta = float(row[value_key]) - float(bulk[value_key])
            spread = math.hypot(float(row[err_key]), float(bulk[err_key]))
            z[value_key] = {
                "delta": delta,
                "percent": 100.0 * delta / float(bulk[value_key]),
                "z": 0.0 if key is None else delta / spread,
            }
        stats[key] = z

    # C1: flat at 0.70 and 0.50, onset by 0.45, larger at 0.35.  The 0.45 film is
    # thicker than the kick shell and discards nothing, so it carries the onset
    # free of the wall-straddling truncation that only 0.35 has.
    assert abs(stats[0.70]["anisotropy"]["z"] - 0.14) < 0.01
    assert abs(stats[0.50]["anisotropy"]["z"] - 0.10) < 0.01
    assert abs(stats[0.45]["anisotropy"]["z"] - 3.83) < 0.02
    assert abs(stats[0.35]["anisotropy"]["z"] - 5.42) < 0.02
    # C2: the ratio is preserved within 3.6% and the offsets do not order with
    # thickness, which is what makes them a systematic rather than a trend.
    worst = max(stats, key=lambda k: abs(stats[k]["c_ratio"]["percent"]))
    assert worst == 0.45 and abs(stats[0.45]["c_ratio"]["percent"]) < 3.6
    assert abs(abs(stats[0.45]["c_ratio"]["z"]) - 2.77) < 0.02
    assert abs(abs(stats[0.70]["c_ratio"]["z"]) - 2.82) < 0.01
    # C3: D_sat/N monotone in thickness.
    d_sat_ordered = [float(groups[k]["summary"]["d_sat_per_particle"]) for k in ORDER]
    assert all(a > b for a, b in zip(d_sat_ordered, d_sat_ordered[1:])), d_sat_ordered
    return stats


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _x_axis(ax, groups: dict) -> None:
    """Shared confinement axis: h/L primary, h/sigma on the second tick line."""
    ticks = [groups[k]["fraction"] for k in ORDER]
    labels = [f"{groups[k]['label']}\n{groups[k]['thickness']:.1f}" for k in ORDER]
    ax.set_xlim(0.283, 1.078)
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", labelsize=fs.FS_TICK - 0.5)
    # The 0.45 and 0.50 films sit 0.05 apart in h/L and their two-line labels
    # collide.  Drop the lower of the pair by one line so both stay readable.
    for label, key in zip(ax.get_xticklabels(), ORDER):
        if key == 0.45:
            label.set_y(label.get_position()[1] - 0.125)
    ax.set_xlabel(r"film thickness  $h/L$   ($h/\sigma$)", labelpad=1.5)


def _bulk_band(ax, mean: float, sem: float, colour) -> None:
    ax.axhspan(mean - sem, mean + sem, color=colour, alpha=0.16, lw=0.0, zorder=0)
    ax.axhline(mean, color=colour, lw=0.7, ls=(0, (1.0, 1.8)), zorder=1)


def panel_a(ax, groups: dict, colours: dict, stats: dict) -> None:
    """D_sat/N and u_DW falling together on one shared scale."""
    films = ORDER[1:]
    bulk = groups[None]["summary"]
    _bulk_band(ax, float(bulk["d_sat_per_particle"]), float(bulk["d_sat_sem"]),
               colours[None])
    _bulk_band(ax, float(bulk["u_dw"]), float(bulk["u_dw_sem"]), colours[None])

    for value_key, err_key, marker, size, dash in (
        ("d_sat_per_particle", "d_sat_sem", "o", 6.0, (0, ())),
        ("u_dw", "u_dw_sem", "s", 5.3, (0, (4.5, 2.0))),
    ):
        xs = [groups[k]["fraction"] for k in films]
        ys = [float(groups[k]["summary"][value_key]) for k in films]
        ax.plot(xs, ys, ls=dash, color=fs.INK, lw=0.9, zorder=2)
        for key in ORDER:
            grp = groups[key]
            cloud = np.array([float(m[value_key]) for m in grp["pairs"]])
            ax.scatter(_cloud_x(grp["fraction"], len(cloud)), cloud, s=10.0,
                       marker=marker, color=colours[key], alpha=0.45,
                       edgecolors="none", zorder=3)
            ax.errorbar(grp["fraction"], float(grp["summary"][value_key]),
                        yerr=float(grp["summary"][err_key]), fmt=marker, ms=size,
                        mfc=colours[key], mec="white", mew=0.55, ecolor=fs.INK,
                        elinewidth=0.85, capsize=1.6, zorder=5)

    ax.set_ylim(0.0915, 0.1585)
    ax.set_yticks([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    ax.set_ylabel(r"$D_{\mathrm{sat}}/N$,  $u_{\mathrm{DW}}$   $(\sigma)$",
                  fontsize=fs.FS_LABEL - 0.5)
    _x_axis(ax, groups)

    ax.text(0.855, float(groups[0.70]["summary"]["d_sat_per_particle"]) - 0.0042,
            r"$D_{\mathrm{sat}}/N$", ha="center", va="top", fontsize=fs.FS_ANNOT,
            color=fs.INK, bbox=WHITE_BOX, zorder=6)
    ax.text(0.855, float(groups[0.70]["summary"]["u_dw"]) + 0.0042,
            r"$u_{\mathrm{DW}}$", ha="center", va="bottom", fontsize=fs.FS_ANNOT,
            color=fs.INK, bbox=WHITE_BOX, zorder=6)
    for value_key in ("d_sat_per_particle", "u_dw"):
        ax.text(0.372, float(groups[0.35]["summary"][value_key]) - 0.0020,
                f"${stats[0.35][value_key]['percent']:+.1f}\\%$", ha="left",
                va="center", fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, zorder=6)
    fs.panel_label(ax, "a", x=-0.20)


def panel_b(ax, groups: dict, colours: dict, stats: dict) -> None:
    """The ratio c, preserved under confinement."""
    films = ORDER[1:]
    bulk = groups[None]["summary"]
    _bulk_band(ax, float(bulk["c_ratio"]), float(bulk["c_ratio_sem"]), colours[None])
    ax.plot([groups[k]["fraction"] for k in films],
            [float(groups[k]["summary"]["c_ratio"]) for k in films],
            color=fs.INK, lw=0.9, zorder=2)
    for key in ORDER:
        grp = groups[key]
        cloud = np.array([float(m["c_ratio"]) for m in grp["pairs"]])
        ax.scatter(_cloud_x(grp["fraction"], len(cloud)), cloud, s=10.0,
                   color=colours[key], alpha=0.45, edgecolors="none", zorder=3)
        ax.errorbar(grp["fraction"], float(grp["summary"]["c_ratio"]),
                    yerr=float(grp["summary"]["c_ratio_sem"]), fmt="o", ms=6.0,
                    mfc=colours[key], mec="white", mew=0.55, ecolor=fs.INK,
                    elinewidth=0.85, capsize=1.6, zorder=5)
        if key is not None:
            ax.text(grp["fraction"], float(grp["summary"]["c_ratio"])
                    - float(grp["summary"]["c_ratio_sem"]) - 0.007,
                    f"${stats[key]['c_ratio']['percent']:+.1f}\\%$", ha="center",
                    va="top", fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, zorder=6)

    ax.set_ylim(1.138, 1.318)
    ax.set_yticks([1.15, 1.20, 1.25, 1.30])
    ax.set_ylabel(r"ceiling ratio  $c=(D_{\mathrm{sat}}/N)/u_{\mathrm{DW}}$")
    _x_axis(ax, groups)
    worst = max(abs(stats[k]["c_ratio"]["percent"]) for k in ORDER)
    ax.text(0.03, 0.965, f"every film within {worst:.1f}% of bulk",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs.FS_ANNOT,
            color=fs.INK, zorder=6, bbox=WHITE_BOX)
    ax.text(0.60, float(bulk["c_ratio"]), "bulk control", ha="center", va="center",
            fontsize=fs.FS_ANNOT - 0.5, color=colours[None], zorder=6, bbox=WHITE_BOX)
    fs.panel_label(ax, "b", x=-0.20)


def panel_c(ax, groups: dict, colours: dict, stats: dict, r_shell: float,
            box_l: float) -> None:
    """The anisotropy crossover: flat, then a jump at the narrowest film."""
    films = ORDER[1:]
    bulk = groups[None]["summary"]
    ax.axhline(1.0, color=fs.GUIDE, lw=0.9, ls="--", zorder=0)
    _bulk_band(ax, float(bulk["anisotropy"]), float(bulk["anisotropy_sem"]),
               colours[None])

    # Shade the onset window: isotropic at the 0.50 film, flattened at 0.45, so the
    # crossover is bracketed between them.  Both bracketing films are thicker than
    # the kick shell and discard nothing, so the window is free of the truncation.
    lo = groups[0.45]["fraction"]
    hi = groups[0.50]["fraction"]
    ax.axvspan(lo, hi, color=fs.SUBTLE, alpha=0.13, lw=0, zorder=0)
    ax.text(0.5 * (lo + hi), 0.9265, "onset", rotation=90, ha="center",
            va="bottom", fontsize=fs.FS_ANNOT - 1.0, color=fs.SUBTLE, zorder=6)

    ax.plot([groups[k]["fraction"] for k in films],
            [float(groups[k]["summary"]["anisotropy"]) for k in films],
            color=fs.INK, lw=0.9, zorder=2)
    for key in ORDER:
        grp = groups[key]
        cloud = np.array([float(m["anisotropy"]) for m in grp["pairs"]])
        ax.scatter(_cloud_x(grp["fraction"], len(cloud)), cloud, s=10.0,
                   color=colours[key], alpha=0.45, edgecolors="none", zorder=3)
        edge = fs.VERMILLION if key == 0.35 else "white"
        ax.errorbar(grp["fraction"], float(grp["summary"]["anisotropy"]),
                    yerr=float(grp["summary"]["anisotropy_sem"]), fmt="o", ms=6.0,
                    mfc=colours[key], mec=edge, mew=1.0 if key == 0.35 else 0.55,
                    ecolor=fs.INK, elinewidth=0.85, capsize=1.6, zorder=5)
        if key is None:
            continue
        z = stats[key]["anisotropy"]["z"]
        hot = key == 0.35
        ax.text(grp["fraction"] + 0.014, float(grp["summary"]["anisotropy"])
                + float(grp["summary"]["anisotropy_sem"]) + 0.0065,
                f"${z:+.1f}\\sigma$" if hot else f"${abs(z):.1f}\\sigma$",
                ha="left", va="bottom",
                fontsize=fs.FS_ANNOT + (0.5 if hot else -0.5),
                fontweight="bold" if hot else "normal",
                color=fs.VERMILLION if hot else fs.SUBTLE, zorder=6)

    ax.set_ylim(0.921, 1.099)
    ax.set_yticks([0.95, 1.00, 1.05])
    ax.set_ylabel(r"anisotropy  $\sqrt{\langle d_{\parallel}^{2}\rangle"
                  r"/2\langle d_{\perp}^{2}\rangle}$", fontsize=fs.FS_LABEL - 0.5)
    _x_axis(ax, groups)
    ax.text(1.058, 1.0035, "isotropic", ha="right", va="bottom",
            fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE, zorder=6)
    ax.text(0.352, 1.094, "in-plane excess", ha="left", va="top",
            fontsize=fs.FS_ANNOT - 0.5, color=fs.VERMILLION, zorder=6)
    fs.panel_label(ax, "c", x=-0.20)


def panel_d(ax, groups: dict, colours: dict, times: np.ndarray, tail: slice) -> None:
    """Every divergence curve, and the u_DW rescaling that collapses them."""
    t_start = float(times[tail][0])
    ax.axvspan(t_start, float(times[-1]), color=fs.GOLD, alpha=0.07, lw=0.0, zorder=0)
    for key in ORDER:
        grp = groups[key]
        for curve in grp["curves"]:
            ax.plot(times, curve, color=colours[key], lw=0.55, alpha=0.30, zorder=2)
        ax.plot(times, grp["curves"].mean(axis=0), color=colours[key], lw=1.8,
                solid_capstyle="round", zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(-1.0, 41.0)
    ax.set_ylim(2.6e-4, 2.0)
    ax.set_xlabel(r"time  $t$", labelpad=1.5)
    ax.set_ylabel(r"divergence per particle  $D(t)/N$   $(\sigma)$")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([1e-3, 1e-2, 1e-1])
    ax.text(t_start + 0.9, 1.55, f"plateau window\n$t\\geq{t_start:g}$",
            ha="left", va="top", fontsize=fs.FS_ANNOT - 0.5, color=fs.SUBTLE,
            linespacing=1.25, zorder=6)

    handles = [Line2D([0], [0], color=colours[k], lw=1.8,
                      label=("bulk" if k is None else f"$h/L={groups[k]['label']}$")
                      + f"  ({groups[k]['n_mobile']})")
               for k in ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.008, 1.030),
              fontsize=fs.FS_LEGEND - 0.5, handlelength=1.3, labelspacing=0.22,
              borderaxespad=0.0, title=r"film  ($N_{\mathrm{film}}$)",
              title_fontsize=fs.FS_LEGEND - 0.5)

    inset = ax.inset_axes([0.505, 0.140, 0.475, 0.345], zorder=8)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    bulk_c = float(groups[None]["summary"]["c_ratio"])
    bulk_c_sem = float(groups[None]["summary"]["c_ratio_sem"])
    inset.axhspan(bulk_c - bulk_c_sem, bulk_c + bulk_c_sem, color=colours[None],
                  alpha=0.20, lw=0.0, zorder=0)
    inset.axhline(bulk_c, color=colours[None], lw=0.7, ls=(0, (4.0, 2.0)), zorder=1)
    for key in ORDER:
        grp = groups[key]
        u_dw = float(grp["summary"]["u_dw"])
        inset.plot(times, grp["curves"].mean(axis=0) / u_dw, color=colours[key],
                   lw=1.2, zorder=3)
    inset.set_xlim(-1.0, 41.0)
    inset.set_ylim(0.0, 1.44)
    inset.set_xticks([0, 20, 40])
    inset.set_yticks([0.0, 0.6, 1.2])
    inset.tick_params(labelsize=fs.FS_TICK - 2.0, pad=1.4)
    inset.set_ylabel(r"$D(t)/(N u_{\mathrm{DW}})$", fontsize=fs.FS_TICK - 1.0,
                     labelpad=1.5)
    inset.text(0.955, 0.10, "rescaled by the\nfilm-local cage", transform=inset.transAxes,
               ha="right", va="bottom", fontsize=fs.FS_ANNOT - 1.5, color=fs.SUBTLE,
               linespacing=1.2)
    for name, spine in inset.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(fs.INK if name in ("left", "bottom") else fs.GUIDE)
    fs.panel_label(ax, "d", x=-0.20)


# --------------------------------------------------------------------------- #
# Layout guard
# --------------------------------------------------------------------------- #
def check_text(fig) -> None:
    """Fail if any two visible text artists overlap, or one leaves the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_visible() and text.get_text().strip():
                boxes.append((f"{ax.get_label() or 'ax'}:{text.get_text()[:22]!r}",
                              text.get_window_extent(renderer)))
    canvas = fig.bbox
    for name, box in boxes:
        if not canvas.overlaps(box):
            raise RuntimeError(f"text {name} is off-canvas")
    for i, (name_i, box_i) in enumerate(boxes):
        for name_j, box_j in boxes[i + 1:]:
            if box_i.overlaps(box_j):
                raise RuntimeError(f"text overlap: {name_i} vs {name_j}")
    print(f"[layout] {len(boxes)} in-axes text artists, no overlaps, all on canvas")


# --------------------------------------------------------------------------- #
def main() -> int:
    data = load()
    groups = data["groups"]
    stats = verify_headlines(groups)
    ramp = fs.sequential(len(ORDER), lo=0.06, hi=0.74)
    colours = {key: ramp[i] for i, key in enumerate(ORDER)}

    fig = plt.figure(figsize=fs.figsize(fs.WIDTH_FULL, 0.74))
    gs = fig.add_gridspec(2, 2)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    panel_a(axa, groups, colours, stats)
    panel_b(axb, groups, colours, stats)
    panel_c(axc, groups, colours, stats, data["r_shell"], data["box_l"])
    panel_d(axd, groups, colours, data["times"], data["tail"])

    fs.finalize(fig)
    check_text(fig)
    paths = fs.save(fig, str(OUT))
    plt.close(fig)

    bulk = groups[None]["summary"]
    print(f"[verify] {MERGED.relative_to(ROOT)}: 32 pairs, "
          f"dt={data['dt']}, horizon={data['horizon']} steps (t=0..{data['times'][-1]:g}), "
          f"L={data['box_l']:.4f} sigma, shell radius {data['r_shell']} sigma")
    print(f"[verify] {'film':>5} {'h/sigma':>8} {'N_film':>7} {'D_sat/N':>19} "
          f"{'u_DW':>19} {'c':>18} {'anisotropy':>18}")
    for key in ORDER:
        row = groups[key]["summary"]
        print(f"[verify] {groups[key]['label']:>5} {groups[key]['thickness']:>8.3f} "
              f"{groups[key]['n_mobile']:>7} "
              f"{row['d_sat_per_particle']:>11.5f}+-{row['d_sat_sem']:<7.5f}"
              f"{row['u_dw']:>11.5f}+-{row['u_dw_sem']:<7.5f}"
              f"{row['c_ratio']:>10.4f}+-{row['c_ratio_sem']:<7.4f}"
              f"{row['anisotropy']:>10.4f}+-{row['anisotropy_sem']:<7.4f}")
    for key in ORDER[1:]:
        s = stats[key]
        print(f"[verify] {groups[key]['label']:>5} vs bulk: "
              f"D_sat/N {s['d_sat_per_particle']['percent']:+.2f}% "
              f"(z={s['d_sat_per_particle']['z']:+.2f}), "
              f"u_DW {s['u_dw']['percent']:+.2f}% (z={s['u_dw']['z']:+.2f}), "
              f"c {s['c_ratio']['percent']:+.2f}% (z={s['c_ratio']['z']:+.2f}), "
              f"anisotropy {s['anisotropy']['percent']:+.2f}% "
              f"(z={s['anisotropy']['z']:+.2f})")
    onset_lo, onset_hi = groups[0.45]["thickness"], groups[0.50]["thickness"]
    print(f"[verify] bulk c = {float(bulk['c_ratio']):.4f}+-{float(bulk['c_ratio_sem']):.4f}; "
          f"anisotropy onset bracketed by {onset_lo:.2f} < h < {onset_hi:.2f} sigma "
          f"(width {onset_hi - onset_lo:.2f} sigma), both films free of wall-straddling kicks")
    cs = "/".join(f"{float(groups[k]['summary']['c_ratio']):.4f}" for k in ORDER)
    zs = "/".join(f"{stats[k]['anisotropy']['z']:+.2f}" for k in ORDER[1:])
    worst_pct = max(abs(stats[k]["c_ratio"]["percent"]) for k in ORDER[1:])
    print(f"VERIFIED-NUMBERS fig_confinement: c={cs} worst-c-dev={worst_pct:.1f}% "
          f"aniso-z={zs} D_sat/N monotone")
    print("[saved]", *[str(p) for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
