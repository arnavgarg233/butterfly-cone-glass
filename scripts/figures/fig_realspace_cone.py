#!/usr/bin/env python3
"""Real-space butterfly cone from a persisted matched twin pair.

The figure is a direct re-analysis of one matched twin pair in the ``L=20
sigma`` block (``N=8000``, ``T=0.108``) -- the same block class that tiles into
the million-particle box.  A localized ``O_shell`` kick is applied to twin B
only; twin A is the bit-identical control.  The panels show the fixed-``z`` slab
at early times, chosen so the divergence front (front speed ``v_b~2.87 sigma``
per time unit) is still interior to the box and the butterfly cone reads as an
expanding bright core inside dark, untouched glass rather than a saturated box.

The per-particle snapshot data are produced by ``realspace_cone_data.py`` (which
reuses the exact deterministic twin construction of ``giant_cone_run.py``) and
stored at ``runs/realspace_viz/cone_particles.npz``; this script only reads and
renders them, with no interpolation or synthetic particle field.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import figstyle as fs  # noqa: E402


DATA_FILE = Path("runs/realspace_viz/cone_particles.npz")
SLAB_HALF_WIDTH = 2.0


def load_pair(root: Path = ROOT) -> dict[str, Any]:
    """Load and validate the persisted twin snapshot used by the figure."""

    path = root / DATA_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/figures/realspace_cone_data.py first"
        )
    with np.load(path, allow_pickle=False) as data:
        positions_kicked = np.asarray(data["positions_kicked"], dtype=np.float32)
        positions_control = np.asarray(data["positions_control"], dtype=np.float32)
        divergences_arr = np.asarray(data["divergences"], dtype=float)
        diameters = np.asarray(data["diameters"], dtype=float)
        box = np.asarray(data["box"], dtype=float)
        center = np.asarray(data["center"], dtype=float)
        times = np.asarray(data["times"], dtype=float)
        front_speed = float(np.asarray(data["v_b"]))

    if positions_kicked.shape != positions_control.shape:
        raise ValueError("kicked/control snapshots do not share a shape")
    if positions_kicked.ndim != 3 or positions_kicked.shape[-1] != 3:
        raise ValueError("snapshot positions must have shape (frames, particles, 3)")
    n_frames, n_particles, _ = positions_kicked.shape
    if divergences_arr.shape != (n_frames, n_particles):
        raise ValueError("divergence array does not match snapshot shape")
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError("persisted box is invalid")
    if diameters.shape != (n_particles,):
        raise ValueError("persisted diameters do not match particle count")
    if center.shape != (3,) or np.any(center < 0.0) or np.any(center >= box):
        raise ValueError("persisted kick center is outside the box")
    if times.shape != (n_frames,):
        raise ValueError("persisted times do not match snapshot count")

    divergences = [divergences_arr[i] for i in range(n_frames)]
    return {
        "positions_kicked": positions_kicked,
        "positions_control": positions_control,
        "diameters": diameters,
        "box": box,
        "center": center,
        "times": times,
        "snapshot_indices": list(range(n_frames)),
        "divergences": divergences,
        "front_speed": front_speed,
    }


def minimum_image_delta(kicked, control, box):
    """Minimum-image displacement between a kicked and a control configuration.

    Specified by ``docs/superpowers/plans/2026-07-19-realspace-cone-table1.md``
    step 3 and exercised by ``tests/test_realspace_cone.py``.  The primitives
    were never landed alongside the figure, so that test could not import and
    its collection error blocked the whole suite.
    """

    delta = np.asarray(kicked, float) - np.asarray(control, float)
    box_array = np.asarray(box, float)
    return delta - box_array * np.round(delta / box_array)


def per_particle_divergence(kicked, control, box):
    """Per-particle minimum-image divergence norm ``|Delta r_i|``."""

    return np.linalg.norm(minimum_image_delta(kicked, control, box), axis=-1)


def _periodic_front(ax, center: np.ndarray, radius: float, box: np.ndarray) -> None:
    """Draw the front circle and its periodic images, clipped to the slab box."""

    angles = np.linspace(0.0, 2.0 * np.pi, 720)
    circle = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    n_images = int(np.ceil(radius / min(box[:2]))) + 1
    for ix in range(-n_images, n_images + 1):
        for iy in range(-n_images, n_images + 1):
            shifted = circle + center[:2] + np.array([ix * box[0], iy * box[1]])
            ax.plot(
                shifted[:, 0],
                shifted[:, 1],
                color=fs.VERMILLION,
                lw=1.3,
                ls=(0, (4.0, 2.0)),
                alpha=0.95,
                zorder=4,
                clip_on=True,
                solid_capstyle="round",
            )


def _slab_mask(positions: np.ndarray, center: np.ndarray, box: np.ndarray) -> np.ndarray:
    z_delta = positions[:, 2] - center[2]
    z_delta -= box[2] * np.round(z_delta / box[2])
    return np.abs(z_delta) < SLAB_HALF_WIDTH


def make_figure(data: dict[str, Any]):
    """Build the four-panel real-space strip without writing files."""

    box = data["box"]
    center = data["center"]
    diameters = data["diameters"]
    positions = data["positions_kicked"]
    divergences = data["divergences"]
    times = data["times"]
    log_values = [np.log10(np.maximum(divergence, 1.0e-8)) for divergence in divergences]
    all_log = np.concatenate(log_values)
    norm = Normalize(vmin=float(np.floor(all_log.min())), vmax=float(np.ceil(all_log.max())))
    cmap = plt.get_cmap("cividis").copy()

    fig, axes = plt.subplots(
        1,
        len(times),
        figsize=fs.figsize(fs.WIDTH_FULL, 0.31),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    mean_diameter = float(np.mean(diameters))
    xlim = (0.0, float(box[0]))
    ylim = (0.0, float(box[1]))
    for panel, (ax, time, position, divergence, log_value) in enumerate(
        zip(axes, times, [positions[i] for i in data["snapshot_indices"]], divergences, log_values)
    ):
        mask = _slab_mask(position, center, box)
        sizes = 20.0 * np.square(diameters[mask] / mean_diameter)
        ax.scatter(
            position[mask, 0],
            position[mask, 1],
            s=sizes,
            c=log_value[mask],
            cmap=cmap,
            norm=norm,
            linewidths=0.0,
            alpha=0.94,
            rasterized=True,
            zorder=3,
        )
        _periodic_front(ax, center, data["front_speed"] * time, box)
        ax.plot(
            center[0],
            center[1],
            marker="*",
            ms=8.5,
            color=fs.VERMILLION,
            mec="white",
            mew=0.45,
            zorder=5,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"$t={time:g}$", loc="left", pad=2.0)
        fs.panel_label(ax, chr(ord("a") + panel))
        ax.set_xticks([0.0, round(float(box[0]) / 2.0, 1), round(float(box[0]), 1)])
        ax.set_yticks([0.0, round(float(box[1]) / 2.0, 1), round(float(box[1]), 1)])
        ax.tick_params(labelsize=fs.FS_TICK - 0.5)
        if panel:
            ax.set_yticklabels([])
        if panel == 0:
            ax.set_ylabel(r"$y/\sigma$")
        ax.set_xlabel(r"$x/\sigma$")

    axes[0].text(
        0.035,
        0.965,
        r"$|z-z_{\rm kick}|<2\sigma$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=fs.FS_ANNOT,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, edgecolor="none"),
    )
    front_label = rf"$r=v_b t$, $v_b={data['front_speed']:.2f}\,\sigma$"
    axes[-1].text(
        0.965,
        0.035,
        front_label,
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        color=fs.VERMILLION,
        fontsize=fs.FS_ANNOT,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, edgecolor="none"),
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, fraction=0.035)
    colorbar.set_label(r"$\log_{10}|\Delta r_i|/\sigma$", labelpad=4.0)
    colorbar.ax.tick_params(labelsize=fs.FS_TICK - 0.5)
    fs.finalize(fig)
    return fig


def main() -> int:
    data = load_pair()
    print(
        f"[verify] {DATA_FILE}: N={data['positions_kicked'].shape[1]}, "
        f"frames={len(data['times'])}, box={np.array2string(data['box'], precision=3)}, "
        f"kick center={np.array2string(data['center'], precision=5)}, "
        f"v_b={data['front_speed']:.4f}"
    )
    for idx, (time, divergence) in enumerate(zip(data["times"], data["divergences"])):
        positive = divergence[divergence > 0.0]
        slab = _slab_mask(data["positions_kicked"][idx], data["center"], data["box"])
        print(
            f"[verify] t={time:g}: divergence range "
            f"[{divergence.min():.6g}, {divergence.max():.6g}] sigma; "
            f"positive range [{positive.min():.6g}, {positive.max():.6g}] sigma; "
            f"slab particles={int(slab.sum())}"
        )
    out = ROOT / "results/figures/fig_realspace_cone"
    fig = make_figure(data)
    fs.save(fig, str(out), formats=("pdf", "png"))
    plt.close(fig)
    print(f"[write] {out}.pdf")
    print(f"[write] {out}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
