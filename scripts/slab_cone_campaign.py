#!/usr/bin/env python3
"""Butterfly cone inside a confined film: the confinement leg's campaign.

Deterministic twin pairs run inside a pinned-wall slab (see
``butterfly_cone.rcce.slab``), so this measures whether the cone and its ceiling survive
spatial restriction.  The three questions it answers, all from the same runs:

* **C1 anisotropy** -- resolve the divergence into film-normal and in-plane
  components.  A cone that feels the boundary should stop being isotropic.
* **C2 ceiling locality** -- does ``D_sat/N = c * u_DW`` still hold with both
  sides measured inside the film?
* **C3 front crossover** -- how the saturated divergence scales as the film
  narrows toward the bulk cone width.

Every geometry is run alongside a **bulk control on the same code path**, which
is what makes confined-versus-bulk comparison meaningful: the cage estimator
here (single-branch MSD against the parent) is not the flagship's cross-branch
variance estimator, so the leg carries its own baseline rather than borrowing
the banked number.

Confinement itself needs no engine change: the integrator already gates motion
on ``active_mask``, so freezing the wall region is the whole protocol, and the
mobile film is in equilibrium by construction because the wall labels come from
an already equilibrated parent (random pinning; Cammarota and Biroli, PNAS 109,
8850 (2012)).

Run:
    ./.venv/bin/python scripts/slab_cone_campaign.py --quick
    ./.venv/bin/python scripts/slab_cone_campaign.py --pairs 8 --horizon 8000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dw_identity import msd_relative_to_reference, u_dw_from_msd_plateau  # noqa: E402
from butterfly_cone.engine.integrate import MDIntegrator  # noqa: E402
from butterfly_cone.engine.potential import minimum_image  # noqa: E402
from butterfly_cone.perturb.operators import o_shell  # noqa: E402
from butterfly_cone.rcce.slab import (  # noqa: E402
    SlabSpec,
    anisotropy_ratio,
    resolve_divergence_components,
    select_slab,
)
from slab_stationarity_gate import load_parent  # noqa: E402

DEFAULT_PARENT = ROOT / "runs" / "gardner" / "bridge-Tladder--c4-unpert" / "parent_state.pt"


def _maxwell_velocities(system, temperature: float, seed: int) -> torch.Tensor:
    """Per-pair Maxwell momenta, zeroed on frozen particles and COM-corrected."""

    generator = torch.Generator().manual_seed(int(seed))
    velocities = torch.randn(
        system.positions.shape, generator=generator, dtype=torch.float64
    ) * math.sqrt(float(temperature))
    active = system.active_mask[:, None]
    velocities = torch.where(active, velocities, torch.zeros_like(velocities))
    n_active = int(system.active_mask.sum())
    if n_active > 0:
        mean = velocities[system.active_mask].mean(dim=0, keepdim=True)
        velocities = torch.where(active, velocities - mean, torch.zeros_like(velocities))
    return velocities


def run_pair(
    parent_path: Path,
    *,
    fraction: float | None,
    delta: float,
    seed: int,
    temperature: float,
    dt: float,
    horizon: int,
    stride: int,
    interface: float,
    r_pert: float,
    kick_clip_fraction: float | None = None,
) -> dict[str, object]:
    """One deterministic twin pair, confined to a film of the given fraction.

    ``kick_clip_fraction`` decouples WHICH displacements survive from WHICH
    particles are frozen.  Left at ``None`` the two coincide, which is the
    production behaviour.  Set to a fraction it clips the kick to that film's
    slab while freezing whatever ``fraction`` says, so a bulk run can be given
    the identical z-truncated kick that a thin film forces.  That is the matched
    control for the anisotropy: the narrowest film is the only geometry where the
    shell straddles the wall, so bulk-vs-film otherwise varies the perturbation
    and the boundary at the same time.
    """

    control, _ = load_parent(parent_path)
    box_z = float(control.box[2])
    box_np = control.box.detach().cpu().numpy().copy()

    if fraction is None:
        mobile_mask = torch.ones(control.positions.shape[0], dtype=torch.bool)
        thickness = box_z
    else:
        thickness = fraction * box_z
        spec = SlabSpec(axis=2, center=0.5 * box_z, thickness=thickness, interface=interface)
        selection = select_slab(control, spec)
        mobile_mask = selection.mobile_mask.clone()

    control.active_mask = mobile_mask.clone()
    control.velocities = _maxwell_velocities(control, temperature, seed)
    reference = control.positions.detach().cpu().numpy().copy()

    # Kick at the film midplane so the perturbation is interior to the mobile
    # region rather than sitting on a frozen wall.
    center = control.positions.mean(dim=0).clone()
    center[2] = 0.5 * box_z
    kicked, provenance = o_shell(control.clone(), center, delta, r_pert, seed=seed)
    kicked.active_mask = mobile_mask.clone()
    kicked.velocities = control.velocities.clone()

    # RESTRICT THE KICK TO THE MOBILE FILM.  o_shell picks its shell by distance
    # in the full periodic box and knows nothing about the slab, so once the film
    # is thinner than the shell diameter ($2 r_{pert}$) it also displaces frozen
    # wall particles.  Those are then held fixed at displaced positions, which
    # gives the two arms different confining boundaries rather than different
    # initial conditions inside the film: a persistent forcing instead of a
    # decaying perturbation, and it contaminates precisely the narrowest film.
    # Zero the displacement on wall particles and re-centre what remains, so the
    # perturbation still conserves the centre of mass of the particles it moves.
    if kick_clip_fraction is None:
        kick_mask = mobile_mask
    else:
        clip_spec = SlabSpec(
            axis=2,
            center=0.5 * box_z,
            thickness=kick_clip_fraction * box_z,
            interface=interface,
        )
        kick_mask = select_slab(control, clip_spec).mobile_mask.clone()

    displacement = minimum_image(kicked.positions - control.positions, control.box)
    n_kick_in_wall = int(((displacement.abs().sum(dim=1) > 0.0) & ~kick_mask).sum())
    displacement = torch.where(kick_mask[:, None], displacement, torch.zeros_like(displacement))
    moved = displacement.abs().sum(dim=1) > 0.0
    if bool(moved.any()):
        displacement[moved] -= displacement[moved].mean(dim=0, keepdim=True)
    kicked.positions = torch.remainder(control.positions + displacement, control.box)
    kicked.unwrapped_positions = control.unwrapped_positions + displacement
    n_perturbed_mobile = int(moved.sum())

    control_md = MDIntegrator(control, dt=dt)
    kicked_md = MDIntegrator(kicked, dt=dt)

    control_frames = [control.positions.detach().cpu().numpy().copy()]
    kicked_frames = [kicked.positions.detach().cpu().numpy().copy()]
    remaining = horizon
    while remaining > 0:
        block = min(stride, remaining)
        control_md.step(block)
        kicked_md.step(block)
        control_frames.append(control.positions.detach().cpu().numpy().copy())
        kicked_frames.append(kicked.positions.detach().cpu().numpy().copy())
        remaining -= block

    control_arr = np.stack(control_frames, axis=0)
    kicked_arr = np.stack(kicked_frames, axis=0)

    # minimum-image displacement field, same semantics as perturb.response
    difference = kicked_arr - control_arr
    difference -= box_np * np.rint(difference / box_np)
    per_particle = np.linalg.norm(difference, axis=-1)

    mobile_np = mobile_mask.detach().cpu().numpy()
    n_mobile = int(mobile_np.sum())

    # D(t) summed over the MOBILE film only: frozen particles cannot diverge and
    # would otherwise dilute the per-particle normalisation.
    divergence = per_particle[:, mobile_np].sum(axis=1)

    tail = slice(max(1, len(divergence) // 2), None)

    # Anisotropy from SECOND moments over the mobile film on the plateau, which
    # is calibrated to 1 for an isotropic field.  A mean-of-norms ratio carries
    # a spurious (pi/2)/sqrt(2) = 1.1107 offset; see anisotropy_ratio.
    tail_field = torch.from_numpy(difference[tail][:, mobile_np, :])
    anisotropy = anisotropy_ratio(tail_field, axis=2)
    normal, in_plane = resolve_divergence_components(tail_field, axis=2)
    d_sat_per_particle = float(divergence[tail].mean() / n_mobile)

    msd = msd_relative_to_reference(control_arr[:, None, :, :][:, :, mobile_np, :],
                                    reference[mobile_np], box_np)
    u_dw = u_dw_from_msd_plateau(float(msd[tail].mean()))

    return {
        "geometry": "bulk_control" if fraction is None else "slab",
        "thickness_fraction_of_box": fraction,
        "kick_clip_fraction": kick_clip_fraction,
        "thickness_sigma": thickness,
        "n_mobile": n_mobile,
        "n_wall": int(len(mobile_np) - n_mobile),
        "delta": float(delta),
        "seed": int(seed),
        "n_perturbed": provenance.n_perturbed,
        "n_perturbed_mobile": n_perturbed_mobile,
        "n_kick_in_wall_discarded": n_kick_in_wall,
        "delta_u": provenance.delta_u,
        "d_sat_per_particle": d_sat_per_particle,
        "u_dw": u_dw,
        "c_ratio": (d_sat_per_particle / u_dw) if u_dw > 0.0 else float("nan"),
        "normal_rms": float(normal.square().mean().sqrt()),
        "in_plane_rms": float(in_plane.square().mean().sqrt()),
        "anisotropy": anisotropy,
        "divergence_curve": [float(v) for v in divergence],
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-pair rows into one row per geometry with a spread."""

    keys = sorted({row["thickness_fraction_of_box"] for row in rows}, key=lambda v: (v is not None, v))
    out = []
    for key in keys:
        group = [row for row in rows if row["thickness_fraction_of_box"] == key]

        def stat(name: str) -> tuple[float, float]:
            values = np.array([float(row[name]) for row in group])
            spread = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            return float(values.mean()), spread

        d_sat, d_sat_err = stat("d_sat_per_particle")
        u_dw, u_dw_err = stat("u_dw")
        c_ratio, c_err = stat("c_ratio")
        anisotropy, anisotropy_err = stat("anisotropy")
        out.append(
            {
                "thickness_fraction_of_box": key,
                "geometry": group[0]["geometry"],
                "n_mobile": group[0]["n_mobile"],
                "n_pairs": len(group),
                "d_sat_per_particle": d_sat,
                "d_sat_sem": d_sat_err,
                "u_dw": u_dw,
                "u_dw_sem": u_dw_err,
                "c_ratio": c_ratio,
                "c_ratio_sem": c_err,
                "anisotropy": anisotropy,
                "anisotropy_sem": anisotropy_err,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--fractions", type=float, nargs="*", default=[0.35, 0.50, 0.70])
    parser.add_argument("--no-bulk", action="store_true", help="skip the bulk control (for sharding)")
    parser.add_argument("--pairs", type=int, default=4, help="pairs per delta per geometry")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.03])
    parser.add_argument("--temperature", type=float, default=0.075)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--interface", type=float, default=1.0)
    parser.add_argument("--r-pert", type=float, default=2.5)
    parser.add_argument(
        "--kick-clip-fraction",
        type=float,
        default=None,
        help="clip the kick to this film while freezing what --fractions says; "
             "the matched control that varies the boundary without varying the kick",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "runs" / "slab_cone" / "slab_cone_campaign.json"))
    args = parser.parse_args()

    horizon = 600 if args.quick else args.horizon
    pairs = 1 if args.quick else args.pairs
    deltas = args.deltas[:1] if args.quick else args.deltas
    stride = min(args.stride, horizon)
    parent_path = Path(args.parent)

    geometries: list[float | None] = list(args.fractions) if args.no_bulk else [None, *args.fractions]
    rows: list[dict[str, object]] = []
    t0 = time.time()
    index = 0
    for fraction in geometries:
        for delta in deltas:
            for replicate in range(pairs):
                index += 1
                rows.append(
                    run_pair(
                        parent_path,
                        fraction=fraction,
                        delta=delta,
                        seed=args.seed + 1009 * index,
                        temperature=args.temperature,
                        dt=args.dt,
                        horizon=horizon,
                        stride=stride,
                        interface=args.interface,
                        r_pert=args.r_pert,
                        kick_clip_fraction=args.kick_clip_fraction,
                    )
                )
    wall = time.time() - t0

    table = summarise(rows)
    bulk = next((row for row in table if row["geometry"] == "bulk_control"), None)

    print(f"parent {parent_path.name}  horizon {horizon} steps (t={horizon * args.dt:g})  "
          f"{len(rows)} pairs  {wall:.0f}s")
    print(f"{'film':>10} {'n_mob':>6} {'pairs':>6} {'D_sat/N':>18} {'u_DW':>16} "
          f"{'c = D_sat/N/u_DW':>20} {'aniso':>7}")
    for row in table:
        label = "bulk" if row["geometry"] == "bulk_control" else f"{row['thickness_fraction_of_box']:.2f}"
        print(
            f"{label:>10} {row['n_mobile']:>6} {row['n_pairs']:>6} "
            f"{row['d_sat_per_particle']:>11.5f}+-{row['d_sat_sem']:<6.5f} "
            f"{row['u_dw']:>9.5f}+-{row['u_dw_sem']:<6.5f} "
            f"{row['c_ratio']:>13.4f}+-{row['c_ratio_sem']:<6.4f} {row['anisotropy']:>7.4f}"
        )
    print()
    print("anisotropy is sqrt(in-plane^2 / 2 normal^2) on the plateau; 1.0000 is isotropic.")
    if bulk is not None:
        print(f"bulk control on this code path: c = {bulk['c_ratio']:.4f}. Compare confined rows to")
        print("this, not to the banked flagship value, which uses a different cage estimator.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "parent": str(parent_path),
                "controls": {
                    "dt": args.dt,
                    "horizon_steps": horizon,
                    "horizon_time": horizon * args.dt,
                    "temperature": args.temperature,
                    "deltas": deltas,
                    "pairs_per_delta": pairs,
                    "interface": args.interface,
                    "r_pert": args.r_pert,
                    "kick_clip_fraction": args.kick_clip_fraction,
                    "seed": args.seed,
                },
                "summary": table,
                "pairs": rows,
                "wall_seconds": wall,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
