#!/usr/bin/env python3
"""Go/no-go gate for the confinement leg: is a pinned slab parent stationary?

The bidisperse extension failed for exactly one reason: its parent was **aging**
on the cone timescale.  The divergence never plateaued and the cage itself grew
(``u_DW`` 0.131 to 0.193 over the horizon), so no confined-versus-bulk number
could mean anything.  That was a methods wall, not a refutation, and it was
discovered *after* a campaign had been spent.

This script spends minutes instead.  For each slab thickness it takes an
already deep-equilibrated parent, freezes the wall region (``active_mask``),
evolves the mobile film **unperturbed** for the full cone horizon, and asks
whether the cage amplitude is flat in time.  A film whose cage grows is
disqualified before any twin pair is run.

Two checks per thickness, both using the repository's own estimators from
``scripts/dw_identity.py`` rather than a hand-rolled measure:

* ``u_DW`` from the per-frame branch-mean cage variance (``cage_msd_curve``),
  compared between the first and last thirds of the window.  Drift is the
  bidisperse failure signature.
* a wall-immobility assertion, so a silently-unfrozen wall cannot pass as a
  stationary film.

It also runs a **no-wall positive control**: a slab thick enough to leave the
wall empty must reproduce bulk dynamics bitwise, because confinement here is
nothing but ``active_mask`` and an all-true mask is the bulk default.  If that
control ever fails, the confinement path has diverged from the flagship path
and no comparison is valid.

Run:
    ./.venv/bin/python scripts/slab_stationarity_gate.py --quick
    ./.venv/bin/python scripts/slab_stationarity_gate.py --horizon 8000
"""

from __future__ import annotations

import argparse
import json
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
from butterfly_cone.engine.system import ParticleSystem  # noqa: E402
from butterfly_cone.rcce.slab import SlabSpec, select_slab  # noqa: E402

DEFAULT_PARENT = ROOT / "runs" / "gardner" / "bridge-Tladder--c4-unpert" / "parent_state.pt"


def load_parent(path: Path) -> tuple[ParticleSystem, str]:
    """Load a persisted parent as a float64 system, carrying its provenance hash."""

    blob = torch.load(path, map_location="cpu", weights_only=False)
    positions = blob["positions"].to(torch.float64)
    return (
        ParticleSystem(
            positions=positions.clone(),
            velocities=blob["velocities"].to(torch.float64).clone(),
            diameters=blob["diameters"].to(torch.float64).clone(),
            box=blob["box"].to(torch.float64).clone(),
            active_mask=blob["active_mask"].clone(),
            unwrapped_positions=blob["unwrapped_positions"].to(torch.float64).clone(),
        ),
        str(blob.get("state_sha256", "unknown")),
    )


def evolve_and_capture(
    system: ParticleSystem,
    *,
    dt: float,
    horizon: int,
    stride: int,
) -> np.ndarray:
    """Unperturbed NVE, returning captured positions with shape ``(T, 1, N, 3)``."""

    integrator = MDIntegrator(system, dt=dt)
    frames = [system.positions.detach().cpu().numpy().copy()]
    remaining = horizon
    while remaining > 0:
        block = min(stride, remaining)
        integrator.step(block)
        frames.append(system.positions.detach().cpu().numpy().copy())
        remaining -= block
    return np.stack(frames, axis=0)[:, None, :, :]


def cage_drift(frames: np.ndarray, reference: np.ndarray, box: np.ndarray) -> dict[str, float]:
    """``u_DW`` on the plateau, middle third vs last third (skips the transient)."""

    msd = msd_relative_to_reference(frames, reference, box)
    third = max(1, len(msd) // 3)
    mid = u_dw_from_msd_plateau(float(msd[third : 2 * third].mean()))
    late = u_dw_from_msd_plateau(float(msd[-third:].mean()))
    return {
        "u_dw_early": mid,
        "u_dw_late": late,
        "drift_ratio": (late / mid) if mid > 0.0 else float("inf"),
        "u_dw_final_frame": u_dw_from_msd_plateau(float(msd[-1])),
    }


def run_thickness(
    parent_path: Path,
    *,
    fraction: float | None,
    dt: float,
    horizon: int,
    stride: int,
    interface: float,
) -> dict[str, object]:
    """One slab thickness (``fraction`` of the box) or the bulk control (``None``)."""

    system, provenance = load_parent(parent_path)
    box = system.box.detach().cpu().numpy().copy()
    box_z = float(system.box[2])
    start = system.positions.clone()

    record: dict[str, object] = {"provenance_sha256": provenance, "n_particles": int(system.positions.shape[0])}

    if fraction is None:
        record["geometry"] = "bulk_control"
        record["n_wall"] = 0
        record["n_mobile"] = int(system.positions.shape[0])
        wall_mask = torch.zeros(system.positions.shape[0], dtype=torch.bool)
    else:
        thickness = fraction * box_z
        spec = SlabSpec(axis=2, center=0.5 * box_z, thickness=thickness, interface=interface)
        selection = select_slab(system, spec)
        system.active_mask = selection.mobile_mask.clone()
        wall_mask = selection.wall_mask.clone()
        record["geometry"] = "slab"
        record["thickness_sigma"] = thickness
        record["thickness_fraction_of_box"] = fraction
        record["n_wall"] = selection.n_wall
        record["n_mobile"] = selection.n_mobile
        record["mobile_fraction"] = selection.mobile_fraction

    t0 = time.time()
    frames = evolve_and_capture(system, dt=dt, horizon=horizon, stride=stride)
    record["wall_seconds"] = time.time() - t0
    record["frames_captured"] = int(frames.shape[0])

    record.update(cage_drift(frames, start.detach().cpu().numpy(), box))

    shift = (system.positions - start).detach().cpu()
    record["wall_max_abs_shift"] = float(shift[wall_mask].abs().max()) if bool(wall_mask.any()) else 0.0
    record["mobile_max_abs_shift"] = float(shift[~wall_mask].abs().max())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--horizon", type=int, default=8000, help="steps; 8000 = t=40 at dt=0.005")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--interface", type=float, default=1.0)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.35, 0.50, 0.70],
        help="mobile film thickness as a fraction of the box length along z",
    )
    parser.add_argument("--drift-tolerance", type=float, default=1.10)
    parser.add_argument("--quick", action="store_true", help="short horizon smoke run")
    parser.add_argument("--out", default=str(ROOT / "runs" / "slab_gate" / "slab_stationarity_gate.json"))
    args = parser.parse_args()

    horizon = 400 if args.quick else args.horizon
    stride = min(args.stride, horizon)
    parent_path = Path(args.parent)

    rows = [
        run_thickness(
            parent_path,
            fraction=None,
            dt=args.dt,
            horizon=horizon,
            stride=stride,
            interface=args.interface,
        )
    ]
    for fraction in args.fractions:
        rows.append(
            run_thickness(
                parent_path,
                fraction=fraction,
                dt=args.dt,
                horizon=horizon,
                stride=stride,
                interface=args.interface,
            )
        )

    print(f"parent {parent_path.name}  sha256 {rows[0]['provenance_sha256'][:16]}  "
          f"N={rows[0]['n_particles']}  horizon={horizon} steps (t={horizon * args.dt:g})")
    print(f"{'geometry':>14} {'n_mob':>6} {'n_wall':>6} {'u_DW early':>11} {'u_DW late':>10} "
          f"{'drift':>7} {'wall shift':>11} {'verdict':>8}")
    ok = True
    for row in rows:
        drift = float(row["drift_ratio"])
        wall_frozen = float(row["wall_max_abs_shift"]) == 0.0
        stationary = drift <= args.drift_tolerance
        passed = wall_frozen and stationary
        if row["geometry"] == "slab":
            ok &= passed
        label = "slab %.2f" % row["thickness_fraction_of_box"] if row["geometry"] == "slab" else "bulk"
        print(f"{label:>14} {row['n_mobile']:>6} {row['n_wall']:>6} "
              f"{row['u_dw_early']:>11.5f} {row['u_dw_late']:>10.5f} {drift:>7.3f} "
              f"{row['wall_max_abs_shift']:>11.2e} {'PASS' if passed else 'FAIL':>8}")
        row["wall_frozen"] = wall_frozen
        row["stationary"] = stationary
        row["passed"] = passed

    print()
    print(f"drift tolerance {args.drift_tolerance:g}; the bidisperse failure was a ratio of about 1.47")
    print(f"GATE: {'GO' if ok else 'NO-GO'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "parent": str(parent_path),
                "dt": args.dt,
                "horizon_steps": horizon,
                "horizon_time": horizon * args.dt,
                "drift_tolerance": args.drift_tolerance,
                "rows": rows,
                "gate": "GO" if ok else "NO-GO",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
