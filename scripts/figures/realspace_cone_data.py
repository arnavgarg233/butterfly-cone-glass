#!/usr/bin/env python3
"""Snapshot a matched twin pair in the L=20 sigma block for the real-space cone.

The streaming giant-box runner (``scripts/giant_cone_run.py``) keeps only the
shell-averaged ``D(r,t)`` and never stores full trajectories, so it cannot feed
a real-space particle plot.  This helper reuses that runner's *exact* twin
construction (matched-momentum reference + one localized ``O_shell`` centre kick,
bit-identical at ``delta==0``) and its velocity-Verlet integrator, but records
the full per-particle positions of both twins at a handful of early snapshot
times -- early enough that the divergence front is still interior to the box, so
the butterfly cone reads as an expanding bright core rather than a saturated box.

The block (``N=8000``, ``L=20 sigma``, ``T=0.108``) is large enough that with the
measured front speed ``v_b~2.87 sigma`` per time unit the cone stays inside the
box through ``t~3.5``.  Output feeds ``fig_realspace_cone.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from butterfly_cone.engine.integrate import MDIntegrator  # noqa: E402
from butterfly_cone.engine.potential import minimum_image  # noqa: E402
from giant_cone_run import ConeConfig, build_twins, load_parent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path,
                        default=ROOT / "runs/giant_blocks/block_000/parent_T0.108.npz")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/realspace_viz/cone_particles.npz")
    parser.add_argument("--temperature", type=float, default=0.108)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--r-pert", type=float, default=2.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--snapshot-times", type=float, nargs="+",
                        default=[0.8, 1.6, 2.4, 3.2])
    args = parser.parse_args(argv)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]

    config = ConeConfig(
        parent=args.parent, out_dir=args.out.parent, temperature=args.temperature,
        operator="shell", delta=args.delta, r_pert=args.r_pert,
        dt=args.dt, device=device, dtype=args.dtype, deterministic=False,
    )
    parent = load_parent(config.parent, device=device, dtype=dtype)
    ref, kick, kick_prov = build_twins(parent, config)
    box = ref.box.detach().cpu().numpy().astype(float)
    center = np.asarray(kick_prov["site"], dtype=float)
    diameters = ref.diameters.detach().cpu().numpy().astype(float)
    n = int(ref.n_particles)
    print(f"[data] N={n} L={box[0]:.3f} T={args.temperature} device={device} "
          f"kick={kick_prov['operator']} n_perturbed={kick_prov['n_perturbed']} "
          f"center={np.array2string(center, precision=3)}", flush=True)

    # bit-identity sanity: at t=0 only the kicked core has nonzero divergence
    diff0 = minimum_image(ref.positions - kick.positions, ref.box)
    d0 = torch.linalg.vector_norm(diff0, dim=1)
    n_changed = int((d0 > 1e-7).sum().item())
    assert n_changed == kick_prov["n_perturbed"], (n_changed, kick_prov["n_perturbed"])

    ref_int = MDIntegrator(ref, dt=config.dt, skin=config.skin, thermostat=None)
    kick_int = MDIntegrator(kick, dt=config.dt, skin=config.skin, thermostat=None)

    times = sorted(float(t) for t in args.snapshot_times)
    step_targets = [int(round(t / config.dt)) for t in times]
    positions_ref: list[np.ndarray] = []
    positions_kick: list[np.ndarray] = []
    divergences: list[np.ndarray] = []
    done = 0
    with torch.no_grad():
        for target, t in zip(step_targets, times):
            advance = target - done
            if advance > 0:
                ref_int.step(advance)
                kick_int.step(advance)
                done = target
            diff = minimum_image(ref.positions - kick.positions, ref.box)
            d = torch.linalg.vector_norm(diff, dim=1).detach().cpu().numpy().astype(np.float32)
            positions_ref.append(ref.positions.detach().cpu().numpy().astype(np.float32))
            positions_kick.append(kick.positions.detach().cpu().numpy().astype(np.float32))
            divergences.append(d)
            print(f"[data] t={t:g} step={target} "
                  f"divergence[min,med,max]=[{d.min():.3g},{np.median(d):.3g},{d.max():.3g}] "
                  f"front(v_b*t)={2.8688744453676125 * t:.2f} sigma", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema="realspace_cone/1",
        N=np.asarray(n, dtype=np.int64),
        L=np.asarray(box[0], dtype=np.float64),
        box=box,
        temperature=np.asarray(args.temperature, dtype=np.float64),
        v_b=np.asarray(2.8688744453676125, dtype=np.float64),
        diameters=diameters,
        center=center,
        times=np.asarray(times, dtype=np.float64),
        positions_kicked=np.stack(positions_kick),
        positions_control=np.stack(positions_ref),
        divergences=np.stack(divergences),
        kick_json=np.asarray(json.dumps(kick_prov)),
    )
    print(f"[write] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
