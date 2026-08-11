#!/usr/bin/env python3
"""Build and certify a swap-friendly, strongly bimodal ButterflyCone glass parent.

This intentionally changes *only* the diameter population.  The force law is
the flagship C2-smoothed r^-12 IPL with its standard non-additive mixing rule;
the HybridSwapMD schedule is the same 120 Bussi-NVT steps plus N exact diameter
swap proposals used by the flagship and ``second_model_equilibrate.py``.

The run is deliberately gated: a flat energy trace is not enough.  It must also
show cross-peak label exchange and a stationary long-time NVE cage before a cone
campaign is permitted to use the saved parent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@dataclass(frozen=True)
class BimodalSpec:
    """Equal-weight continuous two-peak diameter distribution."""

    small_center: float = 0.85
    large_center: float = 1.15
    peak_width: float = 0.08
    fraction_small: float = 0.50


def make_bimodal_diameters(n_particles: int, spec: BimodalSpec,
                            rng: np.random.Generator) -> np.ndarray:
    """Draw an equal-composition, continuous bimodal population with mean 1.

    Each peak is independently Gaussian.  A final global rescaling removes the
    O(N^-1/2) sample-mean fluctuation without changing bimodality or the
    continuous (all values distinct with probability one) nature of the set.
    """

    if n_particles <= 0 or n_particles % 2:
        raise ValueError("n_particles must be positive and even for an exact 50:50 mixture")
    if not (0.0 < spec.fraction_small < 1.0) or not math.isclose(spec.fraction_small, 0.5):
        raise ValueError("this certification protocol requires fraction_small=0.5")
    if spec.small_center <= 0.0 or spec.large_center <= spec.small_center or spec.peak_width <= 0.0:
        raise ValueError("invalid bimodal peak parameters")

    per_peak = n_particles // 2
    small = rng.normal(loc=spec.small_center, scale=spec.peak_width, size=per_peak)
    large = rng.normal(loc=spec.large_center, scale=spec.peak_width, size=per_peak)
    raw = np.concatenate((small, large))
    if np.any(raw <= 0.0):
        raise RuntimeError("non-positive diameter draw; reduce peak_width or draw a new seed")
    diameters = raw / raw.mean()
    rng.shuffle(diameters)
    return diameters.astype(np.float64, copy=False)


def _rank_labels(diameters: np.ndarray) -> tuple[np.ndarray, float]:
    """Return exact 50:50 small/large labels based on the immutable draw ranks."""

    n_particles = diameters.size
    order = np.argsort(diameters, kind="stable")
    labels = np.zeros(n_particles, dtype=bool)
    labels[order[n_particles // 2:]] = True
    boundary = float(0.5 * (diameters[order[n_particles // 2 - 1]] + diameters[order[n_particles // 2]]))
    return labels, boundary


def size_distribution_summary(diameters: np.ndarray) -> dict[str, Any]:
    """Report raw, reproducible evidence that the stored set is bimodal."""

    values = np.asarray(diameters, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("diameters must be a non-empty one-dimensional array")
    labels, boundary = _rank_labels(values)
    small = values[~labels]
    large = values[labels]
    pooled_width = math.sqrt(0.5 * (float(small.var(ddof=1)) + float(large.var(ddof=1))))
    counts, edges = np.histogram(values, bins=24)
    return {
        "n": int(values.size),
        "n_small": int(small.size),
        "n_large": int(large.size),
        "n_unique": int(np.unique(values).size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "rank_boundary": boundary,
        "small_peak_mean": float(small.mean()),
        "large_peak_mean": float(large.mean()),
        "small_peak_std": float(small.std(ddof=1)),
        "large_peak_std": float(large.std(ddof=1)),
        "peak_separation": float(large.mean() - small.mean()),
        "peak_separation_over_pooled_width": float((large.mean() - small.mean()) / pooled_width),
        "histogram": {"counts": [int(x) for x in counts], "edges": [float(x) for x in edges]},
    }


def cage_stationarity(curve: np.ndarray | list[float], *, tolerance: float = 0.10) -> dict[str, Any]:
    """Test whether a long-window observable has a stationary late plateau.

    The last half of the trace is split into two equal windows.  Both its mean
    drift and least-squares slope across that late window must be <=10% of the
    late mean; this rejects the ``growing cage`` failure mode even when energy
    has already flattened.
    """

    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 12:
        raise ValueError("stationarity needs at least twelve samples")
    if not np.isfinite(values).all():
        return {"stationary": False, "reason": "non-finite curve"}
    late = values[values.size // 2:]
    split = late.size // 2
    first, second = late[:split], late[split:]
    first_mean = float(first.mean())
    second_mean = float(second.mean())
    late_mean = float(late.mean())
    scale = max(abs(late_mean), np.finfo(float).eps)
    drift = abs(second_mean - first_mean) / scale
    x = np.arange(late.size, dtype=np.float64)
    slope = float(np.polyfit(x, late, deg=1)[0])
    slope_over_window = abs(slope) * max(late.size - 1, 1) / scale
    stationary = bool(drift <= tolerance and slope_over_window <= tolerance)
    return {
        "stationary": stationary,
        "tolerance": float(tolerance),
        "n_samples": int(values.size),
        "late_n_samples": int(late.size),
        "late_first_half_mean": first_mean,
        "late_second_half_mean": second_mean,
        "late_mean": late_mean,
        "late_relative_drift": float(drift),
        "late_relative_slope_over_window": float(slope_over_window),
        "late_linear_slope_per_sample": slope,
    }


def _swap_cross_peak_fraction(system, initial_peak_labels: torch.Tensor,
                              rank_boundary: float) -> float:
    current = system.diameters > rank_boundary
    return float(torch.ne(current, initial_peak_labels).to(torch.float64).mean().detach().cpu())


def _write_parent(path: Path, system, *, temperature: float) -> None:
    label = f"{temperature:.3f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        L=np.asarray(float(system.box[0].detach().cpu()), dtype=np.float64),
        N=np.asarray(system.n_particles, dtype=np.int64),
        SAVE_T=np.asarray([temperature], dtype=np.float64),
        R=np.asarray(1, dtype=np.int64),
        **{
            f"pos_{label}_0": system.positions.detach().cpu().to(torch.float64).numpy(),
            f"sig_{label}_0": system.diameters.detach().cpu().to(torch.float64).numpy(),
        },
    )


def _energy_span(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float((array.max() - array.min()) / max(abs(float(array.mean())), 1.0))


def _long_nve_cage_probe(system, *, temperature: float, seed: int, branches: int,
                          steps: int, stride: int, dt: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure the intrinsic branch-cloud cage over a cone-length NVE window."""

    from butterfly_cone.engine.integrate import MDIntegrator, maxwell_boltzmann_velocities
    from butterfly_cone.engine.system import make_generator
    import dw_identity

    if steps <= 0 or stride <= 0 or steps % stride:
        raise ValueError("steps must be positive and divisible by stride")
    frames = steps // stride + 1
    n_particles = system.n_particles
    positions = np.empty((frames, branches, n_particles, 3), dtype=np.float64)
    energy_curves = np.empty((branches, frames), dtype=np.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for branch in range(branches):
            probe = system.clone()
            probe.velocities = maxwell_boltzmann_velocities(
                n_particles, temperature, make_generator(seed + branch), device=probe.device,
                dtype=probe.dtype, active_mask=probe.active_mask,
            )
            integrator = MDIntegrator(probe, dt=dt, thermostat=None)
            positions[0, branch] = probe.unwrapped_positions.detach().cpu().numpy()
            energy_curves[branch, 0] = float(integrator.total_energy().detach().cpu())
            for frame in range(1, frames):
                integrator.step(stride)
                positions[frame, branch] = probe.unwrapped_positions.detach().cpu().numpy()
                energy_curves[branch, frame] = float(integrator.total_energy().detach().cpu())
            print(f"  [cage branch {branch + 1}/{branches}] NVE span="
                  f"{_energy_span(energy_curves[branch].tolist()):.3e}", flush=True)

    box = system.box.detach().cpu().numpy().astype(np.float64)
    reference = system.unwrapped_positions.detach().cpu().numpy().astype(np.float64)
    u2_curve = dw_identity.cage_msd_curve(positions, box, ddof=1)
    msd_curve = dw_identity.msd_relative_to_reference(positions, reference, box)
    u2_stationarity = cage_stationarity(u2_curve)
    msd_stationarity = cage_stationarity(msd_curve)
    nve_spans = [_energy_span(row.tolist()) for row in energy_curves]
    certified = bool(
        u2_stationarity["stationary"]
        and msd_stationarity["stationary"]
        and max(float(x) for x in nve_spans if x is not None) <= 1.0e-3
    )
    report = {
        "kind": "long_unperturbed_nve_branch_cloud",
        "branches": int(branches),
        "steps": int(steps),
        "stride": int(stride),
        "dt": float(dt),
        "physical_time": float(steps * dt),
        "elapsed_seconds": float(time.perf_counter() - started),
        "u2_cage_curve": [float(x) for x in u2_curve],
        "msd_rel_parent_curve": [float(x) for x in msd_curve],
        "u2_cage_stationarity": u2_stationarity,
        "msd_rel_parent_stationarity": msd_stationarity,
        "nve_relative_energy_spans": [float(x) for x in nve_spans if x is not None],
        "nve_energy_stable": bool(max(float(x) for x in nve_spans if x is not None) <= 1.0e-3),
        "stationary": certified,
    }
    raw = {
        "steps": np.arange(frames, dtype=np.int64) * stride,
        "u2_cage_curve": u2_curve,
        "msd_rel_parent_curve": msd_curve,
        "energy_curves": energy_curves,
    }
    return report, raw


def run_equilibration(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    from butterfly_cone.engine import potential
    from butterfly_cone.engine.integrate import BussiThermostat, MDIntegrator, maxwell_boltzmann_velocities
    from butterfly_cone.engine.swap import HybridSwapMD
    from butterfly_cone.engine.system import make_generator, make_system, relax_overlaps

    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    spec = BimodalSpec(small_center=args.small_center, large_center=args.large_center,
                       peak_width=args.peak_width)
    np_rng = np.random.default_rng(args.seed)
    diameter_values = make_bimodal_diameters(args.n, spec, np_rng)
    distribution = size_distribution_summary(diameter_values)

    system = make_system(args.n, generator=make_generator(args.seed + 1), device=args.device,
                         dtype=dtype, placement="random", density=1.0)
    with torch.no_grad():
        system.diameters.copy_(torch.as_tensor(diameter_values, device=system.device, dtype=system.dtype))
    initial_peak_np, boundary = _rank_labels(diameter_values)
    initial_peak = torch.as_tensor(initial_peak_np, device=system.device, dtype=torch.bool)
    relax = relax_overlaps(system, steps=args.relax_steps, max_displacement=0.002)
    system.velocities = maxwell_boltzmann_velocities(
        args.n, args.temperature, make_generator(args.seed + 2), device=system.device,
        dtype=system.dtype, active_mask=system.active_mask,
    )
    thermostat = BussiThermostat(args.temperature, 0.5, make_generator(args.seed + 3))
    integrator = MDIntegrator(system, dt=args.dt, thermostat=thermostat)
    hybrid = HybridSwapMD(integrator, temperature=args.temperature, generator=make_generator(args.seed + 4),
                          md_steps=args.md_steps, swap_attempts=args.n)

    energy_blocks: list[float] = []
    current_block: list[float] = []
    log: list[dict[str, Any]] = []
    converged = False
    started = time.perf_counter()
    with torch.no_grad():
        for sweep in range(1, args.max_sweeps + 1):
            hybrid.cycle()
            current_block.append(float(integrator.potential_energy.detach().cpu()) / args.n)
            if sweep % args.check_every != 0 and sweep != args.max_sweeps:
                continue
            epp = float(np.mean(current_block))
            current_block.clear()
            energy_blocks.append(epp)
            tail = energy_blocks[-args.energy_blocks:]
            span = _energy_span(tail) if len(tail) == args.energy_blocks else None
            plateau = bool(span is not None and span <= args.energy_span_tol)
            cross_peak_fraction = _swap_cross_peak_fraction(system, initial_peak, boundary)
            swap_healthy = bool(hybrid.statistics.acceptance_rate >= args.swap_acceptance_min
                                and cross_peak_fraction >= args.cross_peak_mixing_min)
            converged = bool(sweep >= args.min_sweeps and plateau and swap_healthy)
            row = {
                "sweep": int(sweep), "md_steps": int(sweep * args.md_steps), "E_per_particle": epp,
                "energy_span_last_blocks": span, "energy_plateau": plateau,
                "swap_attempts": int(hybrid.statistics.attempts),
                "swap_accepted": int(hybrid.statistics.accepted),
                "swap_acceptance": float(hybrid.statistics.acceptance_rate),
                "cross_peak_label_fraction": cross_peak_fraction,
                "swap_healthy": swap_healthy,
            }
            log.append(row)
            print(f"[equilibrate] sweep={sweep:4d} E/N={epp:.6f} span={span} "
                  f"acc={hybrid.statistics.acceptance_rate:.3f} cross={cross_peak_fraction:.3f} "
                  f"plateau={plateau} healthy={swap_healthy}", flush=True)
            if converged:
                break

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "continuous-bimodal swap-friendly r^-12 glass parent",
        "distribution": {"requested": asdict(spec), "measured": distribution},
        "potential": {
            "family": "flagship C2-smoothed r^-12 IPL",
            "exponent": 12,
            "cutoff_ratio": float(potential.CUTOFF_RATIO),
            "nonadditivity": float(potential.NONADDITIVITY),
            "monkeypatch": "none",
        },
        "protocol": {
            "n": int(args.n), "temperature": float(args.temperature), "seed": int(args.seed),
            "device": args.device, "dtype": args.dtype, "initial_placement": "random",
            "overlap_relaxation_steps": int(args.relax_steps), "dt": float(args.dt),
            "thermostat": "Bussi stochastic velocity rescaling", "thermostat_tau": 0.5,
            "md_steps_per_hybrid_sweep": int(args.md_steps), "swap_attempts_per_sweep": int(args.n),
            "min_sweeps": int(args.min_sweeps), "max_sweeps": int(args.max_sweeps),
            "energy_blocks": int(args.energy_blocks), "energy_span_tol": float(args.energy_span_tol),
            "swap_acceptance_min": float(args.swap_acceptance_min),
            "cross_peak_mixing_min": float(args.cross_peak_mixing_min),
        },
        "overlap_relaxation": {
            "steps_completed": int(relax.steps_completed), "initial_energy": float(relax.initial_energy),
            "final_energy": float(relax.final_energy), "final_max_force": float(relax.final_max_force),
        },
        "equilibration": {
            "converged_before_aging_probe": converged,
            "elapsed_seconds": float(time.perf_counter() - started),
            "log": log,
        },
    }
    return report, system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "bimodal")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--small-center", type=float, default=0.85)
    parser.add_argument("--large-center", type=float, default=1.15)
    parser.add_argument("--peak-width", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--relax-steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--md-steps", type=int, default=120)
    parser.add_argument("--min-sweeps", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=1400)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--energy-blocks", type=int, default=8)
    parser.add_argument("--energy-span-tol", type=float, default=3.0e-3)
    parser.add_argument("--swap-acceptance-min", type=float, default=0.05)
    parser.add_argument("--cross-peak-mixing-min", type=float, default=0.15)
    parser.add_argument("--cage-branches", type=int, default=8)
    parser.add_argument("--cage-steps", type=int, default=6000)
    parser.add_argument("--cage-stride", type=int, default=100)
    parser.add_argument("--cage-dt", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n % 2:
        raise SystemExit("--n must be even")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report, system = run_equilibration(args)
    parent_path = args.out_dir / f"parent_T{args.temperature:.3f}.npz"
    raw_path = args.out_dir / "aging_probe_raw.npz"
    if not report["equilibration"]["converged_before_aging_probe"]:
        report["aging_probe"] = {"stationary": False, "reason": "equilibration/swap gate did not converge"}
        (args.out_dir / "equilibration.json").write_text(json.dumps(report, indent=2) + "\n")
        print("[STOP] no cone: hybrid equilibration or continuous cross-peak swap gate failed", flush=True)
        return 3

    cage, raw = _long_nve_cage_probe(
        system, temperature=args.temperature, seed=args.seed + 1000, branches=args.cage_branches,
        steps=args.cage_steps, stride=args.cage_stride, dt=args.cage_dt,
    )
    report["aging_probe"] = cage
    np.savez_compressed(raw_path, **raw)
    certified = bool(cage["stationary"])
    report["verdict"] = {
        "parent_certified_for_cone": certified,
        "reason": "stationary long NVE cage plus healthy continuous swap mixing" if certified
                  else "non-stationary cage or NVE instability; cone intentionally not run",
    }
    (args.out_dir / "equilibration.json").write_text(json.dumps(report, indent=2) + "\n")
    if certified:
        _write_parent(parent_path, system, temperature=args.temperature)
        print(f"[CERTIFIED] parent={parent_path} raw_cage={raw_path}", flush=True)
        return 0
    print("[STOP] no cone: long unperturbed cage is not stationary", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
