#!/usr/bin/env python3
"""Run and audit a matched-seed O_shell cone for a cage-certified breadth parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from breadth_lib import MODEL_SPECS, inject_breadth_model, stationary_tail  # noqa: E402


def cone_controls(*, n_sites: int = 2, branches: int = 12, horizon: int = 6000,
                  stride: int = 100, dt: float = 0.01) -> dict[str, Any]:
    """Frozen controls and workload for the breadth cones."""

    if min(n_sites, branches, horizon, stride) <= 0 or horizon % stride:
        raise ValueError("invalid cone controls")
    return {
        "operator": "O_shell", "deltas": [0.01, 0.03], "n_sites": int(n_sites),
        "branches": int(branches), "horizon": int(horizon), "stride": int(stride),
        "dt": float(dt), "physical_horizon": float(horizon * dt), "r_pert": 2.5,
        "n_perturbed_ensembles": int(2 * n_sites),
        "n_total_ensembles": int(1 + 2 * n_sites),
        "matched_momentum_seeds": True, "thermostat": None,
        "integrator": "velocity_verlet_nve",
    }


def _relative_difference(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(0.5 * (a + b)), np.finfo(float).eps))


def _seed_list(path: Path) -> list[int]:
    data = json.loads((path / "branch_provenance.json").read_text(encoding="utf-8"))
    return [int(row["momentum_seed"]) for row in data["branches"]]


def _audit(base: Path, r0: dict[str, Any], dw: dict[str, Any], cache) -> dict[str, Any]:
    """Recompute the requested seed, plateau, kick, and exponential checks."""

    import gardner_r0
    from butterfly_cone.perturb.response import total_divergence

    unpert = next(iter(cache.unpert_stores))
    unpert_ref = next(ref for ref in gardner_r0.discover(base) if ref.kind == "unpert")
    reference_seeds = _seed_list(unpert_ref.path)
    seed_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    for field in cache.fields:
        pert_ref = next(
            ref for ref in gardner_r0.discover(base)
            if ref.kind == "pert" and ref.config == field.config
            and ref.site == field.site and ref.delta_index == field.delta_index
        )
        seeds = _seed_list(pert_ref.path)
        seed_rows.append({
            "ensemble": field.label, "n": len(seeds),
            "identical_to_unperturbed": seeds == reference_seeds,
        })
        curve = np.asarray(total_divergence(field.m_field), dtype=np.float64)
        plateau_rows.append({
            "ensemble": field.label, "delta_index": field.delta_index,
            "delta": field.delta, "curve": curve.tolist(),
            "late_stationarity": stationary_tail(curve, tolerance=0.10),
            "late_mean_per_particle": float(curve[curve.size // 2 :].mean() / field.N),
        })

    ensembles = r0["ensembles"]
    by_delta: dict[str, dict[str, float]] = {}
    for delta_index in (0, 1):
        rows = [row for row in ensembles if int(row["delta_index"]) == delta_index]
        by_delta[str(delta_index)] = {
            "delta": float((0.01, 0.03)[delta_index]),
            "n": len(rows),
            "lambda_mean": float(np.mean([row["lam"] for row in rows])),
            "D_sat_mean": float(np.mean([row["D_sat"] for row in rows])),
            "lambda_r2_min": float(np.min([row["lam_r2"] for row in rows])),
            "resolved_fraction": float(np.mean([bool(row["resolved"]) for row in rows])),
            "growing_fraction": float(np.mean([bool(row["growing"]) for row in rows])),
            "saturated_fraction": float(np.mean([bool(row["saturated"]) for row in rows])),
        }
    kick_lambda = _relative_difference(by_delta["0"]["lambda_mean"], by_delta["1"]["lambda_mean"])
    kick_dsat = _relative_difference(by_delta["0"]["D_sat_mean"], by_delta["1"]["D_sat_mean"])
    intrinsic = float(dw["pairwise_divergence_per_particle"])
    landed = float(dw["landed_D_sat_over_N"])
    plateau_gap = _relative_difference(intrinsic, landed)
    return {
        "matched_seed_audit": {
            "unperturbed_key": list(unpert), "unperturbed_seeds": reference_seeds,
            "per_ensemble": seed_rows,
            "all_identical": bool(all(row["identical_to_unperturbed"] for row in seed_rows)),
        },
        "plateau_stationarity": {
            "per_ensemble": plateau_rows,
            "all_stationary": bool(all(row["late_stationarity"]["stationary"] for row in plateau_rows)),
        },
        "by_kick": by_delta,
        "kick_independence": {
            "lambda_relative_difference": kick_lambda,
            "D_sat_relative_difference": kick_dsat,
            "lambda_within_20_percent": bool(kick_lambda <= 0.20),
            "D_sat_within_10_percent": bool(kick_dsat <= 0.10),
        },
        "exponential_growth": {
            "lambda_r2_min": float(min(row["lam_r2"] for row in ensembles)),
            "resolved_fraction": float(np.mean([bool(row["resolved"]) for row in ensembles])),
            "all_growing": bool(all(row["growing"] for row in ensembles)),
            "all_saturated": bool(all(row["saturated"] for row in ensembles)),
            "pooled_cone_resolved": bool(r0["verdict"]["cone_resolved"]),
        },
        "intrinsic_match": {
            "intrinsic_pairwise_divergence_per_particle": intrinsic,
            "perturbed_D_sat_per_particle": landed,
            "relative_difference": plateau_gap,
            "within_10_percent": bool(plateau_gap <= 0.10),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eq_path = args.out_dir / "equilibration.json"
    if not eq_path.is_file():
        raise FileNotFoundError(f"missing equilibration certificate: {eq_path}")
    equilibration = json.loads(eq_path.read_text(encoding="utf-8"))
    if not equilibration["verdict"]["cone_permitted"]:
        raise RuntimeError("refusing cone: parent did not pass stationary cage gate")
    temperature = float(equilibration["protocol"]["temperature"])
    n_particles = int(equilibration["protocol"]["n"])
    parent = args.out_dir / f"parent_T{temperature:.3f}.npz"
    if not parent.is_file():
        raise FileNotFoundError(parent)
    controls = cone_controls(n_sites=args.n_sites, branches=args.branches,
                             horizon=args.horizon, stride=args.stride, dt=args.dt)
    provenance = inject_breadth_model(args.model)
    import dw_identity
    import gardner_cone_campaign as gcc
    import gardner_probe as gp
    import gardner_r0

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    dtype = "float32" if device == "mps" else "float64"
    cfg = gp.ConfigSpec(path=parent, temperature=temperature, replica=0,
                        n_particles=n_particles)
    options = gcc.CampaignOptions(
        configs=(cfg,), temperature=temperature, operator="O_shell",
        n_sites=args.n_sites, deltas=tuple(controls["deltas"]), branches=args.branches,
        horizon=args.horizon, stride=args.stride, dt=args.dt,
        mega_chunk=1, device=device, dtype=dtype,
        root=args.out_dir / "campaign_root", run_id=f"breadth-{args.model}-cone-s{args.n_sites}",
        project_salt=f"butterfly_cone-breadth-{args.model}-20260718",
    )
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    base = gcc.run_campaign(options)
    cache = gardner_r0.collect_fields(base)
    r0 = gardner_r0.aggregate_report(cache, r2_resolved=gardner_r0.R2_RESOLVED_DEFAULT)
    r0_path = base / "gardner_r0.json"
    r0_path.write_text(json.dumps(r0, indent=2, default=float) + "\n", encoding="utf-8")
    unperturbed = dw_identity._default_config_dirs(base)
    dw = dw_identity.analyze_dw_identity(unperturbed, r0_path, plateau_frac=0.5, ddof=1, tol=0.10)
    dw_path = base / "dw_identity.json"
    dw_path.write_text(json.dumps(dw, indent=2, default=float) + "\n", encoding="utf-8")
    audit = _audit(base, r0, dw, cache)
    cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
    compute = {
        "wall_seconds": float(time.perf_counter() - started),
        "user_cpu_seconds": float(cpu_finished.ru_utime - cpu_started.ru_utime),
        "system_cpu_seconds": float(cpu_finished.ru_stime - cpu_started.ru_stime),
        "branch_md_steps": int(controls["n_total_ensembles"] * args.branches * args.horizon),
        "particle_branch_md_steps": int(
            controls["n_total_ensembles"] * args.branches * args.horizon * n_particles
        ),
    }
    empirical_c = float(dw["identity"]["empirical_c"])
    intrinsic_c = float(dw["pairwise_divergence_per_particle"] / dw["u_DW"])
    result = {
        "schema_version": 1, "kind": "breadth_matched_seed_cone",
        "model": provenance, "temperature": temperature, "N": n_particles,
        "equilibration_certificate": str(eq_path), "parent": str(parent),
        "base_path": str(base), "controls": controls,
        "gardner_r0": r0, "dw_identity": dw, "raw_audit": audit,
        "cage_ceiling": {
            "empirical_c": empirical_c, "intrinsic_pairwise_c": intrinsic_c,
            "law_interval": [1.23, 1.30],
            "empirical_in_interval": bool(1.23 <= empirical_c <= 1.30),
            "intrinsic_in_interval": bool(1.23 <= intrinsic_c <= 1.30),
        },
        "compute": compute,
    }
    result["verdict"] = {
        "law_holds": bool(
            result["cage_ceiling"]["empirical_in_interval"]
            and result["cage_ceiling"]["intrinsic_in_interval"]
            and audit["intrinsic_match"]["within_10_percent"]
        ),
        "matched_seeds": audit["matched_seed_audit"]["all_identical"],
        "stationary_perturbed_plateaus": audit["plateau_stationarity"]["all_stationary"],
        "kick_independent_D_sat": audit["kick_independence"]["D_sat_within_10_percent"],
        "resolved_exponential_cone": audit["exponential_growth"]["pooled_cone_resolved"],
    }
    output = args.out_dir / "cone_analysis.json"
    output.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-sites", type=int, default=2)
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({
        "output": str(args.out_dir / "cone_analysis.json"),
        "lambda": result["gardner_r0"]["pooled"]["lambda"]["mean"],
        "empirical_c": result["cage_ceiling"]["empirical_c"],
        "intrinsic_c": result["cage_ceiling"]["intrinsic_pairwise_c"],
        "verdict": result["verdict"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
