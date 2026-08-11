#!/usr/bin/env python3
"""Measure common harmonic-basin proxies and test pooled lambda/c_T separation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from breadth_lib import harmonic_entropy_from_log_pseudodeterminant  # noqa: E402

# Preserve the repository's hard-coded flagship r^-12 primitive before any
# cross-model monkeypatching.  The generated IPL function is mathematically
# equivalent, but using the original removes even ~1e-8 round-off differences
# when checking the five already-published affine-Born c_T values.
from butterfly_cone.engine import potential as _reference_potential  # noqa: E402

REFERENCE_R12_PAIR_POTENTIAL = _reference_potential.pair_potential


def pearson_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Pearson r from the raw paired values, with degenerate inputs rejected."""

    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Pearson inputs must be aligned 1-D arrays with at least two values")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Pearson inputs must be finite")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def translation_minor_log_pseudodeterminant(
    hessian: sparse.spmatrix, n_particles: int,
) -> dict[str, float | int]:
    """Get log pseudodet(H) from one particle-fixed minor and the N^3 identity."""

    matrix = sparse.csr_matrix(hessian, dtype=np.float64)
    if matrix.shape != (3 * n_particles, 3 * n_particles) or n_particles < 2:
        raise ValueError("Hessian shape must be 3N by 3N with N >= 2")
    minor = matrix[3:, 3:].tocsc()
    factor = splu(minor)
    diagonal = np.asarray(factor.U.diagonal(), dtype=np.float64)
    if diagonal.size != 3 * n_particles - 3 or np.any(diagonal == 0.0):
        raise RuntimeError("translation-fixed Hessian minor is singular")
    log_minor_absdet = float(np.log(np.abs(diagonal)).sum(dtype=np.float64))
    log_pseudodeterminant = float(log_minor_absdet + 3.0 * math.log(n_particles))
    return {
        "n_minor_coordinates": int(diagonal.size),
        "log_abs_determinant_translation_fixed_minor": log_minor_absdet,
        "translation_normalization_log_N_cubed": float(3.0 * math.log(n_particles)),
        "log_pseudodeterminant": log_pseudodeterminant,
        "minimum_abs_U_diagonal": float(np.min(np.abs(diagonal))),
    }


def _dataset() -> list[dict[str, Any]]:
    vb = json.loads((ROOT / "runs/vb_elastic_cone/vb_elastic_cone.json").read_text())
    rows: list[dict[str, Any]] = []
    for record in vb["per_rung"]:
        config = int(record["config"])
        rows.append({
            "state_id": f"flagship_T{float(record['temperature']):.3f}",
            "model": "flagship", "temperature": float(record["temperature"]),
            "exponent": 12, "nonadditivity": 0.2,
            "source_kind": "pt",
            "parent": str(ROOT / f"runs/gardner/bridge-Tladder--c{config}-unpert/parent_state.pt"),
            "lambda": float(record["lam"]), "true_s_c": float(record["s_c"]),
            "saved_c_T": float(record["c_T"]),
            "lambda_source": str(ROOT / "runs/vb_elastic_cone/vb_elastic_cone.json"),
            "equilibration_note": "flagship equilibrated temperature-ladder parent",
        })
    inherited = [
        ("additive_delta0", 0.06, 12, 0.0, "scan_T0060.npz", "cone_sm-n12add-T0060-h4000.json"),
        ("additive_delta0", 0.10, 12, 0.0, "scan_T0100.npz", "cone_sm-n12add-T0100.json"),
        ("hard_r18", 0.03, 18, 0.2, "scan_n18_T0030.npz", "cone_sm-n18-T0030-h4000.json"),
        ("hard_r18", 0.05, 18, 0.2, "scan_n18_T0050.npz", "cone_sm-n18-T0050.json"),
    ]
    for model, temperature, exponent, nonadd, parent_name, cone_name in inherited:
        cone_path = ROOT / "runs/second_model" / cone_name
        cone = json.loads(cone_path.read_text())
        scan_json = ROOT / "runs/second_model" / parent_name.replace(".npz", ".json")
        scan = json.loads(scan_json.read_text())
        rows.append({
            "state_id": f"{model}_T{temperature:.3f}", "model": model,
            "temperature": temperature, "exponent": exponent, "nonadditivity": nonadd,
            "source_kind": "npz", "parent": str(ROOT / "runs/second_model" / parent_name),
            "lambda": float(cone["gardner_r0"]["lambda_mean"]), "true_s_c": None,
            "saved_c_T": None, "lambda_source": str(cone_path),
            "equilibration_metadata_converged": bool(scan["replicas"][0]["converged"]),
            "equilibration_note": (
                "inherited swap parent used by the completed cone; legacy scan's strict composite "
                "flag is false and is retained as a pooled-data caveat"
            ),
        })
    bimodal_path = ROOT / "runs/bimodal/bimodal_result.json"
    bimodal = json.loads(bimodal_path.read_text())
    rows.append({
        "state_id": "bimodal_T0.100", "model": "bimodal", "temperature": 0.10,
        "exponent": 12, "nonadditivity": 0.2, "source_kind": "pt",
        "parent": str(ROOT / "runs/bimodal/campaign_staged_root/runs/gardner/"
                      "bimodal-cone-staged--c0-unpert/parent_state.pt"),
        "lambda": float(bimodal["cone"]["lambda_mean"]), "true_s_c": None,
        "saved_c_T": None, "lambda_source": str(bimodal_path),
        "equilibration_note": "stationary long-NVE bimodal cage certificate",
    })
    for model, temperature in (("soft_r8", 0.04), ("trimodal", 0.08)):
        cone_path = ROOT / f"runs/breadth/{model}/cone_analysis.json"
        if not cone_path.is_file():
            continue
        cone = json.loads(cone_path.read_text())
        rows.append({
            "state_id": f"{model}_T{temperature:.3f}", "model": model,
            "temperature": temperature, "exponent": int(cone["model"]["exponent"]),
            "nonadditivity": float(cone["model"]["nonadditivity"]), "source_kind": "npz",
            "parent": str(ROOT / f"runs/breadth/{model}/parent_T{temperature:.3f}.npz"),
            "lambda": float(cone["gardner_r0"]["pooled"]["lambda"]["mean"]),
            "true_s_c": None, "saved_c_T": None, "lambda_source": str(cone_path),
            "equilibration_note": "stationary full N=384 NVE cage certificate",
        })
    return rows


def _load_system(row: dict[str, Any]):
    from butterfly_cone.engine.system import ParticleSystem

    path = Path(row["parent"])
    if row["source_kind"] == "pt":
        state = torch.load(path, map_location="cpu", weights_only=False)
        positions = state["positions"].detach().cpu().to(torch.float64)
        diameters = state["diameters"].detach().cpu().to(torch.float64)
        box = state["box"].detach().cpu().to(torch.float64)
    else:
        loaded = np.load(path)
        temperature = float(row["temperature"])
        label = f"{temperature:.3f}"
        positions = torch.as_tensor(loaded[f"pos_{label}_0"], dtype=torch.float64)
        diameters = torch.as_tensor(loaded[f"sig_{label}_0"], dtype=torch.float64)
        length = float(np.asarray(loaded["L"]))
        box = torch.full((3,), length, dtype=torch.float64)
    n = int(positions.shape[0])
    return ParticleSystem(
        positions=torch.remainder(positions, box), velocities=torch.zeros((n, 3), dtype=torch.float64),
        diameters=diameters, box=box, active_mask=torch.ones(n, dtype=torch.bool),
        unwrapped_positions=positions.clone(),
    )


def _inject(exponent: int, nonadditivity: float) -> dict[str, Any]:
    from second_model_lib import make_pair_potential
    from butterfly_cone.engine import potential, swap
    from butterfly_cone.branching import batched
    from butterfly_cone.mechanics import hessian

    pair = REFERENCE_R12_PAIR_POTENTIAL if exponent == 12 else make_pair_potential(exponent, 1.25)
    potential.pair_potential = pair
    potential.NONADDITIVITY = float(nonadditivity)
    swap.pair_potential = pair
    batched.pair_potential = pair
    hessian.pair_potential = pair
    hessian.CUTOFF_RATIO = 1.25
    return {"exponent": exponent, "nonadditivity": nonadditivity, "cutoff_ratio": 1.25}


def _measure(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _inject(int(row["exponent"]), float(row["nonadditivity"]))
    from butterfly_cone.mechanics.elastic import born_modulus
    from butterfly_cone.mechanics.hessian import analytic_hessian
    from butterfly_cone.mechanics.inherent_structure import minimize_to_IS

    system = _load_system(row)
    n = system.n_particles
    volume = float(torch.prod(system.box))
    density = float(n / volume)
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    g_inf = float(born_modulus(system, axis=(0, 1)))
    c_t = float(math.sqrt(g_inf / density)) if g_inf > 0 else float("nan")
    inherent = minimize_to_IS(
        system, tol=args.force_tolerance, max_steps=args.max_fire_steps,
        lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_outer_steps=args.lbfgs_outer_steps,
    )
    if not inherent.converged:
        raise RuntimeError(f"{row['state_id']}: IS minimization failed, fmax={inherent.fmax}")
    hessian = analytic_hessian(inherent.system)
    smallest = np.sort(eigsh(hessian, k=8, which="SA", return_eigenvectors=False, tol=1.0e-8))
    zero_modes = smallest[np.argsort(np.abs(smallest))[:3]]
    physical_candidates = np.delete(smallest, np.argsort(np.abs(smallest))[:3])
    if np.max(np.abs(zero_modes)) > 1.0e-3 or np.min(physical_candidates) <= 0.0:
        raise RuntimeError(
            f"{row['state_id']}: unstable/dirty sparse Hessian tail: {smallest.tolist()}"
        )
    determinant = translation_minor_log_pseudodeterminant(hessian, n)
    s_harmonic = harmonic_entropy_from_log_pseudodeterminant(
        float(determinant["log_pseudodeterminant"]), n_particles=n,
        temperature=float(row["temperature"]), planck_constant=1.0, mass=1.0,
    )
    cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
    verification = None
    if row.get("saved_c_T") is not None:
        saved = float(row["saved_c_T"])
        verification = {
            "saved_c_T": saved, "recomputed_c_T": c_t,
            "absolute_difference": float(abs(saved - c_t)),
            "matches_to_1e-10": bool(abs(saved - c_t) <= 1.0e-10),
        }
    return {
        **row, "N": n, "volume": volume, "density": density,
        "G_inf_affine_xy_thermal_parent": g_inf, "c_T": c_t,
        "c_T_saved_verification": verification,
        "harmonic_proxy": {
            "name": "classical_harmonic_basin_entropy_per_particle",
            "symbol": "s_harm_proxy", "value": s_harmonic,
            "planck_constant": 1.0, "mass": 1.0, "expected_translation_modes": 3,
            "convention": "N^-1 sum_[3N-3] [1+ln(T/(hbar*sqrt(eigenvalue/mass)))]; h=1, mass=1",
            "is_not_configurational_entropy": True,
        },
        "inherent_structure": {
            "e_is_per_particle": float(inherent.e_is_per_particle), "fmax": float(inherent.fmax),
            "converged": bool(inherent.converged), "fire_steps": int(inherent.fire_steps),
            "lbfgs_outer_steps": int(inherent.lbfgs_steps),
            "smallest_eigenvalues": smallest.tolist(),
            "translation_eigenvalues": zero_modes.tolist(),
            "minimum_sampled_physical_eigenvalue": float(np.min(physical_candidates)),
            **determinant,
        },
        "compute": {
            "wall_seconds": float(time.perf_counter() - started),
            "user_cpu_seconds": float(cpu_finished.ru_utime - cpu_started.ru_utime),
            "system_cpu_seconds": float(cpu_finished.ru_stime - cpu_started.ru_stime),
        },
    }


def _regression(x: list[float], y: list[float]) -> dict[str, float | int]:
    regression = linregress(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    predicted = regression.intercept + regression.slope * np.asarray(x, dtype=float)
    residual = np.asarray(y, dtype=float) - predicted
    return {
        "n": len(x), "pearson_r": float(regression.rvalue),
        "spearman_r": float(spearmanr(x, y).statistic), "slope": float(regression.slope),
        "intercept": float(regression.intercept), "R2": float(regression.rvalue**2),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
    }


def _pool(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    lam = [float(row["lambda"]) for row in measurements]
    ct = [float(row["c_T"]) for row in measurements]
    proxy = [float(row["harmonic_proxy"]["value"]) for row in measurements]
    true_rows = [row for row in measurements if row.get("true_s_c") is not None]
    flagship_sc_ct = _regression(
        [float(row["true_s_c"]) for row in true_rows],
        [float(row["c_T"]) for row in true_rows],
    )
    models = sorted({str(row["model"]) for row in measurements})
    means = []
    for model in models:
        group = [row for row in measurements if row["model"] == model]
        means.append({
            "model": model, "n_states": len(group),
            "lambda": float(np.mean([row["lambda"] for row in group])),
            "c_T": float(np.mean([row["c_T"] for row in group])),
            "s_harm_proxy": float(np.mean([row["harmonic_proxy"]["value"] for row in group])),
        })
    proxy_ct = _regression(proxy, ct)
    lambda_proxy = _regression(proxy, lam)
    lambda_ct = _regression(ct, lam)
    mean_proxy_ct = _regression([r["s_harm_proxy"] for r in means], [r["c_T"] for r in means])
    mean_lambda_proxy = _regression([r["s_harm_proxy"] for r in means], [r["lambda"] for r in means])
    mean_lambda_ct = _regression([r["c_T"] for r in means], [r["lambda"] for r in means])
    return {
        "schema_version": 1, "kind": "pooled_cross_model_entropy_stiffness_separation",
        "n_states": len(measurements), "n_models": len(models), "models": models,
        "rows": measurements,
        "entropy_scale": {
            "flagship_true_s_c_available": True,
            "true_s_c_available_for_nonflagship": False,
            "pooled_quantity": "classical harmonic-basin entropy per particle",
            "common_convention": "h=1, mass=1, epsilon=1, mean diameter=1, density=1",
            "warning": (
                "The pooled harmonic quantity is a vibrational basin entropy, not s_c. "
                "Without model-specific thermodynamic-integration ladders, a true pooled s_c test is unavailable."
            ),
        },
        "flagship_true_s_c_vs_c_T": flagship_sc_ct,
        "pooled_state_weighted": {
            "harmonic_proxy_vs_c_T": proxy_ct,
            "lambda_vs_harmonic_proxy": lambda_proxy,
            "lambda_vs_c_T": lambda_ct,
        },
        "model_mean_sensitivity": {
            "rows": means, "harmonic_proxy_vs_c_T": mean_proxy_ct,
            "lambda_vs_harmonic_proxy": mean_lambda_proxy, "lambda_vs_c_T": mean_lambda_ct,
        },
        "verdict": {
            "flagship_collinearity_reproduced": bool(abs(flagship_sc_ct["pearson_r"]) >= 0.90),
            "proxy_cT_collinearity_below_0_9": bool(abs(proxy_ct["pearson_r"]) < 0.90),
            "lambda_higher_R2_for": (
                "harmonic_proxy" if lambda_proxy["R2"] > lambda_ct["R2"] else "c_T"
            ),
            "true_cross_model_s_c_c_T_collinearity_broken": None,
            "clean_entropy_vs_stiffness_separation": False,
            "reason": (
                "No common true configurational-entropy scale exists for the nonflagship models; "
                "the proxy analysis cannot by itself identify entropy control."
            ),
        },
    }


def _write_outputs(out_dir: Path, measurements: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = _pool(measurements)
    (out_dir / "pooled_separation.json").write_text(
        json.dumps(pooled, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for model in sorted({row["model"] for row in measurements}):
        subset = [row for row in measurements if row["model"] == model]
        payload = {
            "schema_version": 1, "model": model, "states": subset,
            "provenance": "rows copied without transformation from harmonic_measurements.json",
        }
        (out_dir / f"{model}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return pooled


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "harmonic_measurements.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {
        "schema_version": 1, "measurements": [], "failures": []
    }
    if args.remeasure_all:
        cache = {"schema_version": 1, "measurements": [], "failures": []}
    completed = {row["state_id"] for row in cache["measurements"]}
    requested = set(args.state or [])
    if not args.analyze_only:
        for row in _dataset():
            if requested and row["state_id"] not in requested:
                continue
            if row["state_id"] in completed:
                print(f"[resume] {row['state_id']} already measured", flush=True)
                continue
            print(f"[measure] {row['state_id']} parent={row['parent']}", flush=True)
            try:
                measured = _measure(row, args)
            except Exception as exc:
                cache["failures"].append({"state_id": row["state_id"], "error": repr(exc)})
                cache_path.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
                raise
            cache["measurements"].append(measured)
            cache_path.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
            print(json.dumps({
                "state_id": row["state_id"], "c_T": measured["c_T"],
                "s_harm_proxy": measured["harmonic_proxy"]["value"],
                "fmax": measured["inherent_structure"]["fmax"],
                "wall_seconds": measured["compute"]["wall_seconds"],
            }), flush=True)
    expected = {row["state_id"] for row in _dataset()}
    available = {row["state_id"] for row in cache["measurements"]}
    if not expected.issubset(available):
        missing = sorted(expected - available)
        raise RuntimeError(f"cannot pool until all states are measured; missing={missing}")
    ordered = [next(row for row in cache["measurements"] if row["state_id"] == spec["state_id"])
               for spec in _dataset()]
    return _write_outputs(args.out_dir, ordered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/breadth")
    parser.add_argument("--state", action="append", help="measure only this state id (repeatable)")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--remeasure-all", action="store_true", help="replace the generated measurement cache")
    parser.add_argument("--force-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-fire-steps", type=int, default=40000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-outer-steps", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pooled = run(args)
    print(json.dumps({
        "output": str(args.out_dir / "pooled_separation.json"),
        "n_states": pooled["n_states"], "n_models": pooled["n_models"],
        "verdict": pooled["verdict"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
