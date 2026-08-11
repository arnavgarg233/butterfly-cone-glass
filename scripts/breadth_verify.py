#!/usr/bin/env python3
"""Independent raw-file verification of the breadth campaign's headline numbers."""

from __future__ import annotations

import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from breadth_lib import stationary_tail  # noqa: E402
from breadth_separation import _inject, _load_system, pearson_correlation  # noqa: E402


def _difference(stored: float, recomputed: float, tolerance: float = 1.0e-10) -> dict[str, Any]:
    absolute = float(abs(float(stored) - float(recomputed)))
    return {
        "stored": float(stored), "recomputed": float(recomputed),
        "absolute_difference": absolute, "tolerance": tolerance,
        "matches": bool(absolute <= tolerance),
    }


def _verify_new_model(model: str) -> dict[str, Any]:
    import dw_identity
    import gardner_r0

    model_dir = ROOT / "runs/breadth" / model
    equilibrium = json.loads((model_dir / "equilibration.json").read_text())
    cone = json.loads((model_dir / "cone_analysis.json").read_text())
    raw = np.load(model_dir / "cage_raw.npz")
    positions = np.asarray(raw["positions"], dtype=np.float64)
    box = np.asarray(raw["box"], dtype=np.float64)
    parent = np.asarray(raw["parent_unwrapped"], dtype=np.float64)
    u2 = dw_identity.cage_msd_curve(positions, box, ddof=1)
    msd = dw_identity.msd_relative_to_reference(positions, parent, box)
    start = positions.shape[0] // 2
    pairwise = float(np.mean([
        dw_identity.pairwise_branch_divergence_per_particle(positions[frame], box)
        for frame in range(start, positions.shape[0])
    ]))
    u_dw = math.sqrt(float(np.mean(u2[start:])))
    base = Path(cone["base_path"])
    r0 = gardner_r0.run(base, r2_resolved=gardner_r0.R2_RESOLVED_DEFAULT)
    dw = dw_identity.analyze_dw_identity(
        dw_identity._default_config_dirs(base), base / "gardner_r0.json",
        plateau_frac=0.5, ddof=1, tol=0.10,
    )
    checks = {
        "equilibrium_u_DW": _difference(equilibrium["cage_gate"]["u_DW"], u_dw),
        "equilibrium_intrinsic_pairwise": _difference(
            equilibrium["cage_gate"]["intrinsic_pairwise_divergence_per_particle"], pairwise
        ),
        "cone_lambda": _difference(
            cone["gardner_r0"]["pooled"]["lambda"]["mean"], r0["pooled"]["lambda"]["mean"]
        ),
        "cone_D_sat": _difference(
            cone["gardner_r0"]["pooled"]["D_sat"]["mean"], r0["pooled"]["D_sat"]["mean"]
        ),
        "cone_empirical_c": _difference(cone["cage_ceiling"]["empirical_c"], dw["identity"]["empirical_c"]),
        "cone_intrinsic_c": _difference(
            cone["cage_ceiling"]["intrinsic_pairwise_c"],
            float(dw["pairwise_divergence_per_particle"] / dw["u_DW"]),
        ),
    }
    stationarity = {
        "u2": stationary_tail(u2, tolerance=0.10),
        "msd": stationary_tail(msd, tolerance=0.10),
    }
    return {
        "model": model, "raw_paths": {
            "cage": str(model_dir / "cage_raw.npz"), "cone_branches": str(base.parent),
        },
        "checks": checks, "stationarity_recomputed": stationarity,
        "all_numeric_checks_match": bool(all(check["matches"] for check in checks.values())),
        "stationarity_matches": bool(
            stationarity["u2"]["stationary"] == equilibrium["cage_gate"]["u2_cage_stationarity"]["stationary"]
            and stationarity["msd"]["stationary"] == equilibrium["cage_gate"]["msd_rel_parent_stationarity"]["stationary"]
        ),
    }


def _lambda_from_source(row: dict[str, Any]) -> float:
    path = Path(row["lambda_source"])
    data = json.loads(path.read_text())
    name = path.name
    if name == "vb_elastic_cone.json":
        match = next(item for item in data["per_rung"] if abs(float(item["temperature"]) - row["temperature"]) < 1e-12)
        return float(match["lam"])
    if name == "bimodal_result.json":
        return float(data["cone"]["lambda_mean"])
    if name == "cone_analysis.json":
        return float(data["gardner_r0"]["pooled"]["lambda"]["mean"])
    return float(data["gardner_r0"]["lambda_mean"])


def _verify_pool() -> dict[str, Any]:
    from butterfly_cone.mechanics.elastic import born_modulus

    pooled = json.loads((ROOT / "runs/breadth/pooled_separation.json").read_text())
    ct_checks = []
    lambda_checks = []
    entropy_formula_checks = []
    for row in pooled["rows"]:
        _inject(int(row["exponent"]), float(row["nonadditivity"]))
        system = _load_system(row)
        density = system.n_particles / float(np.prod(system.box.detach().cpu().numpy()))
        recomputed_ct = math.sqrt(float(born_modulus(system, axis=(0, 1))) / density)
        ct_checks.append({"state_id": row["state_id"], **_difference(row["c_T"], recomputed_ct, 1.0e-10)})
        lambda_checks.append({
            "state_id": row["state_id"], **_difference(row["lambda"], _lambda_from_source(row), 1.0e-12)
        })
        determinant = float(row["inherent_structure"]["log_pseudodeterminant"])
        n = int(row["N"])
        temperature = float(row["temperature"])
        n_modes = 3 * n - 3
        hbar = 1.0 / (2.0 * math.pi)
        proxy = (n_modes * (1.0 + math.log(temperature / hbar)) - 0.5 * determinant) / n
        entropy_formula_checks.append({
            "state_id": row["state_id"],
            **_difference(row["harmonic_proxy"]["value"], proxy, 1.0e-12),
            "scope": "recomputed from persisted sparse-LU log-pseudodeterminant diagnostic",
        })
    proxy = [float(row["harmonic_proxy"]["value"]) for row in pooled["rows"]]
    ct = [float(row["c_T"]) for row in pooled["rows"]]
    lam = [float(row["lambda"]) for row in pooled["rows"]]
    correlations = {
        "proxy_vs_c_T": _difference(
            pooled["pooled_state_weighted"]["harmonic_proxy_vs_c_T"]["pearson_r"],
            pearson_correlation(proxy, ct), 1.0e-14,
        ),
        "lambda_vs_proxy": _difference(
            pooled["pooled_state_weighted"]["lambda_vs_harmonic_proxy"]["pearson_r"],
            pearson_correlation(proxy, lam), 1.0e-14,
        ),
        "lambda_vs_c_T": _difference(
            pooled["pooled_state_weighted"]["lambda_vs_c_T"]["pearson_r"],
            pearson_correlation(ct, lam), 1.0e-14,
        ),
    }
    return {
        "c_T_parent_recomputations": ct_checks,
        "lambda_source_recomputations": lambda_checks,
        "harmonic_formula_recomputations": entropy_formula_checks,
        "correlation_recomputations": correlations,
        "all_c_T_match": bool(all(row["matches"] for row in ct_checks)),
        "all_lambda_match": bool(all(row["matches"] for row in lambda_checks)),
        "all_harmonic_formula_match": bool(all(row["matches"] for row in entropy_formula_checks)),
        "all_correlations_match": bool(all(row["matches"] for row in correlations.values())),
        "hessian_raw_limit": (
            "Full Hessian matrices were intentionally not persisted; the entropy check independently "
            "re-evaluates the paper formula from the stored sparse-LU log-pseudodeterminant and the "
            "stored eight-mode stability tail. Parent paths and minimizer diagnostics are persisted."
        ),
    }


def main() -> int:
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    new_models = [_verify_new_model(model) for model in ("soft_r8", "trimodal")]
    pool = _verify_pool()
    cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
    payload = {
        "schema_version": 1, "kind": "breadth_independent_raw_verification",
        "new_models": new_models, "pooled": pool,
        "compute": {
            "wall_seconds": float(time.perf_counter() - started),
            "user_cpu_seconds": float(cpu_finished.ru_utime - cpu_started.ru_utime),
            "system_cpu_seconds": float(cpu_finished.ru_stime - cpu_started.ru_stime),
        },
    }
    payload["all_checks_pass"] = bool(
        all(row["all_numeric_checks_match"] and row["stationarity_matches"] for row in new_models)
        and pool["all_c_T_match"] and pool["all_lambda_match"]
        and pool["all_harmonic_formula_match"] and pool["all_correlations_match"]
    )
    output = ROOT / "runs/breadth/verification.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_checks_pass": payload["all_checks_pass"]}, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
