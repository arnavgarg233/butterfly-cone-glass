#!/usr/bin/env python3
"""Assemble the final breadth verdict, per-model JSONs, README, and compute ledger."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/breadth"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_model_result(model: str) -> dict[str, Any]:
    model_dir = OUT / model
    equilibrium = _read(model_dir / "equilibration.json")
    cone = _read(model_dir / "cone_analysis.json")
    cage = cone["cage_ceiling"]
    audit = cone["raw_audit"]
    return {
        "schema_version": 1, "model": cone["model"], "temperature": cone["temperature"],
        "N": cone["N"], "distribution": equilibrium["distribution"],
        "equilibration": {
            "stationary_no_aging": equilibrium["verdict"]["stationary_no_aging_parent"],
            "sweeps": equilibrium["equilibration"]["sweeps_completed"],
            "swap_acceptance": equilibrium["equilibration"]["log"][-1]["swap_acceptance"],
            "swap_label_correlation": equilibrium["equilibration"]["log"][-1]["swap_label_correlation"],
            "cross_peak_label_fraction": equilibrium["equilibration"]["log"][-1]["cross_peak_label_fraction"],
            "u2_stationarity": equilibrium["cage_gate"]["u2_cage_stationarity"],
            "parent_msd_stationarity": equilibrium["cage_gate"]["msd_rel_parent_stationarity"],
            "max_nve_energy_span": max(equilibrium["cage_gate"]["nve_relative_energy_spans"]),
        },
        "cone": {
            "controls": cone["controls"],
            "lambda": cone["gardner_r0"]["pooled"]["lambda"],
            "D_sat": cone["gardner_r0"]["pooled"]["D_sat"],
            "empirical_c": cage["empirical_c"],
            "intrinsic_pairwise_c": cage["intrinsic_pairwise_c"],
            "law_interval": cage["law_interval"],
            "intrinsic_match": audit["intrinsic_match"],
            "kick_independence": audit["kick_independence"],
            "exponential_growth": audit["exponential_growth"],
            "perturbed_plateaus_all_stationary": audit["plateau_stationarity"]["all_stationary"],
            "matched_seeds": audit["matched_seed_audit"]["all_identical"],
        },
        "verdict": cone["verdict"],
        "raw_paths": {
            "equilibration": str(model_dir / "equilibration.json"),
            "cage": str(model_dir / "cage_raw.npz"),
            "parent": cone["parent"], "cone_base": cone["base_path"],
            "cone_analysis": str(model_dir / "cone_analysis.json"),
        },
        "compute": {"equilibration": equilibrium["compute"], "cone": cone["compute"]},
    }


def _extreme_result() -> dict[str, Any]:
    attempts = []
    for path in sorted((OUT / "extreme_bimodal").glob("*/equilibration.json")):
        data = _read(path)
        cage = data.get("cage_gate", {})
        attempts.append({
            "path": str(path), "temperature": data["protocol"]["temperature"],
            "peak_width": data["model"]["peak_width"],
            "measured_outer_peak_ratio": data["distribution"]["measured_outer_peak_ratio"],
            "equilibration_converged": data["equilibration"]["converged"],
            "cage_stationary": bool(cage.get("stationary", False)),
            "u2_late_relative_drift": cage.get("u2_cage_stationarity", {}).get("late_relative_drift"),
            "u2_late_relative_slope": cage.get("u2_cage_stationarity", {}).get("late_relative_slope_over_window"),
            "parent_msd_late_relative_drift": cage.get("msd_rel_parent_stationarity", {}).get("late_relative_drift"),
            "parent_msd_late_relative_slope": cage.get("msd_rel_parent_stationarity", {}).get("late_relative_slope_over_window"),
            "wall_seconds": data["compute"]["wall_seconds"],
        })
    final_path = OUT / "extreme_bimodal/smoke_w018_T0080/equilibration.json"
    final = _read(final_path)
    return {
        "schema_version": 1, "model": final["model"], "status": "stationarity_null",
        "final_candidate": {
            "path": str(final_path), "temperature": final["protocol"]["temperature"],
            "N": final["protocol"]["n"], "distribution": final["distribution"],
            "equilibration": final["equilibration"], "cage_gate": final["cage_gate"],
        },
        "attempts": attempts,
        "verdict": {
            "stationary_no_aging_parent": False, "cone_run": False,
            "cage_ceiling_tested": False, "law_holds": None,
            "reason": (
                "All seven parameter/horizon candidates failed the stationary-cage gate. "
                "The final width-0.18, T=0.08 candidate mixed across peaks but retained "
                "15.7-21.2% late-window cage/MSD slopes."
            ),
        },
    }


def _ceiling_catalog() -> list[dict[str, Any]]:
    flagship = _read(ROOT / "runs/dw_identity/dw_identity.json")
    additive = _read(ROOT / "runs/second_model/cone_sm-n12add-T0060-h4000.json")
    hard03 = _read(ROOT / "runs/second_model/cone_sm-n18-T0030-h4000.json")
    hard05 = _read(ROOT / "runs/second_model/cone_sm-n18-T0050.json")
    bimodal = _read(ROOT / "runs/bimodal/bimodal_result.json")
    soft = _read(OUT / "soft_r8/cone_analysis.json")
    trimodal = _read(OUT / "trimodal/cone_analysis.json")
    return [
        {
            "model": "flagship_r12", "empirical_c": [flagship["identity"]["empirical_c"]],
            "intrinsic_pairwise_c": [flagship["pairwise_divergence_per_particle"] / flagship["u_DW"]],
            "source": str(ROOT / "runs/dw_identity/dw_identity.json"), "confirmed": True,
        },
        {
            "model": "additive_delta0", "empirical_c": [additive["dw_identity"]["empirical_c"]],
            "intrinsic_pairwise_c": [additive["dw_identity"]["pairwise_divergence_per_particle"] / additive["dw_identity"]["u_DW"]],
            "source": str(ROOT / "runs/second_model/cone_sm-n12add-T0060-h4000.json"),
            "confirmed": True,
            "caveat": "T=0.10 horizon-2000 empirical c=1.202 under-fills; its intrinsic c is 1.284.",
        },
        {
            "model": "hard_r18", "empirical_c": [hard03["dw_identity"]["empirical_c"], hard05["dw_identity"]["empirical_c"]],
            "intrinsic_pairwise_c": [
                hard03["dw_identity"]["pairwise_divergence_per_particle"] / hard03["dw_identity"]["u_DW"],
                hard05["dw_identity"]["pairwise_divergence_per_particle"] / hard05["dw_identity"]["u_DW"],
            ],
            "source": [
                str(ROOT / "runs/second_model/cone_sm-n18-T0030-h4000.json"),
                str(ROOT / "runs/second_model/cone_sm-n18-T0050.json"),
            ], "confirmed": True,
        },
        {
            "model": "bimodal_ratio1.4", "empirical_c": [bimodal["ceiling_measurement"]["empirical_c"]],
            "intrinsic_pairwise_c": [bimodal["ceiling_measurement"]["intrinsic_pairwise_c"]],
            "source": str(ROOT / "runs/bimodal/bimodal_result.json"), "confirmed": True,
        },
        {
            "model": "soft_r8", "empirical_c": [soft["cage_ceiling"]["empirical_c"]],
            "intrinsic_pairwise_c": [soft["cage_ceiling"]["intrinsic_pairwise_c"]],
            "source": str(OUT / "soft_r8/cone_analysis.json"), "confirmed": soft["verdict"]["law_holds"],
        },
        {
            "model": "trimodal", "empirical_c": [trimodal["cage_ceiling"]["empirical_c"]],
            "intrinsic_pairwise_c": [trimodal["cage_ceiling"]["intrinsic_pairwise_c"]],
            "source": str(OUT / "trimodal/cone_analysis.json"), "confirmed": trimodal["verdict"]["law_holds"],
        },
    ]


def _manifest_duration(path: Path) -> float:
    data = _read(path)
    start = datetime.fromisoformat(data["start_time"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(data["end_time"].replace("Z", "+00:00"))
    return float((end - start).total_seconds())


def _compute_ledger() -> dict[str, Any]:
    equilibrium_rows = []
    for path in sorted(OUT.glob("**/equilibration.json")):
        data = _read(path)
        equilibrium_rows.append({"path": str(path), **data["compute"]})
    cones = [_read(OUT / f"{model}/cone_analysis.json")["compute"] for model in ("soft_r8", "trimodal")]
    harmonics = _read(OUT / "harmonic_measurements.json")["measurements"]
    verification = _read(OUT / "verification.json")["compute"]
    failed_manifest = OUT / "soft_r8/campaign_root/runs/gardner/breadth-soft_r8-cone--c0-unpert/manifest.json"
    failed_seconds = _manifest_duration(failed_manifest)
    equil_wall = sum(row["wall_seconds"] for row in equilibrium_rows)
    cone_wall = sum(row["wall_seconds"] for row in cones)
    harmonic_wall = sum(row["compute"]["wall_seconds"] for row in harmonics)
    primary = equil_wall + cone_wall + harmonic_wall + failed_seconds
    return {
        "scope": "new breadth campaign only; inherited flagship/additive/r18/bimodal compute excluded",
        "equilibration_and_stationarity_attempts": {
            "n": len(equilibrium_rows), "wall_seconds": equil_wall,
            "hybrid_md_steps": sum(row["hybrid_md_steps"] for row in equilibrium_rows),
            "swap_attempts": sum(row["swap_attempts"] for row in equilibrium_rows),
            "cage_branch_md_steps": sum(row["cage_branch_md_steps"] for row in equilibrium_rows),
            "rows": equilibrium_rows,
        },
        "successful_cones": {
            "n": 2, "wall_seconds": cone_wall,
            "branch_md_steps": sum(row["branch_md_steps"] for row in cones),
            "particle_branch_md_steps": sum(row["particle_branch_md_steps"] for row in cones),
        },
        "failed_three_site_cone_launch": {
            "wall_seconds": failed_seconds, "completed_unperturbed_branch_md_steps": 12 * 6000,
            "completed_particle_branch_md_steps": 12 * 6000 * 384,
            "reason": "finite box could place only two of three radius-2.5 shell sites at min separation 5",
        },
        "harmonic_mechanics": {"n_states": len(harmonics), "wall_seconds": harmonic_wall},
        "primary_compute_wall_seconds": primary,
        "primary_compute_wall_minutes": primary / 60.0,
        "raw_verification_reanalysis": verification,
        "cost": "$0; local CPU/float64 (MPS unavailable for these launches)",
    }


def _readme(summary: dict[str, Any]) -> str:
    pooled = summary["separation"]["pooled_state_weighted"]
    flag = summary["separation"]["flagship_true_s_c_vs_c_T"]
    soft = summary["new_models"]["soft_r8"]
    tri = summary["new_models"]["trimodal"]
    extreme = summary["new_models"]["extreme_bimodal"]
    compute = summary["compute"]
    lines = [
        "# Breadth and cross-model entropy-vs-stiffness campaign", "",
        "## Decisive result", "",
        f"Two of the three requested new models passed the no-aging gate and both confirm the cage-ceiling law. "
        f"Soft r^-8 gives empirical/intrinsic c = {soft['cone']['empirical_c']:.6f}/{soft['cone']['intrinsic_pairwise_c']:.6f}; "
        f"trimodal gives {tri['cone']['empirical_c']:.6f}/{tri['cone']['intrinsic_pairwise_c']:.6f}. "
        "Both are inside 1.23-1.30, their perturbed plateaus match intrinsic pairwise cage divergence "
        "to 0.21% and 0.18%, all late plateaus are stationary, both kicks agree, and every cone fit is resolved.",
        "",
        f"The extreme-bimodal candidate is a controlled null: {len(extreme['attempts'])} smoke variants were tried. "
        "The final continuous distribution has measured peak ratio 1.660, reaches cross-peak mixing, but its "
        "late cage/MSD slopes are 21.2%/15.7%. No cone was run on an aging parent.",
        "",
        f"The cage-ceiling law is therefore confirmed in {summary['cage_ceiling']['n_confirmed']} distinct glass formers: "
        + ", ".join(row["model"] for row in summary["cage_ceiling"]["catalog"] if row["confirmed"]) + ".",
        "",
        "## Pooled separation", "",
        f"The flagship ladder reproduces r(true s_c, c_T) = {flag['pearson_r']:.6f}. "
        f"Across 12 states and 6 models, r(harmonic-basin proxy, c_T) = {pooled['harmonic_proxy_vs_c_T']['pearson_r']:.6f}; "
        f"r(lambda, harmonic proxy) = {pooled['lambda_vs_harmonic_proxy']['pearson_r']:.6f} "
        f"(R2={pooled['lambda_vs_harmonic_proxy']['R2']:.3f}), whereas r(lambda, c_T) = "
        f"{pooled['lambda_vs_c_T']['pearson_r']:.6f} (R2={pooled['lambda_vs_c_T']['R2']:.3f}). "
        "The model-mean sensitivity gives the same ordering.",
        "",
        "This breaks collinearity on the common harmonic proxy and lambda favors that proxy over affine stiffness. "
        "It does not cleanly settle entropy control: the nonflagship models lack thermodynamic-integration energy "
        "ladders, so their quantity is vibrational harmonic-basin entropy, not configurational entropy. A true "
        "cross-model r(s_c,c_T) is therefore undefined rather than silently mixing conventions.",
        "",
        "## Method and provenance", "",
        "- Models use continuous swap-friendly diameter populations, density 1, C2-smoothed IPL cutoff 1.25, "
        "and the existing HybridSwapMD engine. The new axes are r^-8 softness, three continuous peaks, and a "
        "continuous two-peak center ratio 1.6.",
        "",
        "- Every cone was gated by stationary energy and swap mixing, followed by eight independent NVE cage "
        "branches to physical time 60. Stationarity requires both late half-to-half drift and fitted late-window "
        "slope no larger than 10%, plus NVE energy span no larger than 1e-3.",
        "",
        "- Successful cones use O_shell, deltas 0.01/0.03, two finite-box-separated sites, 12 matched-momentum "
        "branches, dt 0.01, and horizon 6000. The initial three-site launch completed only its unperturbed ensemble "
        "before the site-packing gate failed; that run is retained and counted.",
        "",
        "- c_T is sqrt(G_inf/rho), using the exact affine xy Cauchy-Born estimator on each thermal parent. "
        "The common proxy quenches each parent to fmax<1e-8, assembles the analytic sparse Hessian, verifies three "
        "translation modes and positive sampled physical modes, and evaluates the paper convention with h=1, mass=1.",
        "",
        "- `verification.json` independently reloads cage NPZs and all cone branch trajectories, recomputes "
        "stationarity, lambda, D_sat, both c values, every c_T, every source lambda, and all pooled correlations. "
        f"All checks pass: {summary['verification']['all_checks_pass']}.",
        "",
        "## Caveats", "",
        "- Extreme bimodal is not a law failure; it is unmeasured because no stationary parent was found.",
        "",
        "- Harmonic-basin entropy is itself curvature-sensitive and is not s_c. The proxy result is suggestive, "
        "not a completed answer to the referee's entropy-vs-stiffness objection.",
        "",
        "- The inherited additive and r^-18 parents retain `converged=false` in their legacy strict composite scan "
        "metadata even though completed cones exist; this weakens the pooled inference and is preserved per row.",
        "",
        "- The pool is unbalanced (five flagship states, two each additive/r^-18, one for the others) and spans "
        "N=384, 1000, and 1500. Model-mean correlations are included as a sensitivity check.",
        "",
        "## Compute", "",
        f"Primary new compute: {compute['primary_compute_wall_seconds']:.3f} s "
        f"({compute['primary_compute_wall_minutes']:.2f} min), plus {compute['raw_verification_reanalysis']['wall_seconds']:.3f} s "
        "of independent raw reanalysis. This includes all successful and failed equilibration smokes, both full "
        "parents, both cones, the failed three-site unperturbed ensemble, and 12 harmonic measurements. Cost: $0 local.",
        "",
        "## Files", "",
        "- `breadth_verdict.json`: final machine-readable verdict and ceiling catalog.",
        "- `soft_r8/model_result.json`, `trimodal/model_result.json`, `extreme_bimodal/model_result.json`: new-model results.",
        "- `pooled_separation.json`: all 12 triples, correlations, regressions, and model-mean sensitivity.",
        "- `harmonic_measurements.json`: mechanics and inherent-structure diagnostics.",
        "- `verification.json`: independent raw-output checks.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    new_models = {
        "soft_r8": _new_model_result("soft_r8"),
        "trimodal": _new_model_result("trimodal"),
        "extreme_bimodal": _extreme_result(),
    }
    for model, result in new_models.items():
        path = OUT / model / "model_result.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    catalog = _ceiling_catalog()
    separation = _read(OUT / "pooled_separation.json")
    verification = _read(OUT / "verification.json")
    summary = {
        "schema_version": 1, "kind": "breadth_campaign_final_verdict",
        "new_models": new_models,
        "cage_ceiling": {
            "law_interval": [1.23, 1.30], "n_confirmed": sum(bool(row["confirmed"]) for row in catalog),
            "catalog": catalog,
            "verdict": "six distinct glass formers confirm; extreme bimodal remains unmeasured",
        },
        "separation": separation,
        "verification": verification,
        "compute": _compute_ledger(),
        "final_verdict": {
            "breadth": "2/3 requested new models certify and both confirm; extreme fails no-aging gate",
            "entropy_vs_stiffness": (
                "proxy collinearity is broken and lambda follows the harmonic proxy rather than c_T, "
                "but true entropy-vs-stiffness separation is not established without common true s_c"
            ),
            "referee_objection_addressed": False,
        },
    }
    (OUT / "breadth_verdict.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (OUT / "README.md").write_text(_readme(summary), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT / "breadth_verdict.json"),
        "n_confirmed": summary["cage_ceiling"]["n_confirmed"],
        "referee_objection_addressed": False,
        "compute_minutes": summary["compute"]["primary_compute_wall_minutes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
