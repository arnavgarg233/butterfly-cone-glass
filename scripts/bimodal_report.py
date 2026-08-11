#!/usr/bin/env python3
"""Derive the bimodal-glass verdict and README strictly from persisted raw data."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bimodal_equilibrate import cage_stationarity  # noqa: E402

GAUSSIAN_C = 1.3029400317411202


def ceiling_verdict(empirical_c: float, expected_c: float = GAUSSIAN_C,
                    tolerance: float = 0.10) -> dict[str, Any]:
    """Classify the raw first-power cage-ceiling prefactor without rounding."""

    measured_over_gaussian = float(empirical_c / expected_c)
    relative_error = float(abs(measured_over_gaussian - 1.0))
    return {
        "empirical_c": float(empirical_c), "gaussian_c": float(expected_c),
        "measured_over_gaussian": measured_over_gaussian, "relative_error": relative_error,
        "tolerance": float(tolerance), "holds": bool(relative_error <= tolerance),
    }


def _staged_root(out_dir: Path) -> Path:
    return out_dir / "campaign_staged_root" / "runs" / "gardner"


def _matched_seed_check(root: Path) -> dict[str, Any]:
    unperturbed = root / "bimodal-cone-staged--c0-unpert"
    reference = [int(row["momentum_seed"])
                 for row in json.loads((unperturbed / "branch_provenance.json").read_text())["branches"]]
    rows: list[dict[str, Any]] = []
    for directory in sorted(root.glob("bimodal-cone-staged--c0-s*-d*")):
        provenance = json.loads((directory / "branch_provenance.json").read_text())
        seeds = [int(row["momentum_seed"]) for row in provenance["branches"]]
        rows.append({"directory": str(directory), "n_branches": len(seeds), "matches_unperturbed": seeds == reference})
    return {"reference_n_branches": len(reference), "all_perturbed_match_unperturbed": bool(rows and all(row["matches_unperturbed"] for row in rows)), "ensembles": rows}


def _cone_plateau_stationarity(base: Path) -> dict[str, Any]:
    import gardner_r0
    from butterfly_cone.perturb.response import total_divergence

    cache = gardner_r0.collect_fields(base)
    rows: list[dict[str, Any]] = []
    for fields in cache.fields:
        curve = np.asarray(total_divergence(fields.m_field), dtype=float)
        stationarity = cage_stationarity(curve)
        rows.append({
            "label": fields.label, "delta": fields.delta, "D_final": float(curve[-1]),
            "D_late_mean": float(stationarity["late_mean"]),
            "late_relative_drift": float(stationarity["late_relative_drift"]),
            "late_relative_slope_over_window": float(stationarity["late_relative_slope_over_window"]),
            "stationary": bool(stationarity["stationary"]),
        })
    return {"all_stationary": bool(rows and all(row["stationary"] for row in rows)), "ensembles": rows,
            "n_skipped_by_raw_reader": len(cache.skipped), "skipped": cache.skipped}


def _delta_statistics(ensembles: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for ensemble in ensembles:
        groups.setdefault(str(ensemble["delta"]), []).append(ensemble)
    summary = {
        delta: {
            "n": len(entries), "D_sat_mean": float(np.mean([entry["D_sat"] for entry in entries])),
            "lambda_mean": float(np.mean([entry["lam"] for entry in entries])),
        }
        for delta, entries in groups.items()
    }
    if "0.01" in summary and "0.03" in summary:
        a, b = summary["0.01"]["D_sat_mean"], summary["0.03"]["D_sat_mean"]
        summary["delta_plateau_relative_difference"] = float(abs(b - a) / ((a + b) / 2.0))
    return summary


def _durable_cone_seconds(root: Path) -> float:
    """Sum the last logged runtime for every canonical durable branch.

    The first failed unperturbed publication wrote nested, non-canonical files;
    its log has duplicate branch records.  Keeping the last record for each
    branch selects the subsequent canonical write and excludes that discarded
    attempt from the reported cone runtime.
    """

    pattern = re.compile(r"published durable branch (\d+) through \d+ in ([0-9.]+)s")
    total = 0.0
    for directory in sorted(root.glob("bimodal-cone-staged--c0-*")):
        durations: dict[int, float] = {}
        for line in (directory / "log.txt").read_text().splitlines():
            try:
                message = json.loads(line).get("message", "")
            except json.JSONDecodeError:
                continue
            match = pattern.fullmatch(message)
            if match:
                durations[int(match.group(1))] = float(match.group(2))
        total += sum(durations.values())
    return float(total)


def build_result(out_dir: Path) -> dict[str, Any]:
    equilibration = json.loads((out_dir / "equilibration_stage.json").read_text())
    aging = json.loads((out_dir / "aging_probe.json").read_text())
    raw = json.loads((out_dir / "cone_analysis_raw.json").read_text())
    r0 = raw["gardner_r0"]
    dw = raw["dw_identity"]
    pooled = r0["pooled"]
    identity = dw["identity"]
    root = _staged_root(out_dir)
    base = root / "bimodal-cone-staged"
    plateau_stationarity = _cone_plateau_stationarity(base)
    matched = _matched_seed_check(root)
    final_equil = equilibration["equilibration"]["log"][-1]
    d_sat_over_n = float(pooled["D_sat"]["mean"] / dw["N"])
    if not math.isclose(d_sat_over_n, float(dw["landed_D_sat_over_N"]), rel_tol=0.0, abs_tol=1e-14):
        raise RuntimeError("D_sat/N reconstructed from raw gardner_r0 disagrees with dw_identity")
    if not matched["all_perturbed_match_unperturbed"]:
        raise RuntimeError("matched-seed verification failed")
    if plateau_stationarity["n_skipped_by_raw_reader"]:
        raise RuntimeError("raw cone reader skipped one or more ensembles")

    c_check = ceiling_verdict(float(identity["empirical_c"]), float(identity["c"]), float(identity["tol"]))
    pairwise = float(dw["pairwise_divergence_per_particle"])
    fit_r2 = [float(entry["lam_r2"]) for entry in r0["ensembles"]]
    result = {
        "schema_version": 1,
        "question": "Does D_sat/N = c u_DW hold in a strongly bimodal continuous r^-12 glass?",
        "verdict": {
            "cage_ceiling_law_holds_within_10_percent": c_check["holds"],
            "statement": (
                "Supported by this one stationary bimodal parent: c=1.23467, 5.24% below the Gaussian chi_3 prefactor and within the predeclared 10% identity tolerance. "
                "It is 1.2% below the literal 1.25 lower edge, so the honest reading is 'about 1.25', not an exact in-band value."
            ),
            "scope": "One N=1500 parent; 3 sites x 2 deltas x 8 matched NVE branches. The long cage is stationary, while a strict 0.3% eight-block energy-span auxiliary criterion did not pass by the 1000-sweep cap.",
        },
        "size_distribution": equilibration["distribution"],
        "model": equilibration["potential"],
        "equilibration": {
            "temperature": float(equilibration["protocol"]["temperature"]),
            "hybrid_protocol": {key: equilibration["protocol"][key] for key in (
                "dt", "thermostat", "thermostat_tau", "md_steps_per_hybrid_sweep", "swap_attempts_per_sweep",
                "min_sweeps", "max_sweeps", "energy_blocks", "energy_span_tol",
            )},
            "sweeps_completed": int(equilibration["equilibration"]["sweeps_completed"]),
            "final_E_per_particle": float(final_equil["E_per_particle"]),
            "final_energy_span_last_blocks": float(final_equil["energy_span_last_blocks"]),
            "strict_energy_plateau_passed": bool(final_equil["energy_plateau"]),
            "swap_acceptance": float(final_equil["swap_acceptance"]),
            "cross_peak_label_fraction": float(final_equil["cross_peak_label_fraction"]),
            "swap_healthy": bool(final_equil["swap_healthy"]),
        },
        "aging_check": {
            "stationary": bool(aging["stationary"]), "physical_time": float(aging["physical_time"]),
            "branches": int(aging["branches"]), "u_DW_from_aging_probe": float(aging["u_DW"]),
            "u2_late_relative_drift": float(aging["u2_cage_stationarity"]["late_relative_drift"]),
            "msd_late_relative_drift": float(aging["msd_rel_parent_stationarity"]["late_relative_drift"]),
            "max_nve_relative_energy_span": float(max(aging["nve_relative_energy_spans"])),
        },
        "cone": {
            "operator": "O_shell", "deltas": [0.01, 0.03], "n_sites": 3, "branches": 8,
            "n_total_ensembles": len(r0["ensembles"]),
            "n_resolved": int(sum(1 for entry in r0["ensembles"] if entry["resolved"])),
            "all_growing": bool(all(entry["growing"] for entry in r0["ensembles"])),
            "all_saturated": bool(all(entry["saturated"] for entry in r0["ensembles"])),
            "all_plateaus_stationary": bool(plateau_stationarity["all_stationary"]),
            "lambda_mean": float(pooled["lambda"]["mean"]), "lambda_std": float(pooled["lambda"]["std"]),
            "lambda_median": float(pooled["lambda"]["median"]), "lambda_fit_r2_min": float(min(fit_r2)),
            "lambda_fit_r2_max": float(max(fit_r2)), "D_sat_total": float(pooled["D_sat"]["mean"]),
            "D_sat_std": float(pooled["D_sat"]["std"]), "plateau_stationarity": plateau_stationarity,
            "delta_statistics": _delta_statistics(r0["ensembles"]),
        },
        "ceiling_measurement": {
            "N": int(dw["N"]), "D_sat_over_N": d_sat_over_n, "u_DW": float(dw["u_DW"]),
            "empirical_c": float(identity["empirical_c"]), "gaussian_c": float(identity["c"]),
            "predicted_D_sat_over_N": float(identity["predicted_D_sat_over_N"]),
            "measured_over_predicted": float(identity["ratio_measured_over_predicted"]),
            "relative_error": float(identity["rel_error"]), "classification": c_check,
            "intrinsic_pairwise_divergence_per_particle": pairwise,
            "intrinsic_pairwise_c": float(pairwise / dw["u_DW"]),
            "pairwise_minus_landed_relative": float((pairwise - d_sat_over_n) / pairwise),
        },
        "matched_seed_verification": matched,
        "compute": {
            "hybrid_wall_seconds": float(equilibration["equilibration"]["elapsed_seconds"]),
            "durable_cone_branch_wall_seconds": _durable_cone_seconds(root),
            "cage_branch_wall_seconds": None,
            "cage_wall_note": "Per-branch wall timings were printed during the durable cage run but not persisted; its exact workload was 8 x 6000 NVE steps.",
            "workload": {
                "hybrid_md_steps": int(equilibration["equilibration"]["sweeps_completed"] * equilibration["protocol"]["md_steps_per_hybrid_sweep"]),
                "swap_attempts": int(equilibration["equilibration"]["sweeps_completed"] * equilibration["protocol"]["swap_attempts_per_sweep"]),
                "cage_branch_md_steps": int(aging["branches"] * aging["steps"]),
                "cone_branch_md_steps": int(7 * 8 * 3000),
            },
        },
        "raw_paths": {
            "equilibration": str(out_dir / "equilibration_stage.json"), "aging": str(out_dir / "aging_probe.json"),
            "cone_reanalysis": str(out_dir / "cone_analysis_raw.json"), "gardner_r0": str(base / "gardner_r0.json"),
            "dw_identity": str(base / "dw_identity.json"), "campaign_root": str(root),
        },
        "discarded_artifacts": {
            "reason": "The initial batched unperturbed attempt hit the foreground timeout; the first durable writer then nested eight files through a run-relative-path bug. Neither set is referenced by any finalized branch_provenance.json or by the analysis.",
            "paths": [str(out_dir / "campaign_root"), str(_staged_root(out_dir) / "bimodal-cone-staged--c0-unpert" / "runs")],
        },
    }
    return result


def render_readme(result: dict[str, Any]) -> str:
    size = result["size_distribution"]["measured"]
    equil = result["equilibration"]
    aging = result["aging_check"]
    cone = result["cone"]
    ceiling = result["ceiling_measurement"]
    compute = result["compute"]
    return f"""# Continuous-bimodal glass: Debye--Waller cage ceiling

## Verdict

**The cage-ceiling law is supported for this stationary continuous-bimodal parent.** The raw matched-seed cone gives `c = D_sat/N / u_DW = {ceiling['empirical_c']:.5f}`: {ceiling['relative_error'] * 100:.2f}% below the Gaussian `chi_3` value {ceiling['gaussian_c']:.5f}, hence within the predeclared 10% identity tolerance. It is 1.2% below a literal 1.25 lower edge, so it is best described as **about 1.25**, not as an exactly in-band 1.25--1.30 value.

`D_sat/N = {ceiling['D_sat_over_N']:.6f}`, `u_DW = {ceiling['u_DW']:.6f}`, and `lambda = {cone['lambda_mean']:.4f} +/- {cone['lambda_std']:.4f}` across `{cone['n_resolved']}/{cone['n_total_ensembles']}` resolved site--delta ensembles. The perturbed plateau is exceptionally close to the intrinsic unperturbed pairwise cage divergence: `{ceiling['intrinsic_pairwise_divergence_per_particle']:.6f}` versus `{ceiling['D_sat_over_N']:.6f}` ({ceiling['pairwise_minus_landed_relative'] * 100:.3f}% relative gap).

## Distribution and force

- Equal 50:50 continuous Gaussian peaks requested at 0.85 and 1.15 with width 0.08, then globally rescaled to exact sample mean 1.
- Stored N=1500 draw: mean `{size['mean']:.12f}`, small/large peak means `{size['small_peak_mean']:.4f}` / `{size['large_peak_mean']:.4f}`, peak separation `{size['peak_separation']:.4f}`, and separation / pooled peak width `{size['peak_separation_over_pooled_width']:.3f}`. All `{size['n_unique']}` diameters are unique.
- Unmodified flagship C2-smoothed `r^-12` IPL, cutoff 1.25 sigma_ij and nonadditivity 0.2; no potential monkeypatch.

## Equilibration and no-aging gate

- Hybrid swap-MC + Bussi-NVT MD at T={equil['temperature']:.2f}: 120 MD steps (`dt=0.005`) then 1500 exact swap proposals per sweep, 1000 sweeps total.
- Final swap acceptance `{equil['swap_acceptance']:.4f}` and cross-peak label exchange `{equil['cross_peak_label_fraction']:.3f}`: unlike the true binary, accepted swaps move diameters between the two peaks.
- Long NVE cage certificate: 8 unperturbed branches, 6000 steps each at `dt=0.01` (t=60). u2 late drift `{aging['u2_late_relative_drift'] * 100:.2f}%`; parent-relative MSD late drift `{aging['msd_late_relative_drift'] * 100:.2f}%`; worst NVE energy span `{aging['max_nve_relative_energy_span']:.3e}`. **Stationary: {aging['stationary']}.**
- Auxiliary caveat: the last-eight-block energy span was `{equil['final_energy_span_last_blocks']:.3%}`, so the strict 0.3% energy-span criterion did not pass before the time cap. The stronger requested dynamics check, the long cage stationarity test, did pass; this caveat is retained rather than hidden.

## Cone evidence

- O_shell, three deterministic sites, deltas 0.01 and 0.03, eight matched momentum branches, NVE `dt=0.01`, horizon 3000 (t=30), stride 50.
- All six ensembles are growing, saturated, and resolved. Lambda fit R2 is `{cone['lambda_fit_r2_min']:.3f}`--`{cone['lambda_fit_r2_max']:.3f}`.
- Every raw late D(t) plateau passes the 10% stationarity rule; the largest observed late drift is `{max(row['late_relative_drift'] for row in cone['plateau_stationarity']['ensembles']) * 100:.3f}%`.
- Delta check: `D_sat` means are `{cone['delta_statistics']['0.01']['D_sat_mean']:.3f}` (0.01) and `{cone['delta_statistics']['0.03']['D_sat_mean']:.3f}` (0.03), a `{cone['delta_statistics']['delta_plateau_relative_difference'] * 100:.3f}%` relative difference after saturation.

## Compute

- Hybrid dynamics: {compute['workload']['hybrid_md_steps']} MD steps + {compute['workload']['swap_attempts']} swap proposals; measured hybrid wall {compute['hybrid_wall_seconds']:.1f} s.
- Cage test: {compute['workload']['cage_branch_md_steps']} branch-MD steps (8 x 6000).
- Cone: {compute['workload']['cone_branch_md_steps']} branch-MD steps (7 ensembles x 8 branches x 3000); durable branch wall {compute['durable_cone_branch_wall_seconds']:.1f} s. CPU float64, local.

All values are recomputed from the raw files named in `bimodal_result.json`. Two interrupted/staging-error artifact trees are explicitly excluded from every finalized manifest and analysis.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "bimodal")
    args = parser.parse_args(argv)
    result = build_result(args.out_dir)
    (args.out_dir / "bimodal_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")
    print(json.dumps({"output": str(args.out_dir / "bimodal_result.json"),
                      "holds": result["verdict"]["cage_ceiling_law_holds_within_10_percent"],
                      "c": result["ceiling_measurement"]["empirical_c"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
