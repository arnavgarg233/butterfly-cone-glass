#!/usr/bin/env python
"""Recompute every load-bearing number in the butterfly-cone paper, or fail.

The curated artifacts under ``results/`` are the source of truth. This driver
does not trust the summary fields in them: wherever a reported number is a
function of other stored quantities it recomputes it from those quantities and
compares against the frozen value in ``configs/expected_values.json``. Where a
number is a primitive measurement that no shipped artifact can re-derive, it is
re-read from a named JSON pointer and still compared against the frozen value,
and the record says so.

Exit status is 0 only when every frozen value matches. Any mismatch, any
missing artifact and any missing key is a failure.

Usage:
    verify_claims.py [--expected PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        raise SystemExit(f"missing artifact: results/{name}")
    return json.loads(path.read_text())


def pearson(x, y) -> float:
    return float(stats.pearsonr(np.asarray(x, float), np.asarray(y, float))[0])


def zscore(value: float, value_sem: float, ref: float, ref_sem: float) -> float:
    return (value - ref) / math.hypot(value_sem, ref_sem)


# --------------------------------------------------------------------------- #
# recomputations
# --------------------------------------------------------------------------- #


def arm_label(row: dict) -> str:
    if row["geometry"] == "bulk_control":
        return "bulk"
    return f"{row['thickness_fraction_of_box']:.2f}"


def regroup(pairs: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Rebuild each arm's mean and standard error from the shipped twin pairs.

    Nothing in the summary block is trusted. Every pair's ceiling ratio is
    rebuilt from that pair's own plateau and cage amplitude, and each arm is
    then the mean over its eight pairs with the standard error of that mean,
    which is the estimator the campaign driver uses.
    """
    groups: dict[str, list[dict]] = {}
    for row in pairs:
        groups.setdefault(arm_label(row), []).append(row)
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for label, group in groups.items():
        rebuilt = {}
        for name in ("d_sat_per_particle", "u_dw", "anisotropy"):
            values = np.array([float(row[name]) for row in group])
            rebuilt[name] = (float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values))))
        ratios = np.array(
            [float(row["d_sat_per_particle"]) / float(row["u_dw"]) for row in group]
        )
        rebuilt["c_ratio"] = (float(ratios.mean()), float(ratios.std(ddof=1) / math.sqrt(len(ratios))))
        rebuilt["n_pairs"] = (float(len(group)), 0.0)
        summary[label] = rebuilt
    return summary


def confinement() -> dict:
    merged = load("confinement/slab_cone_merged.json")
    control = load("confinement/matched_clip_bulk_control.json")
    gate = load("confinement/slab_stationarity_gate.json")

    rows = regroup(merged["pairs"])
    bulk = rows["bulk"]
    out: dict[str, float | int | bool] = {}

    for key, row in rows.items():
        out[f"c_ratio.{key}"] = row["c_ratio"][0]
        out[f"c_ratio_sem.{key}"] = row["c_ratio"][1]
        out[f"d_sat_per_particle.{key}"] = row["d_sat_per_particle"][0]
        out[f"u_dw.{key}"] = row["u_dw"][0]
    films = [k for k in rows if k != "bulk"]
    deviations = {
        k: 100.0 * (out[f"c_ratio.{k}"] - out["c_ratio.bulk"]) / out["c_ratio.bulk"]
        for k in films
    }
    worst = max(deviations, key=lambda k: abs(deviations[k]))
    out["ceiling.worst_film"] = float(worst)
    out["ceiling.worst_deviation_percent"] = abs(deviations[worst])
    out["ceiling.n_films"] = len(films)
    out["ceiling.n_pairs"] = merged["n_pairs"]

    # Anisotropy departures from the shared bulk control, in standard errors of
    # the difference. Recomputed from the rebuilt per-arm means and errors.
    for key in films:
        out[f"anisotropy.{key}"] = rows[key]["anisotropy"][0]
        out[f"anisotropy_z.{key}"] = zscore(*rows[key]["anisotropy"], *bulk["anisotropy"])
    out["anisotropy.bulk"] = bulk["anisotropy"][0]
    out["anisotropy_sem.bulk"] = bulk["anisotropy"][1]
    clip = regroup(control["pairs"])["bulk"]
    out["anisotropy.matched_clip_control"] = clip["anisotropy"][0]
    out["anisotropy_z.matched_clip_control"] = zscore(
        *clip["anisotropy"], *bulk["anisotropy"]
    )
    out["matched_clip.kick_clip_fraction"] = control["controls"]["kick_clip_fraction"]

    # The rebuilt arms must reproduce the summary block the campaign wrote.
    stored = {arm_label(row): row for row in merged["summary"]}
    gaps = [
        abs(rows[label][field][index] - float(record[key]))
        for label, record in stored.items()
        for field, key, index in (
            ("d_sat_per_particle", "d_sat_per_particle", 0),
            ("d_sat_per_particle", "d_sat_sem", 1),
            ("u_dw", "u_dw", 0),
            ("u_dw", "u_dw_sem", 1),
            ("c_ratio", "c_ratio", 0),
            ("c_ratio", "c_ratio_sem", 1),
            ("anisotropy", "anisotropy", 0),
            ("anisotropy", "anisotropy_sem", 1),
        )
        if label in rows
    ]
    out["summary_agreement.n_fields"] = len(gaps)
    out["summary_agreement.max_absolute_gap_below_1e_12"] = max(gaps) < 1e-12

    # Film thicknesses follow from the shared parent box edge, so the onset
    # window is a difference of two derived lengths, not a stored number.
    edge = control["pairs"][0]["thickness_sigma"]
    thickness = {k: edge * float(k) for k in films}
    out["box_edge_sigma"] = edge
    for key, value in thickness.items():
        out[f"thickness_sigma.{key}"] = value
    out["onset.isotropic_at_sigma"] = thickness["0.50"]
    out["onset.flattened_at_sigma"] = thickness["0.45"]
    out["onset.window_sigma"] = thickness["0.50"] - thickness["0.45"]

    # Kick containment: only the narrowest film loses shell members to the wall.
    audit = merged["kick_containment_audit"]
    for key, record in audit.items():
        out[f"kick_discarded.{key}"] = int(record["n_kick_in_wall_discarded"][0])

    # Stationarity screen: the unperturbed cage must not grow over the horizon.
    drifts = [row["drift_ratio"] for row in gate["rows"]]
    out["stationarity.n_arms"] = len(drifts)
    out["stationarity.min_drift_ratio"] = min(drifts)
    out["stationarity.max_drift_ratio"] = max(drifts)
    out["stationarity.all_passed"] = all(bool(row["passed"]) for row in gate["rows"])
    out["stationarity.tolerance"] = gate["drift_tolerance"]
    return out


def ceiling_identity() -> dict:
    ladder = load("dw_identity_ladder/dw_identity_ladder.json")
    verdict = load("breadth/breadth_verdict.json")
    out: dict[str, float | int | bool] = {}

    # c(T) per rung is rebuilt from the plateau and the cage amplitude.
    rungs = sorted(ladder["per_temperature"], key=lambda r: -r["temperature"])
    for rung in rungs:
        out[f"ladder_c.T{rung['temperature']:.3f}"] = rung["d_sat_over_n"] / rung["u_DW"]
    c_values = [out[f"ladder_c.T{r['temperature']:.3f}"] for r in rungs]
    s_values = [r["s_c"] for r in rungs]
    out["ladder.n_rungs"] = len(rungs)
    out["ladder.spearman_rho_c_vs_s_conf"] = float(stats.spearmanr(c_values, s_values)[0])
    out["ladder.s_conf_warmest"] = max(s_values)
    out["ladder.s_conf_coldest"] = min(s_values)
    out["ladder.monotone_on_cooling"] = all(a < b for a, b in zip(c_values, c_values[1:]))

    # The Gaussian cage target is a pure constant, not a fit.
    target = 4.0 / math.sqrt(3.0 * math.pi)
    out["gaussian_cage_target"] = target

    catalog = verdict["cage_ceiling"]["catalog"]
    flagship = next(entry for entry in catalog if entry["model"] == "flagship_r12")
    out["flagship_c"] = flagship["empirical_c"][0]
    out["flagship_shortfall_percent"] = 100.0 * (target - flagship["empirical_c"][0]) / target

    empirical = [value for entry in catalog for value in entry["empirical_c"]]
    low, high = verdict["cage_ceiling"]["law_interval"]
    out["breadth.n_models"] = len(catalog)
    out["breadth.n_confirmed"] = sum(1 for entry in catalog if entry["confirmed"])
    out["breadth.min_empirical_c"] = min(empirical)
    out["breadth.max_empirical_c"] = max(empirical)
    out["breadth.all_inside_law_band"] = all(low <= value <= high for value in empirical)
    out["breadth.law_band_low"] = low
    out["breadth.law_band_high"] = high
    return out


def entropy_axis() -> dict:
    pooled = load("breadth/pooled_separation.json")
    rows = pooled["rows"]
    out: dict[str, float | int | bool] = {}

    lam = [row["lambda"] for row in rows]
    proxy = [row["harmonic_proxy"]["value"] for row in rows]
    stiffness = [row["c_T"] for row in rows]
    out["pooled.n_states"] = len(rows)
    out["pooled.n_models"] = len({row["model"] for row in rows})
    out["pooled.r_lambda_vs_proxy"] = pearson(lam, proxy)
    out["pooled.r_proxy_vs_stiffness"] = pearson(proxy, stiffness)
    out["pooled.r_lambda_vs_stiffness"] = pearson(lam, stiffness)

    flagship = sorted(
        (row for row in rows if row["model"] == "flagship"), key=lambda r: r["temperature"]
    )
    out["flagship_ladder.n_rungs"] = len(flagship)
    out["flagship_ladder.lambda_coldest"] = flagship[0]["lambda"]
    out["flagship_ladder.lambda_warmest"] = flagship[-1]["lambda"]
    out["flagship_ladder.lambda_fall_percent"] = 100.0 * (
        1.0 - flagship[0]["lambda"] / flagship[-1]["lambda"]
    )
    out["flagship_ladder.s_conf_coldest"] = flagship[0]["true_s_c"]
    out["flagship_ladder.s_conf_warmest"] = flagship[-1]["true_s_c"]
    out["flagship_ladder.s_conf_fall_percent"] = 100.0 * (
        1.0 - flagship[0]["true_s_c"] / flagship[-1]["true_s_c"]
    )
    out["flagship_ladder.spearman_rho_lambda_vs_s_conf"] = float(
        stats.spearmanr(
            [row["lambda"] for row in flagship], [row["true_s_c"] for row in flagship]
        )[0]
    )
    return out


def cone_geometry() -> dict:
    chaos = load("chaos_relations/chaos_relations.json")
    out: dict[str, float | int | bool] = {}
    out["chaos.ell_c_sigma"] = chaos["headline"]["ell_c_sigma"]
    out["chaos.ell_c_ci_low"] = chaos["ballistic"]["linear_response"]["ell_c_median_ci95"][0]
    out["chaos.ell_c_ci_high"] = chaos["ballistic"]["linear_response"]["ell_c_median_ci95"][1]
    out["chaos.ell_c_intensive"] = bool(chaos["headline"]["ell_c_intensive"])
    out["chaos.closure_pearson"] = chaos["closure"]["pearson_t_pred_vs_onset"]
    out["chaos.closure_n_records"] = chaos["n_records"]
    return out


def raw_audit() -> dict:
    verification = load("breadth/verification.json")
    out: dict[str, float | int | bool] = {}
    checks = [
        check
        for model in verification["new_models"]
        for check in model["checks"].values()
    ]
    out["audit.n_models"] = len(verification["new_models"])
    out["audit.n_checks"] = len(checks)
    out["audit.n_matched"] = sum(1 for check in checks if check["matches"])
    out["audit.max_absolute_difference"] = max(check["absolute_difference"] for check in checks)
    out["audit.tolerance"] = max(check["tolerance"] for check in checks)
    return out


SECTIONS = {
    "confinement": confinement,
    "ceiling_identity": ceiling_identity,
    "entropy_axis": entropy_axis,
    "cone_geometry": cone_geometry,
    "raw_audit": raw_audit,
}


def measure() -> dict:
    measured: dict[str, object] = {}
    for prefix, function in SECTIONS.items():
        for key, value in function().items():
            measured[f"{prefix}.{key}"] = value
    return measured


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", default=str(REPO / "configs/expected_values.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="rewrite the frozen file from the current artifacts (maintenance only)",
    )
    args = parser.parse_args(argv)

    measured = measure()
    expected_path = pathlib.Path(args.expected)

    if args.freeze:
        payload = json.loads(expected_path.read_text()) if expected_path.is_file() else {}
        payload["values"] = {
            key: {
                "value": measured[key],
                "kind": payload.get("values", {}).get(key, {}).get("kind", "recomputed"),
                "source": payload.get("values", {}).get(key, {}).get("source", ""),
            }
            for key in sorted(measured)
        }
        payload["n_values"] = len(measured)
        expected_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"froze {len(measured)} values into {expected_path}")
        return 0

    frozen = json.loads(expected_path.read_text())
    records = []
    failed = 0
    for key in sorted(frozen["values"]):
        record = frozen["values"][key]
        want = record["value"]
        if key not in measured:
            records.append({"key": key, "status": "MISSING", "expected": want})
            failed += 1
            continue
        got = measured[key]
        if isinstance(want, bool) or isinstance(got, bool):
            ok = bool(got) == bool(want)
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=1e-12)
        else:
            ok = got == want
        failed += 0 if ok else 1
        records.append(
            {
                "key": key,
                "status": "ok" if ok else "FAIL",
                "kind": record.get("kind", "recomputed"),
                "expected": want,
                "measured": got,
            }
        )

    extra = sorted(set(measured) - set(frozen["values"]))
    n_recomputed = sum(1 for r in records if r.get("kind") == "recomputed")
    n_reread = len(records) - n_recomputed

    print(f"frozen load-bearing values: {len(records)}")
    print(f"  recomputed from the stored measurements: {n_recomputed}")
    print(f"  re-read from a named pointer:            {n_reread}")
    for record in records:
        if record["status"] != "ok":
            print(f"  {record['status']} {record['key']}: expected {record['expected']!r} "
                  f"measured {record.get('measured')!r}")
    for key in extra:
        print(f"  UNFROZEN {key}: {measured[key]!r}")
    status = "PASS" if failed == 0 and not extra else "FAIL"
    print(f"claim verification: {status} ({len(records) - failed} of {len(records)} matched)")

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "kind": "butterfly_cone_claim_verification",
                    "status": status,
                    "n_values": len(records),
                    "n_recomputed": n_recomputed,
                    "n_reread": n_reread,
                    "n_failed": failed,
                    "unfrozen_keys": extra,
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
