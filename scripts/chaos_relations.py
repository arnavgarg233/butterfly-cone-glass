#!/usr/bin/env python3
"""Lyapunov--KS / chaos-observable relations of the butterfly cone.

Pure re-analysis of already-landed cone artifacts (no MD).  The cone gives
three chaos observables per branch ensemble (``gardner_r0.json``):

* ``lam``    -- pre-saturation Lyapunov rate of log D(t)          [1 / t.u.]
* ``v_b``    -- ballistic front speed of the divergence field     [sigma / t.u.]
* ``D_sat``  -- Debye--Waller-locked saturation plateau           [sigma]
  (plus ``D0`` = t->0 intercept and ``onset_time`` = detected saturation onset)

This module asks what *relations* tie them together, in the spirit of the
velocity-dependent-Lyapunov-exponent (VDLE) framework for classical many-body
chaos [Khemani, Huse & Nahum, PRB 98, 144304 (2018); Das et al., PRL 121,
024101 (2018); Ruidas & Banerjee, arXiv:1906.00016 for liquids]:

(a) **Ballistic relation / chaos length.**  The light cone lambda(v_b) = 0
    defines an emergent length ``ell_c = v_b / lambda`` -- the distance the
    chaos front advances per Lyapunov e-fold (the classical analogue of the
    holographic chaos length v_B / lambda_L [Blake, PRL 117, 091601 (2016)]).
    We measure ell_c per ensemble at the linear-response kick (smallest
    delta), test its N-intensivity, and compare it against every landed
    static length in the project (interparticle spacing, potential cutoff,
    xi_PTS window, DW amplitude, cage scale, plateau length).

(b) **Pesin-type bound.**  For an SRB measure Pesin's identity gives
    h_KS = sum of positive Lyapunov exponents [Pesin, Russ. Math. Surv. 32,
    55 (1977)], hence with only the *maximal* exponent measured the honest
    statement is a lower bound h_KS >= lambda_max.  We state the bound in
    nats and bits per time unit and check its consistency with the landed
    memory-ledger erasure rate (runs/memory_ledger/memory_ledger.json).

(c) **Single-exponent closure of the cone triad.**  If a single exponent
    carries the divergence from the kick to the DW plateau,

        lambda * t_sat = ln(D_sat / D0),

    i.e. measure any two of (lambda, D_sat/D0, t_sat) and the third is fixed.
    We test this per ensemble (t_pred = ln(D_sat/D0)/lambda vs the detected
    onset time), across both landed campaigns and both system sizes.

All numbers come from on-disk artifacts; the script writes
``runs/chaos_relations/chaos_relations.json`` (+ ``.md``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Landed constants (cited artifacts / source, not free parameters)
# ---------------------------------------------------------------------------

#: Number density rho = N / L^3 of the landed Gardner campaigns.  Verified from
#: runs/gardner/gardner-T0075-fss--c0-unpert/parent_state.pt:
#: box = 11.4471^3, N = 1500  ->  rho = 1.0000 (and mean diameter <sigma> = 1.0).
DENSITY = 1.0

#: Mean particle diameter of the polydisperse mixture (same parent_state.pt).
SIGMA_MEAN = 1.0

#: Pair-potential cutoff ratio (src/butterfly_cone/engine/potential.py::CUTOFF_RATIO).
CUTOFF_RATIO = 1.25

#: Cage / overlap resolution used across the project
#: (src/butterfly_cone/instruments/pinning.py::overlap_cutoff and the memory ledger).
CAGE_RESOLUTION = 0.3

#: Landed xi_PTS crossover window [sigma] at T = 0.13 (cavity-free floor).
XI_PTS_WINDOW = (0.7, 1.0)


# ---------------------------------------------------------------------------
# Small numerics helpers (stdlib + numpy only)
# ---------------------------------------------------------------------------


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN when either side is degenerate."""

    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    if ax.size != ay.size:
        raise ValueError("x and y must have equal length")
    if ax.size < 2 or float(ax.std()) == 0.0 or float(ay.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_median_ci(
    values: Sequence[float], *, n_boot: int = 4000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Deterministic bootstrap CI for the median."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medians = np.median(arr[idx], axis=1)
    lo = (1.0 - level) / 2.0
    return (
        float(np.quantile(medians, lo)),
        float(np.quantile(medians, 1.0 - lo)),
    )


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


def load_records(summary_paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load resolved per-ensemble cone records from gardner_r0.json files."""

    records: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        run_name = str(payload.get("run_dir", path))
        for ens in payload.get("ensembles", []):
            lam = ens.get("lam")
            if not ens.get("resolved") or lam is None or not (lam > 0.0):
                continue
            required = ("v_b", "D_sat", "D0", "onset_time", "delta", "N")
            if any(ens.get(key) is None for key in required):
                continue
            rec = {key: ens[key] for key in required}
            rec["lam"] = float(lam)
            rec["t_shield"] = ens.get("t_shield")
            rec["label"] = ens.get("label", "?")
            rec["run"] = run_name
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# (a) Ballistic relation: the chaos length ell_c = v_b / lambda
# ---------------------------------------------------------------------------


def chaos_length_per_record(records: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([r["v_b"] / r["lam"] for r in records], dtype=np.float64)


def candidate_length_table(ell_c_median: float) -> dict[str, dict[str, float]]:
    """Compare ell_c against every landed static length (ratios ell_c / L)."""

    interparticle = DENSITY ** (-1.0 / 3.0)
    candidates = {
        "interparticle_spacing_rho^-1/3": interparticle,
        "potential_cutoff_1.25<sigma>": CUTOFF_RATIO * SIGMA_MEAN,
        "xi_PTS_upper_T0.13": XI_PTS_WINDOW[1],
        "xi_PTS_lower_T0.13": XI_PTS_WINDOW[0],
        "cage_resolution_a": CAGE_RESOLUTION,
    }
    return {
        name: {"length_sigma": float(val), "ell_c_over_length": float(ell_c_median / val)}
        for name, val in candidates.items()
    }


def ballistic_analysis(
    records: Sequence[dict[str, Any]],
    *,
    u_dw: float | None = None,
    d_sat_over_n: float | None = None,
) -> dict[str, Any]:
    """Chaos length ell_c = v_b / lambda: per-delta ladder, linear-response
    value, N-intensivity, and comparison with landed static lengths."""

    deltas = sorted({r["delta"] for r in records})
    per_delta: list[dict[str, Any]] = []
    for delta in deltas:
        sub = [r for r in records if r["delta"] == delta]
        lam = [r["lam"] for r in sub]
        v_b = [r["v_b"] for r in sub]
        ell = chaos_length_per_record(sub)
        per_delta.append(
            {
                "delta": float(delta),
                "n": len(sub),
                "lambda_median": float(np.median(lam)),
                "v_b_median": float(np.median(v_b)),
                "ell_c_median": float(np.median(ell)),
                "pearson_lambda_vb": pearson(lam, v_b),
            }
        )

    # Linear response = smallest kick (the landed lambda convention,
    # scripts/memory_ledger.py::lambda_variants).
    delta_lr = deltas[0]
    lr = [r for r in records if r["delta"] == delta_lr]
    ell_lr = chaos_length_per_record(lr)
    ci_lo, ci_hi = bootstrap_median_ci(ell_lr)

    by_n = {}
    for n_particles in sorted({r["N"] for r in lr}):
        sub_ell = chaos_length_per_record([r for r in lr if r["N"] == n_particles])
        by_n[str(n_particles)] = {
            "n": int(sub_ell.size),
            "ell_c_median": float(np.median(sub_ell)),
        }
    n_values = [v["ell_c_median"] for v in by_n.values()]
    rel_spread = (
        (max(n_values) - min(n_values)) / float(np.mean(n_values)) if len(n_values) > 1 else 0.0
    )

    ell_median = float(np.median(ell_lr))
    table = candidate_length_table(ell_median)
    if u_dw is not None:
        table["u_DW_debye_waller"] = {
            "length_sigma": float(u_dw),
            "ell_c_over_length": float(ell_median / u_dw),
        }
    if d_sat_over_n is not None:
        plateau_len = float(d_sat_over_n)  # D_sat/N *is* a length (first-power norm)
        table["plateau_length_D_sat_over_N"] = {
            "length_sigma": plateau_len,
            "ell_c_over_length": float(ell_median / plateau_len),
        }

    # Does the naive proportionality v_b = lambda * const hold across ensemble
    # scatter?  (It should NOT if ell_c is delta-limited by fit nonlinearity.)
    pooled_pearson = pearson([r["lam"] for r in records], [r["v_b"] for r in records])

    closest = min(table.items(), key=lambda kv: abs(math.log(kv[1]["ell_c_over_length"])))
    return {
        "definition": "ell_c = v_b / lambda (front advance per Lyapunov e-fold)",
        "per_delta": per_delta,
        "linear_response": {
            "delta": float(delta_lr),
            "ell_c_median": ell_median,
            "ell_c_median_ci95": [ci_lo, ci_hi],
            "ell_c_stats": summarize(ell_lr),
            "by_N": by_n,
            "intensive_rel_spread": float(rel_spread),
            "intensive": bool(rel_spread <= 0.2),
        },
        "pooled_pearson_lambda_vb": pooled_pearson,
        "candidate_lengths": table,
        "closest_candidate": closest[0],
        "verdict": {
            "v_b_equals_lambda_times_static_length": bool(
                abs(math.log(closest[1]["ell_c_over_length"])) < math.log(1.2)
            ),
            "ell_c_is_emergent_supra_static": bool(
                ell_median
                > 1.2 * max(v["length_sigma"] for v in candidate_length_table(1.0).values())
            ),
        },
    }


# ---------------------------------------------------------------------------
# (b) Pesin-type bound on the KS entropy rate
# ---------------------------------------------------------------------------


def pesin_bound(
    lambda_max: float,
    *,
    ledger_bits_per_tu: float | None = None,
    scramble_nats: float | None = None,
) -> dict[str, Any]:
    """Honest Pesin statement given only lambda_max.

    Pesin (SRB): h_KS = sum_{lambda_i > 0} lambda_i  >=  lambda_max.
    Only the maximal exponent is measured, so h_KS >= lambda_max is the
    strongest defensible statement (a *floor*, matching the memory-ledger
    caveat).  Bits use log2.
    """

    if not (lambda_max > 0.0):
        raise ValueError("lambda_max must be positive")
    bits = lambda_max / math.log(2.0)
    out: dict[str, Any] = {
        "statement": "h_KS >= lambda_max (Pesin identity, lower bound: only lambda_max measured)",
        "lambda_max_nats_per_tu": float(lambda_max),
        "h_KS_lower_bound_nats_per_tu": float(lambda_max),
        "h_KS_lower_bound_bits_per_tu": float(bits),
    }
    if ledger_bits_per_tu is not None:
        out["memory_ledger_bits_per_tu"] = float(ledger_bits_per_tu)
        out["consistent_with_memory_ledger"] = bool(
            math.isclose(bits, ledger_bits_per_tu, rel_tol=1e-9)
        )
        out["note"] = (
            "The ledger's erasure rate is lambda_max/ln2 by construction "
            "(scripts/memory_ledger.py); agreement is a convention consistency "
            "check, not an independent measurement."
        )
    if scramble_nats is not None:
        out["scrambling_budget_per_direction"] = {
            "nats": float(scramble_nats),
            "bits": float(scramble_nats / math.log(2.0)),
            "definition": "median ln(D_sat/D0) = lambda * t_sat (entropy e-folds "
            "spent between kick and DW plateau, per unstable direction)",
        }
    return out


# ---------------------------------------------------------------------------
# (c) Single-exponent closure: lambda * t_sat = ln(D_sat / D0)
# ---------------------------------------------------------------------------


def closure_time_predicted(record: dict[str, Any]) -> float:
    return math.log(record["D_sat"] / record["D0"]) / record["lam"]


def closure_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Test lambda * t_sat = ln(D_sat/D0) per ensemble and per campaign."""

    t_pred = np.asarray([closure_time_predicted(r) for r in records], dtype=np.float64)
    t_meas = np.asarray([r["onset_time"] for r in records], dtype=np.float64)
    ratio = t_pred / t_meas

    per_run = {}
    for run in sorted({r["run"] for r in records}):
        sub = [r for r in records if r["run"] == run]
        sp = np.asarray([closure_time_predicted(r) for r in sub])
        sm = np.asarray([r["onset_time"] for r in sub])
        per_run[run] = {
            "n": len(sub),
            "pearson": pearson(sp, sm),
            "ratio_median": float(np.median(sp / sm)),
        }

    r_all = pearson(t_pred, t_meas)
    ratio_median = float(np.median(ratio))
    holds = bool(r_all >= 0.9 and abs(math.log(ratio_median)) <= math.log(1.1))
    result: dict[str, Any] = {
        "identity": "lambda * t_sat = ln(D_sat / D0)",
        "n": len(records),
        "pearson_t_pred_vs_onset": r_all,
        "ratio_t_pred_over_onset": summarize(ratio),
        "ratio_median_ci95": list(bootstrap_median_ci(ratio.tolist(), seed=1)),
        "per_run": per_run,
        "holds": holds,
    }
    shield = [
        (closure_time_predicted(r), r["t_shield"]) for r in records if r.get("t_shield") is not None
    ]
    if shield:
        sp = np.asarray([s[0] for s in shield])
        ss = np.asarray([s[1] for s in shield])
        result["vs_t_shield"] = {
            "pearson": pearson(sp, ss),
            "ratio_median": float(np.median(sp / ss)),
        }
    return result


def scrambling_budget_nats(records: Sequence[dict[str, Any]]) -> float:
    """Median total log-expansion ln(D_sat/D0) across ensembles [nats]."""

    return float(np.median([math.log(r["D_sat"] / r["D0"]) for r in records]))


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    records: Sequence[dict[str, Any]],
    *,
    lambda_max: float,
    ledger_bits_per_tu: float | None,
    u_dw: float | None,
    d_sat_over_n: float | None,
    sources: dict[str, str],
) -> dict[str, Any]:
    if not records:
        raise ValueError("no resolved cone ensembles found")
    lr_records = [r for r in records if r["delta"] == min(x["delta"] for x in records)]
    report = {
        "schema_version": 1,
        "n_records": len(records),
        "runs": sorted({r["run"] for r in records}),
        "sources": sources,
        "ballistic": ballistic_analysis(records, u_dw=u_dw, d_sat_over_n=d_sat_over_n),
        "pesin": pesin_bound(
            lambda_max,
            ledger_bits_per_tu=ledger_bits_per_tu,
            scramble_nats=scrambling_budget_nats(lr_records),
        ),
        "closure": closure_analysis(records),
        "references": [
            "Pesin, Russ. Math. Surveys 32, 55 (1977) [Pesin identity h_KS = sum lambda_i^+]",
            "Khemani, Huse & Nahum, PRB 98, 144304 (2018) [VDLE lambda(v), lambda(v_b)=0]",
            "Das, Chakrabarty, Dhar, Kundu, Huse, Moessner, Ray & Bhattacharjee, "
            "PRL 121, 024101 (2018) [classical butterfly cone: lambda + v_b]",
            "Ruidas & Banerjee, arXiv:1906.00016 [many-body chaos (lambda, v_b) in liquids]",
            "Blake, PRL 117, 091601 (2016) [chaos length v_B/lambda_L]",
        ],
    }
    report["headline"] = {
        "ell_c_sigma": report["ballistic"]["linear_response"]["ell_c_median"],
        "ell_c_intensive": report["ballistic"]["linear_response"]["intensive"],
        "h_KS_floor_nats_per_tu": report["pesin"]["h_KS_lower_bound_nats_per_tu"],
        "h_KS_floor_bits_per_tu": report["pesin"]["h_KS_lower_bound_bits_per_tu"],
        "closure_holds": report["closure"]["holds"],
        "closure_pearson": report["closure"]["pearson_t_pred_vs_onset"],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    b = report["ballistic"]
    p = report["pesin"]
    c = report["closure"]
    lr = b["linear_response"]
    lines = [
        "# Chaos relations of the butterfly cone",
        "",
        f"Records: {report['n_records']} resolved ensembles from {', '.join(report['runs'])}.",
        "",
        "## (a) Ballistic relation: chaos length ell_c = v_b / lambda",
        "",
        f"- linear-response (delta={lr['delta']}): **ell_c = {lr['ell_c_median']:.2f} sigma** "
        f"(95% CI [{lr['ell_c_median_ci95'][0]:.2f}, {lr['ell_c_median_ci95'][1]:.2f}], "
        f"n={lr['ell_c_stats']['n']})",
        f"- N-intensive: {lr['intensive']} (rel. spread {lr['intensive_rel_spread']:.3f} across "
        f"N = {', '.join(lr['by_N'])})",
        f"- pooled pearson(lambda, v_b) = {b['pooled_pearson_lambda_vb']:.3f} "
        "(no naive proportionality across scatter; ell_c is delta-ladder limited)",
        "",
        "| candidate length | value [sigma] | ell_c / length |",
        "|---|---|---|",
    ]
    for name, row in b["candidate_lengths"].items():
        lines.append(f"| {name} | {row['length_sigma']:.3f} | {row['ell_c_over_length']:.2f} |")
    lines += [
        "",
        f"Closest candidate: {b['closest_candidate']}; "
        f"identity with a static length: {b['verdict']['v_b_equals_lambda_times_static_length']}; "
        f"ell_c exceeds every landed static length: "
        f"{b['verdict']['ell_c_is_emergent_supra_static']}.",
        "",
        "## (b) Pesin-type bound",
        "",
        f"- {p['statement']}",
        f"- h_KS >= {p['h_KS_lower_bound_nats_per_tu']:.4f} nats/t.u. = "
        f"{p['h_KS_lower_bound_bits_per_tu']:.4f} bits/t.u.",
    ]
    if "scrambling_budget_per_direction" in p:
        s = p["scrambling_budget_per_direction"]
        lines.append(
            f"- scrambling budget per unstable direction: {s['nats']:.2f} nats = "
            f"{s['bits']:.2f} bits (median ln(D_sat/D0), linear response)"
        )
    lines += [
        "",
        "## (c) Single-exponent closure: lambda * t_sat = ln(D_sat/D0)",
        "",
        f"- pearson(t_pred, t_onset) = {c['pearson_t_pred_vs_onset']:.3f} over n={c['n']}",
        f"- median t_pred/t_onset = {c['ratio_t_pred_over_onset']['median']:.3f} "
        f"(95% CI [{c['ratio_median_ci95'][0]:.3f}, {c['ratio_median_ci95'][1]:.3f}])",
        f"- holds: **{c['holds']}**",
        "",
        "## References",
        "",
    ]
    lines += [f"- {ref}" for ref in report["references"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        type=Path,
        nargs="+",
        default=[
            Path("runs/gardner/gardner-T0075-fss/gardner_r0.json"),
            Path("runs/gardner/gardner-T0075-m2/gardner_r0.json"),
        ],
    )
    parser.add_argument(
        "--memory-ledger", type=Path, default=Path("runs/memory_ledger/memory_ledger.json")
    )
    parser.add_argument(
        "--dw-identity", type=Path, default=Path("runs/dw_identity/dw_identity.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/chaos_relations/chaos_relations.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("runs/chaos_relations/chaos_relations.md")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.summaries)

    ledger = json.loads(args.memory_ledger.read_text(encoding="utf-8"))
    lambda_max = float(ledger["headline"]["lambda_max"])
    ledger_bits = float(ledger["headline"]["bits_erased_per_time_per_direction"])

    u_dw = None
    d_sat_over_n = None
    if args.dw_identity.exists():
        dw = json.loads(args.dw_identity.read_text(encoding="utf-8"))
        u_dw = float(dw["u_DW"])
        d_sat_over_n = float(dw["landed_D_sat_over_N"])

    sources = {
        "cone_summaries": [str(p) for p in args.summaries],
        "memory_ledger": str(args.memory_ledger),
        "dw_identity": str(args.dw_identity),
        "density_provenance": "runs/gardner/gardner-T0075-fss--c0-unpert/parent_state.pt "
        "(box=11.4471^3, N=1500 -> rho=1.0000, <sigma>=1.0)",
    }
    report = build_report(
        records,
        lambda_max=lambda_max,
        ledger_bits_per_tu=ledger_bits,
        u_dw=u_dw,
        d_sat_over_n=d_sat_over_n,
        sources=sources,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
