#!/usr/bin/env python3
"""LaTeX tables S1, S3, S4 (estimator block) and S6 of the supplementary material.

Reads the shipped results/revision/summary.json and the three deep-state cone
summaries results/revision/deep_T00xx_gardner_r0.json, and writes s1.tex,
s3.tex, s4b.tex and s6.tex to --out (default runs/revision/si_tables).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/revision"
OUT = REPO / "runs/revision/si_tables"
SC = {0.150: 2.734, 0.130: 2.489, 0.108: 2.234, 0.090: 1.949, 0.075: 1.682}


def spread(x):
    x = np.asarray(x, float)
    return 100.0 * (x.max() - x.min()) / x.mean()


def table_s1(rows):
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"$T$ & parent & $e_{\mathrm{pot}}$ & $\lambdab$ & $\Dsat/N$ & $\vb$ (1 t.u.) & $\vb$ (0.25 t.u.) \\", r"\midrule"]
    native = [r for r in rows if r["run_T"] == r["parent_T"]]
    for T in sorted({r["run_T"] for r in native}, reverse=True):
        grp = sorted([r for r in native if r["run_T"] == T], key=lambda r: r["rep"])
        lam, dsat = [], []
        for r in grp:
            a1, a2 = r["fits"]["abs"]["aggregate"]["cad1"], r["fits"]["abs"]["aggregate"]["cad025"]
            lam.append(a1["lambda_mean"]); dsat.append(a1["D_sat_per_N"])
            lines.append(f"{T:.3f} & {r['rep']} & {r['pe_parent_per_particle']:.4f} & "
                         f"${a1['lambda_mean']:.3f}\\pm{a1['lambda_sem']:.3f}$ & {a1['D_sat_per_N']:.4f} & "
                         f"{a1['v_b_mean']:.2f} & {a2['v_b_mean']:.2f} \\\\")
        lines.append(f"\\multicolumn{{3}}{{r}}{{spread}} & {spread(lam):.1f}\\% & {spread(dsat):.1f}\\% & & \\\\")
        lines.append(r"\midrule")
    deep = {0.067: "0067", 0.060: "0060", 0.055: "0055"}
    for T, tag in deep.items():
        g = json.loads((RESULTS / f"deep_T{tag}_gardner_r0.json").read_text())
        lam = g["across_config"]["lambda"]["values"]
        ds = np.array(g["across_config"]["D_sat"]["values"]) / g["ensembles"][0]["N"]
        for k, (l, d) in enumerate(zip(lam, ds)):
            lines.append(f"{T:.3f} & {k} & -- & {l:.3f} & {d:.4f} & -- & -- \\\\")
        lines.append(f"\\multicolumn{{3}}{{r}}{{spread}} & {spread(lam):.1f}\\% & {spread(ds):.1f}\\% & & \\\\")
        lines.append(r"\midrule" if T != 0.055 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_s3(rows):
    lines = [r"\begin{tabular}{ccccccc}", r"\toprule",
             r"parent & $\Delta U_{\mathrm{abs}}$ & $\Delta U_{\mathrm{scaled}}$ & $\lambdab^{\mathrm{abs}}$ & $\lambdab^{\mathrm{scaled}}$ & $\Dsat/N^{\mathrm{abs}}$ & $\Dsat/N^{\mathrm{scaled}}$ \\", r"\midrule"]
    for r in sorted([r for r in rows if "scaled" in r["fits"]], key=lambda r: r["rep"]):
        a, s = r["fits"]["abs"]["aggregate"]["cad1"], r["fits"]["scaled"]["aggregate"]["cad1"]
        lines.append(f"{r['rep']} & {np.mean(r['injected_energy']['abs']):.3f} & {np.mean(r['injected_energy']['scaled']):.3f} & "
                     f"{a['lambda_mean']:.3f} & {s['lambda_mean']:.3f} & {a['D_sat_per_N']:.4f} & {s['D_sat_per_N']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_s4b(rows):
    lines = [r"\begin{tabular}{ccccccc}", r"\toprule",
             r"$T$ & parents & branch variance & $\sqrt{\mathrm{MSD}/2}$ & scattering & $q_{\mathrm{peak}}$ & $\Delta\langle u^2\rangle$ \\", r"\midrule"]
    native = [r for r in rows if r["run_T"] == r["parent_T"]]
    for T in sorted({r["run_T"] for r in native}, reverse=True):
        grp = [r["cage"] for r in native if r["run_T"] == T]
        m = lambda k: np.mean([g[k] for g in grp])
        lines.append(f"{T:.3f} & {len(grp)} & {m('u_branch_variance'):.4f} & {m('u_sqrt_msd_half'):.4f} & "
                     f"{m('u_scattering'):.4f} & {m('q_peak'):.2f} & ${m('u2_late_growth_percent'):+.0f}\\%$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_s6(rows):
    lines = [r"\begin{tabular}{cccccccc}", r"\toprule",
             r"$T_{\mathrm f}$ & $\scf(T_{\mathrm f})$ & parents & $e_{\mathrm{pot}}$ before & $e_{\mathrm{pot}}$ after & $\lambdab$ & $\Dsat/N$ & $\uDW$ \\", r"\midrule"]
    run = [r for r in rows if abs(r["run_T"] - 0.075) < 1e-9]
    for tf in sorted({round(r["parent_T"], 3) for r in run}, reverse=True):
        grp = [r for r in run if round(r["parent_T"], 3) == tf]
        lam = [r["fits"]["abs"]["aggregate"]["cad1"]["lambda_mean"] for r in grp]
        ds = [r["fits"]["abs"]["aggregate"]["cad1"]["D_sat_per_N"] for r in grp]
        u = [r["cage"]["u_branch_variance"] for r in grp]
        se = np.std(lam, ddof=1) / np.sqrt(len(lam)) if len(lam) > 1 else float("nan")
        lines.append(f"{tf:.3f} & {SC[tf]:.3f} & {len(grp)} & {np.mean([r['pe_parent_per_particle'] for r in grp]):.4f} & "
                     f"{np.mean([r['pe_prepared_per_particle'] for r in grp]):.4f} & ${np.mean(lam):.3f}\\pm{se:.3f}$ & "
                     f"{np.mean(ds):.4f} & {np.mean(u):.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=RESULTS / "summary.json")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = json.loads(a.summary.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("s1", table_s1), ("s3", table_s3), ("s4b", table_s4b), ("s6", table_s6)):
        (a.out / f"{name}.tex").write_text(fn(rows) + "\n")
        print(f"== {name}\n" + fn(rows))


if __name__ == "__main__":
    main()
