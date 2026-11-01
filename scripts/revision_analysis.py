#!/usr/bin/env python3
"""Analyse ``runs/revision/*.npz`` with the published estimators.

For every parent run: growth rate, fit window, residuals and plateau from
``fit_lyapunov`` at the published 1 t.u. cadence and at the 0.25 t.u. bridge
cadence; front speed from ``butterfly_velocity`` at both cadences; the cage
amplitude three ways on the same unperturbed trajectories (cross-branch variance
about the cage centre, sqrt(MSD/2) relative to the start, and the plateau of
F_s(q) at the structure-factor peak, inverted with the plateau MSD equal to
2<u^2>); the late-window growth of <u^2>; injected energies.

Reads the ``.npz``/``.json`` pairs that ``revision_controls.py`` writes to
``runs/revision/`` (the ``.npz`` files are not shipped; data/manifest.json lists
them with sizes and checksums) and writes ``runs/revision/summary.json``. The
shipped copy of that summary is ``results/revision/summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from butterfly_cone.perturb.butterfly import butterfly_velocity, fit_lyapunov

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "runs/revision"
C_STAR = 4.0 / np.sqrt(3.0 * np.pi)


def structure_factor_peak(pos, box, q_lo=4.0, q_hi=11.0, n_max=9):
    grid = np.array([(h, k, l) for h in range(n_max) for k in range(n_max) for l in range(n_max) if h or k or l])
    vec = 2.0 * np.pi / box * grid
    mag = np.linalg.norm(vec, axis=1)
    keep = (mag > q_lo) & (mag < q_hi)
    vec, mag = vec[keep], mag[keep]
    intensity = np.abs(np.exp(1j * (pos % box) @ vec.T).sum(0)) ** 2 / pos.shape[0]
    edges = np.arange(q_lo, q_hi, 0.25)
    which = np.digitize(mag, edges)
    prof = np.array([(edges[i - 1], intensity[which == i].mean()) for i in range(1, len(edges)) if (which == i).sum() > 2])
    return float(prof[prof[:, 1].argmax(), 0]) + 0.125


def fit_record(t, d, n):
    fit = fit_lyapunov(t, d, divergence_power=1.0)
    rec = {"lambda": float(fit.lam), "r2": float(fit.r2), "D_sat_per_N": float(fit.D_sat) / n,
           "growing": bool(fit.onset.growing)}
    if fit.onset.growing and fit.n_fit >= 2:
        i0 = fit.onset.start_index
        i1 = i0 + fit.n_fit
        x, y = t[i0:i1], np.log(d[i0:i1])
        slope, icpt = np.polyfit(x, y, 1)
        rec.update({"t_start": float(x[0]), "t_end": float(x[-1]), "t_onset": float(fit.onset.time),
                    "n_fit": int(fit.n_fit), "residuals": (y - (slope * x + icpt)).tolist(),
                    "fit_times": x.tolist(), "slope_se": float(_slope_se(x, y))})
    return rec


def _slope_se(x, y):
    if x.size < 3:
        return float("nan")
    p = np.polyfit(x, y, 1)
    res = y - np.polyval(p, x)
    return np.sqrt((res**2).sum() / (x.size - 2) / ((x - x.mean()) ** 2).sum())


def cage_three_ways(z, q):
    unw = z["unpert_unwrapped"]            # (seeds, frames, N, 3), latter half of the run
    start = z["positions"]                 # common start of every unperturbed branch
    centre = unw.mean(axis=0, keepdims=True)
    var = ((unw - centre) ** 2).sum(-1)    # per seed, frame, particle
    s = unw.shape[0]
    u2_frame = var.sum(0).mean(-1) / (s - 1)          # unbiased, averaged over particles
    u2_branch = float(u2_frame.mean())
    disp = unw - start[None, None]
    msd_frame = (disp**2).sum(-1).mean(axis=(0, 2))
    r = np.linalg.norm(disp, axis=-1)
    fs_frame = np.where(r > 0, np.sin(q * r) / (q * r), 1.0).mean(axis=(0, 2))
    msd_scatt = -6.0 * np.log(fs_frame.mean()) / q**2   # plateau MSD = 2 <u^2>
    return {
        "u_branch_variance": float(np.sqrt(u2_branch)),
        "u_sqrt_msd_half": float(np.sqrt(msd_frame.mean() / 2.0)),
        "u_scattering": float(np.sqrt(msd_scatt / 2.0)),
        "q_peak": q,
        "u2_late_growth_percent": float(100.0 * (u2_frame[-1] / u2_frame[0] - 1.0)),
        "u2_by_frame": u2_frame.tolist(),
        "cage_times": z["cage_times"].tolist(),
    }


def analyse(path: Path):
    z = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    t = z["times"]
    pos, box = z["positions"], float(z["box"])
    n = pos.shape[0]
    out = {**{k: meta[k] for k in ("label", "parent_T", "rep", "run_T", "seeds", "sites", "delta",
                                   "pe_parent_per_particle", "pe_prepared_per_particle",
                                   "injected_energy", "n_kicked", "kicked_mean_sigma",
                                   "max_relative_energy_drift")}}
    out["cage"] = cage_three_ways(z, structure_factor_peak(pos, box))
    out["fits"] = {}
    for v in meta["variants"]:
        per_site = []
        for k in range(meta["sites"]):
            field = z[f"field_{k}_{v}"]
            d = field.sum(1)
            rec = {"D_t": d.tolist()}
            for label, step in (("cad1", 4), ("cad025", 1)):
                rec[label] = fit_record(t[::step], d[::step], n)
                vel = butterfly_velocity(t[::step], field[::step], pos, z["centres"][k], box)
                rec[label]["v_b"] = float(vel.v_b)
            per_site.append(rec)
        agg = {}
        for label in ("cad1", "cad025"):
            lam = np.array([s[label]["lambda"] for s in per_site])
            dsat = np.array([s[label]["D_sat_per_N"] for s in per_site])
            vb = np.array([s[label]["v_b"] for s in per_site])
            vb = vb[vb > 0]
            agg[label] = {"lambda_mean": float(lam.mean()), "lambda_sem": float(lam.std(ddof=1) / np.sqrt(lam.size)),
                          "D_sat_per_N": float(dsat.mean()),
                          "c": float(dsat.mean() / out["cage"]["u_branch_variance"]),
                          "v_b_mean": float(vb.mean()) if vb.size else float("nan"),
                          "v_b_detected": int(vb.size)}
        out["fits"][v] = {"per_site": per_site, "aggregate": agg}
    out["times"] = t.tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/summary.json")
    a = ap.parse_args()
    rows = [analyse(p) for p in sorted(a.run_dir.glob("*.npz"))]
    (a.out or a.run_dir / "summary.json").write_text(json.dumps(rows))
    print(f"{'label':32s} {'lam1':>6s} {'lam.25':>6s} {'Dsat/N':>7s} {'u_var':>6s} {'u_msd':>6s} {'u_sc':>6s} {'c':>6s} {'du2%':>6s} {'vb1':>5s} {'vb.25':>5s} {'dU':>5s}")
    for r in rows:
        for v, f in r["fits"].items():
            a1, a2 = f["aggregate"]["cad1"], f["aggregate"]["cad025"]
            cg = r["cage"]
            print(f"{r['label'] + ':' + v:32s} {a1['lambda_mean']:6.3f} {a2['lambda_mean']:6.3f} {a1['D_sat_per_N']:7.4f} "
                  f"{cg['u_branch_variance']:6.4f} {cg['u_sqrt_msd_half']:6.4f} {cg['u_scattering']:6.4f} {a1['c']:6.3f} "
                  f"{cg['u2_late_growth_percent']:6.1f} {a1['v_b_mean']:5.2f} {a2['v_b_mean']:5.2f} {np.mean(r['injected_energy'][v]):5.3f}")


if __name__ == "__main__":
    main()
