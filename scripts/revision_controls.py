#!/usr/bin/env python3
"""Annealing and kick-scaling controls added in the revised paper.

One parent glass per call.  The parent is taken from the stored bridge ladder
(``runs/sc_energy_ladder/config_ladder.npz``, written by ``sc_energy_ladder.py``;
four independent swap preparations per temperature; not shipped, see
data/manifest.json).
It is thermalised for ``--prep`` time units at the run temperature with a Bussi
thermostat; the same preparation is applied to every parent, so a parent
equilibrated at its own temperature (native) and a parent equilibrated at a
higher temperature and run colder (re-annealed comparison) differ only in
annealing history.

For every momentum seed the script integrates one unperturbed copy and, per
kick site, one copy kicked by the published ``O_shell`` operator (absolute
amplitude ``delta``).  With ``--scaled`` it adds a copy whose per-particle
displacement is proportional to the particle diameter, normalised so that the
shell RMS displacement equals the absolute kick.  Both copies of a pair carry
identical momenta and evolve in NVE.

Writes ``runs/revision/<label>.npz`` with branch-mean divergence fields, the
unperturbed positions needed for the cage amplitude, injected energies and
energy drift.  Analysis is in ``revision_analysis.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from butterfly_cone.engine.integrate import BussiThermostat, MDIntegrator, maxwell_boltzmann_velocities
from butterfly_cone.engine.system import ParticleSystem
from butterfly_cone.events.trajectory import as_box, minimum_image as np_minimum_image
from butterfly_cone.perturb.operators import R_PERT_DEFAULT, o_shell

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "runs/sc_energy_ladder/config_ladder.npz"
OUT_DIR = REPO / "runs/revision"


def load_parent(temperature: str, index: int, ladder: Path = LADDER):
    data = np.load(ladder)
    box = float(data["L"])
    pos = np.remainder(np.asarray(data[f"pos_{temperature}_{index}"], dtype=np.float64), box)
    return pos, np.asarray(data[f"sig_{temperature}_{index}"], dtype=np.float64), box


def build(pos, sig, box, vel):
    p = torch.tensor(np.remainder(pos, box), dtype=torch.float64)
    return ParticleSystem(
        positions=p,
        velocities=vel.clone(),
        diameters=torch.tensor(sig, dtype=torch.float64),
        box=torch.full((3,), box, dtype=torch.float64),
        active_mask=torch.ones(pos.shape[0], dtype=torch.bool),
        unwrapped_positions=p.clone(),
    )


def sites_farthest(pos, box, n):
    chosen = [0]
    dist = np.linalg.norm(np_minimum_image(pos - pos[0], as_box(box)), axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(np_minimum_image(pos - pos[nxt], as_box(box)), axis=1))
    return chosen


def potential_energy(pos, sig, box):
    itg = MDIntegrator(build(pos, sig, box, torch.zeros((pos.shape[0], 3), dtype=torch.float64)))
    return float(itg.potential_energy)


def kicks(pos, sig, box, centre, delta, seed, scaled):
    """Absolute ``O_shell`` displacement and, optionally, its diameter-scaled twin."""

    quiet = build(pos, sig, box, torch.zeros((pos.shape[0], 3), dtype=torch.float64))
    perturbed, prov = o_shell(quiet, centre, delta, R_PERT_DEFAULT, seed=seed)
    disp = np_minimum_image(perturbed.positions.numpy() - pos, as_box(box))
    out = {"abs": (disp, float(prov.delta_u), int(prov.n_perturbed))}
    if scaled:
        moved = np.linalg.norm(disp, axis=1) > 0.0
        # Same Gaussian draws, rescaled per particle by sigma_i / <sigma>_shell,
        # recentred, then normalised to the absolute kick's shell RMS.
        raw = disp.copy()
        raw[moved] *= (sig[moved] / sig[moved].mean())[:, None]
        raw[moved] -= raw[moved].mean(axis=0, keepdims=True)
        rms_abs = math.sqrt(float((disp[moved] ** 2).sum(1).mean()))
        rms_raw = math.sqrt(float((raw[moved] ** 2).sum(1).mean()))
        raw[moved] *= rms_abs / rms_raw
        du = potential_energy(pos + raw, sig, box) - potential_energy(pos, sig, box)
        out["scaled"] = (raw, float(du), int(moved.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-T", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--run-T", type=float, required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sites", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--scaled", action="store_true")
    ap.add_argument("--prep", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ladder", type=Path, default=LADDER)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    t0 = time.time()

    pos0, sig, box = load_parent(a.parent_T, a.rep, a.ladder)
    n = pos0.shape[0]
    e_parent = potential_energy(pos0, sig, box) / n

    # Common preparation: Bussi NVT at the run temperature for a.prep time units.
    gen = torch.Generator().manual_seed(17_000 + a.rep)
    prep = build(pos0, sig, box, maxwell_boltzmann_velocities(n, a.run_T, gen, device="cpu", dtype=torch.float64))
    itg = MDIntegrator(prep, dt=a.dt, thermostat=BussiThermostat(a.run_T, 0.5, gen))
    pe_trace = []
    for _ in range(int(round(a.prep / a.dt)) // 50):
        itg.step(50)
        pe_trace.append(float(itg.potential_energy) / n)
    pos = np.remainder(prep.positions.numpy().copy(), box)
    e_prepared = potential_energy(pos, sig, box) / n

    centres = [pos[i].copy() for i in sites_farthest(pos, box, a.sites)]
    kick_sets = [kicks(pos, sig, box, c, a.delta, 900_001 + 7919 * k, a.scaled) for k, c in enumerate(centres)]
    variants = ["abs"] + (["scaled"] if a.scaled else [])

    n_frames = a.steps // a.stride + 1
    times = np.arange(n_frames) * a.stride * a.dt
    field_sum = {(k, v): np.zeros((n_frames, n)) for k in range(a.sites) for v in variants}
    cage_frames = np.arange(n_frames // 2, n_frames, 4)
    unpert_unwrapped = np.zeros((a.seeds, cage_frames.size, n, 3))
    drift = []
    box_np = as_box(box)

    for s in range(a.seeds):
        g = torch.Generator().manual_seed(4_000_003 + 10_007 * s + 13 * a.rep)
        vel = maxwell_boltzmann_velocities(n, a.run_T, g, device="cpu", dtype=torch.float64)
        systems = {"u": build(pos, sig, box, vel)}
        for k in range(a.sites):
            for v in variants:
                systems[(k, v)] = build(pos + kick_sets[k][v][0], sig, box, vel)
        itgs = {key: MDIntegrator(sysm, dt=a.dt) for key, sysm in systems.items()}
        e0 = {key: float(i.total_energy()) for key, i in itgs.items()}
        ci = 0
        for f in range(n_frames):
            if f > 0:
                for i in itgs.values():
                    i.step(a.stride)
            ref = systems["u"].positions.numpy()
            for k in range(a.sites):
                for v in variants:
                    d = systems[(k, v)].positions.numpy() - ref
                    field_sum[(k, v)][f] += np.linalg.norm(np_minimum_image(d, box_np), axis=1)
            if ci < cage_frames.size and f == cage_frames[ci]:
                unpert_unwrapped[s, ci] = systems["u"].unwrapped_positions.numpy()
                ci += 1
        drift.append(max(abs(float(i.total_energy()) - e0[key]) / abs(e0[key]) for key, i in itgs.items()))
        print(f"seed {s + 1}/{a.seeds} done {time.time() - t0:.0f}s", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"field_{k}_{v}": field_sum[(k, v)] / a.seeds for k in range(a.sites) for v in variants}
    np.savez_compressed(
        a.out_dir / f"{a.label}.npz",
        times=times,
        cage_times=times[cage_frames],
        unpert_unwrapped=unpert_unwrapped,
        positions=pos,
        sigma=sig,
        box=box,
        centres=np.array(centres),
        **arrays,
    )
    meta = {
        "label": a.label,
        "parent_T": float(a.parent_T),
        "rep": a.rep,
        "run_T": a.run_T,
        "seeds": a.seeds,
        "sites": a.sites,
        "delta": a.delta,
        "variants": variants,
        "prep_time": a.prep,
        "pe_parent_per_particle": e_parent,
        "pe_prepared_per_particle": e_prepared,
        "pe_prep_trace": pe_trace,
        "injected_energy": {v: [kick_sets[k][v][1] for k in range(a.sites)] for v in variants},
        "n_kicked": {v: [kick_sets[k][v][2] for k in range(a.sites)] for v in variants},
        "kicked_mean_sigma": [
            float(sig[np.linalg.norm(kick_sets[k]["abs"][0], axis=1) > 0].mean()) for k in range(a.sites)
        ],
        "max_relative_energy_drift": max(drift),
        "wall_seconds": time.time() - t0,
    }
    (a.out_dir / f"{a.label}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("label", "pe_parent_per_particle", "pe_prepared_per_particle",
                                            "injected_energy", "max_relative_energy_drift", "wall_seconds")}))


if __name__ == "__main__":
    main()
