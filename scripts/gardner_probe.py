#!/usr/bin/env python3
"""Causal-Gardner quench-perturb-branch runner (ButterflyCone Wave-18, Task P).

Loads inherited deep configuration(s) through the pilot loader (positions AND the
archived diameters), builds one unperturbed branch ensemble per config and a
matched-seed perturbed ensemble for every (site, delta), then emits the four
discrimination axes (participation ratio, susceptibility, chaos length,
non-self-averaging R_D(N)) and the declared in advance marginal-vs-defect verdict.

The matched-seed counterfactual is guaranteed by sharing ``project_salt`` and
``momentum_seed_domain`` across every ensemble: harness seeds are run_id
independent, so branch k gets bitwise-identical initial velocities in every
ensemble and at delta=0 the perturbed and unperturbed branches coincide exactly.

No new MD, no RCCE: orchestration over ``branching.run_branch_ensemble`` and the
``butterfly_cone.perturb`` analysis layer.  Runs cpu/float64 for certification and
mps/float32 for production.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from butterfly_cone.branching.batched import (
    BatchedMDIntegrator,
    BatchedSystem,
    branch_maxwell_boltzmann_velocities,
)
from butterfly_cone.branching.ensemble import (
    BranchEnsembleResult,
    _capture_steps,
    run_branch_ensemble,
    torch_seed,
)
from butterfly_cone.engine.system import ParticleSystem, make_generator
from butterfly_cone.harness.config import ExperimentConfig
from butterfly_cone.harness.runs import RunManager
from butterfly_cone.instruments.pinning import random_pin
from butterfly_cone.perturb.operators import (
    R_PERT_DEFAULT,
    o_kick,
    o_shell,
    o_strain,
    stratified_sites,
)
from butterfly_cone.perturb.response import (
    AxisSummary,
    DecisionThresholds,
    EnsembleTrajectory,
    chaos_length,
    decide,
    divergence_field,
    non_self_averaging,
    participation_ratio,
    susceptibility,
    total_divergence,
)
from butterfly_cone.pilot.loader import load_inherited_snapshot

DEFAULT_DELTAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class ConfigSpec:
    path: Path
    temperature: float
    replica: int
    n_particles: int


@dataclass(frozen=True)
class ProbeOptions:
    configs: tuple[ConfigSpec, ...]
    operator: str = "O_shell"
    n_sites: int = 8
    deltas: tuple[float, ...] = DEFAULT_DELTAS
    branches: int = 64
    horizon: int = 3000
    stride: int = 100
    dt: float = 0.01
    r_pert: float = R_PERT_DEFAULT
    thermostat_tau: float | None = None
    pin_fraction: float = 0.0
    device: str = "cpu"
    dtype: str = "float64"
    root: Path = ROOT
    run_id: str | None = None
    project_salt: str = "butterfly_cone"


def _json_safe(value):
    """Recursively replace non-finite floats with None for allow_nan=False writes."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return device


def _ensemble(
    parent, run_id: str, options: ProbeOptions, device: str, temperature: float
) -> BranchEnsembleResult:
    run = RunManager.create(
        phase="gardner",
        config=ExperimentConfig(phase="gardner", values={"kind": "branch_ensemble", "run_id": run_id}),
        root=options.root,
        run_id=run_id,
        device=f"{device}/{options.dtype}",
        project_salt=options.project_salt,
    )
    return run_branch_ensemble(
        parent,
        count=options.branches,
        temperature=temperature,
        horizon=options.horizon,
        run=run,
        dt=options.dt,
        stride=options.stride,
        thermostat_tau=options.thermostat_tau,
        momentum_seed_domain="branching.momentum",
    )


def _perturb(system, center, delta, options: ProbeOptions, seed: int):
    if options.operator == "O_kick":
        return o_kick(system, center, delta, seed=seed)
    if options.operator == "O_strain":
        return o_strain(system, center, delta, options.r_pert)
    return o_shell(system, center, delta, options.r_pert, seed=seed)


def _config_tag(index: int) -> str:
    return f"c{index}"


def _apply_pin_fraction(
    parent: ParticleSystem, options: ProbeOptions, manager: RunManager, tag: str
) -> tuple[ParticleSystem, dict | None]:
    """Freeze a random fraction of the parent, threading the mask into branches.

    ``instruments.pinning.random_pin`` returns an ``active_mask`` freezing
    ``round(pin_fraction * N)`` particles; setting it on the parent is enough --
    both ``run_branch_ensemble`` and the mega-batch draw momenta with, and copy,
    ``parent.active_mask``, and every perturbation operator clones it forward, so
    the frozen scaffold propagates to every branch (and stays static there).
    """

    if options.pin_fraction <= 0.0:
        return parent, None
    pin_seed = manager.seed_for(f"gardner.pin.{tag}", 0)
    mask = random_pin(parent, options.pin_fraction, pin_seed)
    pinned = parent.clone()
    pinned.active_mask = mask
    n_frozen = int((~mask).sum().item())
    record = {
        "pin_fraction": float(options.pin_fraction),
        "pin_seed": int(pin_seed),
        "n_particles": int(parent.n_particles),
        "n_frozen": n_frozen,
        "n_mobile": int(mask.sum().item()),
    }
    return pinned, record


def _run_megabatch(
    parent: ParticleSystem,
    perturbed_blocks: Sequence[ParticleSystem],
    *,
    options: ProbeOptions,
    device: str,
    temperature: float,
    box,
    tag: str,
    base: str,
    momentum_seed_domain: str = "branching.momentum",
) -> list["EnsembleTrajectory"]:
    """Integrate every (site, delta) perturbed block of one parent as ONE batch.

    Each of the ``len(perturbed_blocks)`` blocks contributes ``options.branches``
    rows -- one per matched momentum branch -- stacked leading, so a single
    :class:`BatchedSystem` of ``n_blocks * branches`` rows replaces ``n_blocks``
    sequential ensembles (the operators only displace positions, so the shared
    ``(N,)`` diameters, box and active mask are valid for every row).  The
    per-block momentum velocity draws are bitwise-identical to
    ``run_branch_ensemble``'s (same run-id-independent seeds), so block ``b``
    row ``k`` shares branch ``k``'s momenta with the unperturbed ensemble and the
    matched-seed counterfactual holds.  NVE only: a single shared Bussi stream
    could not reproduce each block's per-branch thermostat noise, so the
    thermostat path stays sequential.

    Per-block seed mapping: global row ``g = b * branches + k`` carries perturbed
    block ``b`` and momentum seed ``derive_seed(salt, momentum_domain, k)``.
    Only positions are captured (all four gardner observables need only the
    branch-divergence field), so the returned trajectories omit unwrapped frames.
    """

    count = options.branches
    n = parent.n_particles
    n_blocks = len(perturbed_blocks)
    dtype = parent.dtype
    mega_run = RunManager.create(
        phase="gardner",
        config=ExperimentConfig(
            phase="gardner",
            values={"kind": "megabatch", "tag": tag, "n_blocks": n_blocks, "branches": count},
        ),
        root=options.root,
        run_id=f"{base}--{tag}-mega",
        device=f"{device}/{options.dtype}",
        project_salt=options.project_salt,
    )
    try:
        issued_seeds = tuple(mega_run.seed_for(momentum_seed_domain, k) for k in range(count))
        torch_seeds = tuple(torch_seed(seed) for seed in issued_seeds)
        velocities = branch_maxwell_boltzmann_velocities(
            n,
            temperature,
            [make_generator(seed) for seed in torch_seeds],
            device=parent.device,
            dtype=dtype,
            active_mask=parent.active_mask,
        )  # (branches, N, 3)

        positions = torch.cat(
            [blk.positions.detach().unsqueeze(0).expand(count, n, 3) for blk in perturbed_blocks],
            dim=0,
        )
        unwrapped = torch.cat(
            [
                blk.unwrapped_positions.detach().unsqueeze(0).expand(count, n, 3)
                for blk in perturbed_blocks
            ],
            dim=0,
        )
        vel_all = (
            velocities.unsqueeze(0).expand(n_blocks, count, n, 3).reshape(n_blocks * count, n, 3)
        )
        mega = BatchedSystem(
            positions=positions.contiguous(),
            velocities=vel_all.contiguous(),
            diameters=parent.diameters.detach().clone(),
            box=parent.box.detach().clone(),
            active_mask=parent.active_mask.detach().clone(),
            unwrapped_positions=unwrapped.contiguous(),
        )

        integrator = BatchedMDIntegrator(mega, dt=options.dt, skin=0.3, thermostat=None)
        capture_steps = _capture_steps(options.horizon, options.stride, None)
        frames: list[torch.Tensor] = []
        previous = 0
        for target in capture_steps:
            if target != previous:
                integrator.step(target - previous)
                previous = target
            frames.append(mega.positions.detach().to("cpu").clone())
        stacked = torch.stack(frames).numpy().astype(float)  # (T, n_blocks*count, N, 3)

        trajectories: list[EnsembleTrajectory] = []
        for block in range(n_blocks):
            block_positions = stacked[:, block * count : (block + 1) * count]
            trajectories.append(
                EnsembleTrajectory(
                    positions=block_positions, box=box, momentum_seeds=issued_seeds
                )
            )
        mega_run.write_json(
            "megabatch_provenance.json",
            {
                "format_version": 1,
                "tag": tag,
                "n_blocks": n_blocks,
                "branches": count,
                "rows": n_blocks * count,
                "row_layout": "row g = block*branches + branch",
                "momentum_seed_domain": momentum_seed_domain,
                "momentum_issued_seeds": [int(s) for s in issued_seeds],
                "trajectory_steps": list(capture_steps),
                "integrator": "velocity_verlet_nve",
            },
        )
        mega_run.finish("completed")
    except BaseException:
        mega_run.finish("failed")
        raise
    return trajectories


def run_probe(options: ProbeOptions) -> Path:
    device = _resolve_device(options.device)
    dtype = _torch_dtype(options.dtype)
    if device == "mps" and dtype is torch.float64:
        raise ValueError("MPS does not support float64; use float32")
    if not (0.0 <= options.pin_fraction < 1.0):
        raise ValueError("pin_fraction must lie in [0, 1)")

    base_config = ExperimentConfig(
        phase="gardner",
        values={
            "kind": "probe",
            "operator": options.operator,
            "n_sites": options.n_sites,
            "deltas": list(options.deltas),
            "branches": options.branches,
            "horizon": options.horizon,
            "stride": options.stride,
            "dt": options.dt,
            "r_pert": options.r_pert,
            "thermostat_tau": options.thermostat_tau,
            "pin_fraction": options.pin_fraction,
            "device": f"{device}/{options.dtype}",
            "configs": [
                {"path": str(c.path), "temperature": c.temperature, "replica": c.replica, "N": c.n_particles}
                for c in options.configs
            ],
        },
    )
    manager = RunManager.create(
        phase="gardner",
        config=base_config,
        root=options.root,
        run_id=options.run_id,
        device=f"{device}/{options.dtype}",
        project_salt=options.project_salt,
    )
    base = manager.run_id
    try:
        # Reference delta = MEDIAN of the ladder, not the largest. For O_shell on
        # a deep glass the largest delta injects the most energy and overflows
        # float32 (measured: dU~1e6, non-finite branch dynamics at delta>=0.3),
        # which would null PR/N, R_D, and xi_chaos at exactly the reference point.
        # A mid-ladder point is float32-validated (delta<=0.1 safe by measurement).
        # Downstream chi fits restrict to the validated (finite-energy) delta range.
        deltas = tuple(sorted(options.deltas))
        reference_delta = deltas[len(deltas) // 2]
        per_config_axes: list[dict] = []
        # R_D inputs: N -> {config_tag -> total divergence at reference delta, final frame}
        rd_inputs: dict[int, dict[str, float]] = {}
        # PR/N per N (averaged over configs) for the FSS slope
        pr_by_n: dict[int, list[float]] = {}

        for cfg_index, cfg in enumerate(options.configs):
            snapshot = load_inherited_snapshot(
                cfg.path,
                temperature=cfg.temperature,
                replica=cfg.replica,
                device=device,
                dtype=dtype,
            )
            tag = _config_tag(cfg_index)
            parent, pin_record = _apply_pin_fraction(snapshot.system, options, manager, tag)
            box = parent.box.detach().cpu().numpy().astype(float)
            sigma = parent.diameters.detach().cpu().numpy().astype(float)
            ref_positions = parent.positions.detach().cpu().numpy().astype(float)

            unpert = _ensemble(parent, f"{base}--{tag}-unpert", options, device, cfg.temperature)
            unpert_traj = EnsembleTrajectory.from_result(unpert, box, sigma=sigma)

            site_seed = manager.seed_for(f"gardner.sites.{tag}", 0)
            min_sep = 2.0 * options.r_pert
            sites = stratified_sites(parent.box, options.n_sites, min_sep, seed=site_seed)

            # Build every (site, delta) perturbed block up front, then integrate
            # them as ONE batch (NVE) or sequentially (NVT thermostat confound).
            blocks: list[ParticleSystem] = []
            block_meta: list[tuple[int, int, float, object]] = []
            for site_index in range(options.n_sites):
                center = sites[site_index]
                op_seed = manager.seed_for(f"gardner.op.{tag}.s{site_index}", 0)
                for di, delta in enumerate(deltas):
                    perturbed, prov = _perturb(parent, center, float(delta), options, op_seed + di)
                    blocks.append(perturbed)
                    block_meta.append((site_index, di, float(delta), prov))

            if options.thermostat_tau is None:
                block_trajs = _run_megabatch(
                    parent,
                    blocks,
                    options=options,
                    device=device,
                    temperature=cfg.temperature,
                    box=box,
                    tag=tag,
                    base=base,
                )
            else:
                # NVT thermostat: one shared Bussi stream over all rows could not
                # reproduce each block's matched-seed noise, so integrate blocks
                # sequentially (unchanged per-(site, delta) ensembles).
                block_trajs = []
                for perturbed, (site_index, di, _delta, _prov) in zip(blocks, block_meta):
                    pert = _ensemble(
                        perturbed,
                        f"{base}--{tag}-s{site_index}-d{di}",
                        options,
                        device,
                        cfg.temperature,
                    )
                    block_trajs.append(EnsembleTrajectory.from_result(pert, box, sigma=sigma))

            # D_total(delta): mean over sites of the site total divergence at final frame
            site_totals: dict[float, list[float]] = {d: [] for d in deltas}
            site_pr: dict[float, list[float]] = {d: [] for d in deltas}
            xi_at_ref: list[float] = []
            provenance_rows: list[dict] = []

            for (site_index, di, delta, prov), pert_traj in zip(block_meta, block_trajs):
                D_field = divergence_field(pert_traj, unpert_traj)  # (T, N)
                final = D_field[-1]
                total_final = float(total_divergence(final))
                site_totals[delta].append(total_final)
                site_pr[delta].append(participation_ratio(final))
                if math.isclose(delta, reference_delta):
                    xi = chaos_length(final, ref_positions, box)
                    if np.isfinite(xi):
                        xi_at_ref.append(float(xi))
                provenance_rows.append(
                    {
                        "site": site_index,
                        "delta": float(delta),
                        "delta_u": prov.delta_u,
                        "rms_displacement": prov.rms_displacement,
                        "n_perturbed": prov.n_perturbed,
                        "total_divergence_final": total_final,
                        "pr_over_n_final": participation_ratio(final),
                    }
                )

            mean_total = {d: float(np.mean(site_totals[d])) for d in deltas}
            mean_pr = {d: float(np.mean(site_pr[d])) for d in deltas}
            chi = susceptibility(mean_total)
            xi_ref = float(np.median(xi_at_ref)) if xi_at_ref else float("nan")

            per_config_axes.append(
                {
                    "config": tag,
                    "path": str(cfg.path),
                    "temperature": cfg.temperature,
                    "replica": cfg.replica,
                    "N": cfg.n_particles,
                    "pr_over_n": mean_pr,
                    "total_divergence": mean_total,
                    "susceptibility": {
                        "chi": chi.chi,
                        "exponent": chi.exponent,
                        "linear_plateau": chi.linear_plateau,
                    },
                    "xi_chaos_at_reference_delta": xi_ref,
                    "reference_delta": reference_delta,
                    "pinning": pin_record,
                }
            )
            rd_inputs.setdefault(cfg.n_particles, {})[tag] = mean_total[reference_delta]
            pr_by_n.setdefault(cfg.n_particles, []).append(mean_pr[reference_delta])

            manager.write_json(f"divergence_{tag}.json", _json_safe({"provenance": provenance_rows}))

        # ---- FSS / non-self-averaging (needs >= 2 configs; per-N needs configs) ----
        sizes = sorted(rd_inputs)
        rd_report: dict = {}
        rd_ratio = float("nan")
        pr_slope = float("nan")
        pr_level = float("nan")
        multi_config = all(len(rd_inputs[n]) >= 2 for n in sizes)
        if multi_config:
            nsa = non_self_averaging(rd_inputs, n_boot=2000, seed=0)
            rd_report = {
                str(n): {"point": nsa.point[n], "lo": nsa.per_n[n].lo, "hi": nsa.per_n[n].hi}
                for n in sizes
            }
            if len(sizes) >= 2:
                rd_ratio = nsa.ratio(sizes[-1], sizes[0])
        pr_level_by_n = {n: float(np.mean(v)) for n, v in pr_by_n.items()}
        pr_level = pr_level_by_n[sizes[-1]]
        if len(sizes) >= 2:
            pr_slope = (pr_level_by_n[sizes[-1]] - pr_level_by_n[sizes[0]]) / (sizes[-1] - sizes[0])

        # Corroborating chi / xi trend from the largest-N configs
        largest_axes = [a for a in per_config_axes if a["N"] == sizes[-1]]
        chi_exponents = [a["susceptibility"]["exponent"] for a in largest_axes if np.isfinite(a["susceptibility"]["exponent"])]
        chi_exponent = float(np.median(chi_exponents)) if chi_exponents else float("nan")
        chi_linear = bool(np.isfinite(chi_exponent) and abs(chi_exponent - 1.0) < 0.15)
        xi_values = [a["xi_chaos_at_reference_delta"] for a in largest_axes if np.isfinite(a["xi_chaos_at_reference_delta"])]
        xi_median = float(np.median(xi_values)) if xi_values else float("nan")
        xi_growing = bool(np.isfinite(xi_median) and xi_median > options.r_pert)
        xi_saturates = bool(np.isfinite(xi_median) and xi_median <= 2.0)

        fss_resolvable = multi_config and len(sizes) >= 2
        if fss_resolvable:
            axes = AxisSummary(
                pr_slope=pr_slope,
                pr_level=pr_level,
                rd_ratio=rd_ratio,
                chi_exponent=chi_exponent,
                chi_linear_plateau=chi_linear,
                xi_growing=xi_growing,
                xi_saturates=xi_saturates,
                n_prep_depths_consistent=1,
            )
            verdict = decide(axes)
            verdict_note = None
        else:
            verdict = "bounded"
            verdict_note = (
                "single config / single size: the FSS legs (a-slope and c) are not "
                "resolvable; reported as an interventional bound, not a phase verdict"
            )

        axis_tables = {
            "reference_delta": reference_delta,
            "sizes": sizes,
            "pr_over_n_by_size": pr_level_by_n,
            "pr_slope": pr_slope,
            "non_self_averaging": rd_report,
            "rd_ratio": rd_ratio,
            "chi_exponent_median": chi_exponent,
            "chi_linear_plateau": chi_linear,
            "xi_chaos_median": xi_median,
            "xi_growing": xi_growing,
            "xi_saturates": xi_saturates,
            "per_config": per_config_axes,
        }
        manager.write_json("axes.json", _json_safe(axis_tables))
        manager.write_json(
            "verdict.json",
            _json_safe(
                {
                    "verdict": verdict,
                    "note": verdict_note,
                    "thresholds": DecisionThresholds().__dict__,
                    "fss_resolvable": fss_resolvable,
                }
            ),
        )
        manager.write_text("summary.md", _summary_markdown(options, axis_tables, verdict, verdict_note))
        manager.finish("completed")
        return manager.path
    except BaseException:
        manager.finish("failed")
        raise


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def _summary_markdown(options: ProbeOptions, axes: dict, verdict: str, note: str | None) -> str:
    lines = [
        "# Causal-Gardner quench-and-perturb probe",
        "",
        f"Operator: **{options.operator}**; B={options.branches}; H={options.horizon} steps "
        f"(dt={options.dt}); r_pert={options.r_pert}; NVE (thermostat_tau={options.thermostat_tau}).",
        f"Reference delta: {axes['reference_delta']}.",
        "",
        "## Discrimination axes",
        "",
        "| Axis | Statistic | Value |",
        "|---|---|---|",
        f"| (a) support | PR/N by size | {axes['pr_over_n_by_size']} |",
        f"| (a) support | slope PR/N vs N | {_fmt(axes['pr_slope'])} |",
        f"| (c) self-averaging | R_D ratio (large/small) | {_fmt(axes['rd_ratio'])} |",
        f"| (b) susceptibility | chi exponent (median) | {_fmt(axes['chi_exponent_median'])} |",
        f"| (b) susceptibility | linear plateau | {axes['chi_linear_plateau']} |",
        f"| (d) chaos length | xi_chaos (median) | {_fmt(axes['xi_chaos_median'])} |",
        "",
        "## Per-config",
        "",
        "| Config | N | T | chi exp | PR/N (ref delta) | xi_chaos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in axes["per_config"]:
        lines.append(
            f"| {a['config']} | {a['N']} | {a['temperature']} | "
            f"{_fmt(a['susceptibility']['exponent'])} | "
            f"{_fmt(a['pr_over_n'][axes['reference_delta']])} | "
            f"{_fmt(a['xi_chaos_at_reference_delta'])} |"
        )
    lines += ["", f"## Verdict: **{verdict}**", ""]
    if note:
        lines.append(f"> {note}")
    return "\n".join(lines) + "\n"


def _parse_configs(namespace) -> tuple[ConfigSpec, ...]:
    specs = []
    for entry in namespace.config:
        # format: path:temperature:replica:N
        path, temperature, replica, n = entry.split(":")
        specs.append(
            ConfigSpec(
                path=Path(path),
                temperature=float(temperature),
                replica=int(replica),
                n_particles=int(n),
            )
        )
    return tuple(specs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="path:temperature:replica:N (repeatable)",
    )
    parser.add_argument("--operator", choices=("O_shell", "O_kick", "O_strain"), default="O_shell")
    parser.add_argument("--n-sites", type=int, default=8)
    parser.add_argument("--deltas", type=str, default=",".join(str(d) for d in DEFAULT_DELTAS))
    parser.add_argument("--branches", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--r-pert", type=float, default=R_PERT_DEFAULT)
    parser.add_argument("--thermostat-tau", type=float, default=None)
    parser.add_argument(
        "--pin-fraction",
        type=float,
        default=0.0,
        help="freeze this fraction of the parent (random pinning) into the branch "
        "active_mask; 0.0 disables (unblocks the pinning-chaos ceiling pilot)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-salt", default="butterfly_cone")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    options = ProbeOptions(
        configs=_parse_configs(namespace),
        operator=namespace.operator,
        n_sites=namespace.n_sites,
        deltas=tuple(float(x) for x in namespace.deltas.split(",")),
        branches=namespace.branches,
        horizon=namespace.horizon,
        stride=namespace.stride,
        dt=namespace.dt,
        r_pert=namespace.r_pert,
        thermostat_tau=namespace.thermostat_tau,
        pin_fraction=namespace.pin_fraction,
        device=namespace.device,
        dtype=namespace.dtype,
        root=namespace.root,
        run_id=namespace.run_id,
        project_salt=namespace.project_salt,
    )
    path = run_probe(options)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
