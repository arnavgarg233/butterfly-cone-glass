#!/usr/bin/env python3
"""scripts/gardner_cone_campaign.py -- finer-stride Gardner butterfly-cone campaign.

The pooled butterfly velocity ``v_b`` from :mod:`gardner_r0` is currently
*front-estimator limited*: the coarse ``stride=100`` capture leaves only ~5-6
frames inside the pre-saturation growth window, so the ballistic-front slope
(and therefore ``v_b``) has a large fit-window variance and fails the referee
robustness sweep.  The fix is not more MD -- it is a FINER capture stride so the
same quench-perturb-branch dynamics are sampled ~4x more densely in the growth
window, pinning the front slope.

This runner reproduces the ``gardner-T0075-fss`` cone campaign (same inherited
configs, sites, deltas, operator, matched-seed counterfactual) but with a
configurable, finer ``--stride`` (default ``25`` = 1/4 of the original ``100``,
i.e. 4x the growth-window frames) and emits per-``(site, delta)`` branch
ensembles in the EXACT on-disk schema :mod:`gardner_r0` reads, so the SAME
analysis consumes the finer data and produces a robust ``v_b``.

Memory safety at 4x frames.  A naive "fold every ``(site x delta)`` block of a
parent into ONE mega-batch and materialize the whole ``(T, n_blocks*B, N, 3)``
trajectory" (the A3 mega-batch observable path in :mod:`gardner_probe`) is
memory-lean only because it keeps positions alone and never persists per-branch
trajectories; persisting the full ``gardner_r0`` channels (positions AND
unwrapped) for every block at 4x frames would need tens of GB at ``N=3000``.
This runner therefore uses the mega-batch **in bounded block chunks** with
**streaming per-block capture**: ``--mega-chunk`` blocks are integrated as one
BatchedSystem, every captured frame is sliced per block into float32 CPU buffers
(never the float64 stack), and each block is flushed to disk as a self-contained
ensemble before the next chunk -- so peak memory is ``O(mega_chunk * T * B * N)``
and stays flat as the horizon/stride refine.  ``--mega-chunk 1`` degrades to the
canonical per-block :func:`run_branch_ensemble` (one ensemble resident), the
safest default for large ``N`` / low-RAM machines.

No new physics: pure orchestration over :func:`branching.run_branch_ensemble`,
the batched integrator, and the :mod:`butterfly_cone.perturb` operators.  ``gardner_probe``
/ ``branching`` / ``perturb`` are imported read-only.  NVE only (float32 on MPS
for production, float64 on CPU for certification).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- read-only imports of the existing machinery (never edited here) ----------
import gardner_probe as gp  # noqa: E402  (its module import wires up sys.path/src)
from butterfly_cone.branching.batched import (  # noqa: E402
    BatchedMDIntegrator,
    BatchedSystem,
    branch_maxwell_boltzmann_velocities,
)
from butterfly_cone.branching.ensemble import (  # noqa: E402
    _capture_steps,
    _parent_state_sha256,
    _save_bytes,
    _state_payload,
    run_branch_ensemble,
    torch_seed,
)
from butterfly_cone.engine.system import ParticleSystem, make_generator, make_system  # noqa: E402
from butterfly_cone.harness.config import ExperimentConfig  # noqa: E402
from butterfly_cone.harness.runs import RunManager  # noqa: E402
from butterfly_cone.perturb.operators import R_PERT_DEFAULT, stratified_sites  # noqa: E402
from butterfly_cone.pilot.loader import load_inherited_snapshot  # noqa: E402

# The stride the ``gardner-T0075-fss`` cone campaign was captured at.
ORIGINAL_STRIDE = 100
# Refine by 4x -> 4x the frames everywhere, in particular the growth window.
STRIDE_REFINEMENT = 4
DEFAULT_STRIDE = ORIGINAL_STRIDE // STRIDE_REFINEMENT  # 25
# The SMALL linear-response rungs only.  delta=0.1 is the robustness *violator*
# (largest injected energy; overflows the float32 branch dynamics at the deep
# glass and null-poisons the reference-delta observables), so it is excluded by
# default; the two small rungs stay in the validated linear-response range.
DEFAULT_DELTAS = (0.01, 0.03)
VIOLATOR_DELTA = 0.1


@dataclass(frozen=True)
class CampaignOptions:
    """Everything the finer-stride cone campaign needs, one frozen record."""

    configs: tuple[gp.ConfigSpec, ...]
    temperature: float = 0.075
    operator: str = "O_shell"
    n_sites: int = 6
    deltas: tuple[float, ...] = DEFAULT_DELTAS
    branches: int = 48
    horizon: int = 2000
    stride: int = DEFAULT_STRIDE
    dt: float = 0.01
    r_pert: float = R_PERT_DEFAULT
    pin_fraction: float = 0.0
    mega_chunk: int = 1
    max_frames: int | None = None
    device: str = "mps"
    dtype: str = "float32"
    root: Path = ROOT
    run_id: str | None = None
    project_salt: str = "butterfly_cone"

    def __post_init__(self) -> None:
        if not self.configs:
            raise ValueError("at least one --config is required")
        if self.branches <= 0:
            raise ValueError("branches must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if not self.deltas:
            raise ValueError("at least one delta is required")
        if self.mega_chunk <= 0:
            raise ValueError("mega_chunk must be positive")
        if any(d < 0.0 for d in self.deltas):
            raise ValueError("deltas must be nonnegative")


# ---------------------------------------------------------------------------
# Frame-schedule helpers (the whole point of the campaign is these counts)
# ---------------------------------------------------------------------------


def capture_steps(options: CampaignOptions) -> tuple[int, ...]:
    """The integer capture schedule this campaign will realize on disk."""

    return _capture_steps(options.horizon, options.stride, options.max_frames)


def growth_window_end(horizon: int, fraction: float = 0.25) -> int:
    """End step of the (conservative) pre-saturation growth window.

    The ballistic-front / Lyapunov fits live before the divergence field
    saturates; on the T0075 deep glass that onset sits well within the first
    quarter of the horizon.  A fixed fraction keeps the frame-count comparison
    between two strides honest (same window, different sampling density).
    """

    return max(1, int(round(fraction * horizon)))


def frames_in_growth_window(steps: Sequence[int], horizon: int, fraction: float = 0.25) -> int:
    end = growth_window_end(horizon, fraction)
    return sum(1 for s in steps if s <= end)


def frame_count_report(options: CampaignOptions) -> dict:
    """Original-vs-finer frame counts, overall and in the growth window."""

    fine = capture_steps(options)
    coarse = _capture_steps(options.horizon, ORIGINAL_STRIDE, None)
    end = growth_window_end(options.horizon)
    fine_growth = frames_in_growth_window(fine, options.horizon)
    coarse_growth = frames_in_growth_window(coarse, options.horizon)
    return {
        "original_stride": ORIGINAL_STRIDE,
        "stride": options.stride,
        "horizon": options.horizon,
        "growth_window_end_step": end,
        "frames_total_original": len(coarse),
        "frames_total_finer": len(fine),
        "frames_growth_original": coarse_growth,
        "frames_growth_finer": fine_growth,
        "growth_improvement": (fine_growth / coarse_growth) if coarse_growth else float("nan"),
        "total_improvement": (len(fine) / len(coarse)) if coarse else float("nan"),
    }


# ---------------------------------------------------------------------------
# Streaming, chunked mega-batch persistence (gardner_r0-compatible schema)
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """One perturbed (site, delta) ensemble scheduled for the mega-batch."""

    site_index: int
    delta_index: int
    delta: float
    perturbed: ParticleSystem
    provenance: object
    run: RunManager


def _controls(options: CampaignOptions) -> dict:
    return {
        "count": int(options.branches),
        "temperature": float(options.temperature),
        "horizon": int(options.horizon),
        "dt": float(options.dt),
        "stride": int(options.stride),
        "skin": 0.3,
        "integrator": "velocity_verlet_nve",
        "thermostat_tau": None,
    }


def _write_ensemble(
    run: RunManager,
    parent: ParticleSystem,
    positions: torch.Tensor,       # (T, B, N, 3)
    unwrapped: torch.Tensor,       # (T, B, N, 3)
    velocities: torch.Tensor,      # (T, B, N, 3)
    steps: Sequence[int],
    issued_seeds: Sequence[int],
    torch_seeds: Sequence[int],
    options: CampaignOptions,
    provenance_extra: dict,
) -> None:
    """Publish one ensemble in the exact schema :mod:`gardner_r0` consumes.

    Mirrors :func:`run_branch_ensemble`'s persistence byte-for-byte (same
    ``_state_payload`` / ``_save_bytes`` helpers, same file names, same
    ``branch_provenance.json`` keys) so the finer-stride ensembles are
    indistinguishable to the reanalysis loader from the original coarse ones.
    """

    count = int(positions.shape[1])
    parent_sha = _parent_state_sha256(parent)
    parent_payload = _state_payload(parent)
    parent_payload.update({"format_version": 1, "state_sha256": parent_sha})
    run.write_bytes("parent_state.pt", _save_bytes(parent_payload))

    steps_tensor = torch.tensor(list(steps), dtype=torch.int64)
    diameters = parent.diameters.detach().to("cpu").clone()
    box = parent.box.detach().to("cpu").clone()
    active_mask = parent.active_mask.detach().to("cpu").clone()

    branch_records: list[dict] = []
    for k in range(count):
        label = f"branches/{k:06d}"
        final_file = f"{label}/final_state.pt"
        traj_file = f"{label}/trajectory.pt"
        final_payload = {
            "positions": positions[-1, k].clone(),
            "velocities": velocities[-1, k].clone(),
            "diameters": diameters.clone(),
            "box": box.clone(),
            "active_mask": active_mask.clone(),
            "unwrapped_positions": unwrapped[-1, k].clone(),
            "format_version": 1,
            "branch_index": k,
            "parent_state_sha256": parent_sha,
        }
        run.write_bytes(final_file, _save_bytes(final_payload))
        run.write_bytes(
            traj_file,
            _save_bytes(
                {
                    "format_version": 1,
                    "branch_index": k,
                    "steps": steps_tensor.clone(),
                    "positions": positions[:, k].clone(),
                    "unwrapped_positions": unwrapped[:, k].clone(),
                    "velocities": velocities[:, k].clone(),
                }
            ),
        )
        branch_records.append(
            {
                "index": k,
                "momentum_seed": int(issued_seeds[k]),
                "torch_seed": int(torch_seeds[k]),
                "final_state_file": final_file,
                "trajectory_file": traj_file,
            }
        )

    provenance = {
        "format_version": 1,
        "parent_id": None,
        "parent_state_sha256": parent_sha,
        "parent_state_file": "parent_state.pt",
        "parent": {
            "n_particles": int(parent.n_particles),
            "dtype": str(parent.dtype),
            "device": str(parent.device),
            "active_particles": int(parent.active_mask.sum().item()),
        },
        "controls": _controls(options),
        "capture": {
            "mode": "streaming_megabatch",
            "n_frames": len(steps),
            "max_frames": options.max_frames,
            "max_retained_frames": len(steps),
            "reducer": None,
        },
        "trajectory_steps": list(steps),
        "momentum_seed_domain": "branching.momentum",
        "thermostat": None,
        "branches": branch_records,
        **provenance_extra,
    }
    run.write_json("branch_provenance.json", provenance)
    run.log(
        f"published {count} streaming-megabatch branches through step "
        f"{options.horizon} ({len(steps)} frames, velocity_verlet_nve)"
    )
    run.finish("completed")


def _run_megabatch_chunk(
    parent: ParticleSystem,
    blocks: Sequence[_Block],
    *,
    options: CampaignOptions,
    velocities: torch.Tensor,          # (B, N, 3) shared matched-seed momenta
    issued_seeds: Sequence[int],
    torch_seeds: Sequence[int],
    steps: Sequence[int],
) -> None:
    """Integrate one chunk of blocks as a single BatchedSystem, stream to disk.

    Row layout ``g = block * branches + k`` (identical to
    :func:`gardner_probe._run_megabatch`): block ``b`` row ``k`` carries branch
    ``k``'s matched momenta, so at ``delta -> 0`` it coincides with the
    unperturbed ensemble's branch ``k`` and the counterfactual holds.  Frames are
    captured one at a time and sliced per block into float32 CPU buffers; the
    full float64 stack is never formed.
    """

    count = int(options.branches)
    n = int(parent.n_particles)
    n_blocks = len(blocks)

    positions0 = torch.cat(
        [blk.perturbed.positions.detach().unsqueeze(0).expand(count, n, 3) for blk in blocks],
        dim=0,
    )
    unwrapped0 = torch.cat(
        [
            blk.perturbed.unwrapped_positions.detach().unsqueeze(0).expand(count, n, 3)
            for blk in blocks
        ],
        dim=0,
    )
    vel_all = velocities.unsqueeze(0).expand(n_blocks, count, n, 3).reshape(n_blocks * count, n, 3)

    mega = BatchedSystem(
        positions=positions0.contiguous(),
        velocities=vel_all.contiguous(),
        diameters=parent.diameters.detach().clone(),
        box=parent.box.detach().clone(),
        active_mask=parent.active_mask.detach().clone(),
        unwrapped_positions=unwrapped0.contiguous(),
    )
    integrator = BatchedMDIntegrator(mega, dt=options.dt, skin=0.3, thermostat=None)

    pos_buf: list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]
    unw_buf: list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]
    vel_buf: list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]
    previous = 0
    for target in steps:
        if target != previous:
            integrator.step(target - previous)
            previous = target
        pos_cpu = mega.positions.detach().to("cpu")
        unw_cpu = mega.unwrapped_positions.detach().to("cpu")
        vel_cpu = mega.velocities.detach().to("cpu")
        for b in range(n_blocks):
            lo = b * count
            hi = lo + count
            pos_buf[b].append(pos_cpu[lo:hi].clone())
            unw_buf[b].append(unw_cpu[lo:hi].clone())
            vel_buf[b].append(vel_cpu[lo:hi].clone())

    for b, blk in enumerate(blocks):
        _write_ensemble(
            blk.run,
            blk.perturbed,
            torch.stack(pos_buf[b]),
            torch.stack(unw_buf[b]),
            torch.stack(vel_buf[b]),
            steps,
            issued_seeds,
            torch_seeds,
            options,
            provenance_extra={
                "perturbation": {
                    "operator": options.operator,
                    "site": int(blk.site_index),
                    "delta_index": int(blk.delta_index),
                    "delta": float(blk.delta),
                    "r_pert": float(options.r_pert),
                    "delta_u": getattr(blk.provenance, "delta_u", None),
                    "rms_displacement": getattr(blk.provenance, "rms_displacement", None),
                    "n_perturbed": getattr(blk.provenance, "n_perturbed", None),
                },
                "megabatch": {"chunk_blocks": n_blocks, "row_layout": "g = block*branches + branch"},
            },
        )
        # Free this block's frames before the next chunk / block flush.
        pos_buf[b] = []
        unw_buf[b] = []
        vel_buf[b] = []


# ---------------------------------------------------------------------------
# Per-config orchestration
# ---------------------------------------------------------------------------


def _load_parent(cfg: gp.ConfigSpec, options: CampaignOptions, device: str, dtype: torch.dtype) -> ParticleSystem:
    snapshot = load_inherited_snapshot(
        cfg.path,
        temperature=cfg.temperature,
        replica=cfg.replica,
        device=device,
        dtype=dtype,
    )
    return snapshot.system


def _make_run(options: CampaignOptions, run_id: str, kind: str, device: str) -> RunManager:
    return RunManager.create(
        phase="gardner",
        config=ExperimentConfig(phase="gardner", values={"kind": kind, "run_id": run_id}),
        root=options.root,
        run_id=run_id,
        device=f"{device}/{options.dtype}",
        project_salt=options.project_salt,
    )


def _run_config(
    cfg_index: int,
    cfg: gp.ConfigSpec,
    parent: ParticleSystem,
    options: CampaignOptions,
    device: str,
    base: str,
    base_manager: RunManager,
) -> dict:
    """Produce the unpert + every (site, delta) perturbed ensemble for one config."""

    tag = gp._config_tag(cfg_index)
    parent, pin_record = gp._apply_pin_fraction(parent, options, base_manager, tag)

    # --- unperturbed matched-seed reference (canonical producer) -------------
    unpert_run = _make_run(options, f"{base}--{tag}-unpert", "branch_ensemble", device)
    unpert = run_branch_ensemble(
        parent,
        count=options.branches,
        temperature=options.temperature,
        horizon=options.horizon,
        run=unpert_run,
        dt=options.dt,
        stride=options.stride,
        thermostat_tau=None,
        max_frames=options.max_frames,
        momentum_seed_domain="branching.momentum",
    )
    issued_seeds = unpert.branch_seeds
    torch_seeds = unpert.branch_torch_seeds

    # --- sites + perturbed blocks (same domains as gardner_probe) ------------
    site_seed = base_manager.seed_for(f"gardner.sites.{tag}", 0)
    sites = stratified_sites(parent.box, options.n_sites, 2.0 * options.r_pert, seed=site_seed)
    deltas = tuple(sorted(options.deltas))

    blocks: list[_Block] = []
    for site_index in range(options.n_sites):
        center = sites[site_index]
        op_seed = base_manager.seed_for(f"gardner.op.{tag}.s{site_index}", 0)
        for di, delta in enumerate(deltas):
            perturbed, prov = gp._perturb(parent, center, float(delta), options, op_seed + di)
            run = _make_run(options, f"{base}--{tag}-s{site_index}-d{di}", "branch_ensemble", device)
            blocks.append(_Block(site_index, di, float(delta), perturbed, prov, run))

    steps = capture_steps(options)

    if options.mega_chunk == 1:
        # Canonical per-block path: one ensemble resident at a time (safest).
        for blk in blocks:
            run_branch_ensemble(
                blk.perturbed,
                count=options.branches,
                temperature=options.temperature,
                horizon=options.horizon,
                run=blk.run,
                dt=options.dt,
                stride=options.stride,
                thermostat_tau=None,
                max_frames=options.max_frames,
                momentum_seed_domain="branching.momentum",
            )
    else:
        velocities = branch_maxwell_boltzmann_velocities(
            parent.n_particles,
            options.temperature,
            [make_generator(ts) for ts in torch_seeds],
            device=parent.device,
            dtype=parent.dtype,
            active_mask=parent.active_mask,
        )
        for start in range(0, len(blocks), options.mega_chunk):
            chunk = blocks[start : start + options.mega_chunk]
            _run_megabatch_chunk(
                parent,
                chunk,
                options=options,
                velocities=velocities,
                issued_seeds=issued_seeds,
                torch_seeds=torch_seeds,
                steps=steps,
            )

    return {
        "config": tag,
        "path": str(cfg.path),
        "N": cfg.n_particles,
        "temperature": cfg.temperature,
        "replica": cfg.replica,
        "n_sites": options.n_sites,
        "n_deltas": len(deltas),
        "n_perturbed_ensembles": len(blocks),
        "pinning": pin_record,
    }


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------


def _base_run_id(options: CampaignOptions, device: str) -> str:
    if options.run_id:
        return options.run_id
    t_tag = f"T{options.temperature:.4f}".replace("0.", "").replace(".", "")
    return f"conecamp-{t_tag}-fss-s{options.stride}"


def run_campaign(options: CampaignOptions) -> Path:
    """Run the whole finer-stride cone campaign; return the base root run dir.

    Point :mod:`gardner_r0` at the returned directory (``--run-dir``); its
    ``discover`` gathers every ``<base>--c*-s*-d*`` / ``<base>--c*-unpert``
    sibling by root prefix, and reads this dir's ``config.yaml`` for the
    delta ladder and ``dt``.
    """

    device = gp._resolve_device(options.device)
    dtype = gp._torch_dtype(options.dtype)
    if device == "mps" and dtype is torch.float64:
        raise ValueError("MPS does not support float64; use --dtype float32")

    base_manager = RunManager.create(
        phase="gardner",
        config=ExperimentConfig(
            phase="gardner",
            values={
                "kind": "cone_campaign",
                "operator": options.operator,
                "temperature": options.temperature,
                "n_sites": options.n_sites,
                "deltas": [float(d) for d in sorted(options.deltas)],
                "excluded_violator_delta": VIOLATOR_DELTA,
                "branches": options.branches,
                "horizon": options.horizon,
                "stride": options.stride,
                "original_stride": ORIGINAL_STRIDE,
                "dt": options.dt,
                "r_pert": options.r_pert,
                "pin_fraction": options.pin_fraction,
                "mega_chunk": options.mega_chunk,
                "max_frames": options.max_frames,
                "device": f"{device}/{options.dtype}",
                "configs": [
                    {
                        "path": str(c.path),
                        "temperature": c.temperature,
                        "replica": c.replica,
                        "N": c.n_particles,
                    }
                    for c in options.configs
                ],
            },
        ),
        root=options.root,
        run_id=_base_run_id(options, device),
        device=f"{device}/{options.dtype}",
        project_salt=options.project_salt,
    )
    base = base_manager.run_id
    try:
        per_config: list[dict] = []
        for cfg_index, cfg in enumerate(options.configs):
            parent = _load_parent(cfg, options, device, dtype)
            per_config.append(
                _run_config(cfg_index, cfg, parent, options, device, base, base_manager)
            )

        frames = frame_count_report(options)
        base_manager.write_json(
            "campaign.json",
            {
                "base": base,
                "device": f"{device}/{options.dtype}",
                "n_configs": len(options.configs),
                "frames": frames,
                "per_config": per_config,
                "analysis_hint": (
                    "python scripts/gardner_r0.py --run-dir "
                    f"{base_manager.path}"
                ),
            },
        )
        base_manager.write_text("campaign_summary.md", _summary_markdown(options, base, frames, per_config))
        base_manager.finish("completed")
        return base_manager.path
    except BaseException:
        base_manager.finish("failed")
        raise


def _summary_markdown(options: CampaignOptions, base: str, frames: dict, per_config: list[dict]) -> str:
    lines = [
        "# Gardner finer-stride butterfly-cone campaign",
        "",
        f"Base run: `{base}`  (operator {options.operator}, B={options.branches}, "
        f"H={options.horizon}, dt={options.dt}, r_pert={options.r_pert}, NVE).",
        f"Deltas (small linear-response rungs; delta={VIOLATOR_DELTA} violator excluded): "
        f"{sorted(options.deltas)}.",
        "",
        "## Frame density (the v_b firming lever)",
        "",
        "| | original (stride %d) | finer (stride %d) |"
        % (ORIGINAL_STRIDE, options.stride),
        "|---|---:|---:|",
        f"| total frames | {frames['frames_total_original']} | {frames['frames_total_finer']} |",
        f"| growth-window frames (step <= {frames['growth_window_end_step']}) | "
        f"{frames['frames_growth_original']} | {frames['frames_growth_finer']} |",
        "",
        f"Growth-window improvement: **{frames['growth_improvement']:.2f}x** "
        f"(total {frames['total_improvement']:.2f}x).",
        "",
        "## Ensembles produced",
        "",
        "| config | N | T | sites | deltas | perturbed ensembles |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in per_config:
        lines.append(
            f"| {c['config']} | {c['N']} | {c['temperature']} | {c['n_sites']} | "
            f"{c['n_deltas']} | {c['n_perturbed_ensembles']} |"
        )
    lines += [
        "",
        "Reanalyze the finer-stride data with the SAME estimator:",
        "",
        "```",
        f"python scripts/gardner_r0.py --run-dir runs/gardner/{base}",
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Smoke mode: a tiny, self-contained synthetic config so the pipeline is
# exercised end-to-end without the multi-GB inherited-glass archives.
# ---------------------------------------------------------------------------


def write_synthetic_config(
    path: Path, *, temperature: float, replica: int, n: int, seed: int
) -> None:
    """Write a loader-compatible ``.npz`` from a synthetic lattice system."""

    system = make_system(n, generator=make_generator(seed), dtype=torch.float64)
    key = f"{float(temperature):.3f}_{replica}"
    np.savez(
        path,
        L=np.asarray(float(system.box[0].item())),
        N=np.asarray(n),
        **{
            f"pos_{key}": system.positions.detach().cpu().numpy().astype(np.float64),
            f"sig_{key}": system.diameters.detach().cpu().numpy().astype(np.float64),
        },
    )


def smoke_options(root: Path, *, device: str = "cpu", **overrides) -> CampaignOptions:
    """A tiny CPU/float64 campaign that produces gardner_r0-compatible outputs."""

    cfg_dir = root / "smoke_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "cooled_smoke.npz"
    n = int(overrides.pop("n_particles", 60))
    write_synthetic_config(cfg_path, temperature=0.075, replica=0, n=n, seed=4242)
    base = dict(
        configs=(gp.ConfigSpec(path=cfg_path, temperature=0.075, replica=0, n_particles=n),),
        temperature=0.075,
        operator="O_shell",
        n_sites=1,
        deltas=(0.01, 0.03),
        branches=4,
        horizon=40,
        stride=10,
        dt=0.005,
        r_pert=1.5,
        mega_chunk=1,
        device=device,
        dtype="float64" if device == "cpu" else "float32",
        root=root,
    )
    base.update(overrides)
    return CampaignOptions(**base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_deltas(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip() != "")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gardner_cone_campaign.py",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        action="append",
        default=None,
        help="path:temperature:replica:N (repeatable). Omit only with --smoke.",
    )
    p.add_argument("--temperature", type=float, default=0.075)
    p.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"finer capture stride (original was {ORIGINAL_STRIDE}); default {DEFAULT_STRIDE} = 4x frames",
    )
    p.add_argument(
        "--deltas",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DELTAS),
        help=f"comma list; default the small linear-response rungs (excludes the delta={VIOLATOR_DELTA} violator)",
    )
    p.add_argument("--n-sites", type=int, default=6)
    p.add_argument("--branches", type=int, default=48)
    p.add_argument("--horizon", type=int, default=2000)
    p.add_argument("--operator", choices=("O_shell", "O_kick", "O_strain"), default="O_shell")
    p.add_argument("--r-pert", type=float, default=R_PERT_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--pin-fraction", type=float, default=0.0)
    p.add_argument(
        "--mega-chunk",
        type=int,
        default=1,
        help="blocks folded per streaming mega-batch integration (1 = canonical "
        "per-block; >1 = A3 mega-batch, peak memory ~ chunk*T*B*N)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="cap captured frames (thins the schedule)")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT,
        help="project root under which runs/gardner/<base>... is written (default: repo root)",
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--project-salt", default="butterfly_cone")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic CPU/float64 end-to-end run")
    return p


def _configs_from_namespace(ns) -> tuple[gp.ConfigSpec, ...]:
    specs: list[gp.ConfigSpec] = []
    for entry in ns.config or ():
        path, temperature, replica, n = entry.split(":")
        specs.append(
            gp.ConfigSpec(
                path=Path(path),
                temperature=float(temperature),
                replica=int(replica),
                n_particles=int(n),
            )
        )
    return tuple(specs)


def options_from_namespace(ns) -> CampaignOptions:
    if ns.smoke and not ns.config:
        opts = smoke_options(Path(ns.out), device=ns.device if ns.device != "auto" else "cpu")
        if ns.run_id:
            opts = CampaignOptions(**{**opts.__dict__, "run_id": ns.run_id})
        return opts
    return CampaignOptions(
        configs=_configs_from_namespace(ns),
        temperature=ns.temperature,
        operator=ns.operator,
        n_sites=ns.n_sites,
        deltas=_parse_deltas(ns.deltas),
        branches=ns.branches,
        horizon=ns.horizon,
        stride=ns.stride,
        dt=ns.dt,
        r_pert=ns.r_pert,
        pin_fraction=ns.pin_fraction,
        mega_chunk=ns.mega_chunk,
        max_frames=ns.max_frames,
        device=ns.device,
        dtype=ns.dtype,
        root=Path(ns.out),
        run_id=ns.run_id,
        project_salt=ns.project_salt,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    options = options_from_namespace(ns)
    path = run_campaign(options)
    frames = frame_count_report(options)
    print(path)
    print(
        f"growth-window frames: {frames['frames_growth_original']} -> "
        f"{frames['frames_growth_finer']} ({frames['growth_improvement']:.2f}x); "
        f"reanalyze: python scripts/gardner_r0.py --run-dir {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
