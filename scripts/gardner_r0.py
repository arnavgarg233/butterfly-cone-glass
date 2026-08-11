#!/usr/bin/env python
"""scripts/gardner_r0.py -- Gardner R0 zero-GPU butterfly reanalysis runner.

PURE REANALYSIS of Gardner FSS branch trajectories already on disk -- no new MD.
For every stored ``(config, site, delta)`` perturbed branch ensemble the runner:

1. loads the perturbed trajectories and the matched-seed *unperturbed* ensemble
   of the same config (``torch.load`` -> cpu -> float64);
2. builds the raw matched-seed divergence field ``M(t)`` via
   :func:`butterfly_cone.perturb.response.divergence_field` and the drift-robust structural
   field ``S(t)`` via :func:`butterfly_cone.perturb.response.cage_relative_divergence_field`;
3. feeds both channels to :func:`butterfly_cone.perturb.butterfly.analyze_config`, reading
   the Lyapunov exponent ``lambda`` from the PRE-saturation window, the ballistic
   butterfly velocity ``v_b``, the saturation plateau ``D_sat`` and the structural
   shielding time ``t_shield`` vs the raw crossing ``t_raw``;
4. aggregates per-config and pooled ``lambda``/``v_b``/``t_shield`` with spreads,
   runs the N=1500-vs-3000 intensive-consistency check, and prints the verdict:
   is there a clean pre-saturation Lyapunov cone (``lambda > 0`` resolved) and
   does the structural channel show shielding (``t_shield > t_raw``)?

-----------------------------------------------------------------------------
trajectory.pt schema (format_version 1), discovered on the real run:

    <ensemble>/branches/NNNNNN/trajectory.pt : dict
        format_version       int   = 1
        branch_index         int
        steps                (T,)        int64   -- MD step index of each frame
        positions            (T, N, 3)   float32 -- wrapped positions
        unwrapped_positions  (T, N, 3)   float32 -- continuous positions
        velocities           (T, N, 3)   float32

    <ensemble>/branches/NNNNNN/final_state.pt : dict
        positions, velocities, diameters(N,), box(3,), active_mask(N,) bool,
        unwrapped_positions, format_version, branch_index, parent_state_sha256

    <ensemble>/parent_state.pt : dict
        positions(N,3), velocities, diameters(N,), box(3,), active_mask(N,),
        unwrapped_positions, format_version, state_sha256

    <ensemble>/branch_provenance.json :
        branches: [ {index, momentum_seed, torch_seed, trajectory_file,
                     final_state_file}, ... ]
        controls: {dt, stride, horizon, temperature, ...}
        trajectory_steps: [...]

Static per-particle data (box, diameters=sigma, reference positions) live in
``parent_state.pt`` / ``final_state.pt`` -- NOT in trajectory.pt.  The per-branch
``momentum_seed`` is the identity matched between a config's perturbed and
unperturbed ensembles (the counterfactual contract); it is passed through so
``response.assert_matched_seeds`` genuinely guards the differencing.  The
perturbation site center is recovered self-containedly from the divergence-
weighted centroid of the localized initial divergence field ``M(t=0)``.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

# Make the built src/butterfly_cone modules importable when run as a plain script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from butterfly_cone.events.trajectory import minimum_image  # noqa: E402
from butterfly_cone.perturb import butterfly, response  # noqa: E402

LOG = logging.getLogger("gardner_r0")

# Ensemble-directory naming: <root>--c{config}-s{site}-d{delta}  (perturbed)
#                            <root>--c{config}-unpert             (unperturbed)
PERT_RE = re.compile(r"^(?P<root>.+)--c(?P<c>\d+)-s(?P<s>\d+)-d(?P<d>\d+)$")
UNPERT_RE = re.compile(r"^(?P<root>.+)--c(?P<c>\d+)-unpert$")

R2_RESOLVED_DEFAULT = 0.9   # min log-linear R^2 for a lambda to count as "resolved"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleRef:
    path: Path
    root: str
    config: int
    site: int | None       # None for an unpert ensemble
    delta_index: int | None
    kind: str              # "pert" | "unpert"

    @property
    def label(self) -> str:
        if self.kind == "unpert":
            return f"c{self.config}-unpert"
        return f"c{self.config}-s{self.site}-d{self.delta_index}"


def _classify(path: Path) -> EnsembleRef | None:
    """Return an :class:`EnsembleRef` if ``path`` is a Gardner branch ensemble."""

    if not path.is_dir() or not (path / "branches").is_dir():
        return None
    m = PERT_RE.match(path.name)
    if m:
        return EnsembleRef(
            path=path, root=m.group("root"), config=int(m.group("c")),
            site=int(m.group("s")), delta_index=int(m.group("d")), kind="pert",
        )
    m = UNPERT_RE.match(path.name)
    if m:
        return EnsembleRef(
            path=path, root=m.group("root"), config=int(m.group("c")),
            site=None, delta_index=None, kind="unpert",
        )
    return None


def discover(run_dir: Path) -> list[EnsembleRef]:
    """Discover every branch ensemble relevant to ``run_dir``.

    Handles three forms of ``--run-dir``:

    * a run *root* (e.g. ``gardner-T0075-fss``) -- ensembles are its siblings
      ``<basename>--*``;
    * a *parent* directory (e.g. ``runs/gardner``) -- ensembles are its children;
    * a single *ensemble* dir (e.g. ``gardner-T0075-fss--c0-s0-d1``) -- the whole
      root set it belongs to (siblings sharing its root prefix).
    """

    run_dir = run_dir.resolve()
    found: dict[Path, EnsembleRef] = {}

    def _add(p: Path) -> None:
        ref = _classify(p)
        if ref is not None:
            found[ref.path] = ref

    self_ref = _classify(run_dir)
    if self_ref is not None:
        # run_dir is itself an ensemble: gather the whole root set from siblings.
        for sib in sorted(run_dir.parent.iterdir()):
            sref = _classify(sib)
            if sref is not None and sref.root == self_ref.root:
                found[sref.path] = sref
        return sorted(found.values(), key=lambda r: r.path.name)

    # run_dir is a root or a parent: try its children first.
    if run_dir.is_dir():
        for child in sorted(run_dir.iterdir()):
            _add(child)
    if found:
        return sorted(found.values(), key=lambda r: r.path.name)

    # No ensemble children -> run_dir is a root whose ensembles are siblings.
    base = run_dir.name
    parent = run_dir.parent
    if parent.is_dir():
        for sib in sorted(parent.iterdir()):
            if sib.name.startswith(base + "--"):
                _add(sib)
    return sorted(found.values(), key=lambda r: r.path.name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float64).numpy()


@dataclass
class StaticState:
    box: np.ndarray            # (3,)
    sigma: np.ndarray          # (N,)  per-particle diameters
    positions: np.ndarray      # (N, 3) reference (parent) positions


def load_static(ens_dir: Path) -> StaticState:
    """Read box / diameters / reference positions from the parent (or a branch)."""

    parent = ens_dir / "parent_state.pt"
    src = parent
    if not parent.exists():
        branches = sorted((ens_dir / "branches").glob("*/final_state.pt"))
        if not branches:
            raise FileNotFoundError(f"no parent_state.pt or final_state.pt under {ens_dir}")
        src = branches[0]
    state = _load_pt(src)
    return StaticState(
        box=_f64(state["box"]).reshape(-1),
        sigma=_f64(state["diameters"]).reshape(-1),
        positions=_f64(state["positions"]),
    )


@dataclass
class BranchStore:
    """Per-branch position frames keyed by branch index, plus shared metadata."""

    positions: dict[int, np.ndarray]            # index -> (T, N, 3)
    unwrapped: dict[int, np.ndarray]            # index -> (T, N, 3)
    seeds: dict[int, int]                       # index -> momentum seed
    steps: np.ndarray                           # (T,)

    def indices(self) -> set[int]:
        return set(self.positions)


def _provenance_entries(ens_dir: Path) -> list[tuple[int, int, Path]]:
    """Return ``(branch_index, momentum_seed, trajectory_path)`` for an ensemble.

    Prefers ``branch_provenance.json`` (carrying the matched momentum seed); falls
    back to globbing ``branches/*/trajectory.pt`` with the branch index used as a
    stand-in seed (matched-seed check then reduces to an index-alignment check).
    """

    prov = ens_dir / "branch_provenance.json"
    entries: list[tuple[int, int, Path]] = []
    if prov.exists():
        data = json.loads(prov.read_text())
        for b in data.get("branches", []):
            idx = int(b["index"])
            traj = ens_dir / b.get("trajectory_file", f"branches/{idx:06d}/trajectory.pt")
            seed = int(b.get("momentum_seed", idx))
            entries.append((idx, seed, traj))
    else:
        LOG.warning("%s: no branch_provenance.json; seeding by branch index", ens_dir.name)
        for bdir in sorted((ens_dir / "branches").iterdir()):
            if not bdir.is_dir():
                continue
            try:
                idx = int(bdir.name)
            except ValueError:
                continue
            entries.append((idx, idx, bdir / "trajectory.pt"))
    return sorted(entries, key=lambda e: e[0])


def load_branches(ens_dir: Path, *, max_branches: int | None = None) -> BranchStore:
    """Load every present branch trajectory of an ensemble (missing ones skipped)."""

    positions: dict[int, np.ndarray] = {}
    unwrapped: dict[int, np.ndarray] = {}
    seeds: dict[int, int] = {}
    steps: np.ndarray | None = None
    for idx, seed, traj in _provenance_entries(ens_dir):
        if not traj.exists():
            LOG.info("%s: branch %06d trajectory missing -- skipping", ens_dir.name, idx)
            continue
        obj = _load_pt(traj)
        positions[idx] = _f64(obj["positions"])
        unwrapped[idx] = _f64(obj["unwrapped_positions"])
        seeds[idx] = seed
        if steps is None:
            steps = obj["steps"].detach().cpu().numpy().astype(np.int64)
        if max_branches is not None and len(positions) >= max_branches:
            break
    if steps is None:
        raise FileNotFoundError(f"{ens_dir.name}: no readable branch trajectories")
    return BranchStore(positions=positions, unwrapped=unwrapped, seeds=seeds, steps=steps)


def _build_ensemble(
    store: BranchStore,
    indices: list[int],
    static: StaticState,
    times: np.ndarray,
) -> response.EnsembleTrajectory:
    pos = np.stack([store.positions[i] for i in indices], axis=1)      # (T, B, N, 3)
    unw = np.stack([store.unwrapped[i] for i in indices], axis=1)
    seeds = tuple(store.seeds[i] for i in indices)
    return response.EnsembleTrajectory(
        positions=pos, box=static.box, momentum_seeds=seeds,
        unwrapped_positions=unw, sigma=static.sigma, times=times,
    )


# ---------------------------------------------------------------------------
# Analysis of one perturbed ensemble against its config's unpert ensemble
# ---------------------------------------------------------------------------


def _recover_center(m0: np.ndarray, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Divergence-weighted centroid of the localized initial field ``M(t=0)``.

    The O_shell perturbation only displaces particles inside ``r_pert`` of the
    site, so ``M(0)`` is nonzero exactly on that shell; its minimum-image weighted
    centroid recovers the site center with no external metadata.
    """

    total = float(m0.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 0.5 * box
    anchor = int(np.argmax(m0))
    disp = minimum_image(positions - positions[anchor][None, :], box)
    weights = (m0 / total)[:, None]
    center = positions[anchor] + (weights * disp).sum(axis=0)
    return np.remainder(center, box)


@dataclass
class EnsembleResult:
    label: str
    root: str
    config: int
    site: int | None
    delta_index: int | None
    delta: float | None
    N: int
    n_branches: int
    lam: float
    lam_slope: float
    lam_r2: float
    lam_n_fit: int
    growing: bool
    saturated: bool
    onset_time: float
    D0: float
    D_sat: float
    v_b: float
    v_b_r2: float
    v_b_n_fit: int
    t_shield: float
    t_raw: float
    lag: float
    shielded: bool
    resolved: bool
    center: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class EnsembleFields:
    """The *cached* per-ensemble inputs to :func:`butterfly.analyze_config`.

    The expensive work -- ``torch.load`` of every branch, the matched-seed
    differencing, and (the real cost) the 48-branch cage-relative structural
    field -- is done exactly ONCE per ensemble and stored here as plain numpy.
    Every downstream re-fit (the aggregate, the fit-window/percentile robustness
    sweep, the shielding-threshold band) then re-runs only the cheap numpy fits
    on these arrays; nothing re-integrates or re-loads a trajectory.
    """

    label: str
    root: str
    config: int
    site: int | None
    delta_index: int | None
    delta: float | None
    N: int
    n_branches: int
    times: np.ndarray            # (T,)
    m_field: np.ndarray          # (T, N) raw matched-seed divergence
    s_field: np.ndarray          # (T, N) cage-relative structural divergence
    center: np.ndarray           # (3,) recovered perturbation-site centroid
    positions: np.ndarray        # (N, 3) reference positions
    box: np.ndarray              # (3,)


def build_ensemble_fields(
    pert_ref: EnsembleRef,
    pert_store: BranchStore,
    unpert_store: BranchStore,
    static: StaticState,
    *,
    dt: float,
    delta: float | None,
) -> EnsembleFields:
    """Do the once-only expensive work: both divergence channels + site center."""

    common = sorted(pert_store.indices() & unpert_store.indices())
    if not common:
        raise ValueError("no matched branch indices between perturbed and unpert ensembles")
    times = static_times(pert_store.steps, dt)

    pert = _build_ensemble(pert_store, common, static, times)
    unpert = _build_ensemble(unpert_store, common, static, times)

    m_field = response.divergence_field(pert, unpert)               # (T, N) raw
    s_field = response.cage_relative_divergence_field(pert, unpert)  # (T, N) structural
    center = _recover_center(m_field[0], static.positions, static.box)
    return EnsembleFields(
        label=pert_ref.label, root=pert_ref.root, config=pert_ref.config,
        site=pert_ref.site, delta_index=pert_ref.delta_index, delta=delta,
        N=int(m_field.shape[1]), n_branches=len(common),
        times=times, m_field=m_field, s_field=s_field,
        center=center, positions=static.positions, box=static.box,
    )


def result_from_fields(
    fields: EnsembleFields,
    *,
    r2_resolved: float,
    slope_frac: float = butterfly.SLOPE_FRAC_DEFAULT,
    window: int = butterfly.SLOPE_WINDOW_DEFAULT,
    front_percentile: float = 100.0,
    front_threshold_frac: float = butterfly.FRONT_THRESHOLD_FRAC,
    shield_threshold_frac: float = butterfly.SHIELD_THRESHOLD_FRAC,
) -> EnsembleResult:
    """Re-run the (cheap) two-channel butterfly fits on cached fields.

    Every analysis knob is overridable so the robustness sweep can vary the
    fit-window (``slope_frac``/``window``, driving both lambda and v_b), the
    ballistic-front percentile/threshold (v_b), and the shielding threshold
    (t_shield) without touching the cached arrays.
    """

    report = butterfly.analyze_config(
        fields.times, fields.m_field, S_field=fields.s_field,
        positions=fields.positions, center=fields.center, box=fields.box,
        front_percentile=front_percentile, front_threshold_frac=front_threshold_frac,
        shield_threshold_frac=shield_threshold_frac,
        slope_frac=slope_frac, window=window,
    )
    lyap = report.lyapunov
    vel = report.velocity
    shield = report.shielding
    resolved = bool(lyap.onset.growing and lyap.lam > 0.0
                    and np.isfinite(lyap.r2) and lyap.r2 >= r2_resolved)

    return EnsembleResult(
        label=fields.label, root=fields.root, config=fields.config,
        site=fields.site, delta_index=fields.delta_index, delta=fields.delta,
        N=fields.N, n_branches=fields.n_branches,
        lam=float(lyap.lam), lam_slope=float(lyap.slope), lam_r2=float(lyap.r2),
        lam_n_fit=int(lyap.n_fit), growing=bool(lyap.onset.growing),
        saturated=bool(lyap.onset.saturated), onset_time=float(lyap.onset.time),
        D0=float(lyap.D0), D_sat=float(report.D_sat),
        v_b=float(vel.v_b) if vel else float("nan"),
        v_b_r2=float(vel.r2) if vel else float("nan"),
        v_b_n_fit=int(vel.n_fit) if vel else 0,
        t_shield=float(shield.t_shield) if shield else float("nan"),
        t_raw=float(shield.t_onset_raw) if shield else float("nan"),
        lag=float(shield.lag) if shield else float("nan"),
        shielded=bool(shield.shielded) if shield else False,
        resolved=resolved, center=[float(x) for x in fields.center],
    )


def analyze_pair(
    pert_ref: EnsembleRef,
    pert_store: BranchStore,
    unpert_store: BranchStore,
    static: StaticState,
    *,
    dt: float,
    delta: float | None,
    r2_resolved: float,
) -> EnsembleResult:
    """Compute the two-channel butterfly report for one perturbed ensemble."""

    fields = build_ensemble_fields(
        pert_ref, pert_store, unpert_store, static, dt=dt, delta=delta
    )
    return result_from_fields(fields, r2_resolved=r2_resolved)


def static_times(steps: np.ndarray, dt: float) -> np.ndarray:
    return steps.astype(np.float64) * float(dt)


# ---------------------------------------------------------------------------
# Metadata helpers (dt, per-delta value) from the run config -- best effort
# ---------------------------------------------------------------------------


def _load_root_config(refs: list[EnsembleRef]) -> dict[str, Any]:
    """Best-effort load of the run-root ``config.yaml`` ``values`` block.

    Provides ``dt`` and the ``deltas`` list so the ``d{index}`` suffix can be
    turned into a physical delta.  Never fatal -- returns ``{}`` on any miss.
    """

    if not refs:
        return {}
    root = refs[0].root
    parent = refs[0].path.parent
    for candidate in (parent / root, parent / root / "config.yaml"):
        cfg = candidate if candidate.name == "config.yaml" else candidate / "config.yaml"
        if cfg.exists():
            try:
                import yaml
                data = yaml.safe_load(cfg.read_text()) or {}
                return data.get("values", data) if isinstance(data, dict) else {}
            except Exception as exc:  # pragma: no cover - yaml/parse robustness
                LOG.warning("could not parse %s: %s", cfg, exc)
                return {}
    return {}


def _ensemble_dt(ens_dir: Path, fallback: float) -> float:
    prov = ens_dir / "branch_provenance.json"
    if prov.exists():
        try:
            data = json.loads(prov.read_text())
            dt = data.get("controls", {}).get("dt")
            if dt:
                return float(dt)
        except Exception:  # pragma: no cover
            pass
    return fallback


# ---------------------------------------------------------------------------
# Pooling / verdict
# ---------------------------------------------------------------------------


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan")}
    # The reported "pooled +/- std" is an uncertainty spread over an ensemble
    # sample, so use the sample (ddof=1) standard deviation; a single observation
    # has no spread (reported as 0.0).  Bessel correction applies only to this
    # spread -- min/max/median are plain descriptors.
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size), "mean": float(arr.mean()),
        "std": std, "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()),
    }


def _intensive_by_N(results: list[EnsembleResult], attr: str,
                    only_resolved: bool = False) -> dict[str, Any]:
    by_n: dict[int, list[float]] = {}
    for r in results:
        if only_resolved and not r.resolved:
            continue
        v = getattr(r, attr)
        if np.isfinite(v):
            by_n.setdefault(r.N, []).append(v)
    means = {n: float(np.mean(vs)) for n, vs in by_n.items() if vs}
    if not means:
        return {"values": {}, "consistent": False, "rel_spread": float("nan"), "n_used": 0}
    check = butterfly.intensive_check(means)
    return {
        "values": {str(n): v for n, v in means.items()},
        "mean": check.mean, "rel_spread": check.rel_spread,
        "consistent": bool(check.consistent), "tol": check.tol, "n_used": check.n_used,
    }


def _rel_spread(values: Iterable[float]) -> float:
    """``(max - min) / |mean|`` over finite values -- the referee robustness knob."""

    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    mean = float(arr.mean())
    if mean == 0.0:
        return float("inf")
    return float((arr.max() - arr.min()) / abs(mean))


def _per_config(results: list[EnsembleResult]) -> dict[str, Any]:
    """Aggregate each config to one number, then the spread ACROSS configs.

    The n=2-configs caveat lives here: on the FSS root only c0/c1 carry branch
    data (both N=1500), so ``across_config`` is an honest 2-config spread, not a
    population statistic.
    """

    by_cfg: dict[str, list[EnsembleResult]] = {}
    for r in results:
        by_cfg.setdefault(f"c{r.config}", []).append(r)
    per: dict[str, Any] = {}
    lam_means: list[float] = []
    vb_means: list[float] = []
    dsat_means: list[float] = []
    ts_means: list[float] = []
    for cfg in sorted(by_cfg):
        grp = by_cfg[cfg]
        lam_m = _stats(r.lam for r in grp if r.resolved)["mean"]
        vb_m = _stats(r.v_b for r in grp)["mean"]
        dsat_m = _stats(r.D_sat for r in grp)["mean"]
        ts_m = _stats(r.t_shield for r in grp)["mean"]
        per[cfg] = {
            "N": grp[0].N, "n_ensembles": len(grp),
            "lambda_mean": lam_m, "v_b_mean": vb_m,
            "D_sat_mean": dsat_m, "t_shield_mean": ts_m,
            "frac_resolved": float(np.mean([1.0 if r.resolved else 0.0 for r in grp])),
        }
        lam_means.append(lam_m)
        vb_means.append(vb_m)
        dsat_means.append(dsat_m)
        ts_means.append(ts_m)
    across = {
        "n_configs": len(by_cfg),
        "lambda": {"values": lam_means, "rel_spread": _rel_spread(lam_means),
                   "mean": float(np.nanmean(lam_means)) if lam_means else float("nan")},
        "v_b": {"values": vb_means, "rel_spread": _rel_spread(vb_means),
                "mean": float(np.nanmean(vb_means)) if vb_means else float("nan")},
        "D_sat": {"values": dsat_means, "rel_spread": _rel_spread(dsat_means),
                  "mean": float(np.nanmean(dsat_means)) if dsat_means else float("nan")},
        "t_shield": {"values": ts_means, "rel_spread": _rel_spread(ts_means),
                     "mean": float(np.nanmean(ts_means)) if ts_means else float("nan")},
    }
    return {"per_config": per, "across_config": across}


def summarize(results: list[EnsembleResult]) -> dict[str, Any]:
    resolved = [r for r in results if r.resolved]
    lam_pool = _stats(r.lam for r in resolved) if resolved else _stats(r.lam for r in results)
    vb_pool = _stats(r.v_b for r in results)
    dsat_pool = _stats(r.D_sat for r in results)
    ts_pool = _stats(r.t_shield for r in results)
    traw_pool = _stats(r.t_raw for r in results)
    tsat_pool = _stats(r.onset_time for r in results)
    lag_pool = _stats(r.lag for r in results)

    frac_resolved = (len(resolved) / len(results)) if results else 0.0
    shielded_frac = (
        float(np.mean([1.0 if r.shielded else 0.0 for r in results])) if results else 0.0
    )
    cone_resolved = bool(
        lam_pool["n"] > 0 and lam_pool["median"] > 0.0 and frac_resolved >= 0.5
    )
    shielding = bool(np.isfinite(lag_pool["mean"]) and lag_pool["mean"] > 0.0)

    intensive = {
        "lambda": _intensive_by_N(results, "lam", only_resolved=True),
        "v_b": _intensive_by_N(results, "v_b"),
        "t_shield": _intensive_by_N(results, "t_shield"),
    }

    by_N: dict[str, Any] = {}
    for n in sorted({r.N for r in results}):
        grp = [r for r in results if r.N == n]
        by_N[str(n)] = {
            "n_ensembles": len(grp),
            "lambda_mean": _stats(r.lam for r in grp if r.resolved)["mean"],
            "v_b_mean": _stats(r.v_b for r in grp)["mean"],
            "t_shield_mean": _stats(r.t_shield for r in grp)["mean"],
        }

    per_cfg = _per_config(results)

    return {
        "pooled": {
            "lambda": lam_pool, "v_b": vb_pool, "D_sat": dsat_pool,
            "t_shield": ts_pool, "t_raw": traw_pool, "t_sat": tsat_pool,
            "lag": lag_pool,
            "frac_resolved": frac_resolved, "shielded_fraction": shielded_frac,
        },
        "by_N": by_N,
        "per_config": per_cfg["per_config"],
        "across_config": per_cfg["across_config"],
        "intensive": intensive,
        "verdict": {
            "cone_resolved": cone_resolved,
            "lambda_pooled_median": lam_pool["median"],
            "lambda_pooled_mean": lam_pool["mean"],
            "v_b_pooled_mean": vb_pool["mean"],
            "D_sat_pooled_mean": dsat_pool["mean"],
            "frac_lambda_resolved": frac_resolved,
            "shielding": shielding,
            "shielded_fraction": shielded_frac,
            "lag_mean": lag_pool["mean"],
            "intensive_lambda_consistent": intensive["lambda"]["consistent"],
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{nd}f}"


def render_markdown(results: list[EnsembleResult], summary: dict[str, Any],
                    run_dir: Path) -> str:
    lines = [
        f"# Gardner R0 butterfly-cone reanalysis",
        "",
        f"- run-dir: `{run_dir}`",
        f"- ensembles analyzed: {len(results)}",
        "",
        "| config | N | lambda | v_b | t_shield | D_sat |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda x: (x.config, x.site or 0, x.delta_index or 0)):
        lines.append(
            f"| {r.label} | {r.N} | {_fmt(r.lam)} | {_fmt(r.v_b)} "
            f"| {_fmt(r.t_shield)} | {_fmt(r.D_sat, 2)} |"
        )
    v = summary["verdict"]
    pooled = summary["pooled"]
    lam = pooled["lambda"]
    vb = pooled["v_b"]
    dsat = pooled["D_sat"]
    lines += [
        "",
        "## Pooled verdict",
        "",
        f"- pre-saturation Lyapunov cone: **{'RESOLVED' if v['cone_resolved'] else 'not resolved'}** "
        f"(pooled lambda = {_fmt(lam['mean'])} +/- {_fmt(lam['std'])}, "
        f"median {_fmt(lam['median'])}; {_fmt(v['frac_lambda_resolved']*100,1)}% of ensembles resolved)",
        f"- pooled v_b = {_fmt(vb['mean'])} +/- {_fmt(vb['std'])}; "
        f"pooled D_sat = {_fmt(dsat['mean'], 2)} +/- {_fmt(dsat['std'], 2)}",
        f"- structural shielding (t_shield > t_raw): "
        f"**{'YES' if v['shielding'] else 'no'}** "
        f"(mean lag = {_fmt(v['lag_mean'])}, shielded in {_fmt(v['shielded_fraction']*100,1)}% of ensembles)",
        f"- intensive lambda consistent across N: "
        f"**{v['intensive_lambda_consistent']}** "
        f"(rel spread = {_fmt(summary['intensive']['lambda']['rel_spread'])})",
    ]
    across = summary.get("across_config")
    if across:
        pc = summary.get("per_config", {})
        lines += ["", "## Per-config spread (n=%d configs)" % across["n_configs"], "",
                  "| config | N | lambda | v_b | D_sat | t_shield | frac_resolved |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for cfg in sorted(pc):
            g = pc[cfg]
            lines.append(
                f"| {cfg} | {g['N']} | {_fmt(g['lambda_mean'])} | {_fmt(g['v_b_mean'])} "
                f"| {_fmt(g['D_sat_mean'], 2)} | {_fmt(g['t_shield_mean'])} | {_fmt(g['frac_resolved'], 2)} |"
            )
        lines += [
            "",
            f"- across-config rel spread: lambda {_fmt(across['lambda']['rel_spread'])}, "
            f"v_b {_fmt(across['v_b']['rel_spread'])}, D_sat {_fmt(across['D_sat']['rel_spread'])} "
            f"(n={across['n_configs']} configs -- honest small-n spread, not a population statistic)",
        ]
    by_n = summary["by_N"]
    if by_n:
        lines += ["", "## By system size", "",
                  "| N | ensembles | lambda | v_b | t_shield |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for n, g in by_n.items():
            lines.append(
                f"| {n} | {g['n_ensembles']} | {_fmt(g['lambda_mean'])} "
                f"| {_fmt(g['v_b_mean'])} | {_fmt(g['t_shield_mean'])} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class FieldCache:
    """Everything the three deliverables share, loaded/integrated exactly once.

    ``fields`` are the per-ensemble cached channels (aggregate + sweep + shielding
    band all re-fit these); ``unpert_stores`` keep the raw unperturbed branch
    trajectories per config so the UNPERTURBED cage-decorrelation clock ``tau`` can
    be read from the same data without a second load.
    """

    fields: list[EnsembleFields]
    unpert_stores: dict[tuple[str, int], BranchStore]
    unpert_static: dict[tuple[str, int], StaticState]
    unpert_dt: dict[tuple[str, int], float]
    skipped: list[dict[str, str]]
    default_dt: float
    n_pert_found: int
    run_dir: Path


def collect_fields(
    run_dir: Path, *, max_branches: int | None = None, min_branches: int = 1,
) -> FieldCache:
    """The one expensive pass: load every ensemble and integrate both channels.

    Each config's unperturbed ensemble is loaded once and kept (for both the
    matched-seed differencing and the ``tau`` clock); each perturbed ensemble is
    loaded, differenced into cached M/S fields, and then released.  Downstream the
    fits are pure numpy -- nothing here is repeated by the sweeps.
    """

    refs = discover(run_dir)
    pert_refs = [r for r in refs if r.kind == "pert"]
    unpert_by_cfg: dict[tuple[str, int], EnsembleRef] = {
        (r.root, r.config): r for r in refs if r.kind == "unpert"
    }
    LOG.info("discovered %d perturbed + %d unpert ensembles under %s",
             len(pert_refs), len(unpert_by_cfg), run_dir)

    cfg_meta = _load_root_config(refs)
    deltas = cfg_meta.get("deltas") if isinstance(cfg_meta, dict) else None
    default_dt = float(cfg_meta.get("dt", 1.0)) if isinstance(cfg_meta, dict) else 1.0

    fields: list[EnsembleFields] = []
    skipped: list[dict[str, str]] = []
    unpert_stores: dict[tuple[str, int], BranchStore] = {}
    unpert_static: dict[tuple[str, int], StaticState] = {}
    unpert_dt: dict[tuple[str, int], float] = {}

    by_config: dict[tuple[str, int], list[EnsembleRef]] = {}
    for r in pert_refs:
        by_config.setdefault((r.root, r.config), []).append(r)

    for key in sorted(by_config):
        unpert_ref = unpert_by_cfg.get(key)
        if unpert_ref is None:
            for pr in by_config[key]:
                LOG.warning("%s: no matching unpert ensemble -- skipping", pr.label)
                skipped.append({"ensemble": pr.label, "reason": "no unpert partner"})
            continue
        try:
            static = load_static(unpert_ref.path)
            unpert_store = load_branches(unpert_ref.path, max_branches=max_branches)
        except Exception as exc:
            LOG.warning("%s: failed to load unpert (%s) -- skipping config", unpert_ref.label, exc)
            for pr in by_config[key]:
                skipped.append({"ensemble": pr.label, "reason": f"unpert load: {exc}"})
            continue

        dt = _ensemble_dt(unpert_ref.path, default_dt)
        unpert_stores[key] = unpert_store
        unpert_static[key] = static
        unpert_dt[key] = dt
        for pr in sorted(by_config[key], key=lambda x: (x.site or 0, x.delta_index or 0)):
            try:
                pert_store = load_branches(pr.path, max_branches=max_branches)
                if len(pert_store.indices() & unpert_store.indices()) < min_branches:
                    raise ValueError(
                        f"only {len(pert_store.indices() & unpert_store.indices())} matched "
                        f"branches (< min_branches={min_branches})"
                    )
                delta = None
                if deltas and pr.delta_index is not None and pr.delta_index < len(deltas):
                    delta = float(deltas[pr.delta_index])
                fld = build_ensemble_fields(
                    pr, pert_store, unpert_store, static, dt=dt, delta=delta
                )
                fields.append(fld)
                LOG.info("%s: cached fields N=%d B=%d", fld.label, fld.N, fld.n_branches)
            except Exception as exc:
                LOG.warning("%s: field build failed (%s) -- skipping", pr.label, exc)
                skipped.append({"ensemble": pr.label, "reason": str(exc)})

    return FieldCache(
        fields=fields, unpert_stores=unpert_stores, unpert_static=unpert_static,
        unpert_dt=unpert_dt, skipped=skipped, default_dt=default_dt,
        n_pert_found=len(pert_refs), run_dir=run_dir,
    )


def aggregate_report(cache: FieldCache, *, r2_resolved: float = R2_RESOLVED_DEFAULT) -> dict[str, Any]:
    """Deliverable 1: the persisted lambda/v_b/D_sat aggregate (default knobs)."""

    results = [result_from_fields(f, r2_resolved=r2_resolved) for f in cache.fields]
    for res in results:
        LOG.info("%s: N=%d lambda=%.4f (r2=%.3f) v_b=%.3f D_sat=%.2f t_shield=%.3f lag=%.3f",
                 res.label, res.N, res.lam, res.lam_r2, res.v_b, res.D_sat, res.t_shield, res.lag)
    summary = summarize(results)
    return {
        "run_dir": str(cache.run_dir),
        "default_dt": cache.default_dt,
        "n_ensembles_found": cache.n_pert_found,
        "n_ensembles_analyzed": len(results),
        "n_skipped": len(cache.skipped),
        "skipped": cache.skipped,
        "ensembles": [r.to_dict() for r in results],
        **summary,
    }


def run(run_dir: Path, *, max_branches: int | None = None,
        min_branches: int = 1, r2_resolved: float = R2_RESOLVED_DEFAULT) -> dict[str, Any]:
    """Discover, analyze and aggregate every ensemble under ``run_dir``."""

    cache = collect_fields(run_dir, max_branches=max_branches, min_branches=min_branches)
    return aggregate_report(cache, r2_resolved=r2_resolved)


# ---------------------------------------------------------------------------
# Deliverable 2: fit-window / front-percentile robustness sweep (referee shield)
# ---------------------------------------------------------------------------

SWEEP_SLOPE_FRACS: tuple[float, ...] = (0.3, 0.5, 0.7)
SWEEP_WINDOWS: tuple[int, ...] = (3, 5, 7)
SWEEP_FRONT_PERCENTILES: tuple[float, ...] = (90.0, 95.0, 100.0)
SWEEP_FRONT_THRESHOLD_FRACS: tuple[float, ...] = (0.05, 0.1, 0.2)
ROBUST_REL_TOL = 0.20          # headline may not move more than this across knobs
ROBUST_FRAC_RESOLVED_MIN = 0.5
ROBUST_R2_MIN = 0.9


def robustness_sweep(
    cache: FieldCache, *, r2_resolved: float = R2_RESOLVED_DEFAULT,
    slope_fracs: Sequence[float] = SWEEP_SLOPE_FRACS,
    windows: Sequence[int] = SWEEP_WINDOWS,
    front_percentiles: Sequence[float] = SWEEP_FRONT_PERCENTILES,
    front_threshold_fracs: Sequence[float] = SWEEP_FRONT_THRESHOLD_FRACS,
    rel_tol: float = ROBUST_REL_TOL,
    frac_resolved_min: float = ROBUST_FRAC_RESOLVED_MIN,
    r2_min: float = ROBUST_R2_MIN,
) -> dict[str, Any]:
    """Deliverable 2: does the headline survive the analysis-knob choices?

    Re-runs the (cheap) fits on the CACHED fields across the full grid of
    fit-window (``slope_frac`` x ``window``, driving both lambda and v_b) and
    ballistic-front (``percentile`` x ``threshold_frac``, driving v_b) knobs.
    Each cell reports the pooled lambda/v_b/D_sat, the resolved fraction and the
    median Lyapunov R^2; the verdict requires the headline relative spread across
    all cells to stay < ``rel_tol`` with every cell keeping frac_resolved and R^2
    above their floors.
    """

    cells: list[dict[str, Any]] = []
    for sf in slope_fracs:
        for w in windows:
            for fp in front_percentiles:
                for ft in front_threshold_fracs:
                    results = [
                        result_from_fields(
                            f, r2_resolved=r2_resolved, slope_frac=sf, window=w,
                            front_percentile=fp, front_threshold_frac=ft,
                        )
                        for f in cache.fields
                    ]
                    if not results:
                        continue
                    resolved = [r for r in results if r.resolved]
                    lam = _stats(r.lam for r in resolved) if resolved else _stats(r.lam for r in results)
                    r2_src = resolved if resolved else results
                    cells.append({
                        "slope_frac": sf, "window": w,
                        "front_percentile": fp, "front_threshold_frac": ft,
                        "lambda_median": lam["median"], "lambda_mean": lam["mean"],
                        "v_b_mean": _stats(r.v_b for r in results)["mean"],
                        "D_sat_mean": _stats(r.D_sat for r in results)["mean"],
                        "t_shield_mean": _stats(r.t_shield for r in results)["mean"],
                        "frac_resolved": len(resolved) / len(results),
                        "lam_r2_median": _stats(r.lam_r2 for r in r2_src)["median"],
                    })

    lam_vals = [c["lambda_median"] for c in cells]
    vb_vals = [c["v_b_mean"] for c in cells]
    dsat_vals = [c["D_sat_mean"] for c in cells]
    lam_spread = _rel_spread(lam_vals)
    vb_spread = _rel_spread(vb_vals)
    dsat_spread = _rel_spread(dsat_vals)
    min_frac = float(min((c["frac_resolved"] for c in cells), default=float("nan")))
    min_r2 = float(min((c["lam_r2_median"] for c in cells), default=float("nan")))

    robust = bool(
        cells
        and np.isfinite(lam_spread) and lam_spread < rel_tol
        and np.isfinite(vb_spread) and vb_spread < rel_tol
        and np.isfinite(dsat_spread) and dsat_spread < rel_tol
        and min_frac >= frac_resolved_min
        and min_r2 >= r2_min
    )
    return {
        "grid": {
            "slope_frac": list(slope_fracs), "window": list(windows),
            "front_percentile": list(front_percentiles),
            "front_threshold_frac": list(front_threshold_fracs),
        },
        "n_cells": len(cells),
        "cells": cells,
        "spread": {
            "lambda": {"rel_spread": lam_spread, "min": float(np.min(lam_vals)) if lam_vals else float("nan"),
                       "max": float(np.max(lam_vals)) if lam_vals else float("nan"),
                       "mean": float(np.mean(lam_vals)) if lam_vals else float("nan")},
            "v_b": {"rel_spread": vb_spread, "min": float(np.min(vb_vals)) if vb_vals else float("nan"),
                    "max": float(np.max(vb_vals)) if vb_vals else float("nan"),
                    "mean": float(np.mean(vb_vals)) if vb_vals else float("nan")},
            "D_sat": {"rel_spread": dsat_spread, "min": float(np.min(dsat_vals)) if dsat_vals else float("nan"),
                      "max": float(np.max(dsat_vals)) if dsat_vals else float("nan"),
                      "mean": float(np.mean(dsat_vals)) if dsat_vals else float("nan")},
        },
        "thresholds": {"rel_tol": rel_tol, "frac_resolved_min": frac_resolved_min, "r2_min": r2_min},
        "min_frac_resolved": min_frac,
        "min_lam_r2_median": min_r2,
        "robust": robust,
    }


# ---------------------------------------------------------------------------
# Deliverable 3: channel-S first read + the tau kill-shot pre-emption
# ---------------------------------------------------------------------------

SHIELD_THRESHOLD_BAND: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
TAU_ONE_OVER_E = float(np.exp(-1.0))   # 1/e ~ 0.3679


def _decay_time(t: np.ndarray, y: np.ndarray, level: float) -> tuple[float, bool]:
    """First (interpolated) time a decaying ``y`` drops to ``level``.

    Returns ``(tau, reached)``; when ``y`` never falls to ``level`` within the
    window ``tau`` is the window end (a *lower bound*) and ``reached`` is False.
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    below = y <= level
    if not np.any(below):
        return float(t[-1]), False
    idx = int(np.argmax(below))
    if idx == 0:
        return float(t[0]), True
    y0, y1 = float(y[idx - 1]), float(y[idx])
    t0, t1 = float(t[idx - 1]), float(t[idx])
    if y1 == y0:
        return t1, True
    frac = (level - y0) / (y1 - y0)
    return t0 + frac * (t1 - t0), True


def self_intermediate_scattering(store: BranchStore, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Orientation-averaged self-ISF ``F_s(q, t)`` of an unperturbed ensemble.

    ``F_s = < sinc(q |r_i(t) - r_i(0)|) >`` over particles and branches (the
    isotropic angular average of ``<cos(q.dr)>``), read from the UNPERTURBED
    unwrapped trajectories -- the intrinsic structural-relaxation clock with no
    perturbation involved.
    """

    times = static_times(store.steps, dt)
    acc: np.ndarray | None = None
    nb = 0
    for arr in store.unwrapped.values():
        disp = arr - arr[0][None, :, :]
        r = np.linalg.norm(disp, axis=2)          # (T, N)
        x = q * r
        with np.errstate(invalid="ignore", divide="ignore"):
            sinc = np.where(x > 1e-12, np.sin(x) / np.where(x > 0.0, x, 1.0), 1.0)
        fs = sinc.mean(axis=1)                     # (T,)
        acc = fs if acc is None else acc + fs
        nb += 1
    assert acc is not None
    return times, acc / nb


def self_overlap(store: BranchStore, dt: float, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Plain self-overlap ``Q(t) = < 1[ |r_i(t)-r_i(0)| < a ] >`` (cage scale a)."""

    times = static_times(store.steps, dt)
    acc: np.ndarray | None = None
    nb = 0
    for arr in store.unwrapped.values():
        disp = arr - arr[0][None, :, :]
        r = np.linalg.norm(disp, axis=2)
        q = (r < a).mean(axis=1)
        acc = q if acc is None else acc + q
        nb += 1
    assert acc is not None
    return times, acc / nb


def cage_relative_self_overlap(
    store: BranchStore, static: StaticState, dt: float, a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cage-relative self-overlap: same estimator on cage-relative displacements.

    Reuses the exact ``events.displacements.cage_relative_field`` machinery that
    the structural S channel is built on, so this ``tau_cage`` is the drift-robust
    cage-decorrelation clock referees would demand -- computed once here.
    """

    from butterfly_cone.events.displacements import cage_relative_field
    from butterfly_cone.events.trajectory import Trajectory

    times = static_times(store.steps, dt)
    acc: np.ndarray | None = None
    nb = 0
    for idx, unw in store.unwrapped.items():
        traj = Trajectory(
            unwrapped_positions=unw, times=times, sigma=static.sigma,
            box=static.box, positions=store.positions[idx],
        )
        field = cage_relative_field(traj)          # (T, N, 3)
        r = np.linalg.norm(field, axis=2)          # (T, N)
        q = (r < a).mean(axis=1)
        acc = q if acc is None else acc + q
        nb += 1
    assert acc is not None
    return times, acc / nb


def tau_clock(
    store: BranchStore, static: StaticState, dt: float, *,
    q_band: Sequence[float] | None = None, a: float = 0.3,
    probe_time: float | None = None, cage_relative: bool = True,
) -> dict[str, Any]:
    """Unperturbed structural-relaxation clock tau (F_s at 1/e) with a q band.

    ``q_band`` defaults to the cage wavevector ``q0 = 2*pi/<sigma>`` +/- 20%.  If
    ``F_s`` (or the overlap) never reaches its 1/e (resp. 1/e) level in the ~20
    t.u. window, ``tau`` is reported as a lower bound (``reached=False``).
    """

    mean_sigma = float(np.mean(static.sigma))
    q0 = 2.0 * np.pi / mean_sigma
    if q_band is None:
        q_band = (q0 / 1.2, q0, q0 * 1.2)

    isf: dict[str, Any] = {}
    tau_vals: list[float] = []
    tau_reached: list[bool] = []
    fs_at_probe: dict[str, float] = {}
    fs_at_tmax: dict[str, float] = {}
    times_ref: np.ndarray | None = None
    for q in q_band:
        times, fs = self_intermediate_scattering(store, dt, q)
        times_ref = times
        tau, reached = _decay_time(times, fs, TAU_ONE_OVER_E)
        isf[f"{q:.3f}"] = {
            "q": float(q), "tau": float(tau), "reached_1_over_e": bool(reached),
            "Fs_at_tmax": float(fs[-1]),
        }
        tau_vals.append(float(tau))
        tau_reached.append(bool(reached))
        fs_at_tmax[f"{q:.3f}"] = float(fs[-1])
        if probe_time is not None:
            fs_at_probe[f"{q:.3f}"] = float(np.interp(probe_time, times, fs))

    t_ov, q_ov = self_overlap(store, dt, a)
    tau_q, reached_q = _decay_time(t_ov, q_ov, TAU_ONE_OVER_E)
    overlap = {
        "a": a, "tau": float(tau_q), "reached_1_over_e": bool(reached_q),
        "Q_at_tmax": float(q_ov[-1]),
        "Q_at_probe": float(np.interp(probe_time, t_ov, q_ov)) if probe_time is not None else None,
    }

    cage_overlap: dict[str, Any] | None = None
    if cage_relative:
        t_cg, q_cg = cage_relative_self_overlap(store, static, dt, a)
        tau_cg, reached_cg = _decay_time(t_cg, q_cg, TAU_ONE_OVER_E)
        cage_overlap = {
            "a": a, "tau": float(tau_cg), "reached_1_over_e": bool(reached_cg),
            "Qcage_at_tmax": float(q_cg[-1]),
            "Qcage_at_probe": float(np.interp(probe_time, t_cg, q_cg)) if probe_time is not None else None,
        }

    return {
        "mean_sigma": mean_sigma, "q0": float(q0), "q_band": [float(q) for q in q_band],
        "t_max": float(times_ref[-1]) if times_ref is not None else float("nan"),
        "one_over_e": TAU_ONE_OVER_E,
        "isf": isf,
        "tau_min": float(np.min(tau_vals)) if tau_vals else float("nan"),
        "tau_max": float(np.max(tau_vals)) if tau_vals else float("nan"),
        "tau_all_lower_bounds": (not any(tau_reached)) if tau_reached else False,
        "fs_at_probe": fs_at_probe or None,
        "fs_at_tmax": fs_at_tmax,
        "overlap": overlap,
        "cage_overlap": cage_overlap,
        "probe_time": probe_time,
    }


def _pool_shield(cache: FieldCache, r2_resolved: float, shield_thr: float) -> dict[str, Any]:
    """Pool t_shield / t_raw / t_sat over ensembles at one shielding threshold."""

    results = [
        result_from_fields(f, r2_resolved=r2_resolved, shield_threshold_frac=shield_thr)
        for f in cache.fields
    ]
    ts = _stats(r.t_shield for r in results)
    tr = _stats(r.t_raw for r in results)
    tsat = _stats(r.onset_time for r in results)
    lag = _stats(r.lag for r in results)
    shielded_frac = float(np.mean([1.0 if r.shielded else 0.0 for r in results])) if results else float("nan")
    return {
        "shield_threshold_frac": shield_thr,
        "t_shield": ts, "t_raw": tr, "t_sat": tsat, "lag": lag,
        "shielded_fraction": shielded_frac, "n": len(results),
    }


def channel_s_report(
    cache: FieldCache, *, r2_resolved: float = R2_RESOLVED_DEFAULT,
    threshold_band: Sequence[float] = SHIELD_THRESHOLD_BAND,
) -> dict[str, Any]:
    """Deliverable 3: channel-S read (threshold-invariant t_shield) + tau kill-shot.

    Sweeps the shielding threshold to give t_shield an invariance band, then reads
    the UNPERTURBED cage-decorrelation clock tau on each config's unpert branches
    and compares.  ``t_shield << tau`` proves t_shield is not tau_alpha in disguise
    (the canonical referee kill-shot); ``t_shield`` vs ``t_sat`` is the two-clock
    separation verdict.
    """

    band = [_pool_shield(cache, r2_resolved, thr) for thr in threshold_band]
    ts_means = [b["t_shield"]["mean"] for b in band]
    tr_means = [b["t_raw"]["mean"] for b in band]
    # the default (0.5) threshold is the reference point for t_shield / t_raw / t_sat
    default_b = min(band, key=lambda b: abs(b["shield_threshold_frac"] - butterfly.SHIELD_THRESHOLD_FRAC))
    t_shield_ref = default_b["t_shield"]["mean"]
    t_raw_ref = default_b["t_raw"]["mean"]
    t_sat_ref = default_b["t_sat"]["mean"]

    tau_by_config: dict[str, Any] = {}
    tau_lb: list[float] = []
    for key in sorted(cache.unpert_stores):
        store = cache.unpert_stores[key]
        static = cache.unpert_static[key]
        dt = cache.unpert_dt[key]
        tau_by_config[f"c{key[1]}"] = tau_clock(
            store, static, dt, probe_time=t_shield_ref, a=0.3, cage_relative=True,
        )
        tau_by_config[f"c{key[1]}"]["N"] = int(static.sigma.shape[0])
        tau_lb.append(tau_by_config[f"c{key[1]}"]["tau_min"])

    tau_floor = float(np.min(tau_lb)) if tau_lb else float("nan")
    # separability verdicts (honest, each comparison stated explicitly)
    separates_vs_sat = bool(np.isfinite(t_shield_ref) and np.isfinite(t_sat_ref) and t_shield_ref > t_sat_ref)
    shields_vs_raw = bool(np.isfinite(t_shield_ref) and np.isfinite(t_raw_ref) and t_shield_ref > t_raw_ref)
    not_tau_alpha = bool(np.isfinite(t_shield_ref) and np.isfinite(tau_floor) and t_shield_ref < tau_floor)

    return {
        "threshold_band": {
            "values": list(threshold_band),
            "per_threshold": band,
            "t_shield_mean_min": float(np.nanmin(ts_means)) if ts_means else float("nan"),
            "t_shield_mean_max": float(np.nanmax(ts_means)) if ts_means else float("nan"),
            "t_shield_rel_spread": _rel_spread(ts_means),
            "t_raw_rel_spread": _rel_spread(tr_means),
        },
        "reference": {
            "shield_threshold_frac": butterfly.SHIELD_THRESHOLD_FRAC,
            "t_shield": t_shield_ref, "t_raw": t_raw_ref, "t_sat": t_sat_ref,
        },
        "tau_by_config": tau_by_config,
        "tau_floor_lower_bound": tau_floor,
        "n_unpert_configs": len(tau_by_config),
        "verdict": {
            "channel_s_separates_t_shield_gt_t_sat": separates_vs_sat,
            "structural_shielding_t_shield_gt_t_raw": shields_vs_raw,
            "t_shield_not_tau_alpha": not_tau_alpha,
            "t_shield_ref": t_shield_ref,
            "t_sat_ref": t_sat_ref,
            "t_raw_ref": t_raw_ref,
            "tau_floor_lower_bound": tau_floor,
            "t_shield_over_tau_ratio": (t_shield_ref / tau_floor)
            if (np.isfinite(t_shield_ref) and np.isfinite(tau_floor) and tau_floor != 0.0) else float("nan"),
        },
    }


def render_sweep_md(sweep: dict[str, Any]) -> list[str]:
    sp = sweep["spread"]
    thr = sweep["thresholds"]
    lines = [
        "",
        "## Robustness sweep (referee shield #1)",
        "",
        f"- grid: slope_frac {sweep['grid']['slope_frac']}, window {sweep['grid']['window']}, "
        f"front_percentile {sweep['grid']['front_percentile']}, "
        f"front_threshold_frac {sweep['grid']['front_threshold_frac']}  ({sweep['n_cells']} cells)",
        f"- lambda across cells: {_fmt(sp['lambda']['min'])}..{_fmt(sp['lambda']['max'])} "
        f"(rel spread {_fmt(sp['lambda']['rel_spread'])})",
        f"- v_b across cells: {_fmt(sp['v_b']['min'])}..{_fmt(sp['v_b']['max'])} "
        f"(rel spread {_fmt(sp['v_b']['rel_spread'])})",
        f"- D_sat across cells: {_fmt(sp['D_sat']['min'], 2)}..{_fmt(sp['D_sat']['max'], 2)} "
        f"(rel spread {_fmt(sp['D_sat']['rel_spread'])})",
        f"- min frac_resolved {_fmt(sweep['min_frac_resolved'], 2)} (floor {thr['frac_resolved_min']}); "
        f"min lambda-R^2 median {_fmt(sweep['min_lam_r2_median'])} (floor {thr['r2_min']})",
        f"- per-quantity vs tol {thr['rel_tol']}: "
        f"lambda {'PASS' if sp['lambda']['rel_spread'] < thr['rel_tol'] else 'FAIL'}, "
        f"v_b {'PASS' if sp['v_b']['rel_spread'] < thr['rel_tol'] else 'FAIL'}, "
        f"D_sat {'PASS' if sp['D_sat']['rel_spread'] < thr['rel_tol'] else 'FAIL'}",
        f"- **overall robust (all three < {thr['rel_tol']} AND floors met): {sweep['robust']}**",
    ]
    return lines


def render_channel_s_md(chan: dict[str, Any]) -> list[str]:
    v = chan["verdict"]
    ref = chan["reference"]
    tb = chan["threshold_band"]
    lines = [
        "",
        "## Channel-S read + tau kill-shot (referee shield #2)",
        "",
        f"- t_shield threshold-invariance band (frac {tb['values']}): "
        f"{_fmt(tb['t_shield_mean_min'])}..{_fmt(tb['t_shield_mean_max'])} t.u. "
        f"(rel spread {_fmt(tb['t_shield_rel_spread'])})",
        f"- reference (frac {ref['shield_threshold_frac']}): "
        f"t_shield={_fmt(ref['t_shield'])}, t_raw={_fmt(ref['t_raw'])}, t_sat={_fmt(ref['t_sat'])} t.u.",
        f"- unperturbed structural clock tau (F_s at 1/e), lower-bound floor = "
        f"{_fmt(v['tau_floor_lower_bound'])} t.u.",
    ]
    for cfg in sorted(chan["tau_by_config"]):
        tc = chan["tau_by_config"][cfg]
        lb = " (LOWER BOUND: F_s never reached 1/e in window)" if tc["tau_all_lower_bounds"] else ""
        lines.append(
            f"    - {cfg} (N={tc['N']}): tau in [{_fmt(tc['tau_min'])}, {_fmt(tc['tau_max'])}] t.u.{lb}; "
            f"F_s(t_max) in {[round(x,3) for x in tc['fs_at_tmax'].values()]}; "
            f"cage-overlap tau={_fmt(tc['cage_overlap']['tau']) if tc['cage_overlap'] else '--'}"
        )
    ratio = v["t_shield_over_tau_ratio"]
    lines += [
        "",
        f"- **channel-S separates (t_shield > t_sat): {v['channel_s_separates_t_shield_gt_t_sat']}** "
        f"(t_shield {_fmt(v['t_shield_ref'])} vs t_sat {_fmt(v['t_sat_ref'])} t.u.)",
        f"- structural shielding (t_shield > t_raw): {v['structural_shielding_t_shield_gt_t_raw']} "
        f"(t_shield {_fmt(v['t_shield_ref'])} vs t_raw {_fmt(v['t_raw_ref'])} t.u.)",
        f"- **t_shield is NOT tau_alpha in disguise: {v['t_shield_not_tau_alpha']}** "
        f"(t_shield/tau ~ {_fmt(ratio)}; t_shield {_fmt(v['t_shield_ref'])} << tau floor "
        f"{_fmt(v['tau_floor_lower_bound'])} t.u.)",
    ]
    return lines


def _print_verdict(report: dict[str, Any], sweep: dict[str, Any] | None = None,
                   chan: dict[str, Any] | None = None) -> None:
    v = report["verdict"]
    pooled = report["pooled"]
    lam = pooled["lambda"]
    print("=" * 68)
    print(f"Gardner R0 -- {report['n_ensembles_analyzed']} ensembles analyzed "
          f"({report['n_skipped']} skipped)")
    print("-" * 68)
    print(f"  pre-saturation Lyapunov cone : "
          f"{'RESOLVED' if v['cone_resolved'] else 'NOT resolved'}")
    print(f"      pooled lambda            = {_fmt(lam['mean'])} +/- {_fmt(lam['std'])} "
          f"(median {_fmt(lam['median'])}, {_fmt(v['frac_lambda_resolved']*100,1)}% resolved)")
    print(f"      pooled v_b               = {_fmt(pooled['v_b']['mean'])} "
          f"+/- {_fmt(pooled['v_b']['std'])}")
    print(f"      pooled D_sat             = {_fmt(pooled['D_sat']['mean'], 2)} "
          f"+/- {_fmt(pooled['D_sat']['std'], 2)}")
    print(f"  structural shielding         : "
          f"{'YES (t_shield > t_raw)' if v['shielding'] else 'no'}")
    print(f"      mean lag                 = {_fmt(v['lag_mean'])} "
          f"(shielded in {_fmt(v['shielded_fraction']*100,1)}% of ensembles)")
    print(f"  intensive lambda across N    : "
          f"{'consistent' if v['intensive_lambda_consistent'] else 'INconsistent'} "
          f"(rel spread {_fmt(report['intensive']['lambda']['rel_spread'])})")
    if sweep is not None:
        sp = sweep["spread"]
        print("-" * 68)
        print(f"  robustness sweep ({sweep['n_cells']} cells) : "
              f"{'ROBUST' if sweep['robust'] else 'NOT robust'}")
        print(f"      rel spread lambda/v_b/D_sat = {_fmt(sp['lambda']['rel_spread'])} / "
              f"{_fmt(sp['v_b']['rel_spread'])} / {_fmt(sp['D_sat']['rel_spread'])} "
              f"(tol {sweep['thresholds']['rel_tol']})")
    if chan is not None:
        cv = chan["verdict"]
        print("-" * 68)
        print(f"  channel-S t_shield={_fmt(cv['t_shield_ref'])} t_raw={_fmt(cv['t_raw_ref'])} "
              f"t_sat={_fmt(cv['t_sat_ref'])} tau>={_fmt(cv['tau_floor_lower_bound'])} t.u.")
        print(f"      separates (t_shield>t_sat) : {cv['channel_s_separates_t_shield_gt_t_sat']}")
        print(f"      NOT tau_alpha (t_shield<tau): {cv['t_shield_not_tau_alpha']} "
              f"(t_shield/tau ~ {_fmt(cv['t_shield_over_tau_ratio'])})")
    print("=" * 68)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gardner_r0.py",
        description="Zero-GPU Gardner butterfly-cone reanalysis of stored FSS "
                    "branch trajectories (pure reanalysis; no new MD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Gardner run root (e.g. runs/gardner/gardner-T0075-fss), the "
                        "parent (runs/gardner), or a single ensemble dir "
                        "(gardner-...--c0-s0-d1 -> its whole root set).")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path (a sibling .md summary table is written too). "
                        "Defaults to <run-dir>/gardner_r0.json.")
    p.add_argument("--max-branches", type=int, default=None,
                   help="Cap branches loaded per ensemble (smoke tests / memory).")
    p.add_argument("--min-branches", type=int, default=1,
                   help="Skip an ensemble with fewer matched branches than this.")
    p.add_argument("--r2-resolved", type=float, default=R2_RESOLVED_DEFAULT,
                   help="Min log-linear R^2 for a lambda to count as 'resolved'.")
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip the fit-window/front-percentile robustness sweep.")
    p.add_argument("--no-channel-s", action="store_true",
                   help="Skip the channel-S t_shield band + unperturbed tau clock.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (INFO) logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"error: run-dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    # ONE expensive pass; the aggregate, the sweep and the channel-S band all
    # re-fit the same cached fields (no reload, no re-integration).
    cache = collect_fields(run_dir, max_branches=args.max_branches,
                           min_branches=args.min_branches)
    report = aggregate_report(cache, r2_resolved=args.r2_resolved)

    sweep: dict[str, Any] | None = None
    chan: dict[str, Any] | None = None
    if report["n_ensembles_analyzed"] > 0 and not args.no_sweep:
        sweep = robustness_sweep(cache, r2_resolved=args.r2_resolved)
    if report["n_ensembles_analyzed"] > 0 and not args.no_channel_s:
        chan = channel_s_report(cache, r2_resolved=args.r2_resolved)

    out = args.out or (run_dir / "gardner_r0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    md_lines = render_markdown(
        [EnsembleResult(**e) for e in report["ensembles"]], report, run_dir).rstrip("\n").split("\n")
    if sweep is not None:
        sweep_path = out.with_name(out.stem + "_sweep.json")
        sweep_path.write_text(json.dumps(sweep, indent=2))
        md_lines += render_sweep_md(sweep)
        print(f"wrote {sweep_path}")
    if chan is not None:
        chan_path = out.with_name(out.stem + "_channelS.json")
        chan_path.write_text(json.dumps(chan, indent=2))
        md_lines += render_channel_s_md(chan)
        print(f"wrote {chan_path}")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n")

    _print_verdict(report, sweep=sweep, chan=chan)
    print(f"\nwrote {out}\nwrote {md_path}")
    if report["n_ensembles_analyzed"] == 0:
        print("warning: no ensembles were analyzed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
