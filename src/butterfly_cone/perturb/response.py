"""Response / FSS analysis for the causal-Gardner protocol-A probe.

Everything downstream of the branch integrator lives here: the matched-seed
branch-divergence field, the four declared in advance discrimination observables
(participation ratio, perturbation-size susceptibility, chaos length, and
non-self-averaging R_D(N)), and the frozen-threshold marginal-vs-defect decision
rule of gardner-package.md Sec 1.5.

The analysis operates purely on saved trajectory arrays in numpy float64,
reusing ``events.trajectory``'s numpy minimum-image convention (which matches the
engine constant-for-constant).  The non-self-averaging CI reuses
``stats.intervals.bootstrap_pairs`` with the CONFIG as the exchangeable unit --
never the branch, never the site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from butterfly_cone.events.trajectory import as_box, as_float_array
from butterfly_cone.stats._common import BranchTable
from butterfly_cone.stats.intervals import BootstrapResult, bootstrap_pairs


# ---------------------------------------------------------------------------
# Branch-divergence field (the crux -- matched-seed counterfactual)
# ---------------------------------------------------------------------------


@dataclass
class EnsembleTrajectory:
    """Wrapped branch positions of one branch ensemble, matched by branch index.

    ``positions`` has shape ``(T, B, N, 3)`` (frames, branches, particles, xyz);
    ``momentum_seeds`` is the per-branch Maxwell-Boltzmann seed tuple (the
    identity that MUST match between a paired unperturbed/perturbed ensemble).
    """

    positions: np.ndarray
    box: np.ndarray
    momentum_seeds: tuple[int, ...]
    unwrapped_positions: np.ndarray | None = None
    sigma: np.ndarray | None = None
    times: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = as_float_array(self.positions)
        if self.positions.ndim != 4 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape (T, B, N, 3)")
        self.box = as_box(self.box)
        self.momentum_seeds = tuple(int(s) for s in self.momentum_seeds)
        if len(self.momentum_seeds) != self.positions.shape[1]:
            raise ValueError("momentum_seeds must have one entry per branch")
        if self.unwrapped_positions is not None:
            self.unwrapped_positions = as_float_array(self.unwrapped_positions)

    @classmethod
    def from_result(cls, result: Any, box: Any, *, sigma: Any = None, times: Any = None) -> "EnsembleTrajectory":
        """Build from a :class:`branching.ensemble.BranchEnsembleResult`."""

        trajectory = result.trajectory
        positions = trajectory.positions.detach().cpu().numpy().astype(float)
        unwrapped = trajectory.unwrapped_positions.detach().cpu().numpy().astype(float)
        return cls(
            positions=positions,
            box=box,
            momentum_seeds=tuple(int(s) for s in result.branch_seeds),
            unwrapped_positions=unwrapped,
            sigma=None if sigma is None else as_float_array(sigma),
            times=None if times is None else as_float_array(times),
        )


def assert_matched_seeds(pert: EnsembleTrajectory, unpert: EnsembleTrajectory) -> None:
    """Guard the counterfactual: paired ensembles must share momentum seeds.

    A seed mismatch injects momentum chaos and silently fakes a marginal signal,
    so this check is mandatory before any differencing.
    """

    if pert.momentum_seeds != unpert.momentum_seeds:
        raise ValueError(
            "matched-seed contract violated: perturbed and unperturbed ensembles "
            "have different momentum seeds -- differencing would measure momentum "
            "chaos, not the delta-perturbation response"
        )


def branch_divergence(
    pert: EnsembleTrajectory,
    unpert: EnsembleTrajectory,
    box: Any = None,
) -> np.ndarray:
    """Per-branch divergence magnitude ``d_i^k(t)`` with shape ``(T, B, N)``."""

    assert_matched_seeds(pert, unpert)
    box_arr = as_box(pert.box if box is None else box)
    if pert.positions.shape != unpert.positions.shape:
        raise ValueError("paired ensembles must have identical (T, B, N, 3) shapes")
    # In-place minimum image (same elementwise ops as events.trajectory
    # minimum_image, so bit-identical): halves the number of (T, B, N, 3)
    # temporaries on the campaign hot path.
    # [micro-bench, (40, 8, 1500, 3) f64: divergence_field 75 ms -> 20 ms, ~3.7x]
    difference = pert.positions - unpert.positions
    images = difference / box_arr
    np.rint(images, out=images)
    images *= box_arr
    difference -= images
    return np.linalg.norm(difference, axis=3)


def divergence_field(
    pert: EnsembleTrajectory,
    unpert: EnsembleTrajectory,
    box: Any = None,
) -> np.ndarray:
    """Branch-mean divergence field ``D_i(t) = <d_i^k(t)>_k`` with shape ``(T, N)``."""

    return branch_divergence(pert, unpert, box).mean(axis=1)


def total_divergence(D_field: np.ndarray) -> np.ndarray:
    """Total divergence ``D(t) = sum_i D_i(t)`` from a ``(T, N)`` field -> ``(T,)``.

    A 1-D per-particle field returns a scalar sum.
    """

    values = as_float_array(D_field)
    return values.sum(axis=-1)


def cage_relative_divergence_field(
    pert: EnsembleTrajectory,
    unpert: EnsembleTrajectory,
) -> np.ndarray:
    """Drift-robust divergence: differences of cage-relative displacement fields.

    Robustness cross-check (Sec 4 red flag e): subtracts each particle's
    reference-frame neighbour-mean displacement before differencing the paired
    ensembles, removing uniform drift and slow affine shear.
    """

    from butterfly_cone.events.displacements import cage_relative_field
    from butterfly_cone.events.trajectory import Trajectory

    assert_matched_seeds(pert, unpert)
    if pert.unwrapped_positions is None or unpert.unwrapped_positions is None:
        raise ValueError("cage-relative divergence requires unwrapped positions")
    if pert.sigma is None:
        raise ValueError("cage-relative divergence requires per-particle sigma")
    n_frames, n_branches, _, _ = pert.positions.shape
    times = pert.times if pert.times is not None else np.arange(n_frames, dtype=float)
    accumulated: np.ndarray | None = None
    for k in range(n_branches):
        pert_traj = Trajectory(
            unwrapped_positions=pert.unwrapped_positions[:, k],
            times=times,
            sigma=pert.sigma,
            box=pert.box,
            positions=pert.positions[:, k],
        )
        unpert_traj = Trajectory(
            unwrapped_positions=unpert.unwrapped_positions[:, k],
            times=times,
            sigma=pert.sigma,
            box=pert.box,
            positions=unpert.positions[:, k],
        )
        diff = cage_relative_field(pert_traj) - cage_relative_field(unpert_traj)
        magnitude = np.linalg.norm(diff, axis=2)
        accumulated = magnitude if accumulated is None else accumulated + magnitude
    assert accumulated is not None
    return accumulated / n_branches


# ---------------------------------------------------------------------------
# (a) Response participation ratio -- spatial support
# ---------------------------------------------------------------------------


def participation_ratio(D_field: np.ndarray) -> float:
    """Return PR/N = ``(sum D_i^2)^2 / (N * sum D_i^4)`` for one per-particle field.

    Space-filling (uniform) response -> 1; a single-site spike -> 1/N.  A
    zero field returns 0.0 (no support).
    """

    values = as_float_array(D_field)
    if values.ndim != 1:
        raise ValueError("participation_ratio expects a 1-D per-particle field D_i")
    n = values.size
    sum_sq = float(np.sum(values**2))
    sum_quart = float(np.sum(values**4))
    if sum_quart <= 0.0:
        return 0.0
    return (sum_sq * sum_sq) / (n * sum_quart)


# ---------------------------------------------------------------------------
# (b) Perturbation-size susceptibility -- delta-scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Susceptibility:
    chi: float
    exponent: float
    linear_plateau: bool


def susceptibility(
    D_of_delta: Mapping[float, float] | Sequence[tuple[float, float]],
    *,
    plateau_tol: float = 0.15,
    delta_range: tuple[float, float] | None = None,
) -> Susceptibility:
    """Fit the delta-scaling of the total divergence ``<D>(delta)``.

    Returns ``chi`` (the through-origin linear-response slope ``<D> ~ chi*delta``),
    ``exponent`` (the log-log power-law exponent), and ``linear_plateau`` (True
    iff ``|exponent - 1| < plateau_tol``, i.e. a linear-response plateau exists).

    ``delta_range`` restricts the fit to the float32-validated sub-range (Sec 4).
    """

    items = sorted(dict(D_of_delta).items()) if isinstance(D_of_delta, Mapping) else sorted(D_of_delta)
    deltas = np.array([float(d) for d, _ in items], dtype=float)
    totals = np.array([float(v) for _, v in items], dtype=float)
    if delta_range is not None:
        low, high = delta_range
        keep = (deltas >= low) & (deltas <= high)
        deltas, totals = deltas[keep], totals[keep]
    if deltas.size < 2:
        raise ValueError("susceptibility needs at least two delta points")

    positive = (deltas > 0.0) & (totals > 0.0)
    if int(positive.sum()) >= 2:
        slope, _ = np.polyfit(np.log(deltas[positive]), np.log(totals[positive]), 1)
        exponent = float(slope)
    else:
        exponent = float("nan")
    denom = float(np.sum(deltas**2))
    chi = float(np.sum(deltas * totals) / denom) if denom > 0.0 else float("nan")
    linear_plateau = bool(np.isfinite(exponent) and abs(exponent - 1.0) < plateau_tol)
    return Susceptibility(chi=chi, exponent=exponent, linear_plateau=linear_plateau)


# ---------------------------------------------------------------------------
# (d) Chaos length -- spatial decay of the response
# ---------------------------------------------------------------------------


def _pair_autocorrelation(
    fields: np.ndarray,
    positions: np.ndarray,
    box: np.ndarray,
    edges: np.ndarray,
    *,
    direction: np.ndarray | None = None,
    along: bool | None = None,
    cos_tol: float = 0.7,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Binned two-point correlation of field fluctuations vs min-image distance.

    ``fields`` has shape ``(R, N)`` (R independent realizations sharing positions);
    returns ``(bin_sums, bin_counts)`` of the summed ``<dD_i dD_j>`` product per
    distance bin, normalisable to ``C(r)`` by the caller.  When ``direction`` is
    given, only pairs whose separation is (``along``) parallel or (not ``along``)
    transverse to that unit vector within ``cos_tol`` are counted.
    """

    return _pair_autocorrelations(
        fields, positions, box, edges,
        direction=direction, alongs=(along,), cos_tol=cos_tol, chunk=chunk,
    )[0]


def _pair_autocorrelations(
    fields: np.ndarray,
    positions: np.ndarray,
    box: np.ndarray,
    edges: np.ndarray,
    *,
    direction: np.ndarray | None,
    alongs: Sequence[bool | None],
    cos_tol: float = 0.7,
    chunk: int = 256,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One O(N^2) pair sweep binned under several angular selectors at once.

    The pair geometry (min-image displacements, distances, fluctuation products,
    direction cosines) is computed once per chunk and shared by every selector
    in ``alongs``, so the directional along+transverse split costs one sweep
    instead of two.  Each selector's masked elements keep the original selection
    order, making the binned sums bit-identical to independent sweeps.
    [micro-bench, N=800, R=6, 18 bins, directional split:
     chaos_length 171 ms -> 49 ms, ~3.5x; isotropic 87 ms -> 40 ms, ~2.2x]
    """

    n = positions.shape[0]
    n_bins = edges.size - 1
    r_max = float(edges[-1])
    # Subtract a single global mean (not a per-realization spatial mean): removing
    # each realization's DC mode biases the apparent decay of a long-correlation
    # field toward shorter lengths.
    fluct = fields - float(fields.mean())
    # Hoisted invariants (the unit vector was renormalised per chunk before).
    unit = None if direction is None else direction / np.linalg.norm(direction)
    cols = np.arange(n)
    results = [(np.zeros(n_bins), np.zeros(n_bins)) for _ in alongs]
    for start in range(0, n - 1, chunk):
        stop = min(start + chunk, n - 1)
        rows = np.arange(start, stop)
        disp = positions[rows, None, :] - positions[None, :, :]
        # In-place minimum image (bit-identical to events.trajectory minimum_image).
        images = disp / box
        np.rint(images, out=images)
        images *= box
        disp -= images
        dist = np.linalg.norm(disp, axis=2)
        # keep only ordered pairs i < j
        upper = rows[:, None] < cols[None, :]
        base_range = (dist < r_max) & (dist > 0.0) & upper
        cosang = None
        if unit is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = np.abs(np.einsum("rjk,k->rj", disp, unit) / np.where(dist > 0.0, dist, 1.0))
        # product of fluctuations, averaged over realizations
        prod = np.einsum("ri,rj->ij", fluct[:, rows], fluct) / fluct.shape[0]
        for slot, along in enumerate(alongs):
            if unit is None:
                in_range = base_range
            else:
                in_range = base_range & ((cosang >= cos_tol) if along else (cosang <= (1.0 - cos_tol)))
            r_sel = dist[in_range]
            p_sel = prod[in_range]
            idx = np.floor(r_sel / r_max * n_bins).astype(int)
            idx = np.clip(idx, 0, n_bins - 1)
            bin_sums, bin_counts = results[slot]
            np.add.at(bin_sums, idx, p_sel)
            np.add.at(bin_counts, idx, 1.0)
    return results


def _fit_decay_length(centers: np.ndarray, corr: np.ndarray, counts: np.ndarray) -> float:
    """Fit ``C(r) = A exp(-r/xi)`` on positive bins; return xi (inf if no decay)."""

    valid = (counts > 0.0) & (corr > 0.0) & np.isfinite(corr)
    if int(valid.sum()) < 2:
        return float("nan")
    x = centers[valid]
    y = np.log(corr[valid])
    weights = counts[valid]
    slope, _ = np.polyfit(x, y, 1, w=weights)
    if slope >= 0.0:
        return float("inf")
    return float(-1.0 / slope)


def chaos_length(
    D_field: np.ndarray,
    positions: np.ndarray,
    box: Any,
    *,
    r_max: float | None = None,
    n_bins: int = 20,
    direction: np.ndarray | Sequence[float] | None = None,
    cos_tol: float = 0.7,
) -> float | dict[str, float]:
    """Fit the chaos length xi from the field's radial autocorrelation.

    ``D_field`` is ``(N,)`` (one realization) or ``(R, N)`` (R realizations that
    share ``positions``; their autocorrelations are pooled to reduce noise).
    With ``direction`` (a unit vector) the pairs are split into components along
    vs transverse to it and a ``{"along": xi, "transverse": xi}`` dict is
    returned -- the O_strain directional chaos length.
    """

    fields = as_float_array(D_field)
    if fields.ndim == 1:
        fields = fields[None, :]
    positions = as_float_array(positions)
    box = as_box(box)
    n = positions.shape[0]
    if fields.shape[1] != n:
        raise ValueError("D_field and positions disagree on N")
    if r_max is None:
        r_max = 0.5 * float(np.min(box))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    variance = float(np.mean((fields - float(fields.mean())) ** 2))
    if variance <= 0.0:
        return float("nan") if direction is None else {"along": float("nan"), "transverse": float("nan")}

    if direction is None:
        sums, counts = _pair_autocorrelation(fields, positions, box, edges)
        corr = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0) / variance
        return _fit_decay_length(centers, corr, counts)

    unit = as_float_array(direction)
    # Single pair sweep for both angular selectors (see _pair_autocorrelations).
    binned = _pair_autocorrelations(
        fields, positions, box, edges, direction=unit, alongs=(True, False), cos_tol=cos_tol
    )
    result: dict[str, float] = {}
    for (label, _), (sums, counts) in zip((("along", True), ("transverse", False)), binned):
        corr = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0) / variance
        result[label] = _fit_decay_length(centers, corr, counts)
    return result


# ---------------------------------------------------------------------------
# (c) Non-self-averaging R_D(N) -- config is the exchangeable unit
# ---------------------------------------------------------------------------


def r_d(D_values: Sequence[float]) -> float:
    """Between-config normalised variance ``R_D = Var[D] / <D>^2``."""

    values = np.asarray(list(D_values), dtype=float)
    if values.size == 0:
        raise ValueError("need at least one config value")
    mean = float(values.mean())
    if mean == 0.0:
        return float("nan")
    return float(values.var(ddof=0) / (mean * mean))


def _r_d_statistic(table: BranchTable) -> float:
    """R_D over a branch table, collapsing each config (pair_id) to one D first.

    This is the estimator handed to ``bootstrap_pairs``; because the bootstrap
    resamples pair_ids (configs) and never rows, the config is the exchangeable
    unit even when a config carries many branch/site rows.
    """

    # Single first-occurrence pass (== group_by insertion order + rows[0]); the
    # statistic runs once per bootstrap replicate, so skipping the per-config
    # row-list construction matters.  [micro-bench, 2 sizes x 200 configs,
    # n_boot=300: non_self_averaging 793 ms -> 198 ms, ~4x -- the remainder
    # is the shared stats.intervals resampling machinery]
    per_config: list[float] = []
    seen: set[object] = set()
    for row in table:
        pair_id = row["pair_id"]
        if pair_id not in seen:
            seen.add(pair_id)
            per_config.append(float(row["D"]))
    return r_d(per_config)


@dataclass(frozen=True)
class NonSelfAveraging:
    per_n: dict[int, BootstrapResult]
    point: dict[int, float]

    def ratio(self, n_large: int, n_small: int) -> float:
        """``R_D(n_large) / R_D(n_small)`` -- ~0.5 self-averaging, ~1 marginal."""

        small = self.point[n_small]
        if small == 0.0 or not np.isfinite(small):
            return float("nan")
        return self.point[n_large] / small


def _build_config_table(D_by_config: Mapping[Any, float]) -> list[dict[str, Any]]:
    return [{"pair_id": str(cfg), "D": float(value)} for cfg, value in D_by_config.items()]


def non_self_averaging(
    D_by_config_by_N: Mapping[int, Mapping[Any, float]],
    *,
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> NonSelfAveraging:
    """Config-bootstrapped ``R_D(N)`` for each system size.

    ``D_by_config_by_N`` maps ``N -> {config_id -> total divergence D}``.  Each
    size gets a ``bootstrap_pairs`` CI with the config as the bootstrap unit
    (``stats.intervals``), and the point ``R_D`` is stored for the FSS ratio.
    """

    per_n: dict[int, BootstrapResult] = {}
    point: dict[int, float] = {}
    for n, by_config in D_by_config_by_N.items():
        table = _build_config_table(by_config)
        if len(table) < 1:
            raise ValueError(f"N={n} has no configs")
        boot = bootstrap_pairs(
            table, _r_d_statistic, n_boot=n_boot, alpha=alpha, seed=seed + int(n)
        )
        per_n[int(n)] = boot
        point[int(n)] = boot.point
    return NonSelfAveraging(per_n=per_n, point=point)


# ---------------------------------------------------------------------------
# (e) Declared in advance marginal-vs-defect decision rule (Sec 1.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionThresholds:
    """Frozen thresholds for the Sec 1.5 verdict rule (illustrative defaults)."""

    pr_slope_flat_tol: float = 1e-3          # PR/N slope >= -tol counts as flat/rising
    pr_level_min: float = 0.1                # PR/N level for a supported (marginal) phase
    rd_ratio_marginal_min: float = 0.8       # R_D(3000)/R_D(1500) ~ 1 (not shrinking)
    rd_ratio_defect_max: float = 0.65        # R_D ratio ~ 0.5 (1/N self-averaging)
    chi_linear_tol: float = 0.15             # |exponent - 1| below this => linear
    min_prep_depths_consistent: int = 2


@dataclass(frozen=True)
class AxisSummary:
    """Operationalised inputs to the verdict, one number per axis."""

    pr_slope: float            # slope of PR/N vs N over 1500 -> 3000
    pr_level: float            # PR/N at the largest N
    rd_ratio: float            # R_D(large)/R_D(small)
    chi_exponent: float        # (b) power-law exponent
    chi_linear_plateau: bool   # (b) linear-response plateau present
    xi_growing: bool           # (d) xi grows past R_pert as delta -> 0
    xi_saturates: bool         # (d) xi saturates <~ 2 sigma
    n_prep_depths_consistent: int = 2


def decide(axes: AxisSummary, thresholds: DecisionThresholds = DecisionThresholds()) -> str:
    """Return the declared in advance verdict: ``marginal`` | ``sparse_defects`` | ``bounded``.

    Primary discriminator is (a)+(c) jointly; (b) and (d) corroborate.
    """

    anomalous_chi = not axes.chi_linear_plateau
    pr_flat_or_rising = axes.pr_slope >= -thresholds.pr_slope_flat_tol
    pr_supported = axes.pr_level >= thresholds.pr_level_min
    rd_not_self_averaging = (
        np.isfinite(axes.rd_ratio) and axes.rd_ratio >= thresholds.rd_ratio_marginal_min
    )
    rd_self_averaging = (
        np.isfinite(axes.rd_ratio) and axes.rd_ratio <= thresholds.rd_ratio_defect_max
    )
    prep_consistent = axes.n_prep_depths_consistent >= thresholds.min_prep_depths_consistent

    marginal = (
        pr_flat_or_rising
        and pr_supported
        and rd_not_self_averaging
        and (anomalous_chi or axes.xi_growing)
        and prep_consistent
    )
    if marginal:
        return "marginal"

    sparse_defects = (
        (not pr_flat_or_rising)
        and rd_self_averaging
        and axes.chi_linear_plateau
        and axes.xi_saturates
    )
    if sparse_defects:
        return "sparse_defects"

    return "bounded"
