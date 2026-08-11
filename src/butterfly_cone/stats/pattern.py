"""Spatial pattern-vs-null statistics for the firebreak experiment (Task M).

Turns a branch's per-particle mobility field into the three whole-field
statistics the demonstrative campaign scores, then reduces them to
per-parent paired contrasts that feed the existing Gate-0 CI/null machinery
(`intervals.bootstrap_pairs`, `intervals.paired_diff_normal_approx`)
**unchanged**:

- **W -- template-projection score** (`projection_score`): how much of a
  branch's mobility mass sits under a fixed spatial template (a firebreak
  line/wall/ring's geometry).
- **B -- firebreak crossing flux / protected-region suppression**
  (`crossing_flux`, `protected_region_rate`): whether a connected
  rearrangement spans the barrier, and how much event mass lands in the
  protected region.
- **P -- directional/anisotropy score, §D** (`anisotropy_score`): whether
  rearrangement reach is preferentially along a channel axis.

## Score <-> §B-endpoint mapping

`projection_score` is normalized so that when the mobility field equals a
uniform background `q_bg` plus a bump of height `Δq` confined to the
template's support, `S - S_bg ≈ Δq` *exactly* (up to grid discretization):
with `w_template` L1-normalized (`Σ_r w = 1`) and support entirely inside
the bump, `S = Σ_r w(r)·m(r) = q_bg·Σw + Δq·Σ_{r in bump} w(r) = q_bg + Δq`
when the bump exactly covers the template support, and `S_bg = q_bg`. So
`S` (and, symmetrically, `protected_region_rate` and `anisotropy_score`'s
`A`) is reported in the *same event-probability / mobility units* as
Gate-0's `E[Δq]` endpoint -- `D_pattern` and `ATE_suppress` below are
therefore directly comparable to `estimands.ATEEstimate.pair_weighted` in
magnitude, even though they are computed from a wholly different
(continuous, spatial) endpoint.

## Continuous endpoint, not binary (read this before reusing `estimands`)

Gate-0's `estimands.paired_contrasts` / `PairArmCounts` machinery is built
for a **binary** per-branch endpoint (`as01` enforces a 0/1 value) reduced
to a per-arm proportion `k/n`. The projection/protected-rate/anisotropy
scores here are **continuous** floats -- one already-computed number per
branch, not a count. Reusing the binary path would either raise (`as01`
rejects non-0/1 floats) or, worse, silently misinterpret a continuous score
as a count-of-events-per-branch. This module therefore does NOT import or
call `estimands.PairArmCounts`, `estimands.collect_pair_arm_counts`, or
`estimands.paired_contrasts`. It reuses only the two endpoint-agnostic
pieces of `intervals` -- `bootstrap_pairs` (resamples *pairs*, calls an
arbitrary `table -> float` statistic) and `paired_diff_normal_approx`
(takes an already-collapsed list of per-pair numbers) -- both of which were
already written generically enough to need no change. `pattern_table`'s
rows accordingly carry `pair_id` and `arm_family` (required by
`bootstrap_pairs`' `group_by`) but deliberately omit `arm_sign`: there is
no "+"/"-" branch pair within one (parent, arm) here, only one
already-averaged scalar per (parent, arm).

## Parent is the exchangeability unit (two sentences)

Branches within a (parent, arm) share the same patterned template geometry,
grid, and (for a given parent) the same local cavity/channel environment,
so they are correlated and a branch is not an independent draw -- only the
parent is. `pattern_table` enforces this structurally rather than by
convention: it collapses each (parent, arm)'s `B` branch scores to exactly
one mean **before** the row ever reaches `intervals.bootstrap_pairs`, so
there is no branch-level column left to accidentally resample even by a
future caller's mistake (contrast with `estimands`, which keeps branch rows
and instead trusts every caller to `group_by` pairs first).

## Dual `CavitySpec` hazard (flagged per the task spec)

`rcce.cavity.CavitySpec` (`core_radius`/`buffer_radius`) and
`events.config.CavitySpec` (`r_core`/`r_annulus`) are two *different*
dataclasses with overlapping names. This module imports **neither** --
every public function here takes plain `numpy` arrays/masks and raw
coordinates (a template weight array, a boolean protected-region mask, a
`(axis, lo, hi)` barrier slab, a channel-axis vector and origin point).
`Grid` below is this module's own minimal cavity-free grid geometry, built
only from a box and a cell size.

## Module README note

Editing `src/butterfly_cone/stats/README.md` is out of this module's ownership
scope (task boundary: own only `pattern.py` and `tests/stats/test_pattern.py`,
and existing `stats/` files must not be modified). The four things the task
spec's "Definition of done" asks a module README to cover -- the score <->
endpoint mapping, the parent-as-unit argument, the continuous-vs-binary
note, and the dual-`CavitySpec` flag -- are therefore documented here, in
this module's own top-of-file docstring, instead.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from butterfly_cone.events import (
    ClusterConfig,
    HopConfig,
    Trajectory,
    build_events,
    cage_relative_field,
    detect_hops,
    is_persistent,
    magnitude_field,
    minimum_image,
)

from . import intervals
from ._common import ARM_FAMILY_COL, PAIR_ID_COL, BranchTable, BranchRow, require_columns

Mobility = Literal["cage_mag", "event_density"]
Side = Literal["lo", "hi"]

__all__ = [
    "Grid",
    "coarse_grain_mobility",
    "projection_score",
    "BarrierSlab",
    "protected_region_rate",
    "crossing_flux",
    "AnisotropyResult",
    "anisotropy_score",
    "pattern_table",
    "pattern_contrast_bootstrap",
    "d_pattern",
    "ate_suppress",
    "PatternShuffleNullResult",
    "pattern_shuffle_null",
]


# --------------------------------------------------------------------------- #
# Grid geometry (deliberately cavity-free -- see the dual-CavitySpec note)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grid:
    """A uniform Cartesian grid covering the periodic box `[0, box)`.

    Cells are ~`cell_size` on a side (rounded up per axis so the box is
    covered exactly); a cavity-scale cell (`cell_size ≈ σ`) is the natural
    resolution per the task spec's Definitions. This is the module's own
    minimal grid geometry -- no `CavitySpec` of either flavour is used or
    needed to define it.
    """

    box: np.ndarray
    cell_size: float

    def __post_init__(self) -> None:
        box = np.asarray(self.box, dtype=float)
        object.__setattr__(self, "box", box)
        if box.shape != (3,) or np.any(box <= 0.0) or not np.isfinite(box).all():
            raise ValueError("box must be a finite, positive length-3 array")
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")

    @property
    def n_cells(self) -> tuple[int, int, int]:
        return tuple(max(1, int(np.ceil(b / self.cell_size))) for b in self.box)

    @property
    def n_total(self) -> int:
        nx, ny, nz = self.n_cells
        return nx * ny * nz

    def cell_index(self, positions: np.ndarray) -> np.ndarray:
        """Flat cell index (C order: x slowest, z fastest) for each position.

        Positions are wrapped into `[0, box)` first, matching `Trajectory`'s
        wrapped-position convention.
        """
        positions = np.remainder(np.asarray(positions, dtype=float), self.box)
        nx, ny, nz = self.n_cells
        ix = np.clip((positions[..., 0] / self.box[0] * nx).astype(np.int64), 0, nx - 1)
        iy = np.clip((positions[..., 1] / self.box[1] * ny).astype(np.int64), 0, ny - 1)
        iz = np.clip((positions[..., 2] / self.box[2] * nz).astype(np.int64), 0, nz - 1)
        return (ix * ny + iy) * nz + iz

    def cell_centers(self) -> np.ndarray:
        """Cell-center coordinates, shape `(n_total, 3)`, in the same flat
        (x slowest, z fastest) order as `cell_index`."""
        nx, ny, nz = self.n_cells
        dx, dy, dz = self.box[0] / nx, self.box[1] / ny, self.box[2] / nz
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        zs = (np.arange(nz) + 0.5) * dz
        return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Deliverable 1: mobility field coarse-graining
# --------------------------------------------------------------------------- #


def coarse_grain_mobility(
    traj: Trajectory,
    *,
    grid: Grid,
    horizon_frame: int,
    mobility: Mobility = "cage_mag",
    reference_frame: int = 0,
    hop_config: HopConfig | None = None,
    cluster_config: ClusterConfig | None = None,
    persistent_only: bool = True,
) -> np.ndarray:
    """One branch's coarse-grained mobility field `m(r, Δt)`, flat shape `(grid.n_total,)`.

    `mobility="cage_mag"` (frozen choice, §E): per-particle cage-relative
    displacement magnitude at `horizon_frame` (relative to `reference_frame`),
    binned by the particle's position at `horizon_frame` and cell-averaged.
    Empty cells are filled with `0.0` (documented background-zero
    convention, chosen so an all-empty template contributes nothing to
    `projection_score` rather than propagating NaN).

    `mobility="event_density"`: persistent rearranging-cluster centroids
    (`events.hops.detect_hops` -> `events.clusters.build_events`, optionally
    filtered to `events.clusters.is_persistent`), binned by centroid and
    L1-normalized to sum to 1 over the grid -- an event-probability density,
    directly on the same units `projection_score` expects. This reuses
    `events.clusters`' persistent-cluster centroids directly rather than
    routing through `events.attribution.AttributionResult.where` (which
    requires a `CavitySpec` per monitored cavity); this module is a
    whole-field statistic, not a per-cavity one, and this sidesteps the
    dual-`CavitySpec` hazard entirely. Flagged as a deliberate deviation from
    the spec's literal "OR the event-density field from `events.attribution`"
    wording -- `AttributionResult.where` for convention B/D is exactly a
    persistent cluster's centroid (`data.events[0].centroid`), so the values
    are the same; only the cavity-scoped wrapper is skipped.

    Both branches are deterministic given `traj`/`grid`/config (no RNG).
    """
    if mobility == "cage_mag":
        field = cage_relative_field(traj, reference_frame=reference_frame)
        per_particle = magnitude_field(field)[horizon_frame]
        positions = traj.positions[horizon_frame]
        idx = grid.cell_index(positions)
        sums = np.bincount(idx, weights=per_particle, minlength=grid.n_total).astype(float)
        counts = np.bincount(idx, minlength=grid.n_total).astype(float)
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)

    if mobility == "event_density":
        hop_cfg = hop_config or HopConfig()
        clus_cfg = cluster_config or ClusterConfig()
        events = build_events(detect_hops(traj, hop_cfg), traj, clus_cfg)
        if persistent_only:
            events = [e for e in events if is_persistent(e, traj, clus_cfg)]
        if not events:
            return np.zeros(grid.n_total, dtype=float)
        centroids = np.array([e.centroid for e in events])
        idx = grid.cell_index(centroids)
        density = np.bincount(idx, minlength=grid.n_total).astype(float)
        total = float(density.sum())
        return density / total if total > 0.0 else density

    raise ValueError("mobility must be 'cage_mag' or 'event_density'")


# --------------------------------------------------------------------------- #
# Deliverable 2: template-projection score (W)
# --------------------------------------------------------------------------- #


def projection_score(
    mobility_grid: np.ndarray,
    w_template: np.ndarray,
    *,
    validate: bool = True,
    atol: float = 1e-3,
) -> float:
    """`S = Σ_r w_template(r)·m̄(r, Δt)`.

    `w_template` must already be L1-normalized (`Σ_r w = 1`, the caller's
    responsibility per the task spec -- `rcce/stencil.py`'s template
    record); by default this is checked (`validate=True`) so a
    denormalized template fails loudly here rather than silently biasing
    every downstream contrast. See the module docstring for the
    `S - S_bg ≈ Δq` normalization argument.
    """
    mobility_grid = np.asarray(mobility_grid, dtype=float)
    w_template = np.asarray(w_template, dtype=float)
    if mobility_grid.shape != w_template.shape:
        raise ValueError("mobility_grid and w_template must have the same shape")
    if validate and not np.isclose(float(w_template.sum()), 1.0, atol=atol):
        raise ValueError(
            f"w_template must be L1-normalized (sum to 1); got sum={float(w_template.sum())!r}"
        )
    return float(np.sum(w_template * mobility_grid))


# --------------------------------------------------------------------------- #
# Deliverable 3: firebreak statistics (B)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BarrierSlab:
    """A planar barrier slab normal to `axis`, spanning `[lo, hi]` along it."""

    axis: int
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        if self.hi <= self.lo:
            raise ValueError("hi must exceed lo")


def _side_mask(coord: np.ndarray, slab: BarrierSlab, side: Side) -> np.ndarray:
    if side == "lo":
        return coord < slab.lo
    if side == "hi":
        return coord > slab.hi
    raise ValueError("side must be 'lo' or 'hi'")


def protected_region_rate(
    mobility_grid_or_positions: np.ndarray | Sequence,
    protected_mask: np.ndarray,
    *,
    grid: Grid | None = None,
) -> float:
    """`P(event in protected region)` for one branch.

    Accepts either:

    - a per-cell field with the SAME shape as `protected_mask` (a mobility
      grid from `coarse_grain_mobility`, or an event-density grid) -- returns
      the protected-cell-weighted mass fraction `Σ_{r in mask} field(r) / Σ_r field(r)`; or
    - a raw sequence of event/rearrangement-centroid positions `(N, 3)` (or
      objects with a `.centroid` attribute, e.g. `events.clusters.Event`) --
      `grid` (the same `Grid` used to build `protected_mask`) is then
      required to bin them, and the return value is the fraction of
      centroids landing in a protected cell.

    Returns `0.0` for an all-zero field or an empty event list (no events
    observed is not evidence of suppression, but is a well-defined rate).
    """
    mask = np.asarray(protected_mask, dtype=bool)
    positions_or_field = mobility_grid_or_positions
    if hasattr(positions_or_field, "__len__") and len(positions_or_field) > 0 and hasattr(
        positions_or_field[0], "centroid"
    ):
        positions_or_field = [item.centroid for item in positions_or_field]

    candidate = np.asarray(positions_or_field, dtype=float)
    if candidate.shape == mask.shape:
        total = float(candidate.sum())
        if total <= 0.0:
            return 0.0
        return float(candidate[mask].sum() / total)

    if candidate.size == 0:
        return 0.0
    positions = np.atleast_2d(candidate)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            "expected a field with protected_mask's shape, or (N, 3) event positions"
        )
    if grid is None:
        raise ValueError("grid is required when passing raw event positions")
    idx = grid.cell_index(positions)
    flat_mask = mask.reshape(-1)
    return float(np.mean(flat_mask[idx]))


def crossing_flux(
    traj: Trajectory,
    barrier_slab: BarrierSlab,
    source_side: Side,
    target_side: Side,
    *,
    events: Sequence | None = None,
    hop_config: HopConfig | None = None,
    cluster_config: ClusterConfig | None = None,
    persistent_only: bool = True,
) -> float:
    """`Φ` -- indicator (`0.0`/`1.0`) that some connected rearrangement spans
    from `source_side` to `target_side` of `barrier_slab`, for one branch.

    A rearrangement (an `events.clusters.Event`, persistent by default) is
    judged by its members' positions at the event's onset frame: it "spans"
    the barrier when at least one member sits on `source_side` and at least
    one sits on `target_side`. `events` defaults to persistent clusters
    detected from `traj` with default configs (the same code path
    `coarse_grain_mobility(..., mobility="event_density")` uses); pass a
    precomputed `events` sequence to reuse work or to test with planted
    events directly.

    Used mainly by §D-B' (channel-crossing suppression); for the primary
    firebreak-suppression contrast, `protected_region_rate` is primary.
    """
    if source_side == target_side:
        raise ValueError("source_side and target_side must differ")
    if events is None:
        hop_cfg = hop_config or HopConfig()
        clus_cfg = cluster_config or ClusterConfig()
        events = build_events(detect_hops(traj, hop_cfg), traj, clus_cfg)
        if persistent_only:
            events = [e for e in events if is_persistent(e, traj, clus_cfg)]
    for event in events:
        positions = np.array([traj.positions[event.onset_frame, p] for p in event.particles])
        coord = positions[:, barrier_slab.axis]
        if _side_mask(coord, barrier_slab, source_side).any() and _side_mask(
            coord, barrier_slab, target_side
        ).any():
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# Deliverable 4: anisotropy / directional score (P, §D)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnisotropyResult:
    """`A` (primary) plus the along-axis vs. isotropic "reach" it is built from."""

    A: float
    reach_along_axis: float
    reach_isotropic: float
    n_events: int


def anisotropy_score(
    traj: Trajectory,
    channel_axis: Sequence[float],
    origin: Sequence[float],
    *,
    centroids: np.ndarray | None = None,
    hop_config: HopConfig | None = None,
    cluster_config: ClusterConfig | None = None,
    persistent_only: bool = True,
) -> AnisotropyResult:
    """`A` = (rearrangement-centroid reach along `channel_axis`) minus
    (isotropic/transverse reach), from persistent-cluster centroids.

    PLAN AMBIGUITY, resolved here (flagged, since the literal spec wording
    is under-specified about what "the transverse component" means): a
    naive `mean(signed projection) - mean(transverse magnitude)` fails the
    spec's own acceptance criterion for an isotropic cloud, because a
    transverse *magnitude* cannot average to zero the way a signed
    projection can (random signs cancel along the axis; magnitudes never
    cancel), which would report a spuriously negative `A` for isotropic
    motion. Instead both "reach" terms are per-degree-of-freedom RMS
    displacement -- `reach_along_axis = sqrt(mean(projected^2))` (1 degree
    of freedom) and `reach_isotropic = sqrt(mean(|transverse|^2) / 2)` (the
    remaining 2 transverse degrees of freedom in 3D) -- so an isotropic
    cloud with equal per-axis variance reports `reach_along_axis ==
    reach_isotropic` and `A ≈ 0` by construction, while a front confined to
    the channel axis reports `reach_isotropic ≈ 0` and `A > 0`.

    `centroids` defaults to persistent rearrangement-cluster centroids
    detected from `traj` with default configs; pass a precomputed `(N, 3)`
    array to reuse work or test with planted centroids directly.
    """
    axis = np.asarray(channel_axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("channel_axis must be nonzero")
    axis_unit = axis / axis_norm
    origin_arr = np.asarray(origin, dtype=float)

    if centroids is None:
        hop_cfg = hop_config or HopConfig()
        clus_cfg = cluster_config or ClusterConfig()
        events = build_events(detect_hops(traj, hop_cfg), traj, clus_cfg)
        if persistent_only:
            events = [e for e in events if is_persistent(e, traj, clus_cfg)]
        centroid_arr = np.array([e.centroid for e in events]) if events else np.empty((0, 3))
    else:
        centroid_arr = np.atleast_2d(np.asarray(centroids, dtype=float))

    n_events = int(centroid_arr.shape[0])
    if n_events == 0:
        return AnisotropyResult(A=0.0, reach_along_axis=0.0, reach_isotropic=0.0, n_events=0)

    displacement = minimum_image(centroid_arr - origin_arr[None, :], traj.box)
    projected = displacement @ axis_unit
    transverse = displacement - projected[:, None] * axis_unit[None, :]
    transverse_sq = np.einsum("ij,ij->i", transverse, transverse)

    reach_along_axis = float(np.sqrt(np.mean(projected**2)))
    transverse_dof = displacement.shape[1] - 1
    reach_isotropic = float(np.sqrt(np.mean(transverse_sq) / transverse_dof))
    return AnisotropyResult(
        A=reach_along_axis - reach_isotropic,
        reach_along_axis=reach_along_axis,
        reach_isotropic=reach_isotropic,
        n_events=n_events,
    )


# --------------------------------------------------------------------------- #
# Deliverable 5: per-parent reduction + table emission
# --------------------------------------------------------------------------- #


def pattern_table(
    branch_scores_by_parent_and_arm: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    endpoint_col: str = "score",
) -> BranchTable:
    """Reduce each (parent, arm)'s `B` branch scores to one mean row.

    `branch_scores_by_parent_and_arm` maps `parent_id -> arm_family -> [branch
    scores]` (`arm_family` conventionally one of `"targeted"`, `"random"`,
    `"shuffled"`, `"unedited"`, though this follows `stats._common`'s own
    "free text, never hardcoded" convention rather than validating against
    that fixed set). Emits `stats._common.BranchTable` rows with
    `pair_id=parent_id`, `arm_family`, a continuous `endpoint_col` value
    (the (parent, arm) mean score), and a diagnostic `n_branches` count --
    exactly one row per (parent, arm), which is what lets
    `intervals.bootstrap_pairs` (grouping by `pair_id`) resample parents and
    only parents (see the module docstring's parent-as-unit note). No
    `arm_sign` column is emitted; see the continuous-vs-binary note above
    for why.
    """
    rows: list[BranchRow] = []
    for parent_id in sorted(branch_scores_by_parent_and_arm):
        arms = branch_scores_by_parent_and_arm[parent_id]
        for arm_family in sorted(arms):
            scores = list(arms[arm_family])
            if not scores:
                raise ValueError(f"parent {parent_id!r} arm {arm_family!r} has no branch scores")
            rows.append(
                {
                    PAIR_ID_COL: parent_id,
                    ARM_FAMILY_COL: arm_family,
                    endpoint_col: float(np.mean(scores)),
                    "n_branches": len(scores),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Deliverable 6: primary contrast helpers (thin wrappers over `intervals`)
# --------------------------------------------------------------------------- #


def _paired_family_mean_diff(
    table: BranchTable,
    endpoint_col: str,
    target_family: str,
    comparator_family: str,
    pair_col: str,
) -> float:
    """`mean_parents(score_target - score_comparator)` -- the single code
    path shared by the point estimate and every bootstrap replicate.

    `pattern_table` guarantees exactly one row per (pair, family); pairs
    missing either family are skipped (a paired design only counts pairs
    where both arms ran), mirroring `estimands.paired_contrasts`'s
    `on_incomplete="drop"` behaviour.
    """
    by_pair: dict[object, dict[str, float]] = {}
    for row in table:
        require_columns(row, (pair_col, ARM_FAMILY_COL, endpoint_col))
        family = row[ARM_FAMILY_COL]
        if family not in (target_family, comparator_family):
            continue
        by_pair.setdefault(row[pair_col], {})[family] = float(row[endpoint_col])
    diffs = [
        values[target_family] - values[comparator_family]
        for values in by_pair.values()
        if target_family in values and comparator_family in values
    ]
    if not diffs:
        raise ValueError(
            f"no pairs have both {target_family!r} and {comparator_family!r} arms present"
        )
    return float(np.mean(diffs))


def pattern_contrast_bootstrap(
    table: BranchTable,
    endpoint_col: str,
    target_family: str,
    comparator_family: str,
    *,
    pair_col: str = PAIR_ID_COL,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> intervals.BootstrapResult:
    """Bootstrap-over-parents 95% CI for `mean_parents(score_target - score_comparator)`.

    Thin wrapper: builds the `table -> float` statistic
    `_paired_family_mean_diff` and hands it, unchanged, to
    `intervals.bootstrap_pairs` (`pair_col="pair_id"`), exactly as the task
    spec asks for `D_pattern` and `ATE_suppress` below.
    """
    statistic = functools.partial(
        _paired_family_mean_diff,
        endpoint_col=endpoint_col,
        target_family=target_family,
        comparator_family=comparator_family,
        pair_col=pair_col,
    )
    return intervals.bootstrap_pairs(
        table, statistic, pair_col=pair_col, n_boot=n_boot, alpha=alpha, seed=seed
    )


def d_pattern(
    table: BranchTable,
    endpoint_col: str = "score",
    *,
    target_family: str = "targeted",
    comparator_family: str = "random",
    pair_col: str = PAIR_ID_COL,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> intervals.BootstrapResult:
    """`D_pattern = mean_parents(S_targeted - S_random)`, bootstrap CI over parents."""
    return pattern_contrast_bootstrap(
        table,
        endpoint_col,
        target_family,
        comparator_family,
        pair_col=pair_col,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )


def ate_suppress(
    table: BranchTable,
    endpoint_col: str = "score",
    *,
    target_family: str = "targeted",
    comparator_family: str = "random",
    pair_col: str = PAIR_ID_COL,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> intervals.BootstrapResult:
    """`ATE_suppress = mean_parents(protected_rate_firebreak - protected_rate_random)`
    (expect `< 0`), bootstrap CI over parents. Same machinery as `d_pattern`;
    the caller supplies `protected_region_rate` values as `endpoint_col`
    instead of `projection_score` values."""
    return pattern_contrast_bootstrap(
        table,
        endpoint_col,
        target_family,
        comparator_family,
        pair_col=pair_col,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Deliverable 6 (cont.): shuffle-style null for the projection score
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatternShuffleNullResult:
    """Structurally identical to `shuffle_null.ShuffleNullResult` -- same
    fields, same permutation-p-value convention -- for the field-score
    (rather than candidate-outcome) shuffle."""

    observed: float
    null_distribution: tuple[float, ...]
    p_value: float
    n_permutations: int
    seed: int
    alternative: str


def pattern_shuffle_null(
    mobility_grid: np.ndarray,
    w_template: np.ndarray,
    *,
    n_permutations: int = 2000,
    seed: int,
    alternative: str = "two-sided",
) -> PatternShuffleNullResult:
    """Shuffled-template negative control for `projection_score`.

    Mirrors `shuffle_null.shuffle_null_test`'s structure and its central
    requirement: the real and null statistics run through the identical
    code path (`projection_score`), only the input differs. Here the
    per-site template weights are permuted across grid sites (breaking any
    real template-mobility alignment while preserving both the mobility
    field and the template's multiset of weights, i.e. its total mass and
    shape), and `projection_score` is recomputed on that permuted template
    against the SAME (unpermuted) mobility grid -- reusing this module's own
    "permute the per-site target-sign labels / template orientation"
    framing from the task spec.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    mobility_grid = np.asarray(mobility_grid, dtype=float)
    w_template = np.asarray(w_template, dtype=float)
    observed = projection_score(mobility_grid, w_template)
    rng = np.random.default_rng(seed)
    flat_template = w_template.reshape(-1)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_template = rng.permutation(flat_template).reshape(w_template.shape)
        null[i] = projection_score(mobility_grid, permuted_template, validate=False)
    if alternative == "two-sided":
        extreme = np.sum(np.abs(null) >= abs(observed))
    elif alternative == "greater":
        extreme = np.sum(null >= observed)
    else:
        extreme = np.sum(null <= observed)
    p_value = float((1 + extreme) / (n_permutations + 1))
    return PatternShuffleNullResult(
        observed=observed,
        null_distribution=tuple(float(x) for x in null),
        p_value=p_value,
        n_permutations=n_permutations,
        seed=seed,
        alternative=alternative,
    )
