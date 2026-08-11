"""Seed-versus-incoming attribution conventions (PLAN_v2.1 sec 8).

Four pluggable classifiers label an event associated with a monitored cavity as
``seed`` (relaxation initiated inside the core), ``incoming`` (driven by material
entering the core), or ``ambiguous`` (the deciding structure is absent).  Each
returns a structured explanation record.  A comparison utility computes the
sec-8 stability metrics -- pairwise label agreement, region-level seed-propensity
Spearman rho, and top-decile overlap -- as functions ready for the freeze
analysis.

All thresholds are provisional until the advance-declaration freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AttributionConfig, CavitySpec, ClusterConfig
from .clusters import Event, build_events
from .hops import HopEvent
from .strings import StringConfig, string_crosses_boundary, trace_strings
from .trajectory import (
    Trajectory,
    minimum_image,
    minimum_image_centroid,
    neighbor_pairs,
)


@dataclass(frozen=True)
class AttributionResult:
    """A convention's decision plus a structured explanation."""

    convention: str
    label: str  # "seed" | "incoming" | "ambiguous"
    when: float | None  # onset time of the deciding structure
    where: np.ndarray | None  # centroid / location of the deciding structure
    detail: dict = field(default_factory=dict)


@dataclass
class AttributionInput:
    """Everything the conventions need for one cavity/event."""

    traj: Trajectory
    cavity: CavitySpec
    parent_core_tags: np.ndarray  # bool (N,)
    events: list[Event]  # persistent rearranging clusters, onset-ordered
    hop_events: list[HopEvent]  # per-particle rearrangements


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def _distance_to_center(position: np.ndarray, cavity: CavitySpec, box: np.ndarray) -> float:
    delta = minimum_image(position - np.asarray(cavity.center, dtype=float), box)
    return float(np.sqrt(np.dot(delta, delta)))


def in_core(position: np.ndarray, cavity: CavitySpec, box: np.ndarray) -> bool:
    return _distance_to_center(position, cavity, box) < cavity.r_core


def in_annulus(position: np.ndarray, cavity: CavitySpec, box: np.ndarray) -> bool:
    d = _distance_to_center(position, cavity, box)
    return cavity.r_core <= d < cavity.r_annulus


# --------------------------------------------------------------------------- #
# bond breaking (convention A support)
# --------------------------------------------------------------------------- #

def neighbor_loss_field(
    traj: Trajectory,
    first_shell_factor: float,
    reference_frame: int = 0,
) -> np.ndarray:
    """Fraction of reference-frame first-shell neighbours lost, per (frame, particle)."""

    from .trajectory import mixing_diameter

    reference = traj.positions[reference_frame]
    left, right = neighbor_pairs(reference, traj.sigma, traj.box, first_shell_factor)
    counts = np.bincount(
        np.concatenate((left, right)) if left.size else np.empty(0, dtype=np.int64),
        minlength=traj.n_particles,
    ).astype(float)
    if left.size == 0:
        return np.zeros((traj.n_frames, traj.n_particles), dtype=float)
    sig_ij = mixing_diameter(traj.sigma[left], traj.sigma[right])
    cutoff2 = (first_shell_factor * sig_ij) ** 2
    loss = np.zeros((traj.n_frames, traj.n_particles), dtype=float)
    for frame in range(traj.n_frames):
        disp = minimum_image(
            traj.positions[frame, left] - traj.positions[frame, right], traj.box
        )
        d2 = np.einsum("ij,ij->i", disp, disp)
        retained = d2 < cutoff2
        retained_count = np.zeros(traj.n_particles, dtype=float)
        np.add.at(retained_count, left, retained.astype(float))
        np.add.at(retained_count, right, retained.astype(float))
        loss[frame] = np.divide(
            counts - retained_count,
            counts,
            out=np.zeros(traj.n_particles, dtype=float),
            where=counts > 0.0,
        )
    return loss


def sustained_onsets(active: np.ndarray, required: int) -> np.ndarray:
    """First onset frame of a sustained run of ``required`` active frames, per particle.

    Returns -1 for particles that never sustain the run.  ``active`` is (T, N).
    """

    n_frames, n_particles = active.shape
    run = np.zeros(n_particles, dtype=np.int64)
    onset = np.full(n_particles, -1, dtype=np.int64)
    seen = np.zeros(n_particles, dtype=bool)
    for frame in range(n_frames):
        run = np.where(active[frame], run + 1, 0)
        reached = (run == required) & (~seen)
        onset[reached] = frame - required + 1
        seen |= reached
    return onset


def bond_breaking_onsets(traj: Trajectory, config: AttributionConfig) -> np.ndarray:
    """Onset frame of sustained bond breaking per particle (-1 if none)."""

    loss = neighbor_loss_field(traj, config.first_shell_factor, config.reference_frame)
    active = loss > config.bond_break_loss_fraction
    required = traj.frames_for_duration(config.bond_break_persist_time)
    return sustained_onsets(active, required)


# --------------------------------------------------------------------------- #
# conventions
# --------------------------------------------------------------------------- #

def classify_first_nucleus(
    data: AttributionInput,
    config: AttributionConfig = AttributionConfig(),
) -> AttributionResult:
    """Convention A -- earliest persistent bond-breaking nucleus centroid in core."""

    traj = data.traj
    onset = bond_breaking_onsets(traj, config)
    breakers = np.nonzero(onset >= 0)[0]
    if breakers.size == 0:
        return AttributionResult("A_first_nucleus", "ambiguous", None, None,
                                 {"reason": "no bond-breaking nucleus"})
    # Cluster the bond-breaking particles into nuclei by reusing the event linker.
    synthetic = [
        HopEvent(int(p), int(onset[p]), float(traj.times[onset[p]]), 0.0, "bond_break")
        for p in breakers
    ]
    cluster_config = ClusterConfig(
        dt_link=config.nucleus_dt_link,
        r_link=config.nucleus_r_link,
        first_shell_factor=config.first_shell_factor,
        reference_frame=config.reference_frame,
    )
    nuclei = build_events(synthetic, traj, cluster_config)
    earliest = min(nuclei, key=lambda e: (e.onset_frame, min(e.particles)))
    seeded = in_core(earliest.centroid, data.cavity, traj.box)
    return AttributionResult(
        "A_first_nucleus",
        "seed" if seeded else "incoming",
        earliest.onset_time,
        earliest.centroid,
        {
            "nucleus_particles": earliest.particles,
            "centroid_distance": _distance_to_center(earliest.centroid, data.cavity, traj.box),
            "n_nuclei": len(nuclei),
        },
    )


def classify_core_majority(
    data: AttributionInput,
    config: AttributionConfig = AttributionConfig(),
    *,
    theta: float | None = None,
) -> AttributionResult:
    """Convention B -- >theta of the first persistent cluster's particles in core."""

    theta = config.majority_theta if theta is None else theta
    if not data.events:
        return AttributionResult("B_core_majority", "ambiguous", None, None,
                                 {"reason": "no persistent cluster", "theta": theta})
    event = data.events[0]
    box = data.traj.box
    inside = [
        in_core(data.traj.positions[event.onset_frame, p], data.cavity, box)
        for p in event.particles
    ]
    fraction = float(np.mean(inside))
    return AttributionResult(
        "B_core_majority",
        "seed" if fraction > theta else "incoming",
        event.onset_time,
        event.centroid,
        {"fraction_in_core": fraction, "theta": theta, "n_particles": len(event.particles)},
    )


def classify_first_entry(
    data: AttributionInput,
    config: AttributionConfig = AttributionConfig(),
    *,
    string_config: StringConfig = StringConfig(),
) -> AttributionResult:
    """Convention C -- irreversible core activity before annulus->core crossing."""

    traj = data.traj
    box = traj.box
    cavity = data.cavity
    # Earliest irreversible (persistent-detector) activity inside the core.
    core_time = np.inf
    core_where = None
    for hop in sorted(data.hop_events, key=lambda h: (h.onset_frame, h.particle)):
        if in_core(traj.positions[hop.onset_frame, hop.particle], cavity, box):
            core_time = hop.onset_time
            core_where = traj.positions[hop.onset_frame, hop.particle]
            break
    # Earliest connected rearrangement that crosses the annulus into the core.
    # Strings are traced from the reference (pre-onset) frame so the lag window
    # captures the inward motion; membership is judged on the original positions.
    reference = config.reference_frame
    cross_time = np.inf
    for event in sorted(data.events, key=lambda e: (e.onset_frame, min(e.particles))):
        paths = trace_strings(event.particles, traj, reference, string_config)
        for path in paths:
            straddles = string_crosses_boundary(
                path, traj, reference, cavity.center, cavity.r_core
            )
            has_annulus = any(
                in_annulus(traj.positions[reference, p], cavity, box)
                for p in path.particles
            )
            if straddles and has_annulus:
                cross_time = event.onset_time
                break
        if cross_time < np.inf:
            break

    detail = {"core_activity_time": None if core_time == np.inf else core_time,
              "crossing_time": None if cross_time == np.inf else cross_time}
    if core_time == np.inf and cross_time == np.inf:
        return AttributionResult("C_first_entry", "ambiguous", None, None,
                                 {**detail, "reason": "no core activity, no crossing"})
    # Seed only when irreversible core activity is detected strictly BEFORE any
    # connected rearrangement crosses the annulus into the core.  Otherwise the
    # relaxation entered from outside (or coincides with the crossing).
    if core_time < cross_time:
        return AttributionResult("C_first_entry", "seed", core_time, core_where, detail)
    return AttributionResult("C_first_entry", "incoming",
                             None if cross_time == np.inf else cross_time, None, detail)


def classify_material_core(
    data: AttributionInput,
    config: AttributionConfig = AttributionConfig(),
    *,
    dominance: float | None = None,
) -> AttributionResult:
    """Convention D -- earliest persistent cluster dominated by parent-core material."""

    dominance = config.material_dominance if dominance is None else dominance
    if not data.events:
        return AttributionResult("D_material_core", "ambiguous", None, None,
                                 {"reason": "no persistent cluster", "dominance": dominance})
    event = data.events[0]
    tags = np.asarray(data.parent_core_tags, dtype=bool)
    tagged = [bool(tags[p]) for p in event.particles]
    fraction = float(np.mean(tagged))
    return AttributionResult(
        "D_material_core",
        "seed" if fraction > dominance else "incoming",
        event.onset_time,
        event.centroid,
        {"fraction_tagged": fraction, "dominance": dominance, "n_particles": len(event.particles)},
    )


_CONVENTIONS = {
    "A": classify_first_nucleus,
    "B": classify_core_majority,
    "C": classify_first_entry,
    "D": classify_material_core,
}


def classify(convention: str, data: AttributionInput,
             config: AttributionConfig = AttributionConfig()) -> AttributionResult:
    """Dispatch to convention A, B, C, or D."""

    if convention not in _CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(_CONVENTIONS)}")
    return _CONVENTIONS[convention](data, config)


# --------------------------------------------------------------------------- #
# sec-8 stability metrics
# --------------------------------------------------------------------------- #

def label_agreement(labels_a, labels_b) -> float:
    """Fraction of events on which two conventions assign the identical label."""

    labels_a = list(labels_a)
    labels_b = list(labels_b)
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    if not labels_a:
        raise ValueError("need at least one event")
    return float(np.mean([a == b for a, b in zip(labels_a, labels_b)]))


def seed_propensity(labels, regions, n_regions: int) -> np.ndarray:
    """Fraction of events labelled 'seed' per region (NaN for empty regions)."""

    labels = list(labels)
    regions = np.asarray(regions, dtype=int)
    if len(labels) != regions.shape[0]:
        raise ValueError("labels and regions must be the same length")
    propensity = np.full(n_regions, np.nan)
    for region in range(n_regions):
        mask = regions == region
        if np.any(mask):
            seed_flags = [labels[i] == "seed" for i in np.nonzero(mask)[0]]
            propensity[region] = float(np.mean(seed_flags))
    return propensity


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved to their average (1-based)."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman_rho(x, y) -> float:
    """Spearman rank correlation, NaN-safe and tie-aware.

    NaN-paired entries are dropped.  Returns NaN if fewer than two valid pairs
    remain or if either variable is constant.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def region_seed_spearman(labels_a, labels_b, regions, n_regions: int) -> float:
    """Spearman rho between two conventions' region-level seed propensities."""

    prop_a = seed_propensity(labels_a, regions, n_regions)
    prop_b = seed_propensity(labels_b, regions, n_regions)
    return spearman_rho(prop_a, prop_b)


def top_decile_overlap(prop_a, prop_b, *, fraction: float = 0.1) -> float:
    """Overlap fraction of the top-``fraction`` regions by seed propensity.

    Ranks each propensity vector, takes the top ``ceil(fraction * n_valid)``
    regions of each, and returns |intersection| / |top-set size|.  NaN regions
    are excluded.
    """

    prop_a = np.asarray(prop_a, dtype=float)
    prop_b = np.asarray(prop_b, dtype=float)
    if prop_a.shape != prop_b.shape:
        raise ValueError("propensity vectors must have the same shape")
    valid = np.nonzero(np.isfinite(prop_a) & np.isfinite(prop_b))[0]
    if valid.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(fraction * valid.size)))

    def top_set(values: np.ndarray) -> set[int]:
        # Highest propensity first; break ties by region index for determinism.
        ordered = sorted(valid, key=lambda r: (-values[r], r))
        return set(ordered[:k])

    set_a, set_b = top_set(prop_a), top_set(prop_b)
    return float(len(set_a & set_b) / k)


def compare_conventions(results_by_convention: dict[str, list[AttributionResult]],
                        regions, n_regions: int) -> dict:
    """Full sec-8 stability report across an event set for every convention pair.

    ``results_by_convention`` maps a convention name to its per-event results
    (all lists the same length, aligned to ``regions``).
    """

    names = sorted(results_by_convention)
    labels = {name: [r.label for r in results_by_convention[name]] for name in names}
    propensities = {name: seed_propensity(labels[name], regions, n_regions) for name in names}
    report: dict = {"pairwise_agreement": {}, "region_seed_spearman": {}, "top_decile_overlap": {}}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}|{b}"
            report["pairwise_agreement"][key] = label_agreement(labels[a], labels[b])
            report["region_seed_spearman"][key] = spearman_rho(propensities[a], propensities[b])
            report["top_decile_overlap"][key] = top_decile_overlap(propensities[a], propensities[b])
    return report


# --------------------------------------------------------------------------- #
# pipeline convenience
# --------------------------------------------------------------------------- #

def build_attribution_input(
    traj: Trajectory,
    cavity: CavitySpec,
    parent_core_tags: np.ndarray,
    hop_events: list[HopEvent],
    *,
    cluster_config: ClusterConfig | None = None,
    require_persistent: bool = True,
) -> AttributionInput:
    """Assemble an :class:`AttributionInput` from raw hop events."""

    from .clusters import is_persistent

    cluster_config = cluster_config or ClusterConfig()
    events = build_events(hop_events, traj, cluster_config)
    if require_persistent:
        events = [e for e in events if is_persistent(e, traj, cluster_config)]
    events.sort(key=lambda e: (e.onset_frame, min(e.particles)))
    return AttributionInput(
        traj=traj,
        cavity=cavity,
        parent_core_tags=np.asarray(parent_core_tags, dtype=bool),
        events=events,
        hop_events=hop_events,
    )
