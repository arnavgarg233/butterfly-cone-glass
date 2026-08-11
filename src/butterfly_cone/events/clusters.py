"""Spatiotemporal clustering of rearranging particles into events.

Two rearranging particles are linked when their onset times differ by less than
``dt_link`` AND their onset positions lie within ``r_link`` (minimum image).
Connected components are candidate events.  A persistence filter marks events
whose members' cage-relative displacement has not reversed by ``t_check``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClusterConfig, DisplacementConfig
from .displacements import cage_relative_field, magnitude_field
from .hops import HopEvent
from .trajectory import (
    Trajectory,
    minimum_image,
    minimum_image_centroid,
    radius_of_gyration,
)


@dataclass(frozen=True)
class Event:
    """A spatiotemporal cluster of rearranging particles."""

    particles: tuple[int, ...]
    onset_frame: int
    onset_time: float
    centroid: np.ndarray
    radius_of_gyration: float
    duration: float
    member_events: tuple[HopEvent, ...]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def link_events(
    events: list[HopEvent],
    traj: Trajectory,
    config: ClusterConfig,
) -> list[list[int]]:
    """Return connected components as lists of indices into ``events``."""

    n = len(events)
    uf = _UnionFind(n)
    if n < 2:
        return [[i] for i in range(n)]
    onset_times = np.array([e.onset_time for e in events])
    onset_frames = np.array([e.onset_frame for e in events])
    onset_positions = np.array(
        [traj.positions[e.onset_frame, e.particle] for e in events]
    )
    box = np.asarray(traj.box)
    r_link_sq = config.r_link * config.r_link
    # Vectorised link test, bitwise-identical to the per-pair loop:
    #   * time test |t_a - t_b| < dt_link gates FIRST (mirrors the original
    #     `continue`), so distances are only ever evaluated for time-compatible
    #     pairs -- the same work the Python loop's early-continue skipped;
    #   * same posA-posB subtraction order, same minimum-image reduction, same
    #     left-to-right sum-of-squares as np.dot(delta, delta).
    # Blocked over rows so the working set stays O(block * n) instead of
    # materialising the full O(n**2 * dim) displacement tensor.
    row_block = max(1, int(2_000_000 // n))
    all_b = np.arange(n)
    for lo in range(0, n, row_block):
        hi = min(lo + row_block, n)
        a_idx = np.arange(lo, hi)
        # Only pairs with b > a contribute (a < b in the original loop) and must
        # pass the time-window test before any distance is computed.
        dt = np.abs(onset_times[lo:hi, None] - onset_times[None, :])
        candidate = (all_b[None, :] > a_idx[:, None]) & (dt < config.dt_link)
        rows, cols = np.nonzero(candidate)
        if rows.size == 0:
            continue
        pa = a_idx[rows]
        delta = onset_positions[pa] - onset_positions[cols]
        delta = delta - box * np.rint(delta / box)
        d2 = delta[:, 0] ** 2
        for k in range(1, delta.shape[1]):
            d2 = d2 + delta[:, k] ** 2
        linked = d2 < r_link_sq
        for a, b in zip(pa[linked], cols[linked]):
            uf.union(int(a), int(b))
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(uf.find(i), []).append(i)
    # Deterministic order: by earliest onset frame then smallest particle id.
    ordered = sorted(
        components.values(),
        key=lambda idxs: (min(onset_frames[i] for i in idxs), min(events[i].particle for i in idxs)),
    )
    return ordered


def build_events(
    hop_events: list[HopEvent],
    traj: Trajectory,
    config: ClusterConfig = ClusterConfig(),
) -> list[Event]:
    """Cluster hop events into :class:`Event` objects."""

    components = link_events(hop_events, traj, config)
    result: list[Event] = []
    for idxs in components:
        members = [hop_events[i] for i in idxs]
        members.sort(key=lambda e: (e.onset_frame, e.particle))
        particles = tuple(e.particle for e in members)
        onset_member = min(members, key=lambda e: (e.onset_frame, e.particle))
        latest = max(members, key=lambda e: (e.onset_frame, e.particle))
        positions = np.array([traj.positions[onset_member.onset_frame, p] for p in particles])
        centroid = minimum_image_centroid(positions, traj.box)
        rg = radius_of_gyration(positions, traj.box)
        duration = float(latest.onset_time - onset_member.onset_time)
        result.append(
            Event(
                particles=particles,
                onset_frame=onset_member.onset_frame,
                onset_time=onset_member.onset_time,
                centroid=centroid,
                radius_of_gyration=rg,
                duration=duration,
                member_events=tuple(members),
            )
        )
    return result


def is_persistent(
    event: Event,
    traj: Trajectory,
    config: ClusterConfig = ClusterConfig(),
) -> bool:
    """Whether an event's members' cage-relative displacement has not reversed.

    Compares the mean member cage-relative displacement magnitude at the onset
    frame with its value ``reversal_t_check`` later.  The event is persistent
    when the later mean magnitude is at least ``reversal_fraction`` of the onset
    mean magnitude.
    """

    disp_config = DisplacementConfig(first_shell_factor=config.first_shell_factor)
    field = cage_relative_field(traj, disp_config, reference_frame=config.reference_frame)
    magnitude = magnitude_field(field)
    members = np.array(event.particles)
    check_frame = traj.frame_at_or_after(event.onset_frame, config.reversal_t_check)
    onset_mean = float(np.mean(magnitude[event.onset_frame, members]))
    check_mean = float(np.mean(magnitude[check_frame, members]))
    if onset_mean == 0.0:
        return check_mean > 0.0
    return check_mean >= config.reversal_fraction * onset_mean


def persistent_events(
    hop_events: list[HopEvent],
    traj: Trajectory,
    config: ClusterConfig = ClusterConfig(),
) -> list[Event]:
    """Build events and keep only the persistent ones."""

    return [e for e in build_events(hop_events, traj, config) if is_persistent(e, traj, config)]
