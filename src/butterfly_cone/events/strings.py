"""Microstring (follow-the-leader) tracing within an event.

Particle i is a *follower* of j (directed edge i -> j) when i's displacement
vector over the lag window terminates within ``r_string`` of j's original
position -- i.e. i moves into a spot j has vacated.  Each follower keeps its
single closest leader, giving a forest of chains; maximal chains are the
microstrings.  Emits ordered particle sequences, the length distribution, and a
spherical-boundary crossing test used by the attribution conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import StringConfig
from .trajectory import Trajectory, minimum_image


@dataclass(frozen=True)
class StringPath:
    """An ordered microstring: followers first, ultimate leader last."""

    particles: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.particles)


def _leader_of(
    candidates: np.ndarray,
    traj: Trajectory,
    t0_frame: int,
    t1_frame: int,
    config: StringConfig,
) -> dict[int, int]:
    """Map each follower to its single closest leader among ``candidates``."""

    origin = np.array([traj.positions[t0_frame, p] for p in candidates])
    endpoint_unwrapped = np.array(
        [traj.unwrapped_positions[t1_frame, p] for p in candidates]
    )
    # Wrap endpoints into the cell so the minimum-image comparison to leaders'
    # original (wrapped) positions is consistent across the periodic boundary.
    box = traj.box
    endpoint = np.remainder(endpoint_unwrapped, box)
    leader: dict[int, int] = {}
    n = len(candidates)
    for a in range(n):
        best_j = -1
        best_d2 = config.r_string * config.r_string
        for b in range(n):
            if a == b:
                continue
            delta = minimum_image(endpoint[a] - origin[b], box)
            d2 = float(np.dot(delta, delta))
            if d2 < best_d2:
                best_d2 = d2
                best_j = b
        if best_j >= 0:
            leader[int(candidates[a])] = int(candidates[best_j])
    return leader


def trace_strings(
    particles,
    traj: Trajectory,
    t0_frame: int,
    config: StringConfig = StringConfig(),
) -> list[StringPath]:
    """Trace microstrings among ``particles`` starting at ``t0_frame``.

    The follower displacement is measured over the ``lag_time`` window.
    """

    candidates = np.array(sorted(int(p) for p in particles), dtype=np.int64)
    if candidates.size < 2:
        return []
    t1_frame = traj.frame_at_or_after(t0_frame, config.lag_time)
    leader = _leader_of(candidates, traj, t0_frame, t1_frame, config)

    followed = set(leader.values())
    # Chain heads: particles that follow someone but that nobody follows.
    heads = sorted(p for p in leader if p not in followed)
    paths: list[StringPath] = []
    for head in heads:
        chain = [head]
        visited = {head}
        current = head
        while current in leader:
            nxt = leader[current]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(chain) >= 2:
            paths.append(StringPath(particles=tuple(chain)))
    paths.sort(key=lambda p: (-p.length, p.particles))
    return paths


def string_length_distribution(paths: list[StringPath]) -> dict[int, int]:
    """Histogram of string lengths (number of particles per string)."""

    distribution: dict[int, int] = {}
    for path in paths:
        distribution[path.length] = distribution.get(path.length, 0) + 1
    return dict(sorted(distribution.items()))


def string_crosses_boundary(
    path: StringPath,
    traj: Trajectory,
    t0_frame: int,
    center,
    radius: float,
) -> bool:
    """Whether a string straddles a spherical boundary at ``t0_frame``.

    True when some members lie inside the sphere and others outside (minimum
    image to ``center``).
    """

    center = np.asarray(center, dtype=float)
    box = traj.box
    inside = []
    for p in path.particles:
        delta = minimum_image(traj.positions[t0_frame, p] - center, box)
        inside.append(float(np.dot(delta, delta)) < radius * radius)
    return any(inside) and not all(inside)
