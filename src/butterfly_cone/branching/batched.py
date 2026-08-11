"""Vectorized branch-state mechanics built on the canonical ButterflyCone engine.

The batch dimension is always leading: positions and velocities have shape
``(B, N, 3)``.  Diameters come in two modes:

* shared ``(N,)`` -- every branch is a new momentum realization of one parent
  configuration (the original branching contract, unchanged);
* per-replica ``(B, N)`` -- each row belongs to one chain or PT rung, so
  states whose diameters diverged through accepted swap moves (RCCE
  overdispersed chains, parallel-tempering ladders) integrate as one batched
  kernel launch instead of ``B`` sequential engine runs.

The periodic box and the active mask remain shared by every branch.

Determinism and bitwise contracts:

* the ``B == 1`` route always drives the single-system engine objects and is
  bitwise-identical to them, for both diameter modes;
* a per-replica ``(B, N)`` state whose rows are identical clones is
  bitwise-identical to the shared-``(N,)`` batched path (the per-replica
  gather reads the same float values, all downstream arithmetic is shared);
* for ``B > 1`` the batched evaluation is deterministic (identical inputs
  reproduce identical bits on one device) but it is NOT bitwise-identical to
  running the same replicas sequentially through the single-system engine:
  padded batched reductions sum contributions through a different tree, so
  results differ at float rounding.  A batched run is therefore a distinct,
  equally valid realization of the same sampler -- the same class of
  rounding-level difference as the CPU-vs-MPS device gap the project already
  documents and tolerates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from butterfly_cone.engine.integrate import (
    BussiThermostat,
    MDIntegrator,
    _normal_draw,
    maxwell_boltzmann_velocities as _engine_maxwell_boltzmann_velocities,
)
from butterfly_cone.engine.neighbors import VerletList, cell_list_pairs
from butterfly_cone.engine.potential import (
    CUTOFF_RATIO,
    analytic_potential,
    all_pairs,
    minimum_image,
    mixing_diameter,
    pair_potential,
)
from butterfly_cone.engine.system import ParticleSystem


@dataclass
class BatchedSystem:
    """Particle state for independent branches of one parent configuration.

    Diameters are shared ``(N,)`` when every branch is a momentum realization
    of one parent, or per-replica ``(B, N)`` when rows belong to distinct
    chains/PT rungs whose diameters diverged through accepted swap moves.
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    diameters: torch.Tensor
    box: torch.Tensor
    active_mask: torch.Tensor
    unwrapped_positions: torch.Tensor

    def __post_init__(self) -> None:
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape (B, N, 3)")
        batch_size, n_particles, _ = self.positions.shape
        if batch_size <= 0 or n_particles <= 0:
            raise ValueError("B and N must be positive")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must match positions")
        if self.unwrapped_positions.shape != self.positions.shape:
            raise ValueError("unwrapped_positions must match positions")
        if self.diameters.shape not in ((n_particles,), (batch_size, n_particles)):
            raise ValueError("diameters must have shape (N,) or (B, N)")
        if self.box.shape != (3,) or bool(torch.any(self.box <= 0)):
            raise ValueError("box must contain three positive lengths")
        if self.active_mask.shape != (n_particles,) or self.active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a bool tensor of shape (N,)")
        tensors = (
            self.velocities,
            self.diameters,
            self.box,
            self.active_mask,
            self.unwrapped_positions,
        )
        if any(tensor.device != self.positions.device for tensor in tensors):
            raise ValueError("all state tensors must share a device")
        if any(tensor.dtype != self.positions.dtype for tensor in tensors if tensor.dtype != torch.bool):
            raise ValueError("all floating state tensors must share a dtype")

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[1])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    @property
    def sigma(self) -> torch.Tensor:
        """Alias for the diameter realization, shared ``(N,)`` or per-replica ``(B, N)``."""

        return self.diameters

    @property
    def per_replica_diameters(self) -> bool:
        """True when every replica owns its own ``(N,)`` diameter row."""

        return self.diameters.ndim == 2

    def branch_diameters(self, index: int) -> torch.Tensor:
        """Return the ``(N,)`` diameter realization one branch sees (a view)."""

        if index < 0 or index >= self.batch_size:
            raise IndexError("branch index is out of range")
        return self.diameters if self.diameters.ndim == 1 else self.diameters[index]

    def ensure_per_replica_diameters(self) -> torch.Tensor:
        """Materialize ``(B, N)`` diameters, copying a shared row if needed."""

        if self.diameters.ndim == 1:
            self.diameters = self.diameters.detach().unsqueeze(0).repeat(self.batch_size, 1)
        return self.diameters

    def set_branch_diameters(self, index: int, diameters: torch.Tensor) -> None:
        """Replace one replica's diameter row without touching any other row.

        A shared diameter state is promoted to per-replica rows first.  A
        wholesale row replacement can change the row maximum, so callers that
        hold a Verlet list built from the old row must rebuild it; pure swap
        permutations (see :meth:`swap_branch_diameters_`) never require that.
        """

        if index < 0 or index >= self.batch_size:
            raise IndexError("branch index is out of range")
        if diameters.shape != (self.n_particles,):
            raise ValueError("branch diameters must have shape (N,)")
        if diameters.device != self.device or diameters.dtype != self.dtype:
            raise ValueError("branch diameters must share the state device and dtype")
        self.ensure_per_replica_diameters()
        self.diameters[index].copy_(diameters.detach())

    def swap_branch_diameters_(self, index: int, i: int, j: int) -> None:
        """Exchange two diameter entries of one replica in place (swap-MC accept).

        Mirrors the engine's ``_swap_diameter_entries_`` on exactly one row:
        only row ``index`` is read or written, applying the exchange twice
        restores the row bitwise, and the row maximum is permutation-invariant
        so existing Verlet list radii remain valid.  A shared diameter state
        is promoted to per-replica rows first, because an accepted swap is
        precisely the event after which chains stop sharing one realization.
        """

        if index < 0 or index >= self.batch_size:
            raise IndexError("branch index is out of range")
        if not (0 <= i < self.n_particles and 0 <= j < self.n_particles):
            raise IndexError("swap particle index out of range")
        self.ensure_per_replica_diameters()
        row = self.diameters[index]
        entries = torch.tensor((i, j), device=self.device, dtype=torch.int64)
        values = row[entries].clone()
        row[entries] = values.flip(0)

    @classmethod
    def from_system(cls, parent: ParticleSystem, batch_size: int) -> "BatchedSystem":
        """Copy one parent state into ``batch_size`` independent branches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, parent.n_particles, 3)
        positions = parent.positions.detach().clone().expand(shape).clone()
        velocities = parent.velocities.detach().clone().expand(shape).clone()
        unwrapped = parent.unwrapped_positions.detach().clone().expand(shape).clone()
        return cls(
            positions=positions,
            velocities=velocities,
            diameters=parent.diameters.detach().clone(),
            box=parent.box.detach().clone(),
            active_mask=parent.active_mask.detach().clone(),
            unwrapped_positions=unwrapped,
        )

    from_parent = from_system

    @classmethod
    def from_systems(cls, replicas: Sequence[ParticleSystem]) -> "BatchedSystem":
        """Stack independent replicas (chains, PT rungs) into one batched state.

        Positions, velocities, and diameters become per-replica rows, so
        chains whose diameters diverged through accepted swap moves are
        represented exactly.  The box and the active mask must be
        bitwise-shared across replicas, as must device and dtype.
        """

        states = tuple(replicas)
        if not states:
            raise ValueError("at least one replica system is required")
        first = states[0]
        for other in states[1:]:
            if other.n_particles != first.n_particles:
                raise ValueError("replica systems must share the particle count")
            if other.device != first.device or other.dtype != first.dtype:
                raise ValueError("replica systems must share device and dtype")
            if not torch.equal(other.box, first.box):
                raise ValueError("replica systems must share the periodic box")
            if not torch.equal(other.active_mask, first.active_mask):
                raise ValueError("replica systems must share the active mask")
        return cls(
            positions=torch.stack([state.positions.detach() for state in states]),
            velocities=torch.stack([state.velocities.detach() for state in states]),
            diameters=torch.stack([state.diameters.detach() for state in states]),
            box=first.box.detach().clone(),
            active_mask=first.active_mask.detach().clone(),
            unwrapped_positions=torch.stack(
                [state.unwrapped_positions.detach() for state in states]
            ),
        )

    def clone(self) -> "BatchedSystem":
        return BatchedSystem(
            positions=self.positions.detach().clone(),
            velocities=self.velocities.detach().clone(),
            diameters=self.diameters.detach().clone(),
            box=self.box.detach().clone(),
            active_mask=self.active_mask.detach().clone(),
            unwrapped_positions=self.unwrapped_positions.detach().clone(),
        )

    def _branch_view(self, index: int) -> ParticleSystem:
        """Return a read-only-use view suitable for engine neighbor routines."""

        if index < 0 or index >= self.batch_size:
            raise IndexError("branch index is out of range")
        return ParticleSystem(
            positions=self.positions[index],
            velocities=self.velocities[index],
            diameters=self.branch_diameters(index),
            box=self.box,
            active_mask=self.active_mask,
            unwrapped_positions=self.unwrapped_positions[index],
        )

    def branch(self, index: int) -> ParticleSystem:
        """Return a detached single-system copy of one branch."""

        return self._branch_view(index).clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        clone = self.clone()
        return {
            "positions": clone.positions,
            "velocities": clone.velocities,
            "diameters": clone.diameters,
            "box": clone.box,
            "active_mask": clone.active_mask,
            "unwrapped_positions": clone.unwrapped_positions,
        }


@dataclass(frozen=True)
class BatchedPotentialResult:
    """Canonical potential outputs, one scalar energy and virial per branch."""

    energy: torch.Tensor
    forces: torch.Tensor
    virial: torch.Tensor
    interacting_pair_counts: torch.Tensor


def _normalise_pairs(
    pairs: Sequence[torch.Tensor] | torch.Tensor | None,
    *,
    batch_size: int,
    n_particles: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if pairs is None:
        base = all_pairs(n_particles, device)
        return tuple(base for _ in range(batch_size))
    if isinstance(pairs, torch.Tensor):
        if pairs.ndim == 2:
            candidates = tuple(pairs for _ in range(batch_size))
        elif pairs.ndim == 3 and pairs.shape[0] == batch_size:
            candidates = tuple(pairs[index] for index in range(batch_size))
        else:
            raise ValueError("pairs tensor must have shape (2, P) or (B, 2, P)")
    else:
        candidates = tuple(pairs)
        if len(candidates) != batch_size:
            raise ValueError("one pair tensor is required per branch")
    for candidate in candidates:
        if candidate.ndim != 2 or candidate.shape[0] != 2:
            raise ValueError("each pair tensor must have shape (2, P)")
        if candidate.device != device:
            raise ValueError("pair tensors must share the state device")
        if candidate.dtype != torch.int64:
            raise ValueError("pair tensors must use int64 indices")
        if candidate.numel() and (int(candidate.min()) < 0 or int(candidate.max()) >= n_particles):
            raise ValueError("pair index is out of range")
    return candidates


def _batched_particle_sum(
    particle_indices: torch.Tensor,
    contributions: torch.Tensor,
    n_particles: int,
    *,
    pair_capacity: int,
) -> torch.Tensor:
    """Batched analogue of the engine's sorted padded particle reduction.

    ``n_particles`` is used as a sentinel index for padding/non-interacting
    entries.  The sentinel sorts after every physical particle and therefore
    never participates in an output reduction.  ``pair_capacity`` is a
    host-side upper bound on candidate-pair incidences at any particle; the
    Verlet list refreshes it only when its candidate lists change, avoiding a
    device scalar read in every force evaluation.
    """

    batch_size, width = particle_indices.shape
    if pair_capacity < 0:
        raise ValueError("pair_capacity must be nonnegative")
    if width == 0 or pair_capacity == 0:
        return contributions.new_zeros((batch_size, n_particles, contributions.shape[-1]))
    order = torch.argsort(particle_indices, dim=1, stable=True)
    sorted_indices = torch.gather(particle_indices, 1, order)
    sorted_values = torch.gather(
        contributions,
        1,
        order[..., None].expand(-1, -1, contributions.shape[-1]),
    )
    particles = torch.arange(n_particles, device=particle_indices.device, dtype=particle_indices.dtype)
    particles = particles.expand(batch_size, -1).contiguous()
    starts = torch.searchsorted(sorted_indices, particles, right=False)
    ends = torch.searchsorted(sorted_indices, particles, right=True)
    counts = ends - starts
    slots = torch.arange(pair_capacity, device=particle_indices.device, dtype=particle_indices.dtype)
    gather_positions = starts[..., None] + slots
    valid = slots < counts[..., None]
    safe_positions = torch.clamp(gather_positions, max=width - 1)
    batch_indices = torch.arange(batch_size, device=particle_indices.device)[:, None, None]
    padded = sorted_values[batch_indices, safe_positions]
    padded = torch.where(valid[..., None], padded, torch.zeros_like(padded))
    return padded.sum(dim=2)


def _batched_analytic_potential(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    pairs: Sequence[torch.Tensor] | torch.Tensor | None,
    active_mask: torch.Tensor,
    pair_capacity: int | None = None,
) -> BatchedPotentialResult:
    batch_size, n_particles, _ = positions.shape
    pair_sets = _normalise_pairs(
        pairs,
        batch_size=batch_size,
        n_particles=n_particles,
        device=positions.device,
    )
    # The B=1 route deliberately invokes the engine's canonical evaluator.
    # It is the foundation of the documented bitwise trajectory guarantee.
    if batch_size == 1:
        result = analytic_potential(
            positions[0],
            diameters if diameters.ndim == 1 else diameters[0],
            box,
            pairs=pair_sets[0],
            active_mask=active_mask,
        )
        return BatchedPotentialResult(
            energy=result.energy.unsqueeze(0),
            forces=result.forces.unsqueeze(0),
            virial=result.virial.unsqueeze(0),
            interacting_pair_counts=torch.tensor(
                [result.pair_indices.shape[1]], device=positions.device, dtype=torch.int64
            ),
        )

    max_pairs = max((pair_set.shape[1] for pair_set in pair_sets), default=0)
    if pair_capacity is None:
        # Direct potential calls do not own a Verlet cache.  The pair count is
        # a safe (if loose) reduction bound there; the integrator supplies its
        # tighter per-particle incidence capacity.
        pair_capacity = max_pairs
    if pair_capacity < 0:
        raise ValueError("pair_capacity must be nonnegative")
    if max_pairs == 0:
        return BatchedPotentialResult(
            energy=positions.new_zeros((batch_size,)),
            forces=positions.new_zeros((batch_size, n_particles, 3)),
            virial=positions.new_zeros((batch_size,)),
            interacting_pair_counts=torch.zeros(batch_size, device=positions.device, dtype=torch.int64),
        )

    pair_i = torch.zeros((batch_size, max_pairs), device=positions.device, dtype=torch.int64)
    pair_j = torch.zeros_like(pair_i)
    present = torch.zeros((batch_size, max_pairs), device=positions.device, dtype=torch.bool)
    for branch, pair_set in enumerate(pair_sets):
        count = pair_set.shape[1]
        pair_i[branch, :count] = pair_set[0]
        pair_j[branch, :count] = pair_set[1]
        present[branch, :count] = True

    branch_indices = torch.arange(batch_size, device=positions.device)[:, None]
    position_i = positions[branch_indices, pair_i]
    position_j = positions[branch_indices, pair_j]
    displacement = minimum_image(position_i - position_j, box)
    radius = torch.linalg.vector_norm(displacement, dim=2)
    # Shared (N,) diameters gather flat; per-replica (B, N) rows gather with
    # the branch index broadcast.  For identical rows both read the same
    # float values, keeping the shared/cloned bitwise equivalence.
    if diameters.ndim == 1:
        sigma_ij = mixing_diameter(diameters[pair_i], diameters[pair_j])
    else:
        sigma_ij = mixing_diameter(
            diameters[branch_indices, pair_i], diameters[branch_indices, pair_j]
        )
    active_i = active_mask[pair_i]
    active_j = active_mask[pair_j]
    candidate = present & (active_i | active_j)
    interacting = candidate & (radius < CUTOFF_RATIO * sigma_ij)

    # Padding is evaluated exactly at the cutoff, where the imported engine
    # potential is zero.  This avoids infinities from a synthetic r=0 pair.
    evaluation_radius = torch.where(present, radius, CUTOFF_RATIO * sigma_ij)
    pair_energy, derivative = pair_potential(evaluation_radius, sigma_ij, derivatives=1)
    pair_energy = torch.where(interacting, pair_energy, torch.zeros_like(pair_energy))
    derivative = torch.where(interacting, derivative, torch.zeros_like(derivative))
    safe_radius = torch.clamp(radius, min=torch.finfo(positions.dtype).eps)
    force_i = -(derivative / safe_radius)[..., None] * displacement
    force_i = torch.where(interacting[..., None], force_i, torch.zeros_like(force_i))
    pair_virial = -derivative * radius

    sentinel = torch.full_like(pair_i, n_particles)
    contribution_indices = torch.cat(
        (
            torch.where(interacting, pair_i, sentinel),
            torch.where(interacting, pair_j, sentinel),
        ),
        dim=1,
    )
    contribution_values = torch.cat(
        (
            force_i * active_i[..., None],
            -force_i * active_j[..., None],
        ),
        dim=1,
    )
    return BatchedPotentialResult(
        energy=pair_energy.sum(dim=1),
        forces=_batched_particle_sum(
            contribution_indices,
            contribution_values,
            n_particles,
            pair_capacity=pair_capacity,
        ),
        virial=pair_virial.sum(dim=1),
        interacting_pair_counts=interacting.sum(dim=1),
    )


def batched_analytic_potential(
    system: BatchedSystem,
    *,
    pairs: Sequence[torch.Tensor] | torch.Tensor | None = None,
    pair_capacity: int | None = None,
) -> BatchedPotentialResult:
    """Evaluate the engine's smoothed pair potential for every branch.

    For ``B > 1`` pair arithmetic and the deterministic force reduction are
    vectorized over branches.  Pair sets may differ by branch, so the function
    internally pads them with a nonphysical sentinel index.  Diameters may be
    shared ``(N,)`` or per-replica ``(B, N)``; in the per-replica mode every
    branch's ``sigma_ij`` mixing uses its own diameter row.
    """

    return _batched_analytic_potential(
        system.positions,
        system.diameters,
        system.box,
        pairs=pairs,
        active_mask=system.active_mask,
        pair_capacity=pair_capacity,
    )


def batched_forces(
    system: BatchedSystem,
    *,
    pairs: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return only the canonical force tensor with shape ``(B, N, 3)``."""

    return batched_analytic_potential(system, pairs=pairs).forces


def branch_maxwell_boltzmann_velocities(
    n_particles: int,
    temperature: float,
    generators: Sequence[torch.Generator],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    masses: float | torch.Tensor = 1.0,
    active_mask: torch.Tensor | None = None,
    remove_com: bool = True,
) -> torch.Tensor:
    """Draw one exact engine Maxwell--Boltzmann velocity set per branch.

    A branch receives its own caller-owned generator; this is intentionally not
    a batched normal draw, because the harness records a distinct seed for every
    momentum realization.  The function therefore inherits the engine's CPU
    float64 draw and center-of-mass removal behavior exactly.
    """

    streams = tuple(generators)
    if not streams:
        raise ValueError("at least one branch generator is required")
    return torch.stack(
        [
            _engine_maxwell_boltzmann_velocities(
                n_particles,
                temperature,
                generator,
                device=device,
                dtype=dtype,
                masses=masses,
                active_mask=active_mask,
                remove_com=remove_com,
            )
            for generator in streams
        ]
    )


batched_maxwell_boltzmann_velocities = branch_maxwell_boltzmann_velocities


@dataclass
class BatchedVerletList:
    """One canonical Verlet list per independently moving branch.

    Pair lists cannot safely be shared after branches diverge.  Each branch is
    therefore rebuilt with the engine's own half-skin displacement rule, while
    the resulting variable-length pair sets are evaluated together by
    :func:`batched_analytic_potential`.
    """

    skin: float
    lists: list[VerletList]
    pair_capacity: int

    @staticmethod
    def _particle_incidence_capacity(lists: Sequence[VerletList], n_particles: int) -> int:
        """Return the maximum candidate-pair degree, synchronizing only at rebuilds."""

        maxima: list[torch.Tensor] = []
        for neighbors in lists:
            endpoints = neighbors.pair_indices.reshape(-1)
            if endpoints.numel() == 0:
                continue
            sorted_endpoints = torch.sort(endpoints).values
            particles = torch.arange(
                n_particles,
                device=endpoints.device,
                dtype=endpoints.dtype,
            )
            starts = torch.searchsorted(sorted_endpoints, particles, right=False)
            ends = torch.searchsorted(sorted_endpoints, particles, right=True)
            maxima.append((ends - starts).amax())
        return 0 if not maxima else int(torch.stack(maxima).amax().item())

    @classmethod
    def from_system(cls, system: BatchedSystem, skin: float = 0.3) -> "BatchedVerletList":
        if skin <= 0.0:
            raise ValueError("skin must be positive")
        lists = [
            VerletList.from_system(system._branch_view(index), skin=skin)
            for index in range(system.batch_size)
        ]
        return cls(
            skin=float(skin),
            lists=lists,
            pair_capacity=(
                0
                if system.batch_size == 1
                else cls._particle_incidence_capacity(lists, system.n_particles)
            ),
        )

    @property
    def pair_indices(self) -> tuple[torch.Tensor, ...]:
        return tuple(neighbors.pair_indices for neighbors in self.lists)

    @property
    def reference_positions(self) -> torch.Tensor:
        return torch.stack([neighbors.reference_positions for neighbors in self.lists])

    @property
    def rebuild_counts(self) -> torch.Tensor:
        return torch.tensor(
            [neighbors.rebuild_count for neighbors in self.lists],
            device=self.lists[0].pair_indices.device,
            dtype=torch.int64,
        )

    def needs_rebuild(self, positions: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[0] != len(self.lists):
            raise ValueError("positions must have one (N, 3) state per branch")
        if len(self.lists) > 1:
            displacement = minimum_image(positions - self.reference_positions, box)
            maximum = torch.linalg.vector_norm(displacement, dim=2).amax(dim=1)
            return maximum > 0.5 * self.skin
        return torch.tensor(
            [neighbors.needs_rebuild(positions[index], box) for index, neighbors in enumerate(self.lists)],
            device=positions.device,
            dtype=torch.bool,
        )

    def update(self, system: BatchedSystem) -> torch.Tensor:
        if system.batch_size != len(self.lists):
            raise ValueError("neighbor list batch size does not match system")
        if system.batch_size == 1:
            rebuilt = self.lists[0].update(system._branch_view(0))
            return torch.tensor([rebuilt], device=system.device, dtype=torch.bool)
        rebuilt = self.needs_rebuild(system.positions, system.box)
        # ``update`` is called at an integrator chunk boundary.  Reading this
        # compact mask there lets every inner force evaluation stay on device.
        rebuild_indices = torch.nonzero(
            rebuilt.detach().to(device="cpu"), as_tuple=False
        ).flatten().tolist()
        for index in rebuild_indices:
            neighbors = self.lists[index]
            # Branching changes diameter rows only by swap permutations, so
            # the maximum diameter (and this list radius) remains invariant.
            # A wholesale row replacement already requires the caller to
            # replace/rebuild its Verlet list, as documented on BatchedSystem.
            neighbors.pair_indices = cell_list_pairs(
                system.positions[index], system.box, neighbors.list_radius
            )
            neighbors.reference_positions = system.positions[index].detach().clone()
            neighbors.rebuild_count += 1
        if rebuild_indices:
            self.pair_capacity = self._particle_incidence_capacity(self.lists, system.n_particles)
        return rebuilt

    def evaluate(self, system: BatchedSystem, *, refresh: bool = True) -> BatchedPotentialResult:
        if system.batch_size != len(self.lists):
            raise ValueError("neighbor list batch size does not match system")
        if system.batch_size == 1:
            result = self.lists[0].evaluate(system._branch_view(0))
            return BatchedPotentialResult(
                energy=result.energy.unsqueeze(0),
                forces=result.forces.unsqueeze(0),
                virial=result.virial.unsqueeze(0),
                interacting_pair_counts=torch.tensor(
                    [result.pair_indices.shape[1]], device=system.device, dtype=torch.int64
                ),
            )
        if refresh:
            self.update(system)
        return batched_analytic_potential(
            system,
            pairs=self.pair_indices,
            pair_capacity=self.pair_capacity,
        )


class BatchedBussiThermostat:
    """Independent Bussi rescaling for every branch from one explicit stream."""

    def __init__(self, temperature: float, tau: float, generator: torch.Generator) -> None:
        if temperature <= 0.0 or tau <= 0.0:
            raise ValueError("temperature and tau must be positive")
        if str(generator.device) != "cpu":
            raise ValueError("ButterflyCone generators must be CPU generators")
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.generator = generator
        # Reuse the engine object for the B=1 exact-equivalence route.
        self._single_thermostat = BussiThermostat(self.temperature, self.tau, generator)
        self.last_alpha = torch.empty(0, dtype=torch.float64)
        self.heat = torch.empty(0, dtype=torch.float64)
        self._active_mask_key: tuple[str, int, int, int] | None = None
        self._ndof: int | None = None
        self._invalid_kinetic = torch.empty(0, dtype=torch.bool)

    def _sync_single_statistics(self, system: BatchedSystem) -> None:
        self.last_alpha = torch.tensor(
            [self._single_thermostat.last_alpha], device=system.device, dtype=system.dtype
        )
        self.heat = torch.tensor([self._single_thermostat.heat], device=system.device, dtype=system.dtype)

    @staticmethod
    def _mask_key(active_mask: torch.Tensor) -> tuple[str, int, int, int]:
        """Return static active-mask identity without reading device values."""

        return (
            str(active_mask.device),
            active_mask.data_ptr(),
            active_mask.numel(),
            active_mask._version,
        )

    def prepare(self, system: BatchedSystem) -> None:
        """Cache the static B>1 degree-of-freedom count outside the step loop."""

        if system.batch_size == 1:
            return
        active = system.active_mask
        key = self._mask_key(active)
        if key == self._active_mask_key:
            return
        n_active = int(active.sum().item())
        self._ndof = 3 * n_active - (3 if n_active > 1 else 0)
        self._active_mask_key = key

    def _ndof_for(self, system: BatchedSystem) -> int:
        self.prepare(system)
        if self._ndof is None:
            raise RuntimeError("Bussi thermostat degree-of-freedom cache was not initialized")
        return self._ndof

    def check_kinetic(self) -> None:
        """Raise a deferred B>1 kinetic-energy error at an integration boundary."""

        if self._invalid_kinetic.numel() == 0:
            return
        invalid = self._invalid_kinetic.detach().to(device="cpu")
        self._invalid_kinetic.zero_()
        if any(invalid.tolist()):
            raise ValueError("Bussi rescaling requires nonzero kinetic energy in every branch")

    def apply(self, system: BatchedSystem, dt: float) -> torch.Tensor:
        """Rescale velocities, consuming branch-major contiguous random blocks.

        At a thermostat call, branch 0 consumes the first ``ndof`` normal
        values, branch 1 the next block, and so on.  No global/backend RNG is
        used.  The B=1 route calls the engine thermostat itself.
        """

        if system.batch_size == 1:
            branch = system.branch(0)
            self._single_thermostat.apply(branch, dt)
            system.velocities[0].copy_(branch.velocities)
            self._sync_single_statistics(system)
            return self.last_alpha

        active = system.active_mask
        ndof = self._ndof_for(system)
        if ndof <= 0:
            return torch.ones(system.batch_size, device=system.device, dtype=system.dtype)
        kinetic_before = 0.5 * system.velocities[:, active].square().sum(dim=(1, 2))
        valid_kinetic = kinetic_before > 0.0
        if (
            self._invalid_kinetic.shape != valid_kinetic.shape
            or self._invalid_kinetic.device != valid_kinetic.device
        ):
            self._invalid_kinetic = torch.zeros_like(valid_kinetic)
        self._invalid_kinetic |= ~valid_kinetic
        safe_kinetic = torch.where(valid_kinetic, kinetic_before, torch.ones_like(kinetic_before))
        randoms = _normal_draw((system.batch_size, ndof), self.generator).to(system.device, system.dtype)
        gaussian = randoms[:, 0]
        chi_square = randoms[:, 1:].square().sum(dim=1)
        c = math.exp(-float(dt) / self.tau)
        target_kinetic = 0.5 * ndof * self.temperature
        ratio = torch.as_tensor(target_kinetic, device=system.device, dtype=system.dtype) / safe_kinetic
        alpha_squared = (
            c
            + (1.0 - c) * ratio * (chi_square + gaussian.square()) / ndof
            + 2.0 * gaussian * torch.sqrt(c * (1.0 - c) * ratio / ndof)
        )
        alpha = torch.sqrt(torch.clamp(alpha_squared, min=0.0))
        sign_threshold = gaussian + torch.sqrt(
            torch.as_tensor(c / (1.0 - c) * ndof, device=system.device, dtype=system.dtype) / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        alpha = torch.where(valid_kinetic, alpha, torch.ones_like(alpha))
        system.velocities = torch.where(
            active[None, :, None], system.velocities * alpha[:, None, None], system.velocities
        )
        kinetic_after = 0.5 * system.velocities[:, active].square().sum(dim=(1, 2))
        self.last_alpha = alpha
        if self.heat.numel() != system.batch_size:
            self.heat = torch.zeros(system.batch_size, device=system.device, dtype=system.dtype)
        self.heat = self.heat + kinetic_after - kinetic_before
        return alpha


class BatchedMDIntegrator:
    """Velocity-Verlet integrator for a :class:`BatchedSystem`."""

    def __init__(
        self,
        system: BatchedSystem,
        *,
        dt: float = 0.01,
        skin: float = 0.3,
        neighbor_list: BatchedVerletList | None = None,
        thermostat: BatchedBussiThermostat | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.system = system
        self.dt = float(dt)
        self.neighbor_list = BatchedVerletList.from_system(system, skin) if neighbor_list is None else neighbor_list
        if len(self.neighbor_list.lists) != system.batch_size:
            raise ValueError("neighbor list batch size does not match system")
        self.thermostat = thermostat
        self.step_count = 0
        self._single_integrator: MDIntegrator | None = None
        self._single_system: ParticleSystem | None = None
        if system.batch_size == 1:
            # Running the original engine object rather than an algebraically
            # equivalent batched expression makes the B=1 contract bitwise.
            self._single_system = system.branch(0)
            single_thermostat = None if thermostat is None else thermostat._single_thermostat
            self._single_integrator = MDIntegrator(
                self._single_system,
                dt=self.dt,
                neighbor_list=self.neighbor_list.lists[0],
                thermostat=single_thermostat,
            )
            self._sync_single_state()
        else:
            if self.thermostat is not None:
                self.thermostat.prepare(system)
            # The lists were just constructed from this exact state.  Skipping
            # the redundant refresh keeps integrator construction asynchronous.
            result = self.neighbor_list.evaluate(system, refresh=False)
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial

    def _sync_single_state(self) -> None:
        if self._single_integrator is None or self._single_system is None:
            return
        self.system.positions[0].copy_(self._single_system.positions)
        self.system.velocities[0].copy_(self._single_system.velocities)
        self.system.unwrapped_positions[0].copy_(self._single_system.unwrapped_positions)
        self.forces = self._single_integrator.forces.unsqueeze(0)
        self.potential_energy = self._single_integrator.potential_energy.unsqueeze(0)
        self.virial = self._single_integrator.virial.unsqueeze(0)
        self.step_count = self._single_integrator.step_count
        if self.thermostat is not None:
            self.thermostat._sync_single_statistics(self.system)

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if self._single_integrator is not None:
            self._single_integrator.step(steps)
            self._sync_single_state()
            return
        if steps == 0:
            return
        # Variable-length Python-owned Verlet lists can only be refreshed when
        # their compact device mask is read back.  Do that once at the caller's
        # chunk boundary, then keep every inner velocity-Verlet force evaluation
        # asynchronous and use the cached pair-capacity for its reduction.
        self.neighbor_list.update(self.system)
        active = self.system.active_mask[None, :, None]
        for _ in range(steps):
            half_velocity = self.system.velocities + 0.5 * self.dt * self.forces
            displacement = self.dt * half_velocity
            displacement = torch.where(active, displacement, torch.zeros_like(displacement))
            self.system.unwrapped_positions = self.system.unwrapped_positions + displacement
            moved_positions = torch.remainder(self.system.positions + displacement, self.system.box)
            self.system.positions = torch.where(active, moved_positions, self.system.positions)
            result = self.neighbor_list.evaluate(self.system, refresh=False)
            self.system.velocities = half_velocity + 0.5 * self.dt * result.forces
            self.system.velocities = torch.where(
                active, self.system.velocities, torch.zeros_like(self.system.velocities)
            )
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial
            if self.thermostat is not None:
                self.thermostat.apply(self.system, self.dt)
            self.step_count += 1
        if self.thermostat is not None:
            self.thermostat.check_kinetic()

    def total_energy(self) -> torch.Tensor:
        kinetic = 0.5 * self.system.velocities[:, self.system.active_mask].square().sum(dim=(1, 2))
        return self.potential_energy + kinetic
