"""Dense batched-replica MD: B independent same-N replicas in lockstep.

This layer evolves ``B`` independent replicas of the SAME particle count ``N``
through one set of tensor kernels: positions/velocities ``(B, N, 3)``,
per-replica diameters ``(B, N)``, and a per-replica orthorhombic PBC box
``(B, 3)`` (identical rows are fine and are the v1 production case).  Forces
use the DENSE pairwise route -- the full ``(B, N, N)`` distance matrix --
mirroring the brute-force path in :mod:`butterfly_cone.engine.potential` exactly: the
same :func:`~butterfly_cone.engine.potential.pair_potential` smoothing polynomial, the
same ``CUTOFF_RATIO`` and strict ``radius < CUTOFF_RATIO * sigma_ij``
interaction criterion, the same nonadditive
:func:`~butterfly_cone.engine.potential.mixing_diameter`, and the same minimum-image
convention.  Integration is the engine's velocity-Verlet with an optional
Bussi thermostat, batched with per-replica CPU generators.

Determinism contract (project doctrine)
---------------------------------------

Every reduction in this module is fixed-order: either an elementwise op, a
gather with ascending indices, or :func:`_fixed_tree_sum` -- an explicit
pairwise-halving summation whose bracketing depends only on the reduced
length (``N`` or ``ndof``), never on the batch size, thread count, or
device heuristics.  No ``scatter_add`` / ``index_add`` / ``bincount`` /
atomic accumulation appears anywhere.  Consequences, certified by
``scripts/test_batched_engine.py`` on CPU:

1. **Run-to-run bit-reproducibility.**  The same initial state and seeds
   reproduce bit-identical trajectories on one device.
2. **Batch-size independence.**  Replica ``r``'s trajectory is bit-identical
   whether it runs alone at ``B=1`` or embedded in any larger batch, and is
   independent of the *contents* of the other rows.  This holds because every
   kernel is either elementwise over the batch axis or reduces along a
   fixed-shape per-replica axis with the fixed tree, and because thermostat
   noise comes from per-replica generators.
3. **Twin bit-identity.**  Two bit-identical rows with identical thermostat
   streams remain bit-identical forever until explicitly perturbed.

Why this engine is NOT bit-identical to the single-system engine
----------------------------------------------------------------

:func:`butterfly_cone.engine.potential.analytic_potential` accumulates forces through a
sorted-pair-list segment sum: particle ``p``'s force adds its ``i``-side pair
contributions (``j > p`` ascending) and then its ``j``-side contributions
(``i < p`` ascending), padded to the maximum incidence width, and its energy
sums each ``i<j`` pair once in lexicographic pair order.  The dense route
sums the full force row ``j = 0..N-1`` through a pairwise-halving tree and
counts every pair twice before an exact ``0.5`` factor.  These are different
floating-point summation trees over the same pair values, so results agree to
rounding (verified ``allclose`` at float64) but not bitwise -- the same class
of tolerated rounding-level difference as the documented CPU-vs-MPS gap and
as ``butterfly_cone.branching.batched``'s ``B>1`` path.  Unlike
:class:`butterfly_cone.branching.batched.BatchedMDIntegrator`, this engine deliberately
does NOT special-case ``B=1`` to the single engine: using one code path at
every ``B`` is precisely what makes guarantee (2) achievable.

Guarantees (1)-(3) are certified per device by the test suite; re-run it on a
new device/backend before trusting production output there.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .integrate import (
    _normal_draw,
    maxwell_boltzmann_velocities as _engine_maxwell_boltzmann_velocities,
)
from .potential import CUTOFF_RATIO, minimum_image, mixing_diameter, pair_potential
from .system import ParticleSystem

# Conservative count of concurrently-live (B, N, N)-sized float tensors on the
# dense force path (displacement stack, radius, sigma matrix, evaluation
# radius, smoothing-polynomial temporaries, masked energy/derivative, force
# matrix and its halving-tree partials).  Used only by the memory estimator.
DENSE_FLOATS_PER_PAIR = 26


# ---------------------------------------------------------------------------
# Fixed-tree reductions (the doctrine-compliant summation primitive)
# ---------------------------------------------------------------------------


def _fixed_tree_sum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum along ``dim`` via explicit pairwise halving with elementwise adds.

    The bracketing is a function of the reduced length only, so the summation
    tree is identical for every replica row, every batch size, every thread
    count, and every device.  Odd halves are zero-padded on the right; adding
    an exact ``+0.0`` never changes a partial sum.
    """

    if values.shape[dim] == 0:
        return values.sum(dim=dim)  # exact zeros of the reduced shape
    total = values
    while total.shape[dim] > 1:
        length = total.shape[dim]
        half = (length + 1) // 2
        first = total.narrow(dim, 0, half)
        second = total.narrow(dim, half, length - half)
        if second.shape[dim] < half:
            pad_shape = list(second.shape)
            pad_shape[dim] = half - second.shape[dim]
            second = torch.cat((second, second.new_zeros(pad_shape)), dim=dim)
        total = first + second
    return total.squeeze(dim)


def _fixed_tree_row_sum(values: torch.Tensor) -> torch.Tensor:
    """Reduce ``(B, M)`` to ``(B,)`` with the fixed pairwise tree."""

    if values.ndim != 2:
        raise ValueError("row sum expects a (B, M) tensor")
    return _fixed_tree_sum(values, dim=1)


# ---------------------------------------------------------------------------
# Batched replica state
# ---------------------------------------------------------------------------


@dataclass
class ReplicaBatch:
    """Tensor state for ``B`` independent same-``N`` replicas in lockstep.

    ``diameters`` may be passed as ``(N,)`` and ``box`` as ``(3,)``; both are
    canonicalized to per-replica ``(B, N)`` / ``(B, 3)`` rows at construction.
    The active mask is shared ``(N,)`` across replicas (engine convention).
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
        if self.diameters.shape == (n_particles,):
            self.diameters = self.diameters.unsqueeze(0).expand(batch_size, n_particles).clone()
        if self.diameters.shape != (batch_size, n_particles):
            raise ValueError("diameters must have shape (B, N) or (N,)")
        if self.box.shape == (3,):
            self.box = self.box.unsqueeze(0).expand(batch_size, 3).clone()
        if self.box.shape != (batch_size, 3) or bool(torch.any(self.box <= 0)):
            raise ValueError("box must have shape (B, 3) or (3,) with positive lengths")
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
        if any(
            tensor.dtype != self.positions.dtype for tensor in tensors if tensor.dtype != torch.bool
        ):
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

    @classmethod
    def from_system(cls, parent: ParticleSystem, batch_size: int) -> "ReplicaBatch":
        """Copy one parent state into ``batch_size`` identical replicas."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, parent.n_particles, 3)
        return cls(
            positions=parent.positions.detach().unsqueeze(0).expand(shape).clone(),
            velocities=parent.velocities.detach().unsqueeze(0).expand(shape).clone(),
            diameters=parent.diameters.detach().clone(),
            box=parent.box.detach().clone(),
            active_mask=parent.active_mask.detach().clone(),
            unwrapped_positions=parent.unwrapped_positions.detach().unsqueeze(0).expand(shape).clone(),
        )

    @classmethod
    def from_systems(cls, replicas: Sequence[ParticleSystem]) -> "ReplicaBatch":
        """Stack independent same-``N`` systems; boxes and diameters go per-row."""

        states = tuple(replicas)
        if not states:
            raise ValueError("at least one replica system is required")
        first = states[0]
        for other in states[1:]:
            if other.n_particles != first.n_particles:
                raise ValueError("replica systems must share the particle count")
            if other.device != first.device or other.dtype != first.dtype:
                raise ValueError("replica systems must share device and dtype")
            if not torch.equal(other.active_mask, first.active_mask):
                raise ValueError("replica systems must share the active mask")
        return cls(
            positions=torch.stack([state.positions.detach() for state in states]).clone(),
            velocities=torch.stack([state.velocities.detach() for state in states]).clone(),
            diameters=torch.stack([state.diameters.detach() for state in states]).clone(),
            box=torch.stack([state.box.detach() for state in states]).clone(),
            active_mask=first.active_mask.detach().clone(),
            unwrapped_positions=torch.stack(
                [state.unwrapped_positions.detach() for state in states]
            ).clone(),
        )

    def replica(self, index: int) -> ParticleSystem:
        """Return a detached single-system copy of one replica row."""

        if index < 0 or index >= self.batch_size:
            raise IndexError("replica index is out of range")
        return ParticleSystem(
            positions=self.positions[index].detach().clone(),
            velocities=self.velocities[index].detach().clone(),
            diameters=self.diameters[index].detach().clone(),
            box=self.box[index].detach().clone(),
            active_mask=self.active_mask.detach().clone(),
            unwrapped_positions=self.unwrapped_positions[index].detach().clone(),
        )

    def clone(self) -> "ReplicaBatch":
        return ReplicaBatch(
            positions=self.positions.detach().clone(),
            velocities=self.velocities.detach().clone(),
            diameters=self.diameters.detach().clone(),
            box=self.box.detach().clone(),
            active_mask=self.active_mask.detach().clone(),
            unwrapped_positions=self.unwrapped_positions.detach().clone(),
        )

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


# ---------------------------------------------------------------------------
# Dense pairwise potential (mirrors engine.potential.brute_force exactly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaPotentialResult:
    """Per-replica energy ``(B,)``, forces ``(B, N, 3)``, virial ``(B,)``."""

    energy: torch.Tensor
    forces: torch.Tensor
    virial: torch.Tensor
    interacting_pair_counts: torch.Tensor


def replica_mixing_matrix(diameters: torch.Tensor) -> torch.Tensor:
    """Per-replica nonadditive mixing matrix ``sigma_ij`` with shape ``(B, N, N)``.

    Static for fixed diameters; the integrator caches it across force calls.
    """

    if diameters.ndim != 2:
        raise ValueError("diameters must have shape (B, N)")
    return mixing_diameter(diameters[:, :, None], diameters[:, None, :])


def dense_replica_potential(
    positions: torch.Tensor,
    diameters: torch.Tensor,
    box: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    sigma_matrix: torch.Tensor | None = None,
) -> ReplicaPotentialResult:
    """Dense ``(B, N, N)`` evaluation of the engine pair potential.

    Semantics mirror :func:`butterfly_cone.engine.potential.analytic_potential`: strict
    ``radius < CUTOFF_RATIO * sigma_ij`` interaction criterion, frozen-frozen
    pairs excluded, active-frozen pairs kept in the energy, frozen forces
    zeroed.  Every pair appears twice in the dense matrix; the transpose entry
    evaluates to bit-identical values (the squared-distance norm and the
    symmetric mixing rule are direction-independent), so the exact ``0.5``
    factor recovers the once-per-pair energy and virial.

    Non-interacting and diagonal entries are evaluated at the cutoff radius,
    where the smoothing polynomial vanishes, so no infinities ever enter a
    masked lane (same device-safe device trick as ``butterfly_cone.branching.batched``).
    """

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (B, N, 3)")
    batch_size, n_particles, _ = positions.shape
    if diameters.shape != (batch_size, n_particles):
        raise ValueError("diameters must have shape (B, N)")
    if box.shape != (batch_size, 3):
        raise ValueError("box must have shape (B, 3)")
    if active_mask is None:
        active = torch.ones(n_particles, dtype=torch.bool, device=positions.device)
    else:
        active = active_mask.to(device=positions.device, dtype=torch.bool)
        if active.shape != (n_particles,):
            raise ValueError("active_mask must have shape (N,)")

    if sigma_matrix is None:
        sigma_matrix = replica_mixing_matrix(diameters)
    if sigma_matrix.shape != (batch_size, n_particles, n_particles):
        raise ValueError("sigma_matrix must have shape (B, N, N)")

    displacement = minimum_image(
        positions[:, :, None, :] - positions[:, None, :, :],
        box[:, None, None, :],
    )
    radius = torch.linalg.vector_norm(displacement, dim=3)

    off_diagonal = ~torch.eye(n_particles, dtype=torch.bool, device=positions.device)
    pair_active = active[:, None] | active[None, :]
    interacting = (
        (radius < CUTOFF_RATIO * sigma_matrix)
        & off_diagonal[None, :, :]
        & pair_active[None, :, :]
    )

    evaluation_radius = torch.where(interacting, radius, CUTOFF_RATIO * sigma_matrix)
    pair_energy, derivative = pair_potential(evaluation_radius, sigma_matrix, derivatives=1)
    pair_energy = torch.where(interacting, pair_energy, torch.zeros_like(pair_energy))
    derivative = torch.where(interacting, derivative, torch.zeros_like(derivative))

    safe_radius = torch.clamp(radius, min=torch.finfo(positions.dtype).eps)
    force_matrix = -(derivative / safe_radius)[..., None] * displacement
    force_matrix = torch.where(interacting[..., None], force_matrix, torch.zeros_like(force_matrix))

    forces = _fixed_tree_sum(force_matrix, dim=2)
    forces = forces * active[None, :, None].to(forces.dtype)

    energy = 0.5 * _fixed_tree_row_sum(pair_energy.flatten(start_dim=1))
    pair_virial = torch.where(interacting, -derivative * radius, torch.zeros_like(radius))
    virial = 0.5 * _fixed_tree_row_sum(pair_virial.flatten(start_dim=1))
    counts = interacting.sum(dim=(1, 2)) // 2  # exact integer sum

    return ReplicaPotentialResult(
        energy=energy,
        forces=forces,
        virial=virial,
        interacting_pair_counts=counts,
    )


def evaluate_replicas(
    batch: ReplicaBatch, *, sigma_matrix: torch.Tensor | None = None
) -> ReplicaPotentialResult:
    """Evaluate the dense potential for a :class:`ReplicaBatch`."""

    return dense_replica_potential(
        batch.positions,
        batch.diameters,
        batch.box,
        active_mask=batch.active_mask,
        sigma_matrix=sigma_matrix,
    )


# ---------------------------------------------------------------------------
# Per-replica stochastic pieces (exact engine RNG streams, one per replica)
# ---------------------------------------------------------------------------


def replica_maxwell_boltzmann_velocities(
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
    """Stack one exact engine Maxwell-Boltzmann draw per replica generator.

    Replica ``r`` consumes exactly the stream a single-system run with the
    same generator would, so initial velocities are batch-size independent by
    construction.
    """

    streams = tuple(generators)
    if not streams:
        raise ValueError("at least one replica generator is required")
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


class ReplicaBussiThermostat:
    """Engine Bussi-Donadio-Parrinello rescaling with per-replica generators.

    Replica ``r`` draws its ``ndof`` normals from its OWN CPU generator, so
    its noise stream is independent of the batch composition -- the load-
    bearing property behind the batch-size-independence guarantee.  The
    rescaling algebra mirrors :class:`butterfly_cone.engine.integrate.BussiThermostat`
    line by line, vectorized over the batch axis; the kinetic-energy and
    chi-square reductions use the fixed pairwise tree.
    """

    def __init__(
        self, temperature: float, tau: float, generators: Sequence[torch.Generator]
    ) -> None:
        if temperature <= 0.0 or tau <= 0.0:
            raise ValueError("temperature and tau must be positive")
        streams = tuple(generators)
        if not streams:
            raise ValueError("at least one replica generator is required")
        if any(str(generator.device) != "cpu" for generator in streams):
            raise ValueError("ButterflyCone generators must be CPU generators")
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.generators = streams
        self.last_alpha: torch.Tensor | None = None
        self.heat: torch.Tensor | None = None

    def apply(self, batch: ReplicaBatch, dt: float) -> torch.Tensor:
        if len(self.generators) != batch.batch_size:
            raise ValueError("one generator is required per replica")
        active = batch.active_mask
        n_active = int(active.sum().item())
        ndof = 3 * n_active - (3 if n_active > 1 else 0)
        if ndof <= 0:
            return torch.ones(batch.batch_size, device=batch.device, dtype=batch.dtype)
        selected = batch.velocities[:, active, :]
        kinetic_before = 0.5 * _fixed_tree_row_sum(selected.square().flatten(start_dim=1))
        if bool(torch.any(kinetic_before <= 0.0)):
            raise ValueError("Bussi rescaling requires nonzero kinetic energy in every replica")
        randoms = torch.stack(
            [_normal_draw((ndof,), generator) for generator in self.generators]
        ).to(batch.device, batch.dtype)
        gaussian = randoms[:, 0]
        chi_square = _fixed_tree_row_sum(randoms[:, 1:].square())
        c = math.exp(-float(dt) / self.tau)
        target_kinetic = 0.5 * ndof * self.temperature
        ratio = (
            torch.as_tensor(target_kinetic, device=batch.device, dtype=batch.dtype)
            / kinetic_before
        )
        alpha_squared = (
            c
            + (1.0 - c) * ratio * (chi_square + gaussian.square()) / ndof
            + 2.0 * gaussian * torch.sqrt(c * (1.0 - c) * ratio / ndof)
        )
        alpha = torch.sqrt(torch.clamp(alpha_squared, min=0.0))
        sign_threshold = gaussian + torch.sqrt(
            torch.as_tensor(c / (1.0 - c) * ndof, device=batch.device, dtype=batch.dtype) / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        batch.velocities = torch.where(
            active[None, :, None], batch.velocities * alpha[:, None, None], batch.velocities
        )
        selected_after = batch.velocities[:, active, :]
        kinetic_after = 0.5 * _fixed_tree_row_sum(selected_after.square().flatten(start_dim=1))
        self.last_alpha = alpha
        delta_heat = kinetic_after - kinetic_before
        self.heat = delta_heat if self.heat is None else self.heat + delta_heat
        return alpha


# ---------------------------------------------------------------------------
# Batched velocity-Verlet (mirrors engine.integrate.MDIntegrator semantics)
# ---------------------------------------------------------------------------


class ReplicaMDIntegrator:
    """Velocity-Verlet for a :class:`ReplicaBatch` with dense forces.

    Step ordering mirrors :class:`butterfly_cone.engine.integrate.MDIntegrator`: half
    kick, drift with ``torch.remainder`` wrap (frozen particles pinned),
    force evaluation, half kick, then the optional thermostat -- once per
    step.  The mixing matrix is cached because diameters are static during
    dynamics; call :meth:`refresh_diameters` after any diameter change.
    """

    def __init__(
        self,
        batch: ReplicaBatch,
        *,
        dt: float = 0.01,
        thermostat: ReplicaBussiThermostat | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.batch = batch
        self.dt = float(dt)
        self.thermostat = thermostat
        self.step_count = 0
        self._sigma_matrix = replica_mixing_matrix(batch.diameters)
        result = evaluate_replicas(batch, sigma_matrix=self._sigma_matrix)
        self.forces = result.forces
        self.potential_energy = result.energy
        self.virial = result.virial

    def refresh_diameters(self) -> None:
        """Recompute the cached mixing matrix and forces after a diameter edit."""

        self._sigma_matrix = replica_mixing_matrix(self.batch.diameters)
        result = evaluate_replicas(self.batch, sigma_matrix=self._sigma_matrix)
        self.forces = result.forces
        self.potential_energy = result.energy
        self.virial = result.virial

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        batch = self.batch
        active = batch.active_mask[None, :, None]
        box = batch.box[:, None, :]
        for _ in range(steps):
            half_velocity = batch.velocities + 0.5 * self.dt * self.forces
            displacement = self.dt * half_velocity
            displacement = torch.where(active, displacement, torch.zeros_like(displacement))
            batch.unwrapped_positions = batch.unwrapped_positions + displacement
            moved_positions = torch.remainder(batch.positions + displacement, box)
            batch.positions = torch.where(active, moved_positions, batch.positions)
            result = evaluate_replicas(batch, sigma_matrix=self._sigma_matrix)
            batch.velocities = half_velocity + 0.5 * self.dt * result.forces
            batch.velocities = torch.where(
                active, batch.velocities, torch.zeros_like(batch.velocities)
            )
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial
            if self.thermostat is not None:
                self.thermostat.apply(batch, self.dt)
            self.step_count += 1

    def total_energy(self) -> torch.Tensor:
        selected = self.batch.velocities[:, self.batch.active_mask, :]
        kinetic = 0.5 * _fixed_tree_row_sum(selected.square().flatten(start_dim=1))
        return self.potential_energy + kinetic


# ---------------------------------------------------------------------------
# Memory estimation for the dense (B, N, N) route
# ---------------------------------------------------------------------------


def dense_pair_bytes_per_replica(n_particles: int, dtype: torch.dtype = torch.float32) -> int:
    """Conservative peak bytes one replica adds to a dense force evaluation."""

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    itemsize = torch.finfo(dtype).bits // 8
    pair_floats = DENSE_FLOATS_PER_PAIR * n_particles * n_particles * itemsize
    pair_bools = 2 * n_particles * n_particles  # interaction + scratch masks
    state = 5 * 3 * n_particles * itemsize  # positions/velocities/unwrapped/forces/half
    return pair_floats + pair_bools + state


def max_batch_size(
    n_particles: int,
    memory_bytes: int | float,
    dtype: torch.dtype = torch.float32,
    *,
    safety_fraction: float = 0.8,
) -> int:
    """Largest ``B`` whose dense evaluation fits in ``memory_bytes`` of VRAM."""

    if memory_bytes <= 0:
        raise ValueError("memory_bytes must be positive")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    per_replica = dense_pair_bytes_per_replica(n_particles, dtype)
    return int((safety_fraction * float(memory_bytes)) // per_replica)


def memory_report(
    n_particles: int,
    memory_bytes: int | float,
    dtype: torch.dtype = torch.float32,
    *,
    safety_fraction: float = 0.8,
) -> dict:
    """Human-auditable memory estimate for drivers to print/persist."""

    per_replica = dense_pair_bytes_per_replica(n_particles, dtype)
    return {
        "n_particles": int(n_particles),
        "dtype": str(dtype),
        "dense_floats_per_pair": DENSE_FLOATS_PER_PAIR,
        "bytes_per_replica": int(per_replica),
        "mib_per_replica": float(per_replica / 2**20),
        "vram_bytes": int(memory_bytes),
        "safety_fraction": float(safety_fraction),
        "max_batch_size": max_batch_size(
            n_particles, memory_bytes, dtype, safety_fraction=safety_fraction
        ),
    }
