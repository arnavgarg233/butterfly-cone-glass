"""Particle state and reproducible system construction.

Random numbers are always drawn on CPU from the caller-owned generator and
then copied to the requested device.  This gives one RNG stream on CPU and
MPS and avoids relying on backend-global random state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

DIAMETER_RATIO = 2.219


def diameter_bounds(ratio: float = DIAMETER_RATIO) -> tuple[float, float]:
    """Return bounds whose sigma^-3 distribution has arithmetic mean one."""

    if ratio <= 1.0:
        raise ValueError("diameter ratio must exceed one")
    sigma_min = (1.0 + ratio) / (2.0 * ratio)
    sigma_max = (1.0 + ratio) / 2.0
    return sigma_min, sigma_max


def make_generator(seed: int) -> torch.Generator:
    """Create the explicit CPU generator used by every stochastic utility."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _cpu_random(
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    normal: bool = False,
) -> torch.Tensor:
    if str(generator.device) != "cpu":
        raise ValueError("ButterflyCone generators must be CPU generators")
    draw = torch.randn if normal else torch.rand
    return draw(shape, generator=generator, device="cpu", dtype=torch.float64)


def sample_diameters(
    n_particles: int,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    ratio: float = DIAMETER_RATIO,
) -> torch.Tensor:
    """Draw diameters by inverse-CDF sampling from ``P(sigma) ~ sigma^-3``.

    If ``u`` is a CPU float64 uniform draw in ``[0, 1)``, the returned value is
    ``[a^-2 - u (a^-2 - b^-2)]^-1/2`` with ``a=sigma_min`` and
    ``b=sigma_max``.  The final tensor is converted once to ``device,dtype``.
    """

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    lower, upper = diameter_bounds(ratio)
    u = _cpu_random((n_particles,), generator)
    inverse_square = lower**-2 - u * (lower**-2 - upper**-2)
    return torch.rsqrt(inverse_square).to(device=device, dtype=dtype)


def cubic_box(
    n_particles: int,
    density: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n_particles <= 0 or density <= 0.0:
        raise ValueError("n_particles and density must be positive")
    length = (n_particles / density) ** (1.0 / 3.0)
    return torch.full((3,), length, device=device, dtype=dtype)


def lattice_positions(n_particles: int, box: torch.Tensor) -> torch.Tensor:
    """Place particles at cell centers of the smallest enclosing cubic grid."""

    side = math.ceil(n_particles ** (1.0 / 3.0))
    while side**3 < n_particles:
        side += 1
    axis = (torch.arange(side, device=box.device, dtype=box.dtype) + 0.5) / side
    grid = torch.cartesian_prod(axis, axis, axis)
    if grid.ndim == 1:  # torch.cartesian_prod's one-element edge case
        grid = grid.reshape(-1, 3)
    return grid[:n_particles] * box


def random_positions(
    n_particles: int,
    box: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    draws = _cpu_random((n_particles, 3), generator)
    return draws.to(device=box.device, dtype=box.dtype) * box


@dataclass
class ParticleSystem:
    """Tensor state for unit-mass particles in a periodic orthorhombic box."""

    positions: torch.Tensor
    velocities: torch.Tensor
    diameters: torch.Tensor
    box: torch.Tensor
    active_mask: torch.Tensor
    unwrapped_positions: torch.Tensor

    def __post_init__(self) -> None:
        n_particles = int(self.positions.shape[0])
        if self.positions.shape != (n_particles, 3):
            raise ValueError("positions must have shape (N, 3)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must match positions")
        if self.unwrapped_positions.shape != self.positions.shape:
            raise ValueError("unwrapped_positions must match positions")
        if self.diameters.shape != (n_particles,):
            raise ValueError("diameters must have shape (N,)")
        if self.active_mask.shape != (n_particles,) or self.active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a bool tensor of shape (N,)")
        if self.box.shape != (3,) or bool(torch.any(self.box <= 0)):
            raise ValueError("box must contain three positive lengths")
        tensors = (self.velocities, self.diameters, self.box, self.active_mask, self.unwrapped_positions)
        if any(tensor.device != self.positions.device for tensor in tensors):
            raise ValueError("all state tensors must share a device")

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    def clone(self) -> "ParticleSystem":
        return ParticleSystem(
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

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> "ParticleSystem":
        target = state["positions"].device if device is None else torch.device(device)
        return cls(**{name: tensor.detach().clone().to(target) for name, tensor in state.items()})


@dataclass(frozen=True)
class RelaxationReport:
    steps_completed: int
    initial_energy: float
    final_energy: float
    final_max_force: float


def relax_overlaps(
    system: ParticleSystem,
    *,
    steps: int = 1000,
    max_displacement: float = 0.002,
    force_tolerance: float = 1e-3,
    skin: float = 0.3,
) -> RelaxationReport:
    """Deterministically descend the energy with force-aligned capped moves.

    The cap makes the method robust to the enormous forces possible in a raw
    random placement.  This is a preparation/minimization utility, not physical
    time integration.
    """

    if steps < 0 or max_displacement <= 0.0 or force_tolerance < 0.0:
        raise ValueError("invalid relaxation controls")
    # Local import keeps state construction independent of the neighbor module.
    from .neighbors import VerletList

    neighbors = VerletList.from_system(system, skin=skin)
    result = neighbors.evaluate(system)
    initial_energy = float(result.energy)
    completed = 0
    maximum_force = float(torch.linalg.vector_norm(result.forces, dim=1).max())
    for _ in range(steps):
        force_norm = torch.linalg.vector_norm(result.forces, dim=1, keepdim=True)
        maximum_force = float(force_norm.max())
        if maximum_force <= force_tolerance:
            break
        scale = torch.clamp(max_displacement / torch.clamp(force_norm, min=torch.finfo(system.dtype).eps), max=1.0)
        displacement = result.forces * scale
        displacement = torch.where(system.active_mask[:, None], displacement, torch.zeros_like(displacement))
        system.positions = torch.remainder(system.positions + displacement, system.box)
        system.unwrapped_positions = system.unwrapped_positions + displacement
        result = neighbors.evaluate(system)
        completed += 1
    maximum_force = float(torch.linalg.vector_norm(result.forces, dim=1).max())
    return RelaxationReport(completed, initial_energy, float(result.energy), maximum_force)


def make_system(
    n_particles: int = 4096,
    *,
    generator: torch.Generator,
    density: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    placement: str = "lattice",
    active_mask: torch.Tensor | None = None,
    relaxation_steps: int = 0,
    relaxation_max_displacement: float = 0.002,
) -> ParticleSystem:
    """Construct a reproducible zero-velocity particle set.

    ``placement='lattice'`` is the robust production initializer.  The random
    initializer supplies a raw configuration; :func:`relax_overlaps` can then
    be called before dynamics (and is intentionally explicit because its step
    count is part of a simulation protocol).
    """

    box = cubic_box(n_particles, density, device=device, dtype=dtype)
    diameters = sample_diameters(n_particles, generator, device=device, dtype=dtype)
    if placement == "lattice":
        positions = lattice_positions(n_particles, box)
    elif placement == "random":
        positions = random_positions(n_particles, box, generator)
    else:
        raise ValueError("placement must be 'lattice' or 'random'")
    if active_mask is None:
        active = torch.ones(n_particles, device=device, dtype=torch.bool)
    else:
        active = active_mask.detach().clone().to(device=device, dtype=torch.bool)
    system = ParticleSystem(
        positions=positions,
        velocities=torch.zeros_like(positions),
        diameters=diameters,
        box=box,
        active_mask=active,
        unwrapped_positions=positions.clone(),
    )
    if relaxation_steps:
        relax_overlaps(
            system,
            steps=relaxation_steps,
            max_displacement=relaxation_max_displacement,
        )
    return system
