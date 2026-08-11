"""Frozen parent-time cavity geometry and candidate-state provenance.

The particle labels belonging to the sampled buffer are selected once from the
parent configuration.  They remain the active labels even if their centers
later move across either geometric radius during ordinary RCCE sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from butterfly_cone.engine.potential import minimum_image
from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class CavitySpec:
    """A spherical core and buffer in an orthorhombic periodic box."""

    center: tuple[float, float, float]
    core_radius: float
    buffer_radius: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError("cavity center must contain three finite coordinates")
        core = float(self.core_radius)
        buffer = float(self.buffer_radius)
        if not math.isfinite(core) or not math.isfinite(buffer):
            raise ValueError("cavity radii must be finite")
        if core <= 0.0 or buffer <= core:
            raise ValueError("radii must satisfy 0 < core_radius < buffer_radius")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "core_radius", core)
        object.__setattr__(self, "buffer_radius", buffer)

    def to_dict(self) -> dict[str, object]:
        return {
            "center": list(self.center),
            "core_radius": self.core_radius,
            "buffer_radius": self.buffer_radius,
        }


def minimum_image_from_center(
    positions: torch.Tensor,
    center: Sequence[float] | torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    """Return minimum-image vectors pointing from ``center`` to particles."""

    center_tensor = torch.as_tensor(center, device=positions.device, dtype=positions.dtype)
    if center_tensor.shape != (3,):
        raise ValueError("center must have shape (3,)")
    return minimum_image(positions - center_tensor, box)


@dataclass(frozen=True)
class CavitySelection:
    """Parent-time label sets for one cavity."""

    spec: CavitySpec
    parent_distances: torch.Tensor
    core_mask: torch.Tensor
    shell_mask: torch.Tensor
    buffer_mask: torch.Tensor
    exterior_mask: torch.Tensor
    core_indices: torch.Tensor
    shell_indices: torch.Tensor
    buffer_indices: torch.Tensor
    exterior_indices: torch.Tensor

    @property
    def n_core(self) -> int:
        return int(self.core_indices.numel())

    @property
    def n_buffer(self) -> int:
        return int(self.buffer_indices.numel())


def select_cavity(system: ParticleSystem, spec: CavitySpec) -> CavitySelection:
    """Select core, shell, buffer, and exterior labels at parent time.

    A buffer larger than half the shortest box length is rejected because a
    nominal sphere would overlap its own periodic image and cease to define a
    unique local cavity.
    """

    half_shortest_box = 0.5 * float(system.box.min().detach().cpu())
    if spec.buffer_radius > half_shortest_box:
        raise ValueError("buffer_radius must not exceed half the shortest box length")
    displacement = minimum_image_from_center(system.positions, spec.center, system.box)
    distances = torch.linalg.vector_norm(displacement, dim=1)
    core = distances < spec.core_radius
    buffer = distances < spec.buffer_radius
    shell = buffer & ~core
    exterior = ~buffer

    def indices(mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask, as_tuple=False).flatten()

    return CavitySelection(
        spec=spec,
        parent_distances=distances.detach().clone(),
        core_mask=core.detach().clone(),
        shell_mask=shell.detach().clone(),
        buffer_mask=buffer.detach().clone(),
        exterior_mask=exterior.detach().clone(),
        core_indices=indices(core),
        shell_indices=indices(shell),
        buffer_indices=indices(buffer),
        exterior_indices=indices(exterior),
    )


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().cpu()


@dataclass(frozen=True)
class ParentState:
    """Detached full-system parent configuration used by every child chain."""

    parent_id: str
    positions: torch.Tensor
    velocities: torch.Tensor
    diameters: torch.Tensor
    box: torch.Tensor
    active_mask: torch.Tensor
    unwrapped_positions: torch.Tensor

    @classmethod
    def capture(cls, system: ParticleSystem, *, parent_id: str) -> "ParentState":
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("parent_id must be a non-empty string")
        return cls(
            parent_id=parent_id,
            positions=_cpu_clone(system.positions),
            velocities=_cpu_clone(system.velocities),
            diameters=_cpu_clone(system.diameters),
            box=_cpu_clone(system.box),
            active_mask=_cpu_clone(system.active_mask),
            unwrapped_positions=_cpu_clone(system.unwrapped_positions),
        )

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def to_system(
        self,
        *,
        active_mask: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> ParticleSystem:
        target_dtype = self.positions.dtype if dtype is None else dtype
        active = self.active_mask if active_mask is None else active_mask.detach().cpu()
        return ParticleSystem(
            positions=self.positions.to(device=device, dtype=target_dtype).clone(),
            velocities=self.velocities.to(device=device, dtype=target_dtype).clone(),
            diameters=self.diameters.to(device=device, dtype=target_dtype).clone(),
            box=self.box.to(device=device, dtype=target_dtype).clone(),
            active_mask=active.to(device=device, dtype=torch.bool).clone(),
            unwrapped_positions=self.unwrapped_positions.to(device=device, dtype=target_dtype).clone(),
        )


@dataclass(frozen=True)
class CandidateProvenance:
    """Audit record identifying exactly how a candidate was generated."""

    parent_id: str
    cavity_spec: CavitySpec
    chain_id: str
    sweep_index: int
    init_family: str
    seeds: Mapping[str, int]
    temperature: float
    exact_core_composition: bool
    tempering: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.parent_id or not self.chain_id or not self.init_family:
            raise ValueError("parent_id, chain_id, and init_family must be non-empty")
        if isinstance(self.sweep_index, bool) or int(self.sweep_index) < 0:
            raise ValueError("sweep_index must be a non-negative integer")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive and finite")
        clean_seeds = {str(name): int(seed) for name, seed in self.seeds.items()}
        if not clean_seeds or any(not name for name in clean_seeds):
            raise ValueError("at least one named seed is required")
        object.__setattr__(self, "sweep_index", int(self.sweep_index))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "seeds", clean_seeds)
        if self.tempering is not None:
            object.__setattr__(self, "tempering", dict(self.tempering))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "parent_id": self.parent_id,
            "cavity_spec": self.cavity_spec.to_dict(),
            "chain_id": self.chain_id,
            "sweep_index": self.sweep_index,
            "init_family": self.init_family,
            "seeds": dict(self.seeds),
            "temperature": self.temperature,
            "exact_core_composition": self.exact_core_composition,
        }
        if self.tempering is not None:
            result["tempering"] = dict(self.tempering)
        return result


@dataclass(frozen=True)
class CandidateState:
    """A detached full configuration plus frozen cavity membership and provenance."""

    positions: torch.Tensor
    velocities: torch.Tensor
    diameters: torch.Tensor
    box: torch.Tensor
    unwrapped_positions: torch.Tensor
    buffer_mask: torch.Tensor
    core_mask: torch.Tensor
    shell_mask: torch.Tensor
    buffer_indices: torch.Tensor
    core_indices: torch.Tensor
    shell_indices: torch.Tensor
    provenance: CandidateProvenance
    observables: Mapping[str, float]

    @classmethod
    def capture(
        cls,
        system: ParticleSystem,
        *,
        selection: CavitySelection,
        provenance: CandidateProvenance,
        observables: Mapping[str, float] | None = None,
    ) -> "CandidateState":
        if system.n_particles != int(selection.buffer_mask.numel()):
            raise ValueError("selection and system particle counts differ")
        if provenance.cavity_spec != selection.spec:
            raise ValueError("candidate provenance and selection cavity specs differ")
        return cls(
            positions=_cpu_clone(system.positions),
            velocities=_cpu_clone(system.velocities),
            diameters=_cpu_clone(system.diameters),
            box=_cpu_clone(system.box),
            unwrapped_positions=_cpu_clone(system.unwrapped_positions),
            buffer_mask=_cpu_clone(selection.buffer_mask),
            core_mask=_cpu_clone(selection.core_mask),
            shell_mask=_cpu_clone(selection.shell_mask),
            buffer_indices=_cpu_clone(selection.buffer_indices),
            core_indices=_cpu_clone(selection.core_indices),
            shell_indices=_cpu_clone(selection.shell_indices),
            provenance=provenance,
            observables={str(name): float(value) for name, value in (observables or {}).items()},
        )

    def to_system(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> ParticleSystem:
        target_dtype = self.positions.dtype if dtype is None else dtype
        return ParticleSystem(
            positions=self.positions.to(device=device, dtype=target_dtype).clone(),
            velocities=self.velocities.to(device=device, dtype=target_dtype).clone(),
            diameters=self.diameters.to(device=device, dtype=target_dtype).clone(),
            box=self.box.to(device=device, dtype=target_dtype).clone(),
            active_mask=self.buffer_mask.to(device=device, dtype=torch.bool).clone(),
            unwrapped_positions=self.unwrapped_positions.to(device=device, dtype=target_dtype).clone(),
        )
