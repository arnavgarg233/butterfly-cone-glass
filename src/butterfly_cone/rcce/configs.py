"""Read-only loader for inherited glass configurations.

The inherited archives store per-particle positions *and* diameters as
``pos_<T>_<r>`` / ``sig_<T>_<r>`` array pairs (see
``docs/PRIOR_LOCAL_MACHINERY.md`` sigma rule).  Diameters must always be loaded
from the file; regenerating them from a seed is forbidden because no inherited
file records its RNG state and the whole point of RCCE is to condition on the
*actual* frozen surroundings.

This module never writes to the source archive.  It only reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class InheritedConfig:
    """One (temperature, replica) snapshot from an inherited ``.npz`` archive."""

    path: str
    temperature: float
    replica: int
    box_length: float
    n_particles: int
    positions: torch.Tensor
    diameters: torch.Tensor

    @property
    def parent_id(self) -> str:
        stem = Path(self.path).stem
        return f"{stem}:T{self.temperature:.3f}:r{self.replica}"


def available_snapshots(path: str | Path) -> list[tuple[float, int]]:
    """Return the ``(temperature, replica)`` pairs present in the archive."""

    with np.load(Path(path)) as handle:
        pairs: set[tuple[float, int]] = set()
        for key in handle.files:
            if not key.startswith("pos_"):
                continue
            _, temperature, replica = key.split("_")
            pairs.add((float(temperature), int(replica)))
    return sorted(pairs)


def load_inherited_config(
    path: str | Path,
    temperature: float,
    replica: int = 0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    wrap: bool = True,
) -> InheritedConfig:
    """Load one snapshot's positions and diameters (never regenerated).

    ``pos_<T>_<r>`` keys are formatted with three decimals (e.g. ``pos_0.150_0``).
    Positions are optionally wrapped into ``[0, L)`` for cell-list cleanliness;
    wrapping is a periodic no-op for the physics because every force and distance
    uses the minimum-image convention.
    """

    source = Path(path)
    key_pos = f"pos_{temperature:.3f}_{replica}"
    key_sig = f"sig_{temperature:.3f}_{replica}"
    with np.load(source) as handle:
        if key_pos not in handle.files or key_sig not in handle.files:
            raise KeyError(
                f"{source} lacks {key_pos!r}/{key_sig!r}; "
                f"available: {available_snapshots(source)}"
            )
        positions_np = np.asarray(handle[key_pos], dtype=np.float64)
        diameters_np = np.asarray(handle[key_sig], dtype=np.float64)
        box_length = float(np.asarray(handle["L"]))
        n_particles = int(np.asarray(handle["N"]))

    if positions_np.shape != (n_particles, 3):
        raise ValueError(f"{key_pos} has shape {positions_np.shape}, expected {(n_particles, 3)}")
    if diameters_np.shape != (n_particles,):
        raise ValueError(f"{key_sig} has shape {diameters_np.shape}, expected {(n_particles,)}")

    positions = torch.from_numpy(positions_np).to(device=device, dtype=dtype)
    diameters = torch.from_numpy(diameters_np).to(device=device, dtype=dtype)
    if wrap:
        box = torch.full((3,), box_length, device=device, dtype=dtype)
        positions = torch.remainder(positions, box)
    return InheritedConfig(
        path=str(source),
        temperature=float(temperature),
        replica=int(replica),
        box_length=box_length,
        n_particles=n_particles,
        positions=positions,
        diameters=diameters,
    )


def to_particle_system(config: InheritedConfig) -> ParticleSystem:
    """Build a fully active, zero-velocity parent :class:`ParticleSystem`."""

    device = config.positions.device
    dtype = config.positions.dtype
    box = torch.full((3,), config.box_length, device=device, dtype=dtype)
    return ParticleSystem(
        positions=config.positions.detach().clone(),
        velocities=torch.zeros_like(config.positions),
        diameters=config.diameters.detach().clone(),
        box=box,
        active_mask=torch.ones(config.n_particles, device=device, dtype=torch.bool),
        unwrapped_positions=config.positions.detach().clone(),
    )
