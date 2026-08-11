"""Safe loading of the inherited, per-particle ButterflyCone configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class StoredConfiguration:
    """One saved position/diameter realization with positions safely wrapped."""

    source_path: Path
    temperature: float
    replica: int
    source_key: str
    positions: torch.Tensor
    diameters: torch.Tensor
    box: torch.Tensor

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def make_system(self) -> ParticleSystem:
        """Create a zero-velocity system without ever redrawing diameters."""
        return ParticleSystem(
            positions=self.positions.detach().clone(),
            velocities=torch.zeros_like(self.positions),
            diameters=self.diameters.detach().clone(),
            box=self.box.detach().clone(),
            active_mask=torch.ones(self.n_particles, device=self.positions.device, dtype=torch.bool),
            unwrapped_positions=self.positions.detach().clone(),
        )


def _source_key(temperature: float, replica: int) -> str:
    if replica < 0:
        raise ValueError("replica must be non-negative")
    return f"{float(temperature):.3f}_{replica}"


def load_stored_configuration(
    path: Path | str,
    *,
    temperature: float,
    replica: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StoredConfiguration:
    """Load matching ``pos_*`` and ``sig_*`` arrays and cast them together.

    The inherited archives store float32 arrays and no velocities.  This loader
    deliberately has no seed argument: a caller cannot accidentally regenerate
    a diameter realization instead of using the archived one.
    """

    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"configuration archive not found: {archive}")
    key = _source_key(temperature, replica)
    position_key = f"pos_{key}"
    sigma_key = f"sig_{key}"
    with np.load(archive, allow_pickle=False) as data:
        missing = [name for name in ("L", position_key, sigma_key) if name not in data]
        if missing:
            raise KeyError(f"{archive} is missing required arrays: {', '.join(missing)}")
        box_length = float(np.asarray(data["L"]).item())
        positions = np.asarray(data[position_key])
        diameters = np.asarray(data[sigma_key])
        declared_n = int(np.asarray(data["N"]).item()) if "N" in data else int(positions.shape[0])

    if not np.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("stored box length must be finite and positive")
    if positions.shape != (declared_n, 3):
        raise ValueError(f"{position_key} must have shape ({declared_n}, 3), got {positions.shape}")
    if diameters.shape != (declared_n,):
        raise ValueError(f"{sigma_key} must have shape ({declared_n},), got {diameters.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(diameters).all():
        raise ValueError("stored positions and diameters must be finite")
    if np.any(diameters <= 0.0):
        raise ValueError("stored diameters must be positive")

    target = torch.device(device)
    wrapped = np.remainder(positions, box_length)
    position_tensor = torch.as_tensor(wrapped, device=target, dtype=dtype).clone()
    diameter_tensor = torch.as_tensor(diameters, device=target, dtype=dtype).clone()
    box = torch.full((3,), box_length, device=target, dtype=dtype)
    return StoredConfiguration(
        source_path=archive,
        temperature=float(temperature),
        replica=int(replica),
        source_key=key,
        positions=position_tensor,
        diameters=diameter_tensor,
        box=box,
    )
