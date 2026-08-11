"""Public inherited-snapshot loader used by bulk-pilot callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from butterfly_cone.engine.system import ParticleSystem

from .configs import StoredConfiguration, load_stored_configuration


@dataclass(frozen=True)
class InheritedSnapshot:
    """A saved configuration, its live engine state, and its exact archive keys."""

    system: ParticleSystem
    position_key: str
    sigma_key: str
    source: StoredConfiguration


def load_inherited_snapshot(
    path: Path | str,
    *,
    temperature: float,
    replica: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> InheritedSnapshot:
    """Load positions and their matching stored sigma array without regeneration."""

    source = load_stored_configuration(
        path, temperature=temperature, replica=replica, device=device, dtype=dtype
    )
    return InheritedSnapshot(
        system=source.make_system(),
        position_key=f"pos_{source.source_key}",
        sigma_key=f"sig_{source.source_key}",
        source=source,
    )
