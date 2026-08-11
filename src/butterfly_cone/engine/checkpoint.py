"""Deep-copy, full-state checkpoints with bitwise restart semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .integrate import MDIntegrator
from .system import ParticleSystem


def _deep_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _deep_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_clone(item) for item in value)
    return value


@dataclass(frozen=True)
class EngineCheckpoint:
    payload: dict[str, Any]


def capture_checkpoint(
    system: ParticleSystem,
    integrator: MDIntegrator,
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> EngineCheckpoint:
    """Capture detached clones; no tensor aliases live simulation state."""

    if integrator.system is not system:
        raise ValueError("integrator does not own the supplied system")
    generator_states: dict[str, dict[str, Any]] = {}
    for name, generator in (generators or {}).items():
        generator_states[str(name)] = {
            "device": str(generator.device),
            "state": generator.get_state().detach().clone(),
        }
    payload = {
        "format_version": 1,
        "system": system.state_dict(),
        "integrator": integrator.state_dict(),
        "generators": generator_states,
    }
    return EngineCheckpoint(_deep_clone(payload))


def restore_checkpoint(
    checkpoint: EngineCheckpoint,
    *,
    device: torch.device | str | None = None,
) -> tuple[ParticleSystem, MDIntegrator, dict[str, torch.Generator]]:
    payload = _deep_clone(checkpoint.payload)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    system = ParticleSystem.from_state_dict(payload["system"], device=device)
    integrator = MDIntegrator.from_state_dict(system, payload["integrator"], device=system.device)
    generators: dict[str, torch.Generator] = {}
    for name, generator_state in payload["generators"].items():
        generator_device = generator_state["device"]
        generator = torch.Generator(device=generator_device)
        generator.set_state(generator_state["state"].detach().clone().cpu())
        generators[name] = generator
    return system, integrator, generators


def save_checkpoint(path: str | Path, checkpoint: EngineCheckpoint) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_deep_clone(checkpoint.payload), destination)


def load_checkpoint(path: str | Path) -> EngineCheckpoint:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return EngineCheckpoint(_deep_clone(payload))
