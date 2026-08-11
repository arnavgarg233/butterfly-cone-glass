"""Read canonical saved configurations and measured excess-energy ladders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class SavedConfiguration:
    """One labelled equilibrium configuration converted for CPU analysis."""

    temperature: float
    replica: int
    system: ParticleSystem


@dataclass(frozen=True)
class SavedConfigurationSet:
    """Canonical ``SAVE_T``/replica collection from a predecessor NPZ file."""

    n_particles: int
    box: np.ndarray
    records: tuple[SavedConfiguration, ...]

    @property
    def temperatures(self) -> tuple[float, ...]:
        return tuple(sorted({record.temperature for record in self.records}))

    def at_temperature(self, temperature: float) -> tuple[SavedConfiguration, ...]:
        matches = tuple(record for record in self.records if record.temperature == float(temperature))
        if not matches:
            raise KeyError(f"no saved configurations at T={temperature:g}")
        return matches


@dataclass(frozen=True)
class EnergyLadder:
    """Excess potential energy per particle, ordered by increasing beta."""

    temperatures: np.ndarray
    u_ex: np.ndarray
    beta: np.ndarray


def _system_from_arrays(positions: np.ndarray, diameters: np.ndarray, box: np.ndarray) -> ParticleSystem:
    position_tensor = torch.as_tensor(positions, dtype=torch.float64, device="cpu").clone()
    diameter_tensor = torch.as_tensor(diameters, dtype=torch.float64, device="cpu").clone()
    box_tensor = torch.as_tensor(box, dtype=torch.float64, device="cpu").clone()
    n_particles = int(position_tensor.shape[0])
    return ParticleSystem(
        positions=position_tensor,
        velocities=torch.zeros_like(position_tensor),
        diameters=diameter_tensor,
        box=box_tensor,
        active_mask=torch.ones(n_particles, dtype=torch.bool, device="cpu"),
        unwrapped_positions=position_tensor.clone(),
    )


def load_saved_configurations(path: str | Path) -> SavedConfigurationSet:
    """Load the canonical ``pos_T_rep``/``sig_T_rep`` NPZ layout."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"L", "N", "SAVE_T"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"saved-config archive is missing keys: {missing}")
        n_particles = int(data["N"])
        if n_particles <= 0:
            raise ValueError("N must be positive")
        raw_box = np.asarray(data["L"], dtype=np.float64)
        box = (
            np.full(3, float(raw_box), dtype=np.float64)
            if raw_box.ndim == 0
            else raw_box.reshape(-1)
        )
        if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0.0):
            raise ValueError("L must be a positive scalar or length-three box")
        temperatures = np.asarray(data["SAVE_T"], dtype=np.float64).reshape(-1)
        if temperatures.size == 0 or not np.all(np.isfinite(temperatures)):
            raise ValueError("SAVE_T must contain finite temperatures")
        replica_count = int(data["R"]) if "R" in data.files else 1
        if replica_count <= 0:
            raise ValueError("R must be positive")

        records: list[SavedConfiguration] = []
        for temperature in temperatures:
            label = f"{float(temperature):.3f}"
            before = len(records)
            for replica in range(replica_count):
                position_key = f"pos_{label}_{replica}"
                diameter_key = f"sig_{label}_{replica}"
                if position_key not in data.files and diameter_key not in data.files:
                    continue
                if position_key not in data.files or diameter_key not in data.files:
                    raise ValueError(f"incomplete saved configuration for T={label}, replica={replica}")
                positions = np.asarray(data[position_key])
                diameters = np.asarray(data[diameter_key])
                if positions.shape != (n_particles, 3) or diameters.shape != (n_particles,):
                    raise ValueError(f"invalid array shape for T={label}, replica={replica}")
                records.append(
                    SavedConfiguration(
                        temperature=float(temperature),
                        replica=replica,
                        system=_system_from_arrays(positions, diameters, box),
                    )
                )
            if len(records) == before:
                raise ValueError(f"archive declares T={label} but contains no replicas")
    return SavedConfigurationSet(
        n_particles=n_particles,
        box=box.copy(),
        records=tuple(records),
    )


def assemble_energy_ladder(
    temperatures: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    *,
    anchors: Mapping[float, float] | None = None,
) -> EnergyLadder:
    """Merge a warm ladder with cold anchors, with anchors replacing duplicates."""

    temperature_values = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if temperature_values.size == 0 or temperature_values.shape != energy_values.shape:
        raise ValueError("temperature and energy grids must be non-empty and have identical shape")
    if (
        not np.all(np.isfinite(temperature_values))
        or not np.all(np.isfinite(energy_values))
        or np.any(temperature_values <= 0.0)
    ):
        raise ValueError("temperatures must be positive and both grids finite")
    merged = {
        float(temperature): float(energy)
        for temperature, energy in zip(temperature_values, energy_values, strict=True)
    }
    if len(merged) != temperature_values.size:
        raise ValueError("warm ladder contains duplicate temperatures")
    for temperature, energy in (anchors or {}).items():
        if temperature <= 0.0 or not np.isfinite(temperature) or not np.isfinite(energy):
            raise ValueError("anchor temperatures must be positive and anchors finite")
        merged[float(temperature)] = float(energy)
    ordered_temperatures = np.asarray(sorted(merged, reverse=True), dtype=np.float64)
    ordered_energies = np.asarray([merged[float(value)] for value in ordered_temperatures], dtype=np.float64)
    return EnergyLadder(
        temperatures=ordered_temperatures,
        u_ex=ordered_energies,
        beta=1.0 / ordered_temperatures,
    )


def load_energy_ladder(path: str | Path) -> EnergyLadder:
    """Load ``T_grid`` and ``u_grid`` arrays from an NPZ entropy/ladder artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        missing = sorted({"T_grid", "u_grid"}.difference(data.files))
        if missing:
            raise ValueError(f"energy-ladder archive is missing keys: {missing}")
        temperatures = np.asarray(data["T_grid"], dtype=np.float64)
        energies = np.asarray(data["u_grid"], dtype=np.float64)
    return assemble_energy_ladder(temperatures, energies)
