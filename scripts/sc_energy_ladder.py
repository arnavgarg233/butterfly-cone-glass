#!/usr/bin/env python3
"""Generate the ``T_grid``/``u_grid`` excess-energy ladder that unblocks ``s_c(T)``.

``scripts/sc_pipeline.py`` is input-gated: it consumes a canonical
saved-configuration NPZ (``pos_<label>_<rep>``/``sig_<label>_<rep>`` plus
``L``/``N``/``SAVE_T``/``R``) and a measured excess-energy ladder NPZ
(``T_grid``/``u_grid``).  This generator produces both artifacts from the
cooled/canonical configurations (e.g. ``configs_N1500.npz``).

Excess-energy convention (matched to ``src/butterfly_cone/entropy/thermodynamic.py``)
---------------------------------------------------------------------------
The thermodynamic-integration leg computes the per-particle excess entropy as
``s_ex(T) = beta*u_ex(T) - integral_0^beta u_ex(beta') dbeta'`` and integrates
from ``beta = 0`` (the ideal-gas / infinite-temperature reference) using the
analytic soft-sphere head ``A*beta^-3/4 + B*beta^-1/2 + C*beta^-1/4``.

For a classical system the internal energy per particle is ``(3/2)T`` (kinetic,
the ideal-gas reference) plus ``<U_pot>/N``.  The *excess* internal energy over
the ideal gas is therefore exactly the potential energy per particle:

    u_ex(T) = <U_pot>(T) / N        (NO extra reference subtraction).

So ``u_grid`` is the equilibrium potential-energy-per-particle ladder.  It must
span from a high temperature (small ``beta``, where the analytic head is fitted)
down to the coldest analysed configuration temperature, because the TI leg
integrates ``beta`` from ``0`` up to ``1/T``.

Ladder construction
--------------------
* COLD anchors: for each saved temperature, ``u_ex = <U_pot>/N`` is evaluated
  directly on the saved equilibrium configurations with the engine potential and
  averaged over replicas.  These are exact measurements on the analysed configs.
* WARM ladder: high-temperature ``u_ex(T)`` points are *measured* by heating a
  saved configuration with the engine's NVT (velocity-Verlet + Bussi) dynamics
  and averaging ``<U_pot>/N`` at equilibrium.  A high-temperature liquid is
  ergodic under plain NVT dynamics, so diameter-swap moves (needed only for cold
  glassy equilibration, which the saved configs already have) are unnecessary
  here.  Every random draw uses an explicit CPU generator, so the ladder is
  bitwise-deterministic in ``float64``.

The engine and entropy modules are imported read-only for energy evaluation and
key/convention matching; nothing in ``src/butterfly_cone`` is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from butterfly_cone.engine.integrate import (
    BussiThermostat,
    MDIntegrator,
    maxwell_boltzmann_velocities,
)
from butterfly_cone.engine.potential import analytic_potential
from butterfly_cone.engine.system import ParticleSystem, make_generator
from butterfly_cone.entropy.io import load_saved_configurations

DEFAULT_CONFIG = Path("data/configs_N1500.npz")
DEFAULT_OUT_DIR = ROOT / "runs" / "sc_energy_ladder"
DEFAULT_SEED = 20240716

# ButterflyCone effective-mixing calibration is fitted to T in [0.0555, 0.125]; warmer
# saved temperatures require sc_pipeline's --allow-mixing-extrapolation flag.
MIXING_T_MIN = 0.0555
MIXING_T_MAX = 0.125


def config_label(temperature: float) -> str:
    """Return the ``pos_<label>_<rep>`` label exactly as ``entropy.io`` expects."""

    return f"{float(temperature):.3f}"


def potential_energy_per_particle(system: ParticleSystem) -> float:
    """Return ``<U_pot>/N`` for one configuration using the engine potential."""

    energy = analytic_potential(
        system.positions,
        system.diameters,
        system.box,
        active_mask=system.active_mask,
    ).energy
    return float(energy.detach().cpu()) / system.n_particles


def compute_cold_ladder(
    config_path: str | Path,
    *,
    replicas: int | None = None,
) -> list[dict[str, float | int]]:
    """Measure ``u_ex(T) = <U_pot>/N`` on the saved configs, averaged over replicas.

    Returns one record per saved temperature, ordered by descending temperature
    (ascending ``beta``), each with the replica mean/std and replica count.
    """

    collection = load_saved_configurations(config_path)
    records: list[dict[str, float | int]] = []
    for temperature in sorted(collection.temperatures, reverse=True):
        saved = collection.at_temperature(temperature)
        if replicas is not None:
            if replicas <= 0:
                raise ValueError("replicas must be positive")
            saved = saved[:replicas]
        energies = np.asarray(
            [potential_energy_per_particle(record.system) for record in saved],
            dtype=np.float64,
        )
        records.append(
            {
                "temperature": float(temperature),
                "u_ex": float(energies.mean()),
                "u_ex_std": float(energies.std(ddof=1)) if energies.size > 1 else 0.0,
                "n_replicas": int(energies.size),
            }
        )
    return records


def warm_timestep(temperature: float) -> float:
    """Stable velocity-Verlet step for the stiff ``r^-12`` core at temperature T.

    Velocities scale as ``sqrt(T)``, so the step shrinks with temperature to keep
    per-step displacement bounded.  The schedule matches the predecessor ladder
    (``dt=0.002`` when hot, ``0.005`` when warm).
    """

    if temperature > 8.0:
        return 0.002
    if temperature > 1.5:
        return 0.004
    return 0.006


def warm_temperature_grid(t_max: float, t_min: float, n_warm: int) -> np.ndarray:
    """Descending geometric warm ladder from ``t_max`` down to ``t_min``."""

    if not (t_max > t_min > 0.0):
        raise ValueError("require t_max > t_min > 0 for the warm ladder")
    if n_warm < 1:
        raise ValueError("n_warm must be positive")
    return np.geomspace(float(t_max), float(t_min), int(n_warm))


def measure_warm_point(
    base_system: ParticleSystem,
    temperature: float,
    *,
    seed: int,
    equil_steps: int,
    sample_blocks: int,
    block_steps: int,
    skin: float,
) -> float:
    """Heat ``base_system`` to ``temperature`` and average ``<U_pot>/N`` at equilibrium.

    Plain NVT (velocity-Verlet + Bussi) dynamics with an explicit CPU generator;
    bitwise-deterministic in float64.
    """

    if equil_steps < 0 or sample_blocks < 1 or block_steps < 1:
        raise ValueError("invalid warm-sampling controls")
    system = base_system.clone()
    dt = warm_timestep(temperature)
    generator = make_generator(seed)
    system.velocities = maxwell_boltzmann_velocities(
        system.n_particles,
        temperature,
        generator,
        device="cpu",
        dtype=torch.float64,
    )
    thermostat = BussiThermostat(temperature, tau=max(30.0 * dt, 0.5), generator=generator)
    integrator = MDIntegrator(system, dt=dt, skin=skin, thermostat=thermostat)
    if equil_steps:
        integrator.step(equil_steps)
    samples = np.empty(sample_blocks, dtype=np.float64)
    for block in range(sample_blocks):
        integrator.step(block_steps)
        samples[block] = float(integrator.potential_energy.detach().cpu()) / system.n_particles
    return float(samples.mean())


def generate_warm_ladder(
    base_system: ParticleSystem,
    temperatures: Sequence[float] | np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    equil_steps: int = 400,
    sample_blocks: int = 60,
    block_steps: int = 5,
    skin: float = 0.4,
) -> list[dict[str, float]]:
    """Measure the warm ``u_ex(T)`` ladder by heating ``base_system`` at each T."""

    records: list[dict[str, float]] = []
    for index, temperature in enumerate(temperatures):
        u_ex = measure_warm_point(
            base_system,
            float(temperature),
            seed=seed + index,
            equil_steps=equil_steps,
            sample_blocks=sample_blocks,
            block_steps=block_steps,
            skin=skin,
        )
        records.append({"temperature": float(temperature), "u_ex": float(u_ex)})
    return records


def assemble_ladder(
    cold: Sequence[dict[str, float | int]],
    warm: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge warm + cold records into ``(T_grid, u_grid)`` sorted by descending T.

    The order (descending T == ascending beta) matches how the predecessor and
    ``entropy.io.assemble_energy_ladder`` present the ladder; the loader re-sorts
    regardless, so this is purely for a readable artifact.
    """

    temperatures = np.asarray(
        [float(record["temperature"]) for record in list(warm) + list(cold)],
        dtype=np.float64,
    )
    energies = np.asarray(
        [float(record["u_ex"]) for record in list(warm) + list(cold)],
        dtype=np.float64,
    )
    if temperatures.size == 0:
        raise ValueError("ladder is empty")
    if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(energies)):
        raise ValueError("ladder contains non-finite values")
    if np.any(temperatures <= 0.0):
        raise ValueError("ladder temperatures must be positive")
    if np.unique(np.round(temperatures, 9)).size != temperatures.size:
        raise ValueError("ladder contains duplicate temperatures")
    order = np.argsort(-temperatures, kind="stable")
    return temperatures[order], energies[order]


def repackage_config(config_path: str | Path) -> dict[str, np.ndarray]:
    """Re-emit the saved configs under the exact array names sc_pipeline consumes.

    Produces ``L``/``N``/``R``/``SAVE_T`` plus ``pos_<label>_<rep>``/
    ``sig_<label>_<rep>`` as CPU ``float64`` (positions/diameters) so the artifact
    loads through ``entropy.io.load_saved_configurations`` without ambiguity.
    """

    source = Path(config_path)
    with np.load(source, allow_pickle=False) as data:
        required = {"L", "N", "SAVE_T"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"source config is missing keys: {missing}")
        n_particles = int(data["N"])
        raw_box = np.asarray(data["L"], dtype=np.float64)
        box_scalar = float(raw_box) if raw_box.ndim == 0 else raw_box
        replica_count = int(data["R"]) if "R" in data.files else 1
        temperatures = np.asarray(data["SAVE_T"], dtype=np.float64).reshape(-1)
        packaged: dict[str, np.ndarray] = {
            "L": np.asarray(box_scalar, dtype=np.float64),
            "N": np.asarray(n_particles, dtype=np.int64),
            "R": np.asarray(replica_count, dtype=np.int64),
            "SAVE_T": temperatures,
        }
        for temperature in temperatures:
            label = config_label(temperature)
            found = False
            for replica in range(replica_count):
                position_key = f"pos_{label}_{replica}"
                diameter_key = f"sig_{label}_{replica}"
                if position_key not in data.files and diameter_key not in data.files:
                    continue
                if position_key not in data.files or diameter_key not in data.files:
                    raise ValueError(f"incomplete saved configuration for {label}/{replica}")
                positions = np.asarray(data[position_key], dtype=np.float64)
                diameters = np.asarray(data[diameter_key], dtype=np.float64)
                if positions.shape != (n_particles, 3) or diameters.shape != (n_particles,):
                    raise ValueError(f"invalid array shape for {label}/{replica}")
                packaged[position_key] = positions
                packaged[diameter_key] = diameters
                found = True
            if not found:
                raise ValueError(f"source declares T={label} but has no replicas")
    return packaged


def write_energy_grid(path: str | Path, t_grid: np.ndarray, u_grid: np.ndarray) -> Path:
    """Write the ``T_grid``/``u_grid`` NPZ that ``--energy-grid`` consumes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        T_grid=np.asarray(t_grid, dtype=np.float64),
        u_grid=np.asarray(u_grid, dtype=np.float64),
    )
    return destination


def write_config(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write the repackaged canonical-configuration NPZ that ``--config`` consumes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return destination


def sc_pipeline_command(
    config_path: Path,
    energy_path: Path,
    *,
    replicas: int,
    temperatures: Sequence[float],
    out_path: Path,
) -> dict[str, object]:
    """Build the exact sc_pipeline launch command(s) for the written artifacts."""

    python = sys.executable
    script = ROOT / "scripts" / "sc_pipeline.py"
    ordered = sorted(temperatures, reverse=True)
    needs_extrapolation = any(t < MIXING_T_MIN or t > MIXING_T_MAX for t in ordered)
    in_range = [t for t in ordered if MIXING_T_MIN <= t <= MIXING_T_MAX]

    full = [
        f"PYTHONPATH=src {python}",
        str(script),
        f"--config {config_path}",
        f"--energy-grid {energy_path}",
        f"--replicas {replicas}",
    ]
    if needs_extrapolation:
        full.append("--allow-mixing-extrapolation")
    full.append(f"--out {out_path}")
    full_command = " \\\n    ".join(full)

    conservative = None
    if in_range and len(in_range) != len(ordered):
        conservative_parts = [
            f"PYTHONPATH=src {python}",
            str(script),
            f"--config {config_path}",
            f"--energy-grid {energy_path}",
            "--temperatures " + " ".join(f"{t:g}" for t in in_range),
            f"--replicas {replicas}",
            f"--out {out_path}",
        ]
        conservative = " \\\n    ".join(conservative_parts)

    return {
        "full_curve_command": full_command,
        "full_curve_temperatures": [float(t) for t in ordered],
        "requires_allow_mixing_extrapolation": bool(needs_extrapolation),
        "in_calibration_temperatures": [float(t) for t in in_range],
        "conservative_in_range_command": conservative,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    """Generate the energy grid + repackaged config and describe the sc_pipeline run."""

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    energy_out = Path(args.out_energy) if args.out_energy else out_dir / "energy_grid.npz"
    config_out = Path(args.out_config) if args.out_config else out_dir / "config_ladder.npz"
    sc_out = out_dir / "sc_curve.json"

    cold = compute_cold_ladder(config_path, replicas=args.replicas)
    if not cold:
        raise ValueError("no saved configurations found")

    warm: list[dict[str, float]] = []
    if args.warm:
        collection = load_saved_configurations(config_path)
        warmest = max(collection.temperatures)
        base_system = collection.at_temperature(warmest)[0].system
        cold_max = max(record["temperature"] for record in cold)
        warm_t_min = max(args.warm_t_min, cold_max * 1.02)
        warm_grid = warm_temperature_grid(args.t_max, warm_t_min, args.n_warm)
        warm = generate_warm_ladder(
            base_system,
            warm_grid,
            seed=args.seed,
            equil_steps=args.warm_equil_steps,
            sample_blocks=args.warm_sample_blocks,
            block_steps=args.warm_block_steps,
            skin=args.warm_skin,
        )

    t_grid, u_grid = assemble_ladder(cold, warm)
    packaged = repackage_config(config_path)

    energy_path = write_energy_grid(energy_out, t_grid, u_grid)
    config_path_out = write_config(config_out, packaged)

    available = sorted(record["temperature"] for record in cold)
    command = sc_pipeline_command(
        config_path_out,
        energy_path,
        replicas=int(packaged["R"]),
        temperatures=available,
        out_path=sc_out,
    )

    report: dict[str, object] = {
        "convention": (
            "u_grid[i] = <U_pot>(T_grid[i]) / N (potential energy per particle); "
            "this is the excess internal energy over the ideal gas, integrated by "
            "sc_pipeline's TI leg from beta=0. No extra reference subtraction."
        ),
        "source_config": str(config_path),
        "energy_grid_path": str(energy_path),
        "config_path": str(config_path_out),
        "n_ladder_points": int(t_grid.size),
        "warm_enabled": bool(args.warm),
        "cold_ladder": cold,
        "warm_ladder": warm,
        "T_grid": [float(value) for value in t_grid],
        "u_grid": [float(value) for value in u_grid],
        "sc_pipeline": command,
        "seed": int(args.seed),
    }
    if not args.warm and t_grid.size < 8:
        report["warning"] = (
            "warm ladder disabled: only cold anchors written, so the sc_pipeline TI "
            f"head needs --n-head <= {t_grid.size}."
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="canonical saved-config NPZ")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--out-energy", type=Path, help="override energy-grid NPZ path")
    parser.add_argument("--out-config", type=Path, help="override repackaged-config NPZ path")
    parser.add_argument("--replicas", type=int, default=None, help="max replicas averaged per cold T")
    parser.add_argument("--no-warm", dest="warm", action="store_false", help="skip warm-ladder MD")
    parser.set_defaults(warm=True)
    parser.add_argument("--t-max", type=float, default=30.0, help="hottest warm-ladder temperature")
    parser.add_argument("--warm-t-min", type=float, default=0.16, help="coldest warm-ladder temperature")
    parser.add_argument("--n-warm", type=int, default=16, help="number of warm-ladder temperatures")
    parser.add_argument("--warm-equil-steps", type=int, default=400)
    parser.add_argument("--warm-sample-blocks", type=int, default=60)
    parser.add_argument("--warm-block-steps", type=int, default=5)
    parser.add_argument("--warm-skin", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path, help="optional path to dump the JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    report = build(args)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print("\n# u_ex(T) ladder (T -> u_ex = <U_pot>/N):", file=sys.stderr)
    for temperature, energy in zip(report["T_grid"], report["u_grid"]):  # type: ignore[arg-type]
        print(f"#   T={temperature:9.4f}  u_ex={energy:.6f}", file=sys.stderr)
    print("\n# Launch sc_pipeline with:", file=sys.stderr)
    print(report["sc_pipeline"]["full_curve_command"], file=sys.stderr)  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
