"""Seeded conditional-cavity MD/swap sampling.

The full parent system is retained in every chain.  Parent-time buffer labels
are active; every other label is frozen but remains in the force calculation.
Each MD segment begins with a full Maxwell--Boltzmann redraw and uses Bussi
stochastic velocity rescaling on all ``3 * N_active`` momentum degrees of
freedom.  Unlike an isolated bulk system, an active cavity exchanges momentum
with its frozen surroundings, so its active center-of-mass momentum is not a
conserved degree of freedom and is intentionally not removed.

With ``exact_core_composition=True``, swaps are restricted separately to the
parent-time core and shell label sets and any MD/relaxation segment whose final
centers change side of the core boundary is rejected and restored.  This is a
deliberately more restricted, reflecting-boundary-like ensemble for the
composition robustness analysis; it is not the primary RCCE ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol, Sequence

import torch

from butterfly_cone.engine.integrate import MDIntegrator, maxwell_boltzmann_velocities
from butterfly_cone.engine.neighbors import VerletList
from butterfly_cone.engine.potential import PotentialResult
from butterfly_cone.engine.swap import (
    _SwapSweepWorkspace,
    _prepare_swap_sweep,
    SwapStatistics,
    diameter_swap_sweep,
)
from butterfly_cone.engine.system import ParticleSystem, make_generator, relax_overlaps

from .cavity import (
    CandidateProvenance,
    CandidateState,
    CavitySelection,
    ParentState,
    minimum_image_from_center,
)


class ChainInitError(RuntimeError):
    """Chain initialization could not be relaxed below the init-force gate.

    Raised before any production sampling; callers may redraw the chain with
    fresh seeds (a bounded retry replaces one random init with another and
    does not condition on sampled values, so it cannot bias the ensemble).
    """


class ChainNumericalError(RuntimeError):
    """A production sweep produced a non-finite active trajectory.

    The structured context lets a parallel-tempering owner quarantine exactly
    the failed rung without treating an initialization or burn-in failure as a
    production containment event.
    """

    def __init__(self, *, chain_id: str, rung: int, sweep_index: int) -> None:
        self.chain_id = str(chain_id)
        self.rung = int(rung)
        self.sweep_index = int(sweep_index)
        self.reason = "non-finite active trajectory"
        super().__init__(
            f"{self.reason}: chain {self.chain_id!r}, rung {self.rung}, "
            f"production sweep {self.sweep_index}"
        )


class InitFamily(str, Enum):
    """Intentionally overdispersed starts used to expose poor mixing.

    ``PARENT`` retains the observed local state; ``HIGH_T_ANNEAL`` heats only
    buffer labels and then quenches them; ``DISPLACED`` applies large seeded
    buffer displacements before overlap relaxation; and
    ``DIAMETER_RESHUFFLE`` permutes buffer diameters before overlap relaxation.
    """

    PARENT = "parent"
    HIGH_T_ANNEAL = "high_t_anneal"
    DISPLACED = "displaced"
    DIAMETER_RESHUFFLE = "diameter_reshuffle"


class SeedSource(Protocol):
    def seed_for(self, domain: str, index: int) -> int: ...


@dataclass(frozen=True)
class RCCESeeds:
    """Domain-separated random streams for one chain."""

    initialization: int
    momentum: int
    thermostat: int
    swap: int
    tempering: int

    @classmethod
    def allocate(
        cls,
        source: SeedSource,
        *,
        chain_index: int,
        domain_prefix: str = "rcce",
    ) -> "RCCESeeds":
        if isinstance(chain_index, bool) or chain_index < 0:
            raise ValueError("chain_index must be a non-negative integer")
        if not domain_prefix:
            raise ValueError("domain_prefix must be non-empty")
        values = {
            name: source.seed_for(f"{domain_prefix}.{name}", chain_index)
            for name in ("initialization", "momentum", "thermostat", "swap", "tempering")
        }
        return cls(**values)

    def as_dict(self) -> dict[str, int]:
        return {
            "initialization": int(self.initialization),
            "momentum": int(self.momentum),
            "thermostat": int(self.thermostat),
            "swap": int(self.swap),
            "tempering": int(self.tempering),
        }


def _generator(seed: int) -> torch.Generator:
    # Harness seeds are full SHA-256 integers, while torch accepts a 64-bit
    # manual seed.  Modular projection retains deterministic domain separation.
    return make_generator(int(seed) % (2**63 - 1))


@dataclass(frozen=True)
class RCCEConfig:
    temperature: float
    dt: float = 0.001
    thermostat_tau: float = 0.02
    md_steps_per_sweep: int = 10
    swap_attempts_per_sweep: int | None = None
    skin: float = 0.3
    # Relaxation must be able to undo displacement_scale-sized kicks: the
    # engine's own acceptance protocol uses 1000 x 0.002 (total 2 sigma of
    # corrective motion); the earlier 100 x 0.001 default left rare deep
    # overlaps that overflow float32 in the x^-12 core during production MD
    # (observed as non-finite trajectories at ~1e-4 per chain-sweep rate).
    relaxation_steps: int = 1000
    relaxation_max_displacement: float = 0.002
    # Init-quality gate: production sweeps may not start while the maximum
    # active force exceeds this. Gating happens BEFORE any sampling, so it
    # cannot bias the conditional ensemble. Calibration: legitimately prepared
    # exact-core inits reach ~3e3 (membership-constrained relaxation stops
    # early); a dangerous unrelaxed deep overlap (the float32 NaN seed) sits at
    # ~1e6-1e8. 1e5 separates the two populations with wide margins both ways.
    max_init_force: float = 1.0e5
    displacement_scale: float = 0.30
    anneal_temperature_factor: float = 3.0
    anneal_sweeps: int = 5
    quench_sweeps: int = 5
    exact_core_composition: bool = False

    def __post_init__(self) -> None:
        positive = {
            "temperature": self.temperature,
            "dt": self.dt,
            "thermostat_tau": self.thermostat_tau,
            "skin": self.skin,
            "relaxation_max_displacement": self.relaxation_max_displacement,
            "displacement_scale": self.displacement_scale,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError("temperatures, time controls, skin, and displacement controls must be positive")
        if self.md_steps_per_sweep <= 0:
            raise ValueError("md_steps_per_sweep must be positive")
        if self.swap_attempts_per_sweep is not None and self.swap_attempts_per_sweep < 0:
            raise ValueError("swap_attempts_per_sweep must be non-negative or None")
        if self.relaxation_steps <= 0:
            raise ValueError("relaxation_steps must be positive for displaced initializations")
        if self.anneal_temperature_factor <= 1.0:
            raise ValueError("anneal_temperature_factor must exceed one")
        if self.anneal_sweeps <= 0 or self.quench_sweeps <= 0:
            raise ValueError("anneal_sweeps and quench_sweeps must be positive")


@dataclass
class SamplerCost:
    """Executed work, including rejected moves and initialization relaxation."""

    md_steps: int = 0
    md_segments: int = 0
    md_segments_rejected: int = 0
    momentum_refreshes: int = 0
    swap_sweeps: int = 0
    swap_attempts: int = 0
    swap_accepted: int = 0
    relaxation_steps: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "md_steps": self.md_steps,
            "md_segments": self.md_segments,
            "md_segments_rejected": self.md_segments_rejected,
            "momentum_refreshes": self.momentum_refreshes,
            "swap_sweeps": self.swap_sweeps,
            "swap_attempts": self.swap_attempts,
            "swap_accepted": self.swap_accepted,
            "relaxation_steps": self.relaxation_steps,
        }


def conditional_potential(
    system: ParticleSystem,
    neighbor_list: VerletList | None = None,
) -> PotentialResult:
    """Energy/forces varying with active coordinates, including their exterior interactions."""

    neighbors = VerletList.from_system(system) if neighbor_list is None else neighbor_list
    return neighbors.evaluate(system)


def _normal_draw(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float64)


class _ConditionalBussiThermostat:
    """Bussi rescaling with all active cavity momentum DOF retained."""

    def __init__(self, temperature: float, tau: float, generator: torch.Generator) -> None:
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.generator = generator
        self.last_alpha = 1.0
        self.heat = 0.0

    def apply(self, system: ParticleSystem, dt: float) -> float:
        active = system.active_mask
        ndof = 3 * int(active.sum().item())
        if ndof == 0:
            return 1.0
        kinetic_before = 0.5 * system.velocities[active].square().sum()
        if float(kinetic_before) <= 0.0:
            raise ValueError("Bussi rescaling requires nonzero kinetic energy")
        randoms = _normal_draw((ndof,), self.generator).to(system.device, system.dtype)
        gaussian = randoms[0]
        chi_square = randoms[1:].square().sum()
        c = math.exp(-float(dt) / self.tau)
        target_kinetic = 0.5 * ndof * self.temperature
        ratio = torch.as_tensor(target_kinetic, device=system.device, dtype=system.dtype) / kinetic_before
        alpha_squared = (
            c
            + (1.0 - c) * ratio * (chi_square + gaussian.square()) / ndof
            + 2.0 * gaussian * torch.sqrt(c * (1.0 - c) * ratio / ndof)
        )
        alpha = torch.sqrt(torch.clamp(alpha_squared, min=0.0))
        sign_threshold = gaussian + torch.sqrt(
            torch.as_tensor(c / (1.0 - c) * ndof, device=system.device, dtype=system.dtype)
            / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        system.velocities = torch.where(active[:, None], system.velocities * alpha, system.velocities)
        kinetic_after = 0.5 * system.velocities[active].square().sum()
        self.last_alpha = float(alpha)
        self.heat += float(kinetic_after - kinetic_before)
        return self.last_alpha


class RCCEChain:
    """One reproducible Markov chain for a frozen cavity selection."""

    def __init__(
        self,
        parent: ParentState,
        selection: CavitySelection,
        *,
        chain_id: str,
        init_family: InitFamily | str,
        config: RCCEConfig,
        seeds: RCCESeeds,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not chain_id:
            raise ValueError("chain_id must be non-empty")
        if parent.n_particles != int(selection.buffer_mask.numel()):
            raise ValueError("parent and cavity selection particle counts differ")
        if selection.n_buffer == 0:
            raise ValueError("cavity buffer contains no particles")
        self.parent = parent
        self.selection = selection
        self.chain_id = str(chain_id)
        self.init_family = InitFamily(init_family)
        self.config = config
        self.seeds = seeds
        self.system = parent.to_system(
            active_mask=selection.buffer_mask,
            device=device,
            dtype=dtype,
        )
        self.system.velocities = torch.zeros_like(self.system.velocities)
        self.cost = SamplerCost()
        self.sweep_count = 0
        self._init_generator = _generator(seeds.initialization)
        self._momentum_generator = _generator(seeds.momentum)
        self._thermostat_generator = _generator(seeds.thermostat)
        self._swap_generator = _generator(seeds.swap)
        self._neighbors = VerletList.from_system(self.system, skin=config.skin)
        self._last_result = conditional_potential(self.system, self._neighbors)

        exterior = selection.exterior_indices.to(self.system.device)
        buffer = selection.buffer_indices.to(self.system.device)
        core = selection.core_indices.to(self.system.device)
        shell = selection.shell_indices.to(self.system.device)
        self._exterior_positions = self.system.positions[exterior].clone()
        self._exterior_diameters = self.system.diameters[exterior].clone()
        self._buffer_multiset = torch.sort(self.system.diameters[buffer]).values.clone()
        self._core_multiset = torch.sort(self.system.diameters[core]).values.clone()
        self._shell_multiset = torch.sort(self.system.diameters[shell]).values.clone()

        self._initialize_family()
        self._neighbors = VerletList.from_system(self.system, skin=config.skin)
        self._last_result = conditional_potential(self.system, self._neighbors)
        self._assert_invariants()

    def _initialize_family(self) -> None:
        if self.init_family is InitFamily.PARENT:
            return
        if self.init_family is InitFamily.HIGH_T_ANNEAL:
            high_temperature = self.config.temperature * self.config.anneal_temperature_factor
            for _ in range(self.config.anneal_sweeps):
                self._perform_sweep(high_temperature)
            for quench_index in range(self.config.quench_sweeps):
                fraction = (quench_index + 1) / self.config.quench_sweeps
                temperature = high_temperature + fraction * (self.config.temperature - high_temperature)
                self._perform_sweep(temperature)
            return
        if self.init_family is InitFamily.DISPLACED:
            self._apply_random_displacements()
            self._relax_initialization()
            return
        if self.init_family is InitFamily.DIAMETER_RESHUFFLE:
            self._reshuffle_diameters()
            self._relax_initialization()
            return
        raise AssertionError(self.init_family)  # pragma: no cover

    def _apply_random_displacements(self) -> None:
        indices = self.selection.buffer_indices.to(self.system.device)
        draws = torch.rand(
            (indices.numel(), 3),
            generator=self._init_generator,
            device="cpu",
            dtype=torch.float64,
        )
        displacement = (2.0 * draws - 1.0).to(self.system.device, self.system.dtype)
        displacement = displacement * self.config.displacement_scale
        if self.config.exact_core_composition:
            original_core = self.selection.core_mask.to(self.system.device)[indices]
            for _ in range(16):
                proposed = torch.remainder(self.system.positions[indices] + displacement, self.system.box)
                distances = torch.linalg.vector_norm(
                    minimum_image_from_center(proposed, self.selection.spec.center, self.system.box),
                    dim=1,
                )
                invalid = (distances < self.selection.spec.core_radius) != original_core
                if not bool(torch.any(invalid)):
                    break
                displacement[invalid] *= 0.5
        self.system.positions[indices] = torch.remainder(
            self.system.positions[indices] + displacement,
            self.system.box,
        )
        self.system.unwrapped_positions[indices] += displacement

    def _reshuffle_group(self, indices: torch.Tensor) -> None:
        if indices.numel() < 2:
            return
        permutation = torch.randperm(int(indices.numel()), generator=self._init_generator, device="cpu")
        device_indices = indices.to(self.system.device)
        source = self.system.diameters[device_indices].clone()
        self.system.diameters[device_indices] = source[permutation.to(self.system.device)]

    def _reshuffle_diameters(self) -> None:
        if self.config.exact_core_composition:
            self._reshuffle_group(self.selection.core_indices)
            self._reshuffle_group(self.selection.shell_indices)
        else:
            self._reshuffle_group(self.selection.buffer_indices)

    def _relax_initialization(self) -> None:
        final_force = self._relaxation_pass()
        if final_force > self.config.max_init_force:
            # One bounded extension for a rare deep-overlap draw.
            final_force = self._relaxation_pass()
        if final_force > self.config.max_init_force:
            raise ChainInitError(
                f"chain {self.chain_id!r} ({self.init_family.value}): max active force "
                f"{final_force:.3e} exceeds init gate {self.config.max_init_force:.3e} "
                f"after extended relaxation"
            )
        self.system.velocities = torch.zeros_like(self.system.velocities)

    def _relaxation_pass(self) -> float:
        if not self.config.exact_core_composition:
            report = relax_overlaps(
                self.system,
                steps=self.config.relaxation_steps,
                max_displacement=self.config.relaxation_max_displacement,
                skin=self.config.skin,
            )
            self.cost.relaxation_steps += report.steps_completed
        else:
            # One-step transactions avoid discarding all useful relaxation if
            # only the final attempted minimization step reaches the boundary.
            for _ in range(self.config.relaxation_steps):
                positions = self.system.positions.clone()
                unwrapped = self.system.unwrapped_positions.clone()
                report = relax_overlaps(
                    self.system,
                    steps=1,
                    max_displacement=self.config.relaxation_max_displacement,
                    skin=self.config.skin,
                )
                self.cost.relaxation_steps += report.steps_completed
                if not self._has_exact_membership():
                    self.system.positions = positions
                    self.system.unwrapped_positions = unwrapped
                    break
                if report.steps_completed == 0:
                    break
        return float(report.final_max_force)

    def _has_exact_membership(self) -> bool:
        indices = self.selection.buffer_indices.to(self.system.device)
        distances = torch.linalg.vector_norm(
            minimum_image_from_center(
                self.system.positions[indices],
                self.selection.spec.center,
                self.system.box,
            ),
            dim=1,
        )
        current = distances < self.selection.spec.core_radius
        expected = self.selection.core_mask.to(self.system.device)[indices]
        return torch.equal(current, expected)

    def _md_segment(self, temperature: float) -> None:
        positions = self.system.positions.clone()
        unwrapped = self.system.unwrapped_positions.clone()
        velocities = self.system.velocities.clone()
        self.system.velocities = maxwell_boltzmann_velocities(
            self.system.n_particles,
            temperature,
            self._momentum_generator,
            device=self.system.device,
            dtype=self.system.dtype,
            active_mask=self.system.active_mask,
            remove_com=False,
        )
        self.cost.momentum_refreshes += 1
        thermostat = _ConditionalBussiThermostat(
            temperature,
            self.config.thermostat_tau,
            self._thermostat_generator,
        )
        integrator = MDIntegrator(
            self.system,
            dt=self.config.dt,
            neighbor_list=self._neighbors,
            thermostat=thermostat,  # type: ignore[arg-type]
        )
        integrator.step(self.config.md_steps_per_sweep)
        self.cost.md_segments += 1
        self.cost.md_steps += self.config.md_steps_per_sweep
        self._neighbors = integrator.neighbor_list
        self._last_result = conditional_potential(self.system, self._neighbors)
        if self.config.exact_core_composition and not self._has_exact_membership():
            self.system.positions = positions
            self.system.unwrapped_positions = unwrapped
            self.system.velocities = velocities
            self._neighbors = VerletList.from_system(self.system, skin=self.config.skin)
            self._last_result = conditional_potential(self.system, self._neighbors)
            self.cost.md_segments_rejected += 1

    def _swap_group(
        self,
        mask: torch.Tensor,
        attempts: int,
        temperature: float,
        workspace: _SwapSweepWorkspace | None = None,
    ) -> SwapStatistics:
        if attempts <= 0:
            return SwapStatistics(0, 0)
        original = self.system.active_mask
        try:
            self.system.active_mask = mask.to(self.system.device, dtype=torch.bool)
            return diameter_swap_sweep(
                self.system,
                temperature,
                self._swap_generator,
                self._neighbors,
                n_attempts=attempts,
                workspace=workspace,
            )
        finally:
            self.system.active_mask = original

    def _swap_sweep(self, temperature: float) -> None:
        requested = self.config.swap_attempts_per_sweep
        attempts = self.selection.n_buffer if requested is None else requested
        if not self.config.exact_core_composition:
            workspace = (
                _prepare_swap_sweep(self.system, self._neighbors)
                if attempts > 0 and self.selection.n_buffer >= 2
                else None
            )
            statistics = diameter_swap_sweep(
                self.system,
                temperature,
                self._swap_generator,
                self._neighbors,
                n_attempts=attempts,
                workspace=workspace,
            )
        else:
            core_attempts = int(round(attempts * self.selection.n_core / self.selection.n_buffer))
            shell_attempts = attempts - core_attempts
            workspace = (
                _prepare_swap_sweep(self.system, self._neighbors)
                if (core_attempts > 0 and self.selection.n_core >= 2)
                or (shell_attempts > 0 and self.selection.n_buffer - self.selection.n_core >= 2)
                else None
            )
            core_statistics = self._swap_group(
                self.selection.core_mask,
                core_attempts,
                temperature,
                workspace,
            )
            shell_statistics = self._swap_group(
                self.selection.shell_mask,
                shell_attempts,
                temperature,
                workspace,
            )
            statistics = SwapStatistics(
                core_statistics.attempts + shell_statistics.attempts,
                core_statistics.accepted + shell_statistics.accepted,
            )
        self.cost.swap_sweeps += 1
        self.cost.swap_attempts += statistics.attempts
        self.cost.swap_accepted += statistics.accepted
        self._last_result = conditional_potential(self.system, self._neighbors)

    def _perform_sweep(
        self,
        temperature: float,
        *,
        production_rung: int | None = None,
        production_sweep_index: int | None = None,
    ) -> None:
        if (production_rung is None) != (production_sweep_index is None):
            raise ValueError("production rung and sweep index must be supplied together")
        self._md_segment(float(temperature))
        self._swap_sweep(float(temperature))
        self.sweep_count += 1
        self._assert_invariants(
            production_rung=production_rung,
            production_sweep_index=production_sweep_index,
        )

    def sweep(
        self,
        *,
        temperature: float | None = None,
        production_rung: int | None = None,
        production_sweep_index: int | None = None,
    ) -> None:
        target = self.config.temperature if temperature is None else float(temperature)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("sweep temperature must be positive and finite")
        self._perform_sweep(
            target,
            production_rung=production_rung,
            production_sweep_index=production_sweep_index,
        )

    def capture_candidate(
        self,
        *,
        temperature: float | None = None,
        tempering: Mapping[str, object] | None = None,
    ) -> CandidateState:
        target = self.config.temperature if temperature is None else float(temperature)
        provenance = CandidateProvenance(
            parent_id=self.parent.parent_id,
            cavity_spec=self.selection.spec,
            chain_id=self.chain_id,
            sweep_index=self.sweep_count,
            init_family=self.init_family.value,
            seeds=self.seeds.as_dict(),
            temperature=target,
            exact_core_composition=self.config.exact_core_composition,
            tempering=tempering,
        )
        return CandidateState.capture(
            self.system,
            selection=self.selection,
            provenance=provenance,
            observables={"active_potential_energy": float(self._last_result.energy)},
        )

    def run(
        self,
        *,
        burn_in_sweeps: int,
        production_sweeps: int,
        sample_interval: int = 1,
    ) -> list[CandidateState]:
        if burn_in_sweeps < 0 or production_sweeps < 0 or sample_interval <= 0:
            raise ValueError("sweep counts must be non-negative and sample_interval positive")
        for _ in range(burn_in_sweeps):
            self.sweep()
        samples: list[CandidateState] = []
        for production_index in range(production_sweeps):
            self.sweep(production_rung=0, production_sweep_index=production_index)
            if (production_index + 1) % sample_interval == 0:
                samples.append(self.capture_candidate())
        return samples

    def _assert_invariants(
        self,
        *,
        production_rung: int | None = None,
        production_sweep_index: int | None = None,
    ) -> None:
        exterior = self.selection.exterior_indices.to(self.system.device)
        buffer = self.selection.buffer_indices.to(self.system.device)
        if not torch.equal(self.system.positions[exterior], self._exterior_positions):
            raise RuntimeError("frozen exterior position changed")
        if not torch.equal(self.system.diameters[exterior], self._exterior_diameters):
            raise RuntimeError("frozen exterior diameter changed")
        if not torch.equal(torch.sort(self.system.diameters[buffer]).values, self._buffer_multiset):
            raise RuntimeError("buffer diameter multiset changed")
        expected_active = self.selection.buffer_mask.to(self.system.device)
        if not torch.equal(self.system.active_mask, expected_active):
            raise RuntimeError("active mask no longer matches frozen buffer membership")
        if not bool(torch.all(torch.isfinite(self.system.positions))):
            if production_rung is not None and production_sweep_index is not None:
                raise ChainNumericalError(
                    chain_id=self.chain_id,
                    rung=production_rung,
                    sweep_index=production_sweep_index,
                )
            raise RuntimeError("non-finite active trajectory")
        if self.config.exact_core_composition:
            core = self.selection.core_indices.to(self.system.device)
            shell = self.selection.shell_indices.to(self.system.device)
            if not self._has_exact_membership():
                raise RuntimeError("exact-core trajectory crossed the core boundary")
            if not torch.equal(torch.sort(self.system.diameters[core]).values, self._core_multiset):
                raise RuntimeError("core diameter multiset changed")
            if not torch.equal(torch.sort(self.system.diameters[shell]).values, self._shell_multiset):
                raise RuntimeError("shell diameter multiset changed")


@dataclass(frozen=True)
class ParallelTemperingReport:
    attempts: int
    accepted: int
    round_trips: tuple[int, ...]
    live_rungs: tuple[int, ...] = ()
    quarantined_walkers: tuple["WalkerQuarantine", ...] = ()

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


@dataclass(frozen=True)
class WalkerQuarantine:
    """Immutable record of one terminal production failure in a PT ladder."""

    walker_id: int
    walker_origin_chain_id: str
    chain_id: str
    rung: int
    sweep_index: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "walker_id": self.walker_id,
            "walker_origin_chain_id": self.walker_origin_chain_id,
            "chain_id": self.chain_id,
            "rung": self.rung,
            "sweep_index": self.sweep_index,
            "reason": self.reason,
        }


def replica_exchange_log_acceptance(
    energy_left: float,
    energy_right: float,
    temperature_left: float,
    temperature_right: float,
) -> float:
    """Configurational replica-exchange log Metropolis ratio."""

    if temperature_left <= 0.0 or temperature_right <= 0.0:
        raise ValueError("replica temperatures must be positive")
    return (1.0 / temperature_left - 1.0 / temperature_right) * (
        energy_left - energy_right
    )


class ParallelTemperingSampler:
    """Optional replica exchange over active-region temperatures.

    Whole active configurations (positions and the fixed diameter multiset) are
    exchanged. Frozen surroundings are identical and never copied or moved.
    The Metropolis exponent is ``(beta_i-beta_j) * (U_i-U_j)``.
    """

    def __init__(
        self,
        chains: Sequence[RCCEChain],
        *,
        temperatures: Sequence[float],
        seed: int,
    ) -> None:
        if len(chains) < 2 or len(chains) != len(temperatures):
            raise ValueError("parallel tempering requires matching chains and at least two temperatures")
        clean_temperatures = tuple(float(value) for value in temperatures)
        if any(value <= 0.0 or not math.isfinite(value) for value in clean_temperatures):
            raise ValueError("parallel-tempering temperatures must be positive and finite")
        if any(right <= left for left, right in zip(clean_temperatures, clean_temperatures[1:])):
            raise ValueError("parallel-tempering temperatures must be strictly increasing")
        reference = chains[0]
        for chain in chains[1:]:
            if chain.parent.parent_id != reference.parent.parent_id or chain.selection.spec != reference.selection.spec:
                raise ValueError("replicas must share a parent and cavity")
            if chain.config.exact_core_composition != reference.config.exact_core_composition:
                raise ValueError("replicas must use the same composition constraint")
        self.chains = list(chains)
        self.temperatures = clean_temperatures
        self.seed = int(seed)
        self.generator = _generator(seed)
        self._attempts = 0
        self._accepted = 0
        self._round = 0
        self._walker_at_slot = list(range(len(chains)))
        self._origin: list[int | None] = [None] * len(chains)
        self._seen_opposite = [False] * len(chains)
        self._round_trips = [0] * len(chains)
        self._rung_alive = [True] * len(chains)
        self._dead_walkers: set[int] = set()
        self._quarantined_walkers: list[WalkerQuarantine] = []
        self._record_extreme_visits()

    def _record_extreme_visits(self) -> None:
        last_slot = len(self.chains) - 1
        for slot in (0, last_slot):
            if not self._rung_alive[slot]:
                continue
            extreme = 0 if slot == 0 else 1
            walker = self._walker_at_slot[slot]
            origin = self._origin[walker]
            if origin is None:
                self._origin[walker] = extreme
            elif extreme != origin:
                self._seen_opposite[walker] = True
            elif self._seen_opposite[walker]:
                self._round_trips[walker] += 1
                self._seen_opposite[walker] = False

    def _quarantine(self, error: ChainNumericalError) -> None:
        rung = error.rung
        if rung < 0 or rung >= len(self.chains):
            raise ValueError(f"numerical error rung {rung} is outside this ladder")
        if not self._rung_alive[rung]:
            return
        walker = self._walker_at_slot[rung]
        self._rung_alive[rung] = False
        self._dead_walkers.add(walker)
        self._quarantined_walkers.append(
            WalkerQuarantine(
                walker_id=walker,
                walker_origin_chain_id=self.chains[walker].chain_id,
                chain_id=error.chain_id,
                rung=rung,
                sweep_index=error.sweep_index,
                reason=error.reason,
            )
        )

    @property
    def quarantined_walkers(self) -> tuple[WalkerQuarantine, ...]:
        return tuple(self._quarantined_walkers)

    @property
    def live_rungs(self) -> tuple[int, ...]:
        return tuple(rung for rung, alive in enumerate(self._rung_alive) if alive)

    @property
    def live_walkers(self) -> tuple[int, ...]:
        return tuple(
            self._walker_at_slot[rung]
            for rung, alive in enumerate(self._rung_alive)
            if alive
        )

    @property
    def target_rung_dead(self) -> bool:
        return not self._rung_alive[0]

    def is_rung_alive(self, rung: int) -> bool:
        if rung < 0 or rung >= len(self.chains):
            raise IndexError(f"rung {rung} is outside this ladder")
        return self._rung_alive[rung]

    @staticmethod
    def _exchange_active(left: RCCEChain, right: RCCEChain) -> None:
        left_indices = left.selection.buffer_indices.to(left.system.device)
        right_indices = right.selection.buffer_indices.to(right.system.device)
        # This is configurational replica exchange: momenta stay attached to
        # their temperature slots (and are fully redrawn before the next MD
        # segment), so potential energy alone is the correct acceptance term.
        names = ("positions", "unwrapped_positions", "diameters")
        for name in names:
            left_tensor = getattr(left.system, name)
            right_tensor = getattr(right.system, name)
            left_values = left_tensor[left_indices].detach().clone()
            right_values = right_tensor[right_indices].detach().clone().to(left.system.device)
            left_tensor[left_indices] = right_values
            right_tensor[right_indices] = left_values.to(right.system.device)
        for chain in (left, right):
            chain._neighbors = VerletList.from_system(chain.system, skin=chain.config.skin)
            chain._last_result = conditional_potential(chain.system, chain._neighbors)
            chain._assert_invariants()

    def attempt_exchanges(
        self,
        *,
        parity: int,
        production_sweep_index: int | None = None,
    ) -> ParallelTemperingReport:
        if parity not in (0, 1):
            raise ValueError("exchange parity must be zero or one")
        uniforms = torch.rand(
            (max(0, (len(self.chains) - parity) // 2),),
            generator=self.generator,
            device="cpu",
            dtype=torch.float64,
        )
        draw_index = 0
        for left_slot in range(parity, len(self.chains) - 1, 2):
            right_slot = left_slot + 1
            uniform = float(uniforms[draw_index])
            draw_index += 1
            if not self._rung_alive[left_slot] or not self._rung_alive[right_slot]:
                continue
            left, right = self.chains[left_slot], self.chains[right_slot]
            energy_left = float(left._last_result.energy)
            energy_right = float(right._last_result.energy)
            log_acceptance = replica_exchange_log_acceptance(
                energy_left,
                energy_right,
                self.temperatures[left_slot],
                self.temperatures[right_slot],
            )
            self._attempts += 1
            if log_acceptance >= 0.0 or math.log(uniform) < log_acceptance:
                self._exchange_active(left, right)
                self._walker_at_slot[left_slot], self._walker_at_slot[right_slot] = (
                    self._walker_at_slot[right_slot],
                    self._walker_at_slot[left_slot],
                )
                self._accepted += 1
        self._record_extreme_visits()
        return self.report

    def sweep(self, *, production_sweep_index: int | None = None) -> ParallelTemperingReport:
        for rung, (chain, temperature) in enumerate(zip(self.chains, self.temperatures, strict=True)):
            if not self._rung_alive[rung]:
                continue
            try:
                chain.sweep(
                    temperature=temperature,
                    production_rung=rung if production_sweep_index is not None else None,
                    production_sweep_index=production_sweep_index,
                )
            except ChainNumericalError as error:
                if production_sweep_index is None:
                    raise
                self._quarantine(error)
        report = self.attempt_exchanges(
            parity=self._round % 2,
            production_sweep_index=production_sweep_index,
        )
        self._round += 1
        return report

    def run(self, *, rounds: int, sample_interval: int = 1) -> list[CandidateState]:
        if rounds < 0 or sample_interval <= 0:
            raise ValueError("rounds must be non-negative and sample_interval positive")
        samples: list[CandidateState] = []
        for index in range(rounds):
            self.sweep(production_sweep_index=index)
            if (index + 1) % sample_interval == 0 and self._rung_alive[0]:
                walker = self._walker_at_slot[0]
                samples.append(
                    self.chains[0].capture_candidate(
                        temperature=self.temperatures[0],
                        tempering={
                            "walker_id": walker,
                            "walker_origin_chain_id": self.chains[walker].chain_id,
                            "temperature_slot": 0,
                            "temperature_ladder": list(self.temperatures),
                            "tempering_seed": self.seed,
                            "replica_seeds": {
                                chain.chain_id: chain.seeds.as_dict() for chain in self.chains
                            },
                            "live_rungs": list(self.live_rungs),
                            "quarantined_walkers": [
                                quarantine.to_dict() for quarantine in self._quarantined_walkers
                            ],
                        },
                    )
                )
        return [
            sample
            for sample in samples
            if sample.provenance.tempering is not None
            and int(sample.provenance.tempering["walker_id"]) not in self._dead_walkers
        ]

    @property
    def report(self) -> ParallelTemperingReport:
        return ParallelTemperingReport(
            attempts=self._attempts,
            accepted=self._accepted,
            round_trips=tuple(self._round_trips),
            live_rungs=self.live_rungs,
            quarantined_walkers=self.quarantined_walkers,
        )
