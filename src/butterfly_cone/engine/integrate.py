"""Velocity-Verlet dynamics and stochastic velocity-rescaling NVT."""

from __future__ import annotations

import math
from typing import Any

import torch

from .neighbors import VerletList
from .system import ParticleSystem


def _normal_draw(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    if str(generator.device) != "cpu":
        raise ValueError("ButterflyCone generators must be CPU generators")
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float64)


def maxwell_boltzmann_velocities(
    n_particles: int,
    temperature: float,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    masses: float | torch.Tensor = 1.0,
    active_mask: torch.Tensor | None = None,
    remove_com: bool = True,
) -> torch.Tensor:
    """Draw unit-``k_B`` Maxwell velocities from an explicit generator."""

    if n_particles <= 0 or temperature < 0.0:
        raise ValueError("n_particles must be positive and temperature nonnegative")
    target_device = torch.device(device)
    if isinstance(masses, torch.Tensor):
        mass = masses.detach().to(device=target_device, dtype=dtype)
        if mass.shape != (n_particles,):
            raise ValueError("masses tensor must have shape (N,)")
    else:
        if masses <= 0.0:
            raise ValueError("masses must be positive")
        mass = torch.full((n_particles,), float(masses), device=target_device, dtype=dtype)
    if bool(torch.any(mass <= 0)):
        raise ValueError("masses must be positive")
    if active_mask is None:
        active = torch.ones(n_particles, device=target_device, dtype=torch.bool)
    else:
        active = active_mask.to(device=target_device, dtype=torch.bool)
    normal = _normal_draw((n_particles, 3), generator).to(device=target_device, dtype=dtype)
    velocities = normal * torch.sqrt(torch.as_tensor(temperature, device=target_device, dtype=dtype) / mass)[:, None]
    velocities = torch.where(active[:, None], velocities, torch.zeros_like(velocities))
    if remove_com and bool(torch.any(active)):
        active_mass = torch.where(active, mass, torch.zeros_like(mass))
        center_velocity = (active_mass[:, None] * velocities).sum(dim=0) / active_mass.sum()
        velocities = torch.where(active[:, None], velocities - center_velocity, torch.zeros_like(velocities))
    return velocities


class BussiThermostat:
    """Bussi-Donadio-Parrinello canonical stochastic velocity rescaling."""

    def __init__(self, temperature: float, tau: float, generator: torch.Generator) -> None:
        if temperature <= 0.0 or tau <= 0.0:
            raise ValueError("temperature and tau must be positive")
        if str(generator.device) != "cpu":
            raise ValueError("ButterflyCone generators must be CPU generators")
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.generator = generator
        self.last_alpha = 1.0
        self.heat = 0.0

    def apply(self, system: ParticleSystem, dt: float) -> float:
        active = system.active_mask
        n_active = int(active.sum().item())
        ndof = 3 * n_active - (3 if n_active > 1 else 0)
        if ndof <= 0:
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
            torch.as_tensor(c / (1.0 - c) * ndof, device=system.device, dtype=system.dtype) / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        system.velocities = torch.where(active[:, None], system.velocities * alpha, system.velocities)
        kinetic_after = 0.5 * system.velocities[active].square().sum()
        self.last_alpha = float(alpha)
        self.heat += float(kinetic_after - kinetic_before)
        return self.last_alpha

    def state_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "tau": self.tau,
            "generator_state": self.generator.get_state().detach().clone(),
            "last_alpha": self.last_alpha,
            "heat": self.heat,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BussiThermostat":
        generator = torch.Generator(device="cpu")
        generator.set_state(state["generator_state"].detach().clone().cpu())
        thermostat = cls(float(state["temperature"]), float(state["tau"]), generator)
        thermostat.last_alpha = float(state["last_alpha"])
        thermostat.heat = float(state["heat"])
        return thermostat


class MDIntegrator:
    """Stateful velocity-Verlet integrator with an optional Bussi thermostat."""

    def __init__(
        self,
        system: ParticleSystem,
        *,
        dt: float = 0.01,
        skin: float = 0.3,
        neighbor_list: VerletList | None = None,
        thermostat: BussiThermostat | None = None,
        _initialize: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.system = system
        self.dt = float(dt)
        self.neighbor_list = VerletList.from_system(system, skin) if neighbor_list is None else neighbor_list
        self.thermostat = thermostat
        self.step_count = 0
        if _initialize:
            result = self.neighbor_list.evaluate(system)
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        active = self.system.active_mask[:, None]
        for _ in range(steps):
            half_velocity = self.system.velocities + 0.5 * self.dt * self.forces
            displacement = self.dt * half_velocity
            displacement = torch.where(active, displacement, torch.zeros_like(displacement))
            self.system.unwrapped_positions = self.system.unwrapped_positions + displacement
            moved_positions = torch.remainder(self.system.positions + displacement, self.system.box)
            self.system.positions = torch.where(active, moved_positions, self.system.positions)
            result = self.neighbor_list.evaluate(self.system)
            self.system.velocities = half_velocity + 0.5 * self.dt * result.forces
            self.system.velocities = torch.where(active, self.system.velocities, torch.zeros_like(self.system.velocities))
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial
            if self.thermostat is not None:
                self.thermostat.apply(self.system, self.dt)
            self.step_count += 1

    def total_energy(self) -> torch.Tensor:
        kinetic = 0.5 * self.system.velocities[self.system.active_mask].square().sum()
        return self.potential_energy + kinetic

    def state_dict(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "step_count": self.step_count,
            "forces": self.forces.detach().clone(),
            "potential_energy": self.potential_energy.detach().clone(),
            "virial": self.virial.detach().clone(),
            "neighbor_list": self.neighbor_list.state_dict(),
            "thermostat": None if self.thermostat is None else self.thermostat.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        system: ParticleSystem,
        state: dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> "MDIntegrator":
        target = system.device if device is None else torch.device(device)
        neighbors = VerletList.from_state_dict(state["neighbor_list"], device=target)
        thermostat_state = state["thermostat"]
        thermostat = None if thermostat_state is None else BussiThermostat.from_state_dict(thermostat_state)
        integrator = cls(
            system,
            dt=float(state["dt"]),
            neighbor_list=neighbors,
            thermostat=thermostat,
            _initialize=False,
        )
        integrator.step_count = int(state["step_count"])
        integrator.forces = state["forces"].detach().clone().to(target)
        integrator.potential_energy = state["potential_energy"].detach().clone().to(target)
        integrator.virial = state["virial"].detach().clone().to(target)
        return integrator
