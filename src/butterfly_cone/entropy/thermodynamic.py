"""Ideal and excess entropy from convention-matched thermodynamic integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SOFT_SPHERE_HEAD_EXPONENTS = (-0.75, -0.5, -0.25)


@dataclass(frozen=True)
class HighTemperatureHead:
    """Fitted high-temperature energy asymptote and diagnostics."""

    coefficients: tuple[float, float, float]
    beta_min: float
    n_points: int
    relative_rms_residual: float

    def energy(self, beta: float | np.ndarray) -> float | np.ndarray:
        """Evaluate ``A beta^-3/4 + B beta^-1/2 + C beta^-1/4``."""

        values = np.asarray(beta, dtype=np.float64)
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("beta must be finite and positive")
        result = sum(
            coefficient * values**exponent
            for coefficient, exponent in zip(
                self.coefficients,
                SOFT_SPHERE_HEAD_EXPONENTS,
                strict=True,
            )
        )
        if values.ndim == 0:
            return float(result)
        return result

    def integral_to(self, beta: float) -> float:
        """Analytically integrate the fitted energy from beta=0 to ``beta``."""

        if beta < 0.0 or not math.isfinite(beta):
            raise ValueError("beta must be finite and non-negative")
        if beta == 0.0:
            return 0.0
        return float(
            sum(
                coefficient * beta ** (exponent + 1.0) / (exponent + 1.0)
                for coefficient, exponent in zip(
                    self.coefficients,
                    SOFT_SPHERE_HEAD_EXPONENTS,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True)
class FreeEnergyIntegral:
    """Result of ``beta f_ex = integral_0^beta u_ex(beta') d beta'``."""

    beta_target: float
    beta_f_ex: float
    head_integral: float
    grid_integral: float
    u_at_target: float
    head: HighTemperatureHead


@dataclass(frozen=True)
class ExcessEntropy:
    """Excess-entropy result at one temperature, per particle."""

    temperature: float
    beta: float
    u_ex: float
    beta_f_ex: float
    entropy_per_particle: float
    integration: FreeEnergyIntegral


def sackur_tetrode_entropy(
    temperature: float,
    density: float,
    *,
    planck_constant: float = 1.0,
    mass: float = 1.0,
) -> float:
    """Return the one-component three-dimensional ideal-gas entropy per particle."""

    values = (temperature, density, planck_constant, mass)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("temperature, density, planck_constant, and mass must be positive")
    thermal_wavelength = planck_constant / math.sqrt(2.0 * math.pi * mass * temperature)
    return float(2.5 - math.log(density * thermal_wavelength**3))


def _coerce_ladder(
    betas: np.ndarray | object,
    energies: np.ndarray | object,
) -> tuple[np.ndarray, np.ndarray]:
    beta_values = np.asarray(betas, dtype=np.float64).reshape(-1)
    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if beta_values.size == 0 or beta_values.shape != energy_values.shape:
        raise ValueError("beta and energy grids must be non-empty and have identical shape")
    if (
        not np.all(np.isfinite(beta_values))
        or not np.all(np.isfinite(energy_values))
        or np.any(beta_values <= 0.0)
    ):
        raise ValueError("beta must be positive and both grids must be finite")
    order = np.argsort(beta_values, kind="stable")
    beta_values = beta_values[order]
    energy_values = energy_values[order]
    if np.any(np.diff(beta_values) <= 0.0):
        raise ValueError("beta grid values must be unique")
    return beta_values, energy_values


def fit_soft_sphere_high_temperature_head(
    betas: np.ndarray | object,
    energies: np.ndarray | object,
    *,
    n_points: int = 8,
) -> HighTemperatureHead:
    r"""Fit the full soft-sphere ``beta^{-3/4,-1/2,-1/4}`` energy head.

    Multiplying by ``beta**(3/4)`` turns the asymptote into a quadratic
    polynomial in ``x=beta**(1/4)``, avoiding the poorly conditioned raw power
    fit used by the predecessor.
    """

    beta_values, energy_values = _coerce_ladder(betas, energies)
    if n_points < 3 or n_points > beta_values.size:
        raise ValueError("n_points must be between 3 and the ladder length")
    fitted_beta = beta_values[:n_points]
    fitted_energy = energy_values[:n_points]
    x = fitted_beta**0.25
    y = fitted_beta**0.75 * fitted_energy
    coefficients = np.polynomial.polynomial.polyfit(x, y, deg=2)
    prediction = np.polynomial.polynomial.polyval(x, coefficients)
    scale = max(float(np.sqrt(np.mean(y * y))), np.finfo(np.float64).tiny)
    residual = float(np.sqrt(np.mean((prediction - y) ** 2)) / scale)
    return HighTemperatureHead(
        coefficients=tuple(float(value) for value in coefficients),
        beta_min=float(beta_values[0]),
        n_points=int(n_points),
        relative_rms_residual=residual,
    )


def integrate_excess_free_energy(
    beta_target: float,
    beta_grid: np.ndarray | object,
    u_grid: np.ndarray | object,
    *,
    head: HighTemperatureHead | None = None,
    n_head: int = 8,
) -> FreeEnergyIntegral:
    r"""Integrate ``beta f_ex = integral_0^beta u_ex(beta') d beta'``."""

    if beta_target <= 0.0 or not math.isfinite(beta_target):
        raise ValueError("beta_target must be finite and positive")
    betas, energies = _coerce_ladder(beta_grid, u_grid)
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(beta_target), float(betas[-1]))
    if beta_target < betas[0] - tolerance or beta_target > betas[-1] + tolerance:
        raise ValueError("beta_target must lie within the measured beta grid")
    beta_target = min(max(float(beta_target), float(betas[0])), float(betas[-1]))
    fitted_head = (
        fit_soft_sphere_high_temperature_head(betas, energies, n_points=n_head)
        if head is None
        else head
    )
    head_integral = fitted_head.integral_to(float(betas[0]))

    endpoint = int(np.searchsorted(betas, beta_target, side="right"))
    segment_beta = betas[:endpoint]
    segment_energy = energies[:endpoint]
    u_target = float(np.interp(beta_target, betas, energies))
    if segment_beta.size == 0:
        segment_beta = np.array([betas[0]], dtype=np.float64)
        segment_energy = np.array([energies[0]], dtype=np.float64)
    if segment_beta[-1] < beta_target:
        segment_beta = np.append(segment_beta, beta_target)
        segment_energy = np.append(segment_energy, u_target)
    grid_integral = float(np.trapezoid(segment_energy, segment_beta))
    return FreeEnergyIntegral(
        beta_target=beta_target,
        beta_f_ex=float(head_integral + grid_integral),
        head_integral=float(head_integral),
        grid_integral=grid_integral,
        u_at_target=u_target,
        head=fitted_head,
    )


def excess_entropy(
    *,
    temperature: float,
    beta_grid: np.ndarray | object,
    u_grid: np.ndarray | object,
    u_at_temperature: float | None = None,
    head: HighTemperatureHead | None = None,
    n_head: int = 8,
) -> ExcessEntropy:
    """Return ``s_ex = beta u_ex - beta f_ex`` per particle."""

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    beta = 1.0 / temperature
    integration = integrate_excess_free_energy(
        beta,
        beta_grid,
        u_grid,
        head=head,
        n_head=n_head,
    )
    energy = integration.u_at_target if u_at_temperature is None else float(u_at_temperature)
    if not math.isfinite(energy):
        raise ValueError("u_at_temperature must be finite")
    entropy = beta * energy - integration.beta_f_ex
    return ExcessEntropy(
        temperature=float(temperature),
        beta=float(beta),
        u_ex=energy,
        beta_f_ex=integration.beta_f_ex,
        entropy_per_particle=float(entropy),
        integration=integration,
    )
