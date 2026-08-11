"""Classical harmonic basin entropy from a CPU-float64 Hessian spectrum."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HarmonicEntropy:
    """Validated physical modes and their harmonic entropy per particle."""

    entropy_per_particle: float
    frequencies: np.ndarray
    physical_eigenvalues: np.ndarray
    discarded_zero_eigenvalues: np.ndarray
    n_modes: int


def harmonic_vibrational_entropy(
    eigenvalues: np.ndarray | object,
    *,
    temperature: float,
    n_particles: int,
    expected_zero_modes: int = 3,
    zero_tolerance: float = 1.0e-3,
    planck_constant: float = 1.0,
    mass: float = 1.0,
) -> HarmonicEntropy:
    r"""Compute ``N^-1 sum_a [1 + ln(T/(hbar omega_a))]``.

    The ``expected_zero_modes`` eigenvalues nearest zero are removed.  Every
    remaining eigenvalue must be strictly positive: unstable or extra-zero
    physical modes are errors, never silently filtered out.
    """

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if expected_zero_modes < 0 or zero_tolerance < 0.0:
        raise ValueError("invalid zero-mode controls")
    if planck_constant <= 0.0 or mass <= 0.0:
        raise ValueError("planck_constant and mass must be positive")

    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size < expected_zero_modes:
        raise ValueError("spectrum has fewer entries than expected zero modes")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("spectrum must contain finite float64 eigenvalues")

    if expected_zero_modes:
        zero_indices = np.argsort(np.abs(values), kind="stable")[:expected_zero_modes]
        zeros = values[zero_indices]
        if np.any(np.abs(zeros) > zero_tolerance):
            raise ValueError("expected translational modes are not within zero_tolerance")
        physical = np.delete(values, zero_indices)
    else:
        zeros = np.empty(0, dtype=np.float64)
        physical = values.copy()
    if physical.size == 0:
        raise ValueError("spectrum contains no physical modes")
    if np.any(physical <= 0.0):
        count = int(np.count_nonzero(physical <= 0.0))
        raise ValueError(f"spectrum contains {count} non-positive physical modes")

    frequencies = np.sqrt(physical / mass)
    hbar = planck_constant / (2.0 * math.pi)
    mode_entropies = 1.0 + np.log(temperature / (hbar * frequencies))
    entropy = float(np.sum(mode_entropies, dtype=np.float64) / n_particles)
    return HarmonicEntropy(
        entropy_per_particle=entropy,
        frequencies=frequencies,
        physical_eigenvalues=physical,
        discarded_zero_eigenvalues=zeros,
        n_modes=int(physical.size),
    )
