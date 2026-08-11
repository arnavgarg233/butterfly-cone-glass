"""Configurational-entropy bookkeeping with an explicit mixing convention."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EntropyDifference:
    """Per-particle entropy terms in one internally consistent convention."""

    reference_total_entropy: float
    fixed_basin_entropy: float
    reported_mixing_entropy: float
    effective_mixing_entropy: float
    total_entropy: float
    glass_entropy: float
    configurational_entropy: float


def entropy_difference(
    *,
    reference_total_entropy: float,
    fixed_basin_entropy: float,
    reported_mixing_entropy: float = 0.0,
    effective_mixing_entropy: float = 0.0,
) -> EntropyDifference:
    """Return ``s_tot - s_glass`` without a polydispersity gauge mismatch.

    ``reference_total_entropy`` excludes the conventional ideal mixing term.
    ``reported_mixing_entropy`` may use any *discrete* species convention, but
    is inserted in both total and glass entropies and therefore cancels.
    ``effective_mixing_entropy`` is the finite basin-permutation contribution
    measured by a generalised Frenkel--Ladd/diameter-swap construction.  It is
    zero for a monodisperse or strictly fixed-label landscape.
    """

    values = (
        reference_total_entropy,
        fixed_basin_entropy,
        reported_mixing_entropy,
        effective_mixing_entropy,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("entropy inputs must be finite")
    if reported_mixing_entropy < 0.0:
        raise ValueError("reported discrete mixing entropy must be non-negative")
    if effective_mixing_entropy < 0.0:
        raise ValueError("effective mixing entropy must be non-negative")

    total = reference_total_entropy + reported_mixing_entropy
    glass = fixed_basin_entropy + reported_mixing_entropy - effective_mixing_entropy
    # Evaluate the reduced expression directly so an arbitrarily large
    # reporting convention cannot leak cancellation roundoff into s_c.
    configurational = (
        reference_total_entropy - fixed_basin_entropy + effective_mixing_entropy
    )
    return EntropyDifference(
        reference_total_entropy=float(reference_total_entropy),
        fixed_basin_entropy=float(fixed_basin_entropy),
        reported_mixing_entropy=float(reported_mixing_entropy),
        effective_mixing_entropy=float(effective_mixing_entropy),
        total_entropy=float(total),
        glass_entropy=float(glass),
        configurational_entropy=float(configurational),
    )
