"""Causal instruments for the instrument-invariant-kernel rider.

Currently exposes the core-only momentum-conditional instrument ``I_mom``
(keystone GAP-2): an on-support momentum intervention that resamples core
momenta from the conditional Maxwell--Boltzmann law with fixed total momentum
and preserved kinetic temperature, holding positions fixed.
"""

from .momentum import (
    MomentumDraw,
    MomentumInstrumentResult,
    conditional_mb_momenta,
    momentum_instrument,
)

__all__ = [
    "MomentumDraw",
    "MomentumInstrumentResult",
    "conditional_mb_momenta",
    "momentum_instrument",
]
