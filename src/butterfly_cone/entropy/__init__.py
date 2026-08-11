"""Convention-safe configurational-entropy analysis for ButterflyCone.

Offset-bug diagnosis
--------------------
The read-only predecessor formed ``s_c = s_ST + s_excess - s_harm`` after
adding a histogram *differential* mixing entropy to ``s_tot`` and subtracting
the same number again.  Cancelling the arbitrary/divergent reporting entropy
is necessary for continuous polydispersity, but cancelling it this way also
discarded the finite basin-permutation contribution, ``s_mix_eff``.  In the
generalised Frenkel--Ladd convention the consistent bookkeeping is

``s_tot = s_ref + s_mix_reported`` and
``s_glass = s_fixed_basin + s_mix_reported - s_mix_eff``,

so ``s_c = s_ref - s_fixed_basin + s_mix_eff``.  For the exact continuously
polydisperse soft-sphere model used by ButterflyCone, the published calibration is
``s_mix_eff(T) = 1.3601 + 7.6565 T`` over the supercooled range.  Omitting this
positive 1.87--2.19-per-particle term is the dominant reason the predecessor's
entire saved curve was negative.

Two secondary defects are repaired here as well.  The old thermodynamic-
integration head fitted ``A beta**(-3/4) + constant``; the soft-sphere
high-temperature expansion instead contains beta powers ``-3/4, -1/2,
-1/4``.  The old harmonic helper also removed the three algebraically smallest
eigenvalues, which can hide unstable modes, rather than the three modes nearest
zero.  This package uses the full head basis and removes zero modes by absolute
magnitude.  A single explicit phase-space convention (``hbar = h/(2*pi)``)
and per-particle normalization is shared by every leg and checked against
exact one- and two-basin harmonic landscapes.

All Hessian-facing operations require CPU float64 data.
"""

from .configurational import EntropyDifference, entropy_difference
from .mixing import (
    BUTTERFLY_CONE_EFFECTIVE_MIXING,
    EffectiveMixingCalibration,
    binned_diameter_mixing_entropy,
    discrete_mixing_entropy,
)
from .vibrational import HarmonicEntropy, harmonic_vibrational_entropy
from .thermodynamic import (
    ExcessEntropy,
    FreeEnergyIntegral,
    HighTemperatureHead,
    excess_entropy,
    fit_soft_sphere_high_temperature_head,
    integrate_excess_free_energy,
    sackur_tetrode_entropy,
)
from .pipeline import (
    BasinMeasurement,
    EntropyPoint,
    analyze_entropy_point,
    measure_harmonic_basin,
)

__all__ = [
    "EntropyDifference",
    "EntropyPoint",
    "EffectiveMixingCalibration",
    "ExcessEntropy",
    "FreeEnergyIntegral",
    "HarmonicEntropy",
    "HighTemperatureHead",
    "BUTTERFLY_CONE_EFFECTIVE_MIXING",
    "BasinMeasurement",
    "analyze_entropy_point",
    "binned_diameter_mixing_entropy",
    "discrete_mixing_entropy",
    "entropy_difference",
    "excess_entropy",
    "fit_soft_sphere_high_temperature_head",
    "harmonic_vibrational_entropy",
    "integrate_excess_free_energy",
    "measure_harmonic_basin",
    "sackur_tetrode_entropy",
]
