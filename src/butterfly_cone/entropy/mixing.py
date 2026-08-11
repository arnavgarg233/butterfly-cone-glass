"""Discrete reporting entropy and finite basin-permutation corrections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EffectiveMixingCalibration:
    """Linear ``s_mix_eff(T)`` calibration for a named particle model."""

    name: str
    intercept: float
    slope: float
    temperature_min: float
    temperature_max: float
    source: str

    def evaluate(self, temperature: float, *, allow_extrapolation: bool = False) -> float:
        """Evaluate the calibration, rejecting silent extrapolation by default."""

        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        if not allow_extrapolation and not self.temperature_min <= temperature <= self.temperature_max:
            raise ValueError(
                f"temperature {temperature:g} is outside calibrated range "
                f"[{self.temperature_min:g}, {self.temperature_max:g}]"
            )
        value = self.intercept + self.slope * temperature
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("effective mixing calibration produced an invalid entropy")
        return float(value)


# Same P(sigma)~sigma^-3 distribution, diameter ratio, nonadditivity and
# smoothed r^-12 potential as ButterflyCone.  Berthier et al., PNAS 114, 11356 (2017),
# Appendix I.4: s_mix* = b0 + b1 T, b0=1.3601, b1=7.6565.
BUTTERFLY_CONE_EFFECTIVE_MIXING = EffectiveMixingCalibration(
    name="butterfly_cone-continuous-polydisperse-soft-sphere",
    intercept=1.3601,
    slope=7.6565,
    temperature_min=0.0555,
    temperature_max=0.125,
    source="doi:10.1073/pnas.1706860114, Appendix I.4",
)


def discrete_mixing_entropy(counts: Sequence[int] | np.ndarray) -> float:
    r"""Return ``-sum x_a ln(x_a)`` for discrete species counts.

    This is dimensionless and invariant to diameter units.  It is suitable as
    the arbitrary *reported* convention that is added to both total and glass
    entropy, unlike a histogram-density differential entropy.
    """

    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("counts must be a non-empty finite non-negative sequence")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("at least one count must be positive")
    fractions = values[values > 0.0] / total
    return float(-np.sum(fractions * np.log(fractions), dtype=np.float64))


def binned_diameter_mixing_entropy(
    diameters: np.ndarray | Sequence[float],
    *,
    bins: int | Sequence[float] = 40,
) -> float:
    """Return a discrete diameter-bin mixing entropy for reporting only."""

    values = np.asarray(diameters, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("diameters must be a non-empty finite positive sequence")
    if isinstance(bins, int) and bins <= 0:
        raise ValueError("bins must be positive")
    counts, _ = np.histogram(values, bins=bins, density=False)
    return discrete_mixing_entropy(counts)
