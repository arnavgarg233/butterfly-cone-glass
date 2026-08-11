#!/usr/bin/env python3
"""Shared, add-only model definitions and numerical helpers for breadth runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BreadthModelSpec:
    """One swap-friendly model with at most one requested structural change."""

    name: str
    exponent: int
    nonadditivity: float = 0.2
    centers: tuple[float, ...] = ()
    peak_width: float = 0.0
    temperature: float = 0.10
    description: str = ""

    @property
    def distribution(self) -> str:
        if len(self.centers) == 2:
            return "continuous_extreme_bimodal"
        if len(self.centers) == 3:
            return "continuous_trimodal"
        return "continuous_sigma^-3"

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["centers"] = list(self.centers)
        record["distribution"] = self.distribution
        return record


_EXTREME_SMALL = 2.0 / 2.6
_EXTREME_LARGE = 1.6 * _EXTREME_SMALL

MODEL_SPECS: dict[str, BreadthModelSpec] = {
    "soft_r8": BreadthModelSpec(
        name="soft_r8", exponent=8, temperature=0.04,
        description="C2-smoothed r^-8 IPL; flagship continuous diameter distribution and mixing",
    ),
    "trimodal": BreadthModelSpec(
        name="trimodal", exponent=12, centers=(0.78, 1.0, 1.22), peak_width=0.045,
        temperature=0.08,
        description="equal-weight continuous three-peak diameter population at fixed r^-12",
    ),
    "extreme_bimodal": BreadthModelSpec(
        name="extreme_bimodal", exponent=12,
        centers=(_EXTREME_SMALL, _EXTREME_LARGE), peak_width=0.18,
        temperature=0.08,
        description="equal-weight continuous two-peak population with center ratio exactly 1.6",
    ),
}


def make_model_diameters(
    model: str,
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the model's immutable continuous diameter population and peak labels."""

    if model not in MODEL_SPECS:
        raise KeyError(f"unknown breadth model {model!r}")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    spec = MODEL_SPECS[model]
    if not spec.centers:
        ratio = 2.219
        lower = (1.0 + ratio) / (2.0 * ratio)
        upper = (1.0 + ratio) / 2.0
        uniform = rng.random(n_particles)
        values = (lower**-2 - uniform * (lower**-2 - upper**-2)) ** -0.5
        labels = np.floor(np.argsort(np.argsort(values, kind="stable"), kind="stable") * 3 / n_particles)
        labels = np.minimum(labels, 2).astype(np.int64)
    else:
        n_peaks = len(spec.centers)
        if n_particles % n_peaks:
            raise ValueError(f"{model} requires N divisible by {n_peaks}")
        per_peak = n_particles // n_peaks
        pieces = [rng.normal(center, spec.peak_width, per_peak) for center in spec.centers]
        values = np.concatenate(pieces)
        labels = np.repeat(np.arange(n_peaks, dtype=np.int64), per_peak)
        if np.any(values <= 0.0):
            raise RuntimeError("non-positive clustered diameter draw")
        order = rng.permutation(n_particles)
        values, labels = values[order], labels[order]
    values = np.asarray(values / values.mean(), dtype=np.float64)
    return values, np.asarray(labels, dtype=np.int64)


def distribution_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Raw moments and per-peak evidence for one persisted diameter draw."""

    diameters = np.asarray(values, dtype=np.float64)
    groups = np.asarray(labels, dtype=np.int64)
    if diameters.ndim != 1 or groups.shape != diameters.shape:
        raise ValueError("diameters and labels must be aligned one-dimensional arrays")
    peaks = []
    for label in np.unique(groups):
        selected = diameters[groups == label]
        peaks.append({
            "label": int(label), "n": int(selected.size), "mean": float(selected.mean()),
            "std": float(selected.std(ddof=1)) if selected.size > 1 else 0.0,
            "min": float(selected.min()), "max": float(selected.max()),
        })
    counts, edges = np.histogram(diameters, bins=30)
    return {
        "n": int(diameters.size), "n_unique": int(np.unique(diameters).size),
        "mean": float(diameters.mean()), "std": float(diameters.std(ddof=1)),
        "min": float(diameters.min()), "max": float(diameters.max()),
        "peak_means": [row["mean"] for row in peaks],
        "measured_outer_peak_ratio": float(peaks[-1]["mean"] / peaks[0]["mean"]),
        "peaks": peaks,
        "histogram": {"counts": counts.astype(int).tolist(), "edges": edges.tolist()},
    }


def stationary_tail(curve: np.ndarray | list[float], *, tolerance: float) -> dict[str, Any]:
    """Classify a late plateau by half-to-half drift and fitted full-window slope."""

    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 12:
        raise ValueError("stationarity requires at least twelve finite samples")
    if not np.isfinite(values).all():
        return {"stationary": False, "reason": "non-finite curve"}
    late = values[values.size // 2 :]
    split = late.size // 2
    first, second = late[:split], late[split:]
    late_mean = float(late.mean())
    scale = max(abs(late_mean), np.finfo(np.float64).eps)
    drift = abs(float(second.mean()) - float(first.mean())) / scale
    x = np.arange(late.size, dtype=np.float64)
    slope = float(np.polyfit(x, late, 1)[0])
    slope_window = abs(slope) * max(late.size - 1, 1) / scale
    return {
        "stationary": bool(drift <= tolerance and slope_window <= tolerance),
        "tolerance": float(tolerance), "n_samples": int(values.size),
        "late_n_samples": int(late.size), "late_mean": late_mean,
        "late_first_half_mean": float(first.mean()), "late_second_half_mean": float(second.mean()),
        "late_relative_drift": float(drift),
        "late_relative_slope_over_window": float(slope_window),
        "late_linear_slope_per_sample": slope,
    }


def harmonic_entropy_from_log_pseudodeterminant(
    log_pseudodeterminant: float,
    *,
    n_particles: int,
    temperature: float,
    planck_constant: float = 1.0,
    mass: float = 1.0,
) -> float:
    """Paper-convention harmonic entropy from ``log(prod physical eigenvalues)``."""

    if n_particles <= 1 or temperature <= 0.0 or planck_constant <= 0.0 or mass <= 0.0:
        raise ValueError("invalid harmonic-entropy controls")
    n_modes = 3 * n_particles - 3
    hbar = planck_constant / (2.0 * math.pi)
    common = 1.0 + math.log(temperature / hbar) + 0.5 * math.log(mass)
    return float((n_modes * common - 0.5 * log_pseudodeterminant) / n_particles)


def inject_breadth_model(model: str) -> dict[str, Any]:
    """Patch all live dynamics and mechanics bindings to the selected IPL exponent."""

    if model not in MODEL_SPECS:
        raise KeyError(f"unknown breadth model {model!r}")
    from second_model_lib import make_pair_potential
    from butterfly_cone.engine import potential, swap
    from butterfly_cone.branching import batched

    spec = MODEL_SPECS[model]
    pair = make_pair_potential(spec.exponent, 1.25)
    potential.pair_potential = pair
    potential.NONADDITIVITY = float(spec.nonadditivity)
    swap.pair_potential = pair
    batched.pair_potential = pair
    patched = [
        "butterfly_cone.engine.potential.pair_potential", "butterfly_cone.engine.swap.pair_potential",
        "butterfly_cone.branching.batched.pair_potential", "butterfly_cone.engine.potential.NONADDITIVITY",
    ]
    hessian = sys.modules.get("butterfly_cone.mechanics.hessian")
    if hessian is not None:
        hessian.pair_potential = pair
        hessian.CUTOFF_RATIO = 1.25
        patched.extend(["butterfly_cone.mechanics.hessian.pair_potential", "butterfly_cone.mechanics.hessian.CUTOFF_RATIO"])
    return {**spec.as_record(), "cutoff_ratio": 1.25, "patched": patched}
