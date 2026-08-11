#!/usr/bin/env python3
"""Second-model portability: distinct swap-equilibrable glass former via a
parametrized inverse-power-law (IPL) repulsion exponent and/or additive
cross-diameters, injected by monkeypatch so the ENTIRE existing ButterflyCone stack
(swap equilibration, single-system engine, batched branch dynamics, cone probe,
DW re-analysis) runs the second model with zero edits to src/butterfly_cone.

Flagship model (Ninarello-Berthier-Coslovich, src/butterfly_cone/engine/potential.py):
  V(x) = x^-12 + c0 + c2 x^2 + c4 x^4 ,  x = r / sigma_ij ,  x < 1.25
  sigma_ij = 0.5 (sig_i + sig_j) (1 - 0.2 |sig_i - sig_j|)         (nonadd=0.2)
  P(sigma) ~ sigma^-3 , ratio 2.219 , smoothed at cutoff 1.25 sigma_ij.

The smoothing coefficients for a general exponent n (V, V', V'' all vanish at
the cutoff x_c) are, derived from the three C^2 boundary conditions:
  c0 = -(n+2)(n+4)/8 * x_c^-n
  c2 =  n(n+4)/4     * x_c^-(n+2)
  c4 = -n(n+2)/8     * x_c^-(n+4)
At n=12, x_c=1.25 these reproduce the hardcoded C0=-28 x_c^-12, C2=48 x_c^-14,
C4=-21 x_c^-16 (verified numerically in the reproduction probe).

Why still swap-equilibrable: SWAP Monte Carlo only needs a *continuous* diameter
distribution and a pairwise potential whose energy change under a diameter
exchange is cheap and local -- true for any IPL exponent and either mixing rule.
The diameter-swap machinery (engine/swap.py) reads sigma_ij through the SAME
mixing_diameter / pair_potential this module patches, so swaps stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    """A second-model definition relative to the flagship."""

    exponent: int = 12          # repulsion power r^-exponent (flagship 12)
    nonadditivity: float = 0.2  # cross-diameter nonadditivity (flagship 0.2)
    cutoff_ratio: float = 1.25  # cutoff in units of sigma_ij (flagship 1.25)

    @property
    def label(self) -> str:
        return f"n{self.exponent}_nonadd{self.nonadditivity:g}"


def _smoothing_coefficients(n: float, xc: float) -> tuple[float, float, float]:
    c0 = -(n + 2.0) * (n + 4.0) / 8.0 * xc ** (-n)
    c2 = n * (n + 4.0) / 4.0 * xc ** (-(n + 2.0))
    c4 = -n * (n + 2.0) / 8.0 * xc ** (-(n + 4.0))
    return c0, c2, c4


def make_pair_potential(n: int, cutoff_ratio: float = 1.25):
    """Return a smoothed IPL ``pair_potential`` (energy + 1st/2nd derivatives).

    Signature-identical to ``butterfly_cone.engine.potential.pair_potential`` so it can be
    dropped in by monkeypatch.
    """

    nf = float(n)
    xc = float(cutoff_ratio)
    c0, c2, c4 = _smoothing_coefficients(nf, xc)

    def pair_potential(radius, sigma_ij, *, derivatives: int = 0):
        if derivatives not in (0, 1, 2):
            raise ValueError("derivatives must be 0, 1, or 2")
        if radius.shape != sigma_ij.shape:
            raise ValueError("radius and sigma_ij must have identical shapes")
        positive_radius = torch.clamp(radius, min=torch.finfo(radius.dtype).eps)
        x = positive_radius / sigma_ij
        inside = x < xc
        x_mn = x.pow(-nf)
        value_raw = x_mn + c0 + c2 * x.square() + c4 * x.pow(4)
        value = torch.where(inside, value_raw, torch.zeros_like(value_raw))
        if derivatives == 0:
            return value
        dv_dx = -nf * x.pow(-nf - 1.0) + 2.0 * c2 * x + 4.0 * c4 * x.pow(3)
        first = torch.where(inside, dv_dx / sigma_ij, torch.zeros_like(value))
        if derivatives == 1:
            return value, first
        d2v_dx2 = nf * (nf + 1.0) * x.pow(-nf - 2.0) + 2.0 * c2 + 12.0 * c4 * x.square()
        second = torch.where(inside, d2v_dx2 / sigma_ij.square(), torch.zeros_like(value))
        return value, first, second

    pair_potential.__doc__ = f"Smoothed r^-{n} IPL pair potential (cutoff {cutoff_ratio} sigma_ij)."
    return pair_potential


def inject_model(spec: ModelSpec) -> dict:
    """Monkeypatch the ButterflyCone stack to the second model. Returns provenance.

    Patches every module-level binding of ``pair_potential`` (potential, swap,
    batched) plus the live-read ``NONADDITIVITY`` constant.  ``mixing_diameter``
    resolves ``NONADDITIVITY`` from the potential module at call time, so a single
    assignment there propagates through the single-system engine, the swap energy
    kernel, and the batched branch integrator.
    """

    from butterfly_cone.engine import potential as _pot
    from butterfly_cone.engine import swap as _swap
    from butterfly_cone.branching import batched as _batched

    new_fn = make_pair_potential(spec.exponent, spec.cutoff_ratio)

    # Patch the three module-level references to pair_potential.
    _pot.pair_potential = new_fn
    _swap.pair_potential = new_fn
    _batched.pair_potential = new_fn
    # analytic_potential (potential.py) and _energy_only (swap.py) look
    # pair_potential up as a module global, so the two assignments above cover
    # both the reference swap path and the fast swap path.

    # Live-read nonadditivity: mixing_diameter reads potential.NONADDITIVITY at
    # call time everywhere it is used.
    _pot.NONADDITIVITY = float(spec.nonadditivity)

    return {
        "exponent": spec.exponent,
        "nonadditivity": float(spec.nonadditivity),
        "cutoff_ratio": spec.cutoff_ratio,
        "label": spec.label,
        "patched": ["potential.pair_potential", "swap.pair_potential",
                    "batched.pair_potential", "potential.NONADDITIVITY"],
    }
