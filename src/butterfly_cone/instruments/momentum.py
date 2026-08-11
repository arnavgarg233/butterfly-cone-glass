"""Core-only momentum-conditional causal instrument (keystone GAP-2, I_mom).

This is the *second* causal instrument of the instrument-invariant-kernel rider.
It realises an on-support **momentum** intervention ``Z_p``: positions are held
bitwise-fixed while the momenta of a chosen *core* set are resampled from the
exact conditional Maxwell--Boltzmann law at temperature ``T`` subject to a
**fixed total momentum of that set** (the intervention injects no net drift).
An optional per-particle event-propensity gradient prospectively tilts the draw
toward (``+``) or away from (``-``) a short-horizon central event, giving a
physically *different* encouragement of the same event than the structural RCCE
edit -- exactly what the downstream kernel-invariance test needs.

Physics of the fixed-total-momentum projection + temperature rescale
--------------------------------------------------------------------
Let the active (core) set have particles ``i`` with masses ``m_i`` and draw iid
Maxwell velocities ``v_i ~ N(0, (T/m_i) I_3)`` (unit ``k_B``).  We impose two
constraints, in this exact order:

1. **Fixed-total-momentum projection.**  Subtract the mass-weighted
   centre-of-mass velocity of the active set,

       ``V_cm = (sum_i m_i v_i) / (sum_i m_i)``,      ``v_i <- v_i - V_cm``,

   so ``sum_i m_i v_i = 0`` exactly (to float rounding).  For iid Gaussian
   velocities this COM removal is **not** an approximation: it reproduces the
   *exact* conditional Gaussian of ``{v_i}`` given ``sum_i m_i v_i = 0``.  The
   conditional law has ``Cov(v_i^a, v_k^a) = delta_ik T/m_i - T/M`` with
   ``M = sum_i m_i``; one checks directly that ``v_i - V_cm`` has variance
   ``T/m_i - T/M`` and cross-covariance ``-T/M``, matching term for term.  The
   number of momentum degrees of freedom drops from ``3 n_active`` to
   ``ndof = 3 n_active - 3`` (three linear constraints, one per axis).

2. **Temperature-preserving rescale.**  The projected draw has an instantaneous
   kinetic temperature ``T_inst = 2 * KE / ndof`` (``KE = 0.5 sum_i m_i |v_i|^2``)
   that fluctuates around ``T`` with the canonical ``O(1/sqrt(ndof))`` spread.
   We multiply every active velocity by the single scalar

       ``s = sqrt(ndof * T / (2 * KE))``,             ``v_i <- s v_i``,

   so ``T_inst`` equals the target ``T`` *exactly*.  A scalar rescale leaves the
   net momentum untouched (``sum_i m_i (s v_i) = s * 0 = 0``), so constraint (1)
   is preserved.

Does this preserve the conditional canonical?  Constraint (1) alone is exactly
canonical.  Constraint (2) fixes the kinetic energy to its mean, i.e. it is the
*isokinetic* projection of the conditional canonical ensemble onto the constant
kinetic-energy shell.  The isokinetic and canonical ensembles agree on all
observables to leading order in ``1/ndof`` (their marginal single-particle laws
differ only at ``O(1/ndof)``; the velocity *directions* on the shell are drawn
from exactly the canonical conditional).  So the sampler holds the conditional
canonical **to leading order**, and holds the fixed-total-momentum constraint
and the target kinetic temperature **exactly**.  Setting ``fix_total=False``
skips step (1) (``ndof = 3 n_active``); step (2) is always applied.

Determinism
-----------
All randomness is drawn on a CPU float64 generator (via
``engine.integrate.maxwell_boltzmann_velocities``) and cast once to the target
device/dtype, following the ButterflyCone device protocol.  The projection and rescale
are a fixed sequence of reductions, so a given seed reproduces bitwise-identical
momenta on repeated runs on the same device (cpu and mps alike).  Cross-device
equality is *not* claimed (float32 reduction order differs); per-device
repeatability is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from butterfly_cone.engine.integrate import maxwell_boltzmann_velocities
from butterfly_cone.engine.system import ParticleSystem, make_generator

__all__ = [
    "MomentumDraw",
    "MomentumInstrumentResult",
    "conditional_mb_momenta",
    "momentum_instrument",
]

# torch accepts a 64-bit manual seed; harness seeds may be larger.  Modular
# projection keeps deterministic domain separation (matches rcce.sampler).
_SEED_MODULUS = 2**63 - 1


def _as_bool_mask(mask: torch.Tensor, n_particles: int, device: torch.device) -> torch.Tensor:
    if mask.shape != (n_particles,):
        raise ValueError("mask must have shape (N,)")
    return mask.to(device=device, dtype=torch.bool)


def _mass_tensor(
    masses: float | torch.Tensor,
    n_particles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(masses, torch.Tensor):
        mass = masses.detach().to(device=device, dtype=dtype)
        if mass.shape != (n_particles,):
            raise ValueError("masses tensor must have shape (N,)")
    else:
        if masses <= 0.0:
            raise ValueError("masses must be positive")
        mass = torch.full((n_particles,), float(masses), device=device, dtype=dtype)
    if bool(torch.any(mass <= 0)):
        raise ValueError("masses must be positive")
    return mass


def _net_momentum(velocities: torch.Tensor, active: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Net momentum ``sum_i m_i v_i`` of the active set, shape (3,)."""

    contrib = torch.where(active[:, None], mass[:, None] * velocities, torch.zeros_like(velocities))
    return contrib.sum(dim=0)


def _kinetic_energy(velocities: torch.Tensor, active: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    per_particle = 0.5 * mass * velocities.square().sum(dim=1)
    return torch.where(active, per_particle, torch.zeros_like(per_particle)).sum()


def _ndof(n_active: int, fix_total: bool) -> int:
    if n_active <= 0:
        return 0
    if fix_total:
        return 3 * n_active - 3
    return 3 * n_active


def _project_and_rescale(
    velocities: torch.Tensor,
    active: torch.Tensor,
    temperature: float,
    mass: torch.Tensor,
    *,
    fix_total: bool,
) -> torch.Tensor:
    """Apply the fixed-total-momentum projection then the isokinetic rescale.

    Inactive rows are forced to zero.  See the module docstring for the exact
    convention; the two steps preserve, respectively, the exact conditional
    canonical support and the target kinetic temperature, and each preserves the
    other's invariant.
    """

    velocities = torch.where(active[:, None], velocities, torch.zeros_like(velocities))
    n_active = int(active.sum().item())

    # (1) fixed-total-momentum projection: subtract mass-weighted COM velocity.
    if fix_total and n_active > 1:
        total_mass = torch.where(active, mass, torch.zeros_like(mass)).sum()
        center_velocity = _net_momentum(velocities, active, mass) / total_mass
        velocities = torch.where(active[:, None], velocities - center_velocity, torch.zeros_like(velocities))
    elif fix_total and n_active == 1:
        # The only zero-net-momentum state of a single particle is rest; there
        # are no thermal degrees of freedom left to carry temperature.
        return torch.zeros_like(velocities)

    # (2) temperature-preserving isokinetic rescale onto the ndof-T shell.
    ndof = _ndof(n_active, fix_total)
    if ndof > 0:
        kinetic = _kinetic_energy(velocities, active, mass)
        if float(kinetic) > 0.0:
            target = torch.as_tensor(
                0.5 * ndof * float(temperature), device=velocities.device, dtype=velocities.dtype
            )
            scale = torch.sqrt(target / kinetic)
            velocities = torch.where(active[:, None], velocities * scale, torch.zeros_like(velocities))
    return velocities


@dataclass(frozen=True)
class MomentumDraw:
    """A single conditional momentum draw plus its achieved diagnostics."""

    velocities: torch.Tensor
    n_active: int
    ndof: int
    temperature_target: float
    achieved_temperature: float
    net_momentum_residual: float


def conditional_mb_momenta(
    system: ParticleSystem,
    active_mask: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    *,
    fix_total: bool = True,
    masses: float | torch.Tensor = 1.0,
    return_draw: bool = False,
) -> torch.Tensor | MomentumDraw:
    """Draw conditional Maxwell--Boltzmann momenta for the active set.

    Velocities ``v_i ~ N(0, (T/m_i) I)`` are drawn for active particles (zero
    elsewhere), then the fixed-total-momentum projection and the
    temperature-preserving isokinetic rescale of the module docstring are
    applied (the projection is skipped when ``fix_total=False``).  The result
    has, on the active set, net momentum ``0`` and instantaneous kinetic
    temperature ``T`` -- both exactly -- while holding the conditional canonical
    to leading order.

    Returns the full ``(N, 3)`` velocity tensor (device/dtype of ``system``), or
    a :class:`MomentumDraw` with diagnostics when ``return_draw=True``.
    """

    if temperature < 0.0:
        raise ValueError("temperature must be nonnegative")
    device, dtype = system.device, system.dtype
    n_particles = system.n_particles
    active = _as_bool_mask(active_mask, n_particles, device)
    mass = _mass_tensor(masses, n_particles, device, dtype)
    n_active = int(active.sum().item())

    if n_active == 0:
        velocities = torch.zeros((n_particles, 3), device=device, dtype=dtype)
    else:
        # remove_com=False: this module owns the projection+rescale convention.
        velocities = maxwell_boltzmann_velocities(
            n_particles,
            float(temperature),
            generator,
            device=device,
            dtype=dtype,
            masses=mass,
            active_mask=active,
            remove_com=False,
        )
        velocities = _project_and_rescale(
            velocities, active, float(temperature), mass, fix_total=fix_total
        )

    if not return_draw:
        return velocities

    ndof = _ndof(n_active, fix_total)
    kinetic = float(_kinetic_energy(velocities, active, mass))
    achieved_t = (2.0 * kinetic / ndof) if ndof > 0 else 0.0
    residual = float(torch.linalg.vector_norm(_net_momentum(velocities, active, mass)))
    return MomentumDraw(
        velocities=velocities,
        n_active=n_active,
        ndof=ndof,
        temperature_target=float(temperature),
        achieved_temperature=achieved_t,
        net_momentum_residual=residual,
    )


@dataclass(frozen=True)
class MomentumInstrumentResult:
    """A ``+``/``-`` momentum-edited counterfactual pair and its provenance.

    ``plus`` and ``minus`` are clones of the input system with **only** the core
    momenta changed (positions, box, diameters, unwrapped positions and non-core
    velocities are bitwise-identical to the input).  With a ``bias`` field the
    two members are tilted toward (``plus``) and away from (``minus``) the event
    the gradient encodes; without a bias they are a symmetric momentum-reversal
    pair (``+v`` / ``-v``) of the same conditional draw -- identical energy and
    zero net momentum, opposite momentum direction.
    """

    plus: ParticleSystem
    minus: ParticleSystem
    seed: int
    temperature: float
    n_core: int
    ndof: int
    biased: bool
    bias_strength: float
    achieved_temperature_plus: float
    achieved_temperature_minus: float
    net_momentum_residual_plus: float
    net_momentum_residual_minus: float

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "temperature": float(self.temperature),
            "n_core": int(self.n_core),
            "ndof": int(self.ndof),
            "biased": bool(self.biased),
            "bias_strength": float(self.bias_strength),
            "achieved_temperature_plus": float(self.achieved_temperature_plus),
            "achieved_temperature_minus": float(self.achieved_temperature_minus),
            "net_momentum_residual_plus": float(self.net_momentum_residual_plus),
            "net_momentum_residual_minus": float(self.net_momentum_residual_minus),
        }


def _with_core_velocities(
    system: ParticleSystem, core: torch.Tensor, core_velocities: torch.Tensor
) -> ParticleSystem:
    """Clone ``system`` replacing only the core rows of ``velocities``."""

    edited = system.clone()
    new_velocities = torch.where(core[:, None], core_velocities, edited.velocities)
    edited.velocities = new_velocities
    return edited


def momentum_instrument(
    system: ParticleSystem,
    core_mask: torch.Tensor,
    temperature: float,
    seed: int,
    *,
    bias: torch.Tensor | None = None,
    bias_strength: float = 1.0,
    masses: float | torch.Tensor = 1.0,
) -> MomentumInstrumentResult:
    """Produce a ``+``/``-`` momentum-edited counterfactual pair.

    Positions are untouched.  A single conditional draw ``v0`` is taken for the
    ``core_mask`` set (fixed total momentum, temperature ``T``).  When ``bias``
    (a per-particle ``(N, 3)`` event-propensity gradient in velocity space) is
    given, the pair is ``v0 +/- bias_strength * bias`` re-projected and
    re-rescaled, so both members share the input positions, the exact target
    temperature and zero net core momentum, differing only in how the momenta
    are tilted along the event gradient.  When ``bias`` is ``None`` the pair is
    the momentum reversal ``+v0`` / ``-v0``.

    Note on diagnosing the tilt: because kinetic energy is even in the momentum,
    the ``+`` and ``-`` members have identical (regional and total) kinetic
    energy by construction; the instrument's effect is carried by the *odd*
    moments -- the net directed momentum ``sum_i v_i . bias_i`` along the bias
    field -- which is what physically encourages/discourages the central event.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    device, dtype = system.device, system.dtype
    n_particles = system.n_particles
    core = _as_bool_mask(core_mask, n_particles, device)
    mass = _mass_tensor(masses, n_particles, device, dtype)
    n_core = int(core.sum().item())
    generator = make_generator(int(seed) % _SEED_MODULUS)

    base = conditional_mb_momenta(
        system, core, float(temperature), generator, fix_total=True, masses=mass
    )

    if bias is None:
        v_plus = base
        v_minus = torch.where(core[:, None], -base, torch.zeros_like(base))
        biased = False
    else:
        if bias.shape != (n_particles, 3):
            raise ValueError("bias must have shape (N, 3)")
        shift = bias.detach().to(device=device, dtype=dtype) * float(bias_strength)
        shift = torch.where(core[:, None], shift, torch.zeros_like(shift))
        v_plus = _project_and_rescale(base + shift, core, float(temperature), mass, fix_total=True)
        v_minus = _project_and_rescale(base - shift, core, float(temperature), mass, fix_total=True)
        biased = True

    plus = _with_core_velocities(system, core, v_plus)
    minus = _with_core_velocities(system, core, v_minus)

    ndof = _ndof(n_core, True)

    def _achieved_t(vel: torch.Tensor) -> float:
        if ndof <= 0:
            return 0.0
        return float(2.0 * _kinetic_energy(vel, core, mass) / ndof)

    return MomentumInstrumentResult(
        plus=plus,
        minus=minus,
        seed=int(seed),
        temperature=float(temperature),
        n_core=n_core,
        ndof=ndof,
        biased=biased,
        bias_strength=float(bias_strength),
        achieved_temperature_plus=_achieved_t(v_plus),
        achieved_temperature_minus=_achieved_t(v_minus),
        net_momentum_residual_plus=float(torch.linalg.vector_norm(_net_momentum(v_plus, core, mass))),
        net_momentum_residual_minus=float(torch.linalg.vector_norm(_net_momentum(v_minus, core, mass))),
    )
