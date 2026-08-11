"""Primary power gate: f_ICE -> MDE -> branch budget (PLAN_v2.1, Gate-0).

This is the decision gate that turns a *measured* iso-configurational
predictability ``f_ICE`` (the structural/local variance fraction estimated
from the scaled pilot by :mod:`butterfly_cone.stats.varcomp`) into a GO / NO-GO call on
the confirmatory campaign, together with the branch budget the confirmatory
run would need.

Two distinct roles for ``f_ICE`` (they must not be conflated):

1. **Ceiling on the true effect.** ``f_ICE`` is the fraction of outcome
   variance that is structurally (iso-configurationally) determined; it bounds
   how strong a field-targeted vs. matched-random contrast *can* be. If even
   the optimistic structural ceiling cannot reach the Gate-0 threshold
   ``E[Δq] >= 0.15``, no amount of extra branches can rescue the experiment
   -> ``NO-GO-field-too-weak``. This is the only place ``f_ICE`` enters.

2. **The MDE does NOT depend on f_ICE.** The minimum detectable effect is a
   pure sampling-precision quantity: it is set by the per-branch Bernoulli
   variance, the number of exchangeable parent-cavity pairs, and the branches
   per arm-cell. ``f_ICE`` decides whether the *truth* clears the bar; the MDE
   decides whether the *design* can see it.

Estimand
--------
The gated estimand is the negative-control contrast (``estimands.py`` §21.4)

    D_target-random = ATE_targeted - ATE_random,

each ``ATE`` being the pair-averaged paired contrast ``E[Δq_c]`` with
``Δq_c = q̂(X_c+) - q̂(X_c-)``. The parent-cavity pair is the exchangeable
unit (see ``stats/_common.py``); every arm-cell ``(pair, family, sign)`` holds
``n_branches`` independent Bernoulli(base_rate) branches.

MDE formula
-----------
Under the standard two-sided test at level ``alpha`` with power ``1 - beta``,

    MDE = (z_{1-alpha/2} + z_{1-beta}) * SE(D̂),

and, keeping only the branch-Bernoulli sampling term (the pooled-null model
that maximises variance at ``base_rate = 0.5``),

    Var(D̂) = n_arms * p(1-p) / (n_pairs * n_branches),   p = base_rate,

so

    MDE = (z_{1-alpha/2} + z_{1-beta}) * sqrt( n_arms * p(1-p)
                                               / (n_pairs * n_branches) ).

``n_arms = 4`` for ``D_target-random``: two signs (+/-) times two families
(targeted, random). The coefficient is design-invariant -- whether the two
families are run on the *same* pairs (fully paired: ``Var(D_c) = 4p(1-p)/n_b``
per pair) or on two disjoint but geometry-matched pair samples
(``Var = Var(ATE_t) + Var(ATE_r) = 2p(1-p)/(n_b P) + 2p(1-p)/(n_b P)``), the
per-pair variance is ``4 p(1-p)/n_branches``. Pass ``n_arms = 2`` to size a
single one-family ``ATE`` instead.

Because ``MDE ∝ 1/sqrt(n_pairs * n_branches)``, quadrupling either factor
halves the MDE; this is the ``sqrt(n)`` scaling the pilot budgeting relies on.

Budget inversion
----------------
Solving ``MDE = target_effect`` for the branches-per-cell at a fixed
``n_pairs`` gives

    n_branches = (z_{1-alpha/2} + z_{1-beta})^2 * n_arms * p(1-p)
                 / (n_pairs * target_effect^2),

which :func:`required_budget` returns (exact real value plus the ceiling
integer that *guarantees* ``MDE <= target_effect``), along with the total
branch budget ``n_pairs * n_arms * n_branches``.

Fallback ladder
---------------
When the primary gate fails (``NO-GO-field-too-weak``) the declared,
pre-committed escalation order is :data:`FALLBACK_LADDER`: (1) swap the f_ICE
source from the local-only component to the raw/total field variance,
(2) use larger structural edits, (3) quench to a shallower ``T``. Each rung
is a distinct lever on the achievable contrast, tried in order before the
experiment is abandoned.

No numpy / no scipy: the whole module is pure ``math`` (the normal quantile
is Acklam's rational approximation), so it is deterministic and dependency-
free. It imports :mod:`butterfly_cone.stats.varcomp` and :mod:`butterfly_cone.stats.estimands`
read-only, only in the convenience adapters that pull ``f_ICE`` and the
observed effect out of already-fitted pilot objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from .estimands import ContrastResult
    from .varcomp import VarianceComponents


# ---------------------------------------------------------------------------
# Declared constants (Gate-0 policy numbers).
# ---------------------------------------------------------------------------

GATE0_THRESHOLD = 0.15
"""Gate-0 target effect: the campaign must be able to detect ``E[Δq] >= 0.15``."""

DEFAULT_POWER = 0.9
DEFAULT_ALPHA = 0.05

DEFAULT_N_ARMS = 4
"""Arm-cells in ``D_target-random``: {+, -} signs x {targeted, random} families."""

DEFAULT_CEILING_COEFFICIENT = 2.0
"""Structural swing (in SDs) the field-targeted arm is credited with reaching.

The achievable-effect ceiling is ``C * sqrt(f_ICE * Var_total)``; ``C = 2``
corresponds to a full +/-1 structural standard-deviation swing between the
high-field and low-field arms (the random comparator contributes ~0 mean, so
the ceiling on ``D`` is the ceiling on the targeted arm). This is deliberately
optimistic: ``NO-GO-field-too-weak`` should fire only when even the best case
cannot reach the threshold.
"""


# ---------------------------------------------------------------------------
# Fallback ladder (raw -> total field, larger edits, shallower T).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackRung:
    """One rung of the declared escalation ladder."""

    name: str
    action: str
    rationale: str


FALLBACK_LADDER: tuple[FallbackRung, ...] = (
    FallbackRung(
        name="raw_to_total_field",
        action="Re-source f_ICE from the raw / total-field variance instead of the "
        "local-only (iso-configurational) component.",
        rationale="The total-field estimand adds the environmental structure the "
        "local-only component discards, raising the measured predictability ceiling "
        "when the local component alone is too tight.",
    ),
    FallbackRung(
        name="larger_edits",
        action="Increase the perturbation / edit magnitude of the field-targeted arm.",
        rationale="A larger structural edit widens the high-field vs. low-field "
        "separation, lifting the achievable E[Δq] toward the Gate-0 threshold.",
    ),
    FallbackRung(
        name="shallower_T",
        action="Quench to a shallower effective temperature / inherent-structure depth.",
        rationale="At shallower T the field's coupling to fate is stronger and less "
        "thermally washed out, so more of the outcome variance becomes structural "
        "(f_ICE rises).",
    ),
)


class Verdict(str, enum.Enum):
    """The three Gate-0 outcomes."""

    GO = "GO"
    GO_WITH_MORE_BRANCHES = "GO-with-more-branches"
    NO_GO_FIELD_TOO_WEAK = "NO-GO-field-too-weak"


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _validate_base_rate(base_rate: float) -> float:
    value = float(base_rate)
    if not 0.0 < value < 1.0:
        raise ValueError(f"base_rate must be strictly between 0 and 1, got {base_rate!r}")
    return value


def _validate_unit_open(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return result


def _validate_positive_float(value: float, name: str) -> float:
    result = float(value)
    if not result > 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _validate_fraction(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return result


# ---------------------------------------------------------------------------
# Normal quantile (Acklam's rational approximation; no scipy/numpy).
# ---------------------------------------------------------------------------


def norm_ppf(p: float) -> float:
    """Standard-normal quantile: Acklam's rational approximation + one Halley step.

    Acklam's bare rational approximation is accurate to ~1.15e-9 in *relative*
    error; a single Halley refinement using the exact normal CDF (``math.erfc``)
    and PDF drives that to full double precision (absolute error ~1e-15 over the
    supported range).  Used only for the two critical values ``z_{1-alpha/2}``
    and ``z_{1-beta}``.  Deliberately dependency-free (stdlib ``math`` only) so
    the gate returns identical numbers whether or not numpy/scipy are installed.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    # One Halley refinement step (Acklam): correct the rational approximation
    # ``x`` using the exact normal CDF ``Phi(x) = 0.5 * erfc(-x / sqrt(2))`` and
    # PDF ``phi(x) = exp(-x^2/2) / sqrt(2*pi)``.  This makes the ~1e-9 accuracy
    # claim honest by pushing the residual to full double precision.
    residual = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p          # Phi(x) - p
    u = residual * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)  # residual / phi(x)
    return x - u / (1.0 + 0.5 * x * u)


def z_factor(power: float = DEFAULT_POWER, alpha: float = DEFAULT_ALPHA) -> float:
    """The ``z_{1-alpha/2} + z_{1-beta}`` multiplier of the MDE formula."""
    power = _validate_unit_open(power, "power")
    alpha = _validate_unit_open(alpha, "alpha")
    return norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)


# ---------------------------------------------------------------------------
# Core: MDE, budget inversion, achievable-effect ceiling.
# ---------------------------------------------------------------------------


def standard_error(
    n_pairs: int,
    n_branches: int,
    base_rate: float,
    n_arms: int = DEFAULT_N_ARMS,
) -> float:
    """SE of D̂ under the pooled-null branch-Bernoulli model.

    ``SE = sqrt(n_arms * p(1-p) / (n_pairs * n_branches))``.
    """
    n_pairs = _validate_positive_int(n_pairs, "n_pairs")
    n_branches = _validate_positive_int(n_branches, "n_branches")
    n_arms = _validate_positive_int(n_arms, "n_arms")
    base_rate = _validate_base_rate(base_rate)
    variance = n_arms * base_rate * (1.0 - base_rate) / (n_pairs * n_branches)
    return math.sqrt(variance)


def mde(
    n_pairs: int,
    n_branches: int,
    base_rate: float,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    n_arms: int = DEFAULT_N_ARMS,
) -> float:
    """Minimum detectable ``D_target-random`` effect at the given design.

    ``MDE = (z_{1-alpha/2} + z_{1-beta}) * sqrt(n_arms * p(1-p)
    / (n_pairs * n_branches))``. Decreases as ``1/sqrt(n_pairs)`` and
    ``1/sqrt(n_branches)``.
    """
    return z_factor(power, alpha) * standard_error(n_pairs, n_branches, base_rate, n_arms)


@dataclass(frozen=True)
class BudgetRequirement:
    """Branches-per-cell needed to reach ``MDE <= target_effect`` at fixed pairs."""

    target_effect: float
    n_pairs: int
    base_rate: float
    power: float
    alpha: float
    n_arms: int
    n_branches_exact: float
    n_branches: int
    total_branches: int
    mde_at_budget: float

    @property
    def is_detectable(self) -> bool:
        return self.mde_at_budget <= self.target_effect


def required_budget(
    target_effect: float,
    n_pairs: int,
    base_rate: float,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    n_arms: int = DEFAULT_N_ARMS,
) -> BudgetRequirement:
    """Invert the MDE formula for branches-per-cell at a fixed ``n_pairs``.

    ``n_branches = z^2 * n_arms * p(1-p) / (n_pairs * target_effect^2)`` where
    ``z = z_{1-alpha/2} + z_{1-beta}``. Returns the exact real solution, the
    ceiling integer (which guarantees ``MDE <= target_effect``), the resulting
    total branch budget ``n_pairs * n_arms * n_branches``, and the MDE actually
    achieved at the ceiling integer.
    """
    target_effect = _validate_positive_float(target_effect, "target_effect")
    n_pairs = _validate_positive_int(n_pairs, "n_pairs")
    n_arms = _validate_positive_int(n_arms, "n_arms")
    base_rate = _validate_base_rate(base_rate)
    z = z_factor(power, alpha)
    n_branches_exact = (
        z * z * n_arms * base_rate * (1.0 - base_rate) / (n_pairs * target_effect * target_effect)
    )
    n_branches = max(1, math.ceil(n_branches_exact))
    total_branches = n_pairs * n_arms * n_branches
    mde_at_budget = mde(n_pairs, n_branches, base_rate, power, alpha, n_arms)
    return BudgetRequirement(
        target_effect=target_effect,
        n_pairs=n_pairs,
        base_rate=base_rate,
        power=float(power),
        alpha=float(alpha),
        n_arms=n_arms,
        n_branches_exact=n_branches_exact,
        n_branches=n_branches,
        total_branches=total_branches,
        mde_at_budget=mde_at_budget,
    )


def ceiling_effect(
    f_ice: float,
    base_rate: float,
    ceiling_coefficient: float = DEFAULT_CEILING_COEFFICIENT,
    total_variance: float | None = None,
) -> float:
    """Optimistic ceiling on the true ``E[Δq]`` the field can produce.

    ``ceiling = C * sqrt(f_ICE * Var_total)``. ``Var_total`` defaults to the
    binary-outcome marginal variance ``base_rate*(1-base_rate)``; pass the
    variance-component total from :class:`~butterfly_cone.stats.varcomp.VarianceComponents`
    for an exact structural SD in outcome units.
    """
    f_ice = _validate_fraction(f_ice, "f_ice")
    base_rate = _validate_base_rate(base_rate)
    ceiling_coefficient = _validate_positive_float(ceiling_coefficient, "ceiling_coefficient")
    if total_variance is None:
        variance_total = base_rate * (1.0 - base_rate)
    else:
        variance_total = float(total_variance)
        if variance_total < 0.0:
            raise ValueError(f"total_variance must be non-negative, got {total_variance!r}")
    return ceiling_coefficient * math.sqrt(f_ice * variance_total)


# ---------------------------------------------------------------------------
# The gate verdict.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerGateVerdict:
    """Full Gate-0 decision record."""

    verdict: Verdict
    f_ice: float
    ceiling_effect: float
    target_effect: float
    mde_current: float
    n_pairs: int
    n_branches: int
    n_arms: int
    required_n_branches: int | None
    additional_n_branches: int | None
    total_branch_budget: int | None
    feasible: bool
    rationale: str
    fallback_ladder: tuple[FallbackRung, ...]

    @property
    def is_go(self) -> bool:
        """True for GO or GO-with-more-branches (i.e. not a NO-GO)."""
        return self.verdict is not Verdict.NO_GO_FIELD_TOO_WEAK


def power_gate_verdict(
    f_ice: float,
    n_pairs: int,
    n_branches: int,
    base_rate: float,
    target_effect: float = GATE0_THRESHOLD,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    n_arms: int = DEFAULT_N_ARMS,
    ceiling_coefficient: float = DEFAULT_CEILING_COEFFICIENT,
    total_variance: float | None = None,
    max_branches: int | None = None,
) -> PowerGateVerdict:
    """Return the Gate-0 verdict: GO / GO-with-more-branches / NO-GO-field-too-weak.

    - ``NO-GO-field-too-weak``: the optimistic structural ceiling
      ``C * sqrt(f_ICE * Var)`` is below ``target_effect`` -- no branch budget
      can rescue it; the caller should walk :data:`FALLBACK_LADDER`.
    - ``GO``: the field can clear the bar and the *current* ``(n_pairs,
      n_branches)`` design already gives ``MDE <= target_effect``.
    - ``GO-with-more-branches``: the field can clear the bar but the current
      design is under-powered; ``required_n_branches`` / ``additional_n_branches``
      say how many branches-per-cell are needed (``feasible`` reports whether
      that stays within ``max_branches`` if a cap was supplied).
    """
    f_ice = _validate_fraction(f_ice, "f_ice")
    base_rate = _validate_base_rate(base_rate)
    target_effect = _validate_positive_float(target_effect, "target_effect")
    n_pairs = _validate_positive_int(n_pairs, "n_pairs")
    n_branches = _validate_positive_int(n_branches, "n_branches")
    n_arms = _validate_positive_int(n_arms, "n_arms")
    if max_branches is not None:
        max_branches = _validate_positive_int(max_branches, "max_branches")

    ceiling = ceiling_effect(f_ice, base_rate, ceiling_coefficient, total_variance)
    mde_current = mde(n_pairs, n_branches, base_rate, power, alpha, n_arms)

    if ceiling < target_effect:
        rationale = (
            f"Structural ceiling {ceiling:.4f} < target {target_effect:.4f}: even a "
            f"full {ceiling_coefficient:g}-SD field swing at f_ICE={f_ice:.4f} cannot "
            f"reach the Gate-0 threshold. No branch budget can fix a too-weak field; "
            f"escalate the fallback ladder (raw->total field, larger edits, shallower T)."
        )
        return PowerGateVerdict(
            verdict=Verdict.NO_GO_FIELD_TOO_WEAK,
            f_ice=f_ice,
            ceiling_effect=ceiling,
            target_effect=target_effect,
            mde_current=mde_current,
            n_pairs=n_pairs,
            n_branches=n_branches,
            n_arms=n_arms,
            required_n_branches=None,
            additional_n_branches=None,
            total_branch_budget=None,
            feasible=False,
            rationale=rationale,
            fallback_ladder=FALLBACK_LADDER,
        )

    if mde_current <= target_effect:
        rationale = (
            f"Ceiling {ceiling:.4f} >= target {target_effect:.4f} and current MDE "
            f"{mde_current:.4f} <= target: the design already detects the Gate-0 "
            f"effect at n_pairs={n_pairs}, n_branches={n_branches}."
        )
        return PowerGateVerdict(
            verdict=Verdict.GO,
            f_ice=f_ice,
            ceiling_effect=ceiling,
            target_effect=target_effect,
            mde_current=mde_current,
            n_pairs=n_pairs,
            n_branches=n_branches,
            n_arms=n_arms,
            required_n_branches=n_branches,
            additional_n_branches=0,
            total_branch_budget=n_pairs * n_arms * n_branches,
            feasible=True,
            rationale=rationale,
            fallback_ladder=FALLBACK_LADDER,
        )

    budget = required_budget(target_effect, n_pairs, base_rate, power, alpha, n_arms)
    additional = max(0, budget.n_branches - n_branches)
    feasible = max_branches is None or budget.n_branches <= max_branches
    cap_note = "" if max_branches is None else f" (cap {max_branches})"
    rationale = (
        f"Ceiling {ceiling:.4f} >= target {target_effect:.4f} but current MDE "
        f"{mde_current:.4f} > target: field is strong enough, design is under-powered. "
        f"Need {budget.n_branches} branches/cell (+{additional} more) at n_pairs="
        f"{n_pairs}{cap_note}."
    )
    return PowerGateVerdict(
        verdict=Verdict.GO_WITH_MORE_BRANCHES,
        f_ice=f_ice,
        ceiling_effect=ceiling,
        target_effect=target_effect,
        mde_current=mde_current,
        n_pairs=n_pairs,
        n_branches=n_branches,
        n_arms=n_arms,
        required_n_branches=budget.n_branches,
        additional_n_branches=additional,
        total_branch_budget=budget.total_branches,
        feasible=feasible,
        rationale=rationale,
        fallback_ladder=FALLBACK_LADDER,
    )


# ---------------------------------------------------------------------------
# Read-only adapters over already-fitted pilot objects (varcomp / estimands).
# ---------------------------------------------------------------------------


def f_ice_from_components(components: "VarianceComponents") -> float:
    """Extract f_ICE = the local (iso-configurational) predictability fraction.

    ``f_ICE`` is exactly ``varcomp``'s bias-corrected local causal fraction
    ``f_local`` -- the share of outcome variance that is structurally
    determined rather than environmental or thermal.
    """
    return float(components.f_local)


def observed_effect_and_pairs_from_contrast(contrast: "ContrastResult") -> tuple[float, int]:
    """Pull the observed ``D_target-random`` point estimate and ``n_pairs``.

    Uses the pair-weighted contrast (the primary estimate) and the targeted
    family's pair count from an :class:`~butterfly_cone.stats.estimands.ContrastResult`.
    """
    return float(contrast.point_pair_weighted), int(contrast.target.n_pairs)


def verdict_from_pilot(
    components: "VarianceComponents",
    n_pairs: int,
    n_branches: int,
    base_rate: float,
    target_effect: float = GATE0_THRESHOLD,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    n_arms: int = DEFAULT_N_ARMS,
    ceiling_coefficient: float = DEFAULT_CEILING_COEFFICIENT,
    use_component_variance: bool = True,
    max_branches: int | None = None,
) -> PowerGateVerdict:
    """Run the gate straight off a fitted :class:`VarianceComponents`.

    Reads ``f_ICE = components.f_local`` and, when ``use_component_variance``
    is set, the exact variance-component total ``components.total_variance``
    for the structural ceiling (falling back to ``base_rate*(1-base_rate)``
    otherwise).
    """
    f_ice = f_ice_from_components(components)
    total_variance = float(components.total_variance) if use_component_variance else None
    return power_gate_verdict(
        f_ice=f_ice,
        n_pairs=n_pairs,
        n_branches=n_branches,
        base_rate=base_rate,
        target_effect=target_effect,
        power=power,
        alpha=alpha,
        n_arms=n_arms,
        ceiling_coefficient=ceiling_coefficient,
        total_variance=total_variance,
        max_branches=max_branches,
    )


__all__ = [
    "GATE0_THRESHOLD",
    "DEFAULT_POWER",
    "DEFAULT_ALPHA",
    "DEFAULT_N_ARMS",
    "DEFAULT_CEILING_COEFFICIENT",
    "FALLBACK_LADDER",
    "FallbackRung",
    "Verdict",
    "BudgetRequirement",
    "PowerGateVerdict",
    "norm_ppf",
    "z_factor",
    "standard_error",
    "mde",
    "required_budget",
    "ceiling_effect",
    "power_gate_verdict",
    "f_ice_from_components",
    "observed_effect_and_pairs_from_contrast",
    "verdict_from_pilot",
]
