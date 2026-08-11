"""Multiple-comparison control + confirmatory/exploratory claim registry.

Paper-rigor layer for the ~18 adopted rider claims.  A Nature referee will
reject a paper that "tested 18 things, some pass by chance" unless the
family-wise / false-discovery error is explicitly controlled *and* the
confirmatory-vs-exploratory split was frozen before the p-values were seen.
This module supplies both halves.

Corrections
-----------
* :func:`holm_bonferroni` -- the step-down Holm procedure, controlling the
  family-wise error rate (FWER, the probability of *any* false rejection) at
  ``alpha`` under arbitrary dependence between the tests.
* :func:`benjamini_hochberg` -- the step-up Benjamini-Hochberg procedure,
  controlling the false-discovery rate (FDR, the expected fraction of false
  rejections among rejections) at ``alpha`` under independence / PRDS.

Both return per-hypothesis reject/accept decisions and *adjusted p-values*
(the smallest family ``alpha`` at which the hypothesis would be rejected),
computed with the exact textbook running-extremum monotonicity enforcement so
that reject ``<=> adjusted_p <= alpha`` reproduces the step-down / step-up
rule exactly, ties included.

Registry
--------
* :class:`ConfirmatoryRegistry` -- an append-only partition of claims into
  ``CONFIRMATORY`` (declared in advance, FWER-controlled via Holm) and
  ``EXPLORATORY`` (FDR-controlled, reported as hypothesis-generating).  Each
  claim is registered ``(claim_id, family, direction, alpha)`` *before* any
  p-value exists.  Once the registry has been evaluated -- i.e. results have
  been seen -- it is frozen, and a claim can never afterwards be moved
  confirmatory<->exploratory.  Re-registering an existing ``claim_id`` at any
  time is refused.  This mirrors the append-only, hash-chained decision ledger
  in :mod:`butterfly_cone.harness.ledger`.

* :func:`evaluate` -- applies Holm to the confirmatory family and BH to the
  exploratory family (each at its family's registered ``alpha``), freezes the
  registry, and returns per-claim verdicts, a summary table, and the headline
  count of confirmatory claims that survive FWER control.

The ``direction`` field (``"greater"`` / ``"less"`` / ``"two-sided"``) is
recorded for the advance-declaration record; the correction math consumes only the
p-value, which is assumed already computed for that registered direction.

Pure analysis over p-values: numpy only, no simulation, no allocation, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray

# Family labels (frozen vocabulary).
CONFIRMATORY = "confirmatory"
EXPLORATORY = "exploratory"
_FAMILIES = frozenset({CONFIRMATORY, EXPLORATORY})

# Direction labels (descriptive; the p-value already encodes the direction).
GREATER = "greater"
LESS = "less"
TWO_SIDED = "two-sided"
_DIRECTIONS = frozenset({GREATER, LESS, TWO_SIDED})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Base class for confirmatory-registry discipline violations."""


class ClaimRevisionError(RegistryError):
    """Raised on any attempt to re-register / revise an existing claim."""


class RegistryFrozenError(RegistryError):
    """Raised on any attempt to register after the registry has been frozen."""


class UnknownClaimError(RegistryError):
    """Raised when a p-value set does not match the registered claim ids."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"alpha must lie in the open interval (0, 1); got {alpha!r}")
    return value


def _as_pvalue_array(pvalues) -> np.ndarray:
    """Coerce ``pvalues`` to a validated 1-D float array in ``[0, 1]``."""
    array = np.asarray(pvalues, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"p-values must be a 1-D sequence, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("p-values must all be finite")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    return array


# ---------------------------------------------------------------------------
# Correction result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionResult:
    """Per-hypothesis output of a multiple-comparison correction.

    ``pvalues``, ``adjusted`` and ``reject`` are aligned tuples in the *input*
    order (not sorted).  ``adjusted[i]`` is the smallest family ``alpha`` at
    which hypothesis ``i`` is rejected, so ``reject[i] == (adjusted[i] <=
    alpha)`` by construction.
    """

    method: str
    alpha: float
    pvalues: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]

    @property
    def n(self) -> int:
        return len(self.pvalues)

    @property
    def n_reject(self) -> int:
        return int(sum(self.reject))

    def adjusted_array(self) -> np.ndarray:
        return np.asarray(self.adjusted, dtype=float)

    def reject_array(self) -> np.ndarray:
        return np.asarray(self.reject, dtype=bool)


def _empty_result(method: str, alpha: float) -> CorrectionResult:
    return CorrectionResult(
        method=method, alpha=alpha, pvalues=(), adjusted=(), reject=()
    )


# ---------------------------------------------------------------------------
# Holm-Bonferroni (step-down, FWER)
# ---------------------------------------------------------------------------


def holm_bonferroni(pvalues, alpha: float = 0.05) -> CorrectionResult:
    """Holm's step-down procedure controlling the FWER at ``alpha``.

    Let ``p_(1) <= ... <= p_(m)`` be the ordered p-values.  The adjusted
    p-value at ordered rank ``i`` (1-indexed) is the running maximum

        p~_(i) = max_{j<=i} min{ (m - j + 1) * p_(j), 1 },

    and hypothesis ``(i)`` is rejected iff ``p~_(i) <= alpha``.  Because the
    adjusted values are non-decreasing in rank, the first ordered acceptance
    stops all further rejections -- exactly the step-down rule.  Valid under
    *arbitrary* dependence between the tests.
    """
    alpha = _check_alpha(alpha)
    p = _as_pvalue_array(pvalues)
    m = p.size
    if m == 0:
        return _empty_result("holm-bonferroni", alpha)

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    multipliers = m - np.arange(m)  # (m, m-1, ..., 1)
    raw = np.minimum(multipliers * p_sorted, 1.0)
    adj_sorted = np.maximum.accumulate(raw)  # enforce monotone non-decreasing

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return CorrectionResult(
        method="holm-bonferroni",
        alpha=alpha,
        pvalues=tuple(float(x) for x in p),
        adjusted=tuple(float(x) for x in adjusted),
        reject=tuple(bool(x) for x in reject),
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg (step-up, FDR)
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg step-up procedure controlling the FDR at ``alpha``.

    Let ``p_(1) <= ... <= p_(m)`` be the ordered p-values.  The adjusted
    p-value (BH q-value) at ordered rank ``i`` (1-indexed) is the running
    minimum taken from the largest rank downward,

        p~_(i) = min_{j>=i} min{ (m / j) * p_(j), 1 },

    and hypothesis ``(i)`` is rejected iff ``p~_(i) <= alpha``.  This is
    equivalent to rejecting ``(1), ..., (k*)`` where ``k*`` is the largest
    ``k`` with ``p_(k) <= (k / m) * alpha`` -- the step-up rule.  Valid under
    independence (and positive-regression-dependent, PRDS, structures).
    """
    alpha = _check_alpha(alpha)
    p = _as_pvalue_array(pvalues)
    m = p.size
    if m == 0:
        return _empty_result("benjamini-hochberg", alpha)

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    raw = p_sorted * m / ranks
    # Running minimum from the right enforces monotone non-decreasing q-values.
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return CorrectionResult(
        method="benjamini-hochberg",
        alpha=alpha,
        pvalues=tuple(float(x) for x in p),
        adjusted=tuple(float(x) for x in adjusted),
        reject=tuple(bool(x) for x in reject),
    )


# ---------------------------------------------------------------------------
# Confirmatory / exploratory registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimSpec:
    """A single declared in advance claim: identity, family, direction, level."""

    claim_id: str
    family: str
    direction: str
    alpha: float


class ConfirmatoryRegistry:
    """Append-only partition of claims into confirmatory / exploratory.

    Register every claim *before* any p-value exists.  The registry refuses to
    revise an existing claim (re-registering the same ``claim_id`` raises
    :class:`ClaimRevisionError`), and :func:`evaluate` freezes it once results
    have been seen, after which registration raises :class:`RegistryFrozenError`
    -- so a claim can never be moved confirmatory<->exploratory after the fact.
    """

    def __init__(self) -> None:
        self._claims: dict[str, ClaimSpec] = {}
        self._frozen = False

    # -- registration -----------------------------------------------------

    def register(
        self,
        claim_id: str,
        family: str,
        direction: str = TWO_SIDED,
        alpha: float = 0.05,
    ) -> ClaimSpec:
        """Register a new claim; append-only, refused once frozen."""
        if self._frozen:
            raise RegistryFrozenError(
                f"registry is frozen (results seen); cannot register {claim_id!r} "
                "or move any claim confirmatory<->exploratory"
            )
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("claim_id must be a non-empty string")
        if family not in _FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(_FAMILIES)}; got {family!r}"
            )
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_DIRECTIONS)}; got {direction!r}"
            )
        alpha = _check_alpha(alpha)
        if claim_id in self._claims:
            existing = self._claims[claim_id]
            raise ClaimRevisionError(
                f"claim {claim_id!r} already registered as {existing.family!r}; "
                "the registry is append-only and refuses revision"
            )
        spec = ClaimSpec(claim_id=claim_id, family=family, direction=direction, alpha=alpha)
        self._claims[claim_id] = spec
        return spec

    def register_many(self, specs) -> None:
        """Register an iterable of ``(claim_id, family, direction, alpha)`` rows."""
        for row in specs:
            if isinstance(row, ClaimSpec):
                self.register(row.claim_id, row.family, row.direction, row.alpha)
            else:
                self.register(*row)

    # -- freezing ----------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the partition (idempotent).  Called by :func:`evaluate`."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # -- access -----------------------------------------------------------

    @property
    def claims(self) -> tuple[ClaimSpec, ...]:
        """All claims in registration order."""
        return tuple(self._claims.values())

    def family(self, name: str) -> tuple[ClaimSpec, ...]:
        if name not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}; got {name!r}")
        return tuple(s for s in self._claims.values() if s.family == name)

    @property
    def confirmatory(self) -> tuple[ClaimSpec, ...]:
        return self.family(CONFIRMATORY)

    @property
    def exploratory(self) -> tuple[ClaimSpec, ...]:
        return self.family(EXPLORATORY)

    def __contains__(self, claim_id: object) -> bool:
        return claim_id in self._claims

    def __len__(self) -> int:
        return len(self._claims)

    def __iter__(self):
        return iter(self._claims.values())

    def __getitem__(self, claim_id: str) -> ClaimSpec:
        return self._claims[claim_id]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-claim result after the family-correct correction is applied."""

    claim_id: str
    family: str
    direction: str
    alpha: float
    raw_p: float
    adjusted_p: float
    reject: bool


@dataclass(frozen=True)
class Evaluation:
    """Full evaluation of a registry against a p-value set.

    ``verdicts`` are in registration order.  ``confirmatory`` /
    ``exploratory`` are the underlying :class:`CorrectionResult`\\ s (Holm and
    BH respectively), or ``None`` if that family had no claims.
    """

    verdicts: tuple[ClaimVerdict, ...]
    confirmatory: CorrectionResult | None
    exploratory: CorrectionResult | None

    @property
    def n_confirmatory(self) -> int:
        return sum(1 for v in self.verdicts if v.family == CONFIRMATORY)

    @property
    def n_confirmatory_survive(self) -> int:
        return sum(
            1 for v in self.verdicts if v.family == CONFIRMATORY and v.reject
        )

    @property
    def n_exploratory(self) -> int:
        return sum(1 for v in self.verdicts if v.family == EXPLORATORY)

    @property
    def n_exploratory_flagged(self) -> int:
        return sum(1 for v in self.verdicts if v.family == EXPLORATORY and v.reject)

    @property
    def headline(self) -> str:
        if self.confirmatory is None:
            return "0 of 0 confirmatory claims survive FWER control (none registered)"
        return (
            f"{self.n_confirmatory_survive} of {self.n_confirmatory} confirmatory "
            f"claims survive FWER control (Holm, alpha={self.confirmatory.alpha:g})"
        )

    def by_claim(self) -> dict[str, ClaimVerdict]:
        return {v.claim_id: v for v in self.verdicts}

    def table(self) -> tuple[dict, ...]:
        """The summary table: one row per claim with the reported columns."""
        return tuple(
            {
                "claim": v.claim_id,
                "family": v.family,
                "direction": v.direction,
                "raw_p": v.raw_p,
                "adj_p": v.adjusted_p,
                "reject": v.reject,
            }
            for v in self.verdicts
        )

    def summary(self) -> str:
        """A monospace, human-readable rendering of :meth:`table`."""
        header = f"{'claim':<20} {'family':<12} {'raw_p':>10} {'adj_p':>10} {'reject':>7}"
        lines = [header, "-" * len(header)]
        for v in self.verdicts:
            lines.append(
                f"{v.claim_id:<20} {v.family:<12} {v.raw_p:>10.4g} "
                f"{v.adjusted_p:>10.4g} {str(v.reject):>7}"
            )
        lines.append("")
        lines.append(self.headline)
        return "\n".join(lines)


def _family_alpha(specs: list[ClaimSpec]) -> float:
    """The single ``alpha`` shared by a family; the whole family is corrected
    at one level, so mixed levels within a family are refused."""
    alphas = sorted({s.alpha for s in specs})
    if len(alphas) != 1:
        raise ValueError(
            "all claims in a family must share one alpha for family-level "
            f"correction; got {alphas}"
        )
    return alphas[0]


def _apply_family(specs, pvalues: Mapping[str, float], correction):
    """Run ``correction`` over a family; return its result and a per-claim map."""
    if not specs:
        return None, {}
    alpha = _family_alpha(list(specs))
    ids = [s.claim_id for s in specs]
    ps = [float(pvalues[cid]) for cid in ids]
    result = correction(ps, alpha=alpha)
    per = {
        cid: (adj, rej)
        for cid, adj, rej in zip(ids, result.adjusted, result.reject)
    }
    return result, per


def evaluate(registry: ConfirmatoryRegistry, pvalues: Mapping[str, float]) -> Evaluation:
    """Evaluate a registry against a p-value set and freeze it.

    ``pvalues`` is a mapping ``claim_id -> raw p-value`` whose keys must match
    the registered claim ids exactly.  Holm is applied to the confirmatory
    family and BH to the exploratory family, each at that family's registered
    ``alpha``.  Freezing the registry here is what makes a post-hoc
    confirmatory<->exploratory move impossible (see
    :class:`ConfirmatoryRegistry`).
    """
    if not isinstance(pvalues, Mapping):
        raise TypeError("pvalues must be a mapping of claim_id -> p-value")

    claims = registry.claims
    registered = {c.claim_id for c in claims}
    provided = set(pvalues)
    if provided != registered:
        missing = registered - provided
        extra = provided - registered
        raise UnknownClaimError(
            "p-value keys must match the registered claims exactly; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    for cid in registered:
        # Reuse the array validator on the single value (range/finiteness).
        _as_pvalue_array([pvalues[cid]])

    registry.freeze()

    conf_specs = [c for c in claims if c.family == CONFIRMATORY]
    expl_specs = [c for c in claims if c.family == EXPLORATORY]
    conf_result, conf_per = _apply_family(conf_specs, pvalues, holm_bonferroni)
    expl_result, expl_per = _apply_family(expl_specs, pvalues, benjamini_hochberg)

    verdicts: list[ClaimVerdict] = []
    for spec in claims:
        adj, rej = (conf_per if spec.family == CONFIRMATORY else expl_per)[spec.claim_id]
        verdicts.append(
            ClaimVerdict(
                claim_id=spec.claim_id,
                family=spec.family,
                direction=spec.direction,
                alpha=spec.alpha,
                raw_p=float(pvalues[spec.claim_id]),
                adjusted_p=float(adj),
                reject=bool(rej),
            )
        )

    return Evaluation(
        verdicts=tuple(verdicts),
        confirmatory=conf_result,
        exploratory=expl_result,
    )


__all__ = [
    "CONFIRMATORY",
    "EXPLORATORY",
    "GREATER",
    "LESS",
    "TWO_SIDED",
    "RegistryError",
    "ClaimRevisionError",
    "RegistryFrozenError",
    "UnknownClaimError",
    "CorrectionResult",
    "holm_bonferroni",
    "benjamini_hochberg",
    "ClaimSpec",
    "ConfirmatoryRegistry",
    "ClaimVerdict",
    "Evaluation",
    "evaluate",
]
