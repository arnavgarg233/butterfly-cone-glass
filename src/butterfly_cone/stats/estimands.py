"""Per-parent-cavity-pair causal contrasts (PLAN_v2.1 §5, §21).

Estimand -> plan-section map (restated in the module README):

- `paired_contrasts` / `estimate_ate`            -> Claim C, §21.2 (Delta q_c, E[Delta q])
- `estimate_ate` on an H1 endpoint column        -> §21.3 (ATE_targeted / ATE_random)
- `family_contrast` (target="targeted", comparator="random")
                                                  -> §21.4 negative control, D_target-random
- `family_contrast` (comparator="softness")      -> §21 secondary, D_S
- `family_contrast` (comparator="best_incumbent")-> §21 secondary, D_B

The exchangeability unit is always the parent-cavity pair: every function
below reduces each pair to exactly one number before any averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._common import (
    ARM_FAMILY_COL,
    ARM_SIGN_COL,
    PAIR_ID_COL,
    BranchTable,
    SIGNS,
    as01,
    group_by,
    require_columns,
)

Weighting = Literal["pair", "branch"]
OnIncomplete = Literal["raise", "drop"]


@dataclass(frozen=True)
class PairArmCounts:
    """Branch-level (k, n) counts for one (pair, family, sign)."""

    k: int
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not (0 <= self.k <= self.n):
            raise ValueError(f"k={self.k} out of range for n={self.n}")

    @property
    def q_hat(self) -> float:
        return self.k / self.n


@dataclass(frozen=True)
class PairContrast:
    """One parent-cavity pair's paired contrast for a given family/endpoint."""

    pair_id: str
    arm_family: str
    plus: PairArmCounts
    minus: PairArmCounts

    @property
    def q_plus(self) -> float:
        return self.plus.q_hat

    @property
    def q_minus(self) -> float:
        return self.minus.q_hat

    @property
    def delta(self) -> float:
        """Delta q_c = q_hat(X_c+) - q_hat(X_c-)."""
        return self.q_plus - self.q_minus

    @property
    def n_total(self) -> int:
        return self.plus.n + self.minus.n


def collect_pair_arm_counts(
    table: BranchTable, endpoint_col: str, arm_family: str
) -> dict[str, dict[str, PairArmCounts]]:
    """Aggregate branch rows into per-(pair, sign) (k, n) counts.

    Only rows with ``arm_family == arm_family`` are considered. Every row
    used must carry a binary value in `endpoint_col` and a sign in `SIGNS`.
    """
    counts: dict[str, dict[str, list[int]]] = {}
    for row in table:
        require_columns(row, (PAIR_ID_COL, ARM_FAMILY_COL, ARM_SIGN_COL, endpoint_col))
        if row[ARM_FAMILY_COL] != arm_family:
            continue
        sign = row[ARM_SIGN_COL]
        if sign not in SIGNS:
            raise ValueError(f"arm_sign must be one of {SIGNS}, got {sign!r}")
        pair_id = row[PAIR_ID_COL]
        outcome = as01(row[endpoint_col])
        bucket = counts.setdefault(pair_id, {"+": [0, 0], "-": [0, 0]})
        bucket[sign][0] += outcome
        bucket[sign][1] += 1
    return {
        pair_id: {sign: PairArmCounts(k=k, n=n) for sign, (k, n) in signs.items() if n > 0}
        for pair_id, signs in counts.items()
    }


def paired_contrasts(
    table: BranchTable,
    endpoint_col: str,
    arm_family: str,
    on_incomplete: OnIncomplete = "raise",
) -> list[PairContrast]:
    """Build one `PairContrast` per parent-cavity pair with both arms present.

    `on_incomplete="raise"` (default) fails loudly if any pair in
    `arm_family` is missing a `+` or `-` arm, a silently-dropped pair is
    exactly the kind of bug this module exists to prevent. Pass `"drop"` to
    instead skip incomplete pairs (e.g. mid-campaign snapshots where the
    adaptive allocator has not yet run both arms of every pair).
    """
    counts = collect_pair_arm_counts(table, endpoint_col, arm_family)
    contrasts: list[PairContrast] = []
    incomplete: list[str] = []
    for pair_id, signs in counts.items():
        if "+" not in signs or "-" not in signs:
            incomplete.append(pair_id)
            continue
        contrasts.append(
            PairContrast(pair_id=pair_id, arm_family=arm_family, plus=signs["+"], minus=signs["-"])
        )
    if incomplete and on_incomplete == "raise":
        raise ValueError(
            f"pairs missing a '+' or '-' arm for family {arm_family!r}, endpoint "
            f"{endpoint_col!r}: {sorted(incomplete)}"
        )
    contrasts.sort(key=lambda c: c.pair_id)
    return contrasts


def mean_delta(contrasts: list[PairContrast], weighting: Weighting = "pair") -> float:
    """Average Delta q_c across pairs.

    - ``"pair"`` (primary, per the task's statistical-care note): every pair
      counts once regardless of how many branches it carries. This is the
      point estimate that should be reported as *the* result.
    - ``"branch"``: each pair's Delta q_c is weighted by its total branch
      count (``n_total``), so pairs the adaptive allocator gave more
      branches to (because they were more uncertain) pull the average
      harder. Report alongside the pair-weighted estimate only when they
      differ meaningfully; never substitute it for the primary estimate.
    """
    if not contrasts:
        raise ValueError("no contrasts to average")
    if weighting == "pair":
        return sum(c.delta for c in contrasts) / len(contrasts)
    if weighting == "branch":
        total_weight = sum(c.n_total for c in contrasts)
        return sum(c.delta * c.n_total for c in contrasts) / total_weight
    raise ValueError(f"unknown weighting {weighting!r}")


@dataclass(frozen=True)
class ATEEstimate:
    arm_family: str
    endpoint_col: str
    n_pairs: int
    pair_weighted: float
    branch_weighted: float
    contrasts: tuple[PairContrast, ...]


def estimate_ate(
    table: BranchTable,
    endpoint_col: str,
    arm_family: str,
    on_incomplete: OnIncomplete = "raise",
) -> ATEEstimate:
    """E[Delta q] for one arm family on one endpoint column.

    With ``arm_family="targeted"`` and an H1 endpoint column this is
    ATE_targeted (§21.3); with ``arm_family="random"`` it is ATE_random
    (§21.4); with the primary confirmation-branch endpoint it is the §21.2
    manipulation-success estimand E[Delta q_c].
    """
    contrasts = paired_contrasts(table, endpoint_col, arm_family, on_incomplete=on_incomplete)
    return ATEEstimate(
        arm_family=arm_family,
        endpoint_col=endpoint_col,
        n_pairs=len(contrasts),
        pair_weighted=mean_delta(contrasts, "pair"),
        branch_weighted=mean_delta(contrasts, "branch"),
        contrasts=tuple(contrasts),
    )


@dataclass(frozen=True)
class ContrastResult:
    """D = ATE_target_family - ATE_comparator_family (§21.4 and secondary D_S/D_B)."""

    target_family: str
    comparator_family: str
    endpoint_col: str
    target: ATEEstimate
    comparator: ATEEstimate
    fully_matched: bool
    """True iff target and comparator arms were run on the identical set of
    parent-cavity pairs (the paired design). When True, `paired_deltas`
    holds one D_c per shared pair -- the correct input to a paired bootstrap
    or the paired-difference normal approximation. When False (disjoint or
    partially-overlapping pair sets -- e.g. random-edit controls drawn on a
    separate matched-but-not-identical cavity sample), `paired_deltas` is
    None and D's uncertainty must instead be built by combining independent
    per-family bootstrap distributions (see intervals.independent_diff_ci).
    """
    paired_deltas: tuple[float, ...] | None

    @property
    def point_pair_weighted(self) -> float:
        return self.target.pair_weighted - self.comparator.pair_weighted

    @property
    def point_branch_weighted(self) -> float:
        return self.target.branch_weighted - self.comparator.branch_weighted


def family_contrast(
    table: BranchTable,
    endpoint_col: str,
    target_family: str,
    comparator_family: str,
    on_incomplete: OnIncomplete = "raise",
) -> ContrastResult:
    """Generic D = ATE_<target_family> - ATE_<comparator_family> machinery.

    Used for the §21.4 negative control (comparator="random"), and,
    unchanged, for the secondary superiority contrasts D_S
    (comparator="softness") and D_B (comparator="best_incumbent") -- same
    machinery, different comparator arm, exactly as the task spec requires.

    PLAN AMBIGUITY (flagged, not resolved): the plan does not state whether
    the random/softness/best-incumbent comparator interventions are applied
    to the *same* parent-cavity pairs as the targeted intervention (a fully
    paired design) or to a separately-drawn, geometry-matched pair sample.
    This function detects which situation the input table encodes (by
    comparing the pair-id sets of the two families) and reports
    `fully_matched` accordingly; callers should pick the CI method in
    `intervals.py` that matches.
    """
    target = estimate_ate(table, endpoint_col, target_family, on_incomplete=on_incomplete)
    comparator = estimate_ate(table, endpoint_col, comparator_family, on_incomplete=on_incomplete)
    target_pairs = {c.pair_id for c in target.contrasts}
    comparator_pairs = {c.pair_id for c in comparator.contrasts}
    fully_matched = target_pairs == comparator_pairs and len(target_pairs) > 0
    paired_deltas: tuple[float, ...] | None = None
    if fully_matched:
        by_pair_target = {c.pair_id: c.delta for c in target.contrasts}
        by_pair_comparator = {c.pair_id: c.delta for c in comparator.contrasts}
        paired_deltas = tuple(
            by_pair_target[pid] - by_pair_comparator[pid] for pid in sorted(target_pairs)
        )
    return ContrastResult(
        target_family=target_family,
        comparator_family=comparator_family,
        endpoint_col=endpoint_col,
        target=target,
        comparator=comparator,
        fully_matched=fully_matched,
        paired_deltas=paired_deltas,
    )


def negative_control_contrast(
    table: BranchTable,
    endpoint_col: str = "h1",
    target_family: str = "targeted",
    control_family: str = "random",
    on_incomplete: OnIncomplete = "raise",
) -> ContrastResult:
    """D_target-random = ATE_targeted - ATE_random (§21.4, primary pass criterion)."""
    return family_contrast(table, endpoint_col, target_family, control_family, on_incomplete)


def secondary_contrast_softness(
    table: BranchTable,
    endpoint_col: str = "h1",
    target_family: str = "targeted",
    comparator_family: str = "softness",
    on_incomplete: OnIncomplete = "raise",
) -> ContrastResult:
    """D_S = ATE_targeted - ATE_softness (secondary, decides Claim D only)."""
    return family_contrast(table, endpoint_col, target_family, comparator_family, on_incomplete)


def secondary_contrast_incumbent(
    table: BranchTable,
    endpoint_col: str = "h1",
    target_family: str = "targeted",
    comparator_family: str = "best_incumbent",
    on_incomplete: OnIncomplete = "raise",
) -> ContrastResult:
    """D_B = ATE_targeted - ATE_best-incumbent (secondary, decides Claim D only)."""
    return family_contrast(table, endpoint_col, target_family, comparator_family, on_incomplete)
