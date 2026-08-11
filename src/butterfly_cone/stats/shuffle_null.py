"""§21.4 shuffled-field negative control.

The real and null statistics MUST run through the identical code path --
only the field-score permutation differs -- so that no analyst-introduced
asymmetry between "real" and "null" processing can manufacture or hide an
effect. `_contrast_statistic` below is that single shared code path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import estimands
from ._common import PAIR_ID_COL, BranchTable, group_by, with_override

CANDIDATE_ARM_FAMILY = "shuffle_target"

ArmRule = Callable[[BranchTable, str, str], "list[object]"]


def derive_extreme_arms(
    candidates: BranchTable,
    field_col: str,
    pair_col: str = PAIR_ID_COL,
) -> list[object]:
    """Default arm-derivation rule: within each parent-cavity pair's candidate
    pool, the candidate with the highest `field_col` becomes the "+" arm and
    the candidate with the lowest becomes the "-" arm (the equilibrium-
    supported extreme-state selection Claim C's targeted edits perform).

    PLAN AMBIGUITY (flagged, not resolved): PLAN_v2.1 does not specify the
    exact candidate-to-arm selection rule at the level this negative-control
    permutation needs -- that selection logic lives in the RCCE/allocation
    modules, which this package must not import from. This default
    (max/min field score per pair) is a documented placeholder callers
    should override with `arm_rule=` once the real selection rule is
    frozen; the permutation machinery itself is independent of the specific
    rule as long as the same rule is used for the real and null statistics.
    """
    groups = group_by(candidates, pair_col)
    out: list[object] = []
    for pair_id, rows in groups.items():
        if len(rows) < 2:
            continue
        best = max(rows, key=lambda r: r[field_col])
        worst = min(rows, key=lambda r: r[field_col])
        out.append(with_override(best, arm_family=CANDIDATE_ARM_FAMILY, arm_sign="+"))
        out.append(with_override(worst, arm_family=CANDIDATE_ARM_FAMILY, arm_sign="-"))
    return out


def _contrast_statistic(
    candidates: BranchTable,
    field_col: str,
    outcome_col: str,
    pair_col: str,
    arm_rule: ArmRule,
) -> float:
    """The single code path shared by the observed statistic and every
    permutation draw: derive arms from (possibly permuted) field scores,
    then compute the pair-weighted mean Delta q exactly as `estimands` does
    for the real targeted/random contrasts.
    """
    arm_table = arm_rule(candidates, field_col, pair_col)
    contrasts = estimands.paired_contrasts(arm_table, outcome_col, CANDIDATE_ARM_FAMILY, on_incomplete="drop")
    if not contrasts:
        return 0.0
    return estimands.mean_delta(contrasts, "pair")


def permute_field_scores(
    candidates: BranchTable,
    field_col: str,
    rng: np.random.Generator,
) -> list[object]:
    """Reassign `field_col` values across ALL candidate states (global
    permutation, breaking any field-outcome relationship), preserving every
    other column. `rng` is consumed in place -- callers wanting a single
    standalone permutation should build their own
    `numpy.random.default_rng(seed)` and pass it in.
    """
    values = [row[field_col] for row in candidates]
    order = rng.permutation(len(values))
    return [with_override(row, **{field_col: values[order[i]]}) for i, row in enumerate(candidates)]


@dataclass(frozen=True)
class ShuffleNullResult:
    observed: float
    null_distribution: tuple[float, ...]
    p_value: float
    n_permutations: int
    seed: int
    alternative: str


def shuffle_null_test(
    candidates: BranchTable,
    field_col: str,
    outcome_col: str,
    *,
    n_permutations: int = 2000,
    seed: int,
    pair_col: str = PAIR_ID_COL,
    arm_rule: ArmRule = derive_extreme_arms,
    alternative: str = "two-sided",
) -> ShuffleNullResult:
    """Shuffled-field negative control: permute field scores across the
    candidate pool, re-derive arm assignments with the SAME `arm_rule`, and
    recompute the SAME contrast statistic as the real (unpermuted) data --
    `n_permutations` times, seeded -- to build a null distribution and a
    permutation p-value for the observed real-data contrast.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    observed = _contrast_statistic(candidates, field_col, outcome_col, pair_col, arm_rule)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted = permute_field_scores(candidates, field_col, rng)
        null[i] = _contrast_statistic(permuted, field_col, outcome_col, pair_col, arm_rule)
    if alternative == "two-sided":
        extreme = np.sum(np.abs(null) >= abs(observed))
    elif alternative == "greater":
        extreme = np.sum(null >= observed)
    else:
        extreme = np.sum(null <= observed)
    p_value = float((1 + extreme) / (n_permutations + 1))
    return ShuffleNullResult(
        observed=observed,
        null_distribution=tuple(float(x) for x in null),
        p_value=p_value,
        n_permutations=n_permutations,
        seed=seed,
        alternative=alternative,
    )
