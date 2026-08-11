"""§21.5 replication machinery: deterministic cohort split, half re-estimation,
sign+threshold agreement, and a generic "estimate under variant V" runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from ._common import PAIR_ID_COL, BranchTable, stable_hash_int

T = TypeVar("T")


@dataclass(frozen=True)
class CohortSplit:
    seed: int
    half_a: frozenset[str]
    half_b: frozenset[str]


def split_cohort(pair_ids: "list[str] | set[str] | frozenset[str]", seed: int) -> CohortSplit:
    """Deterministic seeded split of the confirmation cohort into two halves.

    Split key is ``hash(seed, pair_id)``, not input order or list position:
    all distinct pair ids are sorted by their hash value (ties -- vanishingly
    unlikely with SHA-256 -- broken by the pair id string itself, for a
    total order), and the sorted sequence is cut at the midpoint. Because
    the sort key depends only on `seed` and each id's own string, permuting
    the input collection's order never changes which half a given pair_id
    lands in, and the same `seed` always reproduces the same split.
    """
    unique_ids = sorted(set(pair_ids))
    if len(unique_ids) < 2:
        raise ValueError("need at least 2 distinct pair ids to split")
    ranked = sorted(unique_ids, key=lambda pid: (stable_hash_int(str(seed), pid), pid))
    midpoint = len(ranked) // 2
    return CohortSplit(seed=seed, half_a=frozenset(ranked[:midpoint]), half_b=frozenset(ranked[midpoint:]))


def filter_table_by_pairs(table: BranchTable, pair_ids: "frozenset[str] | set[str]", pair_col: str = PAIR_ID_COL) -> list[object]:
    return [row for row in table if row[pair_col] in pair_ids]


def reestimate_halves(
    table: BranchTable,
    split: CohortSplit,
    estimator: Callable[[BranchTable], T],
    pair_col: str = PAIR_ID_COL,
) -> tuple[T, T]:
    """Apply `estimator` independently to each half of a cohort split.

    `estimator` is any callable that reduces a branch table to a result
    (typically `functools.partial(estimands.estimate_ate, endpoint_col=...,
    arm_family=...)`); this function only handles the split-and-filter
    bookkeeping so the same estimator code path is reused, unchanged, on
    both halves.
    """
    table_a = filter_table_by_pairs(table, split.half_a, pair_col)
    table_b = filter_table_by_pairs(table, split.half_b, pair_col)
    return estimator(table_a), estimator(table_b)


@dataclass(frozen=True)
class AgreementReport:
    point_a: float
    point_b: float
    sign_agree: bool
    threshold: float | None
    both_exceed_threshold: bool | None


def agreement_report(point_a: float, point_b: float, threshold: float | None = None) -> AgreementReport:
    """Sign+threshold agreement between two half-cohort point estimates.

    `sign_agree` compares `sign(point_a) == sign(point_b)`, with 0 treated
    as its own sign bucket (an exact zero agrees only with another exact
    zero). If `threshold` is given, `both_exceed_threshold` additionally
    requires both halves to individually clear the same prespecified
    threshold (e.g. the §21.2 E[Delta q] >= 0.15 provisional bar) -- this
    package never picks a threshold after seeing the halves' outcomes.
    """

    def _sign(x: float) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    sign_agree = _sign(point_a) == _sign(point_b)
    both_exceed_threshold = None if threshold is None else (point_a >= threshold and point_b >= threshold)
    return AgreementReport(
        point_a=point_a,
        point_b=point_b,
        sign_agree=sign_agree,
        threshold=threshold,
        both_exceed_threshold=both_exceed_threshold,
    )


def run_variants(
    table: BranchTable,
    variant_endpoint_cols: Mapping[str, str],
    estimator: Callable[..., T],
    **estimator_kwargs: object,
) -> dict[str, T]:
    """Generic "estimate under variant V" runner (§21.5 event-threshold x2,
    geometry x2 replications).

    Variants are supplied purely as alternative endpoint-column names in the
    SAME input table (per the task spec) -- e.g.
    ``{"threshold_a": "h1_thresh_a", "threshold_b": "h1_thresh_b"}`` or
    ``{"geom_a": "h1_geom_a", "geom_b": "h1_geom_b"}``. `estimator` is called
    once per variant as ``estimator(table, endpoint_col=<column>,
    **estimator_kwargs)``; this makes no assumption about which estimator is
    being replicated (ATE, a family contrast, ...), only that it accepts
    `endpoint_col` as a keyword.
    """
    return {
        name: estimator(table, endpoint_col=column, **estimator_kwargs)
        for name, column in variant_endpoint_cols.items()
    }
