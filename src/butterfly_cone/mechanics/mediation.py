"""Parent-cluster mechanical contrasts and linear mediation decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from butterfly_cone.stats._common import ARM_FAMILY_COL, ARM_SIGN_COL, PAIR_ID_COL, BranchRow, BranchTable, SIGNS, as01, require_columns
from butterfly_cone.stats.intervals import BootstrapResult, NormalApproxResult, bootstrap_pairs, paired_diff_normal_approx


REQUIRED_FAMILIES: tuple[str, str] = ("targeted", "random")


@dataclass(frozen=True)
class ContrastEstimate:
    """Matched continuous DiD estimate with parent-cluster uncertainty."""

    observable: str
    point: float
    n_pairs: int
    pair_deltas: tuple[float, ...]
    bootstrap: BootstrapResult
    normal_approx: NormalApproxResult | None

    @property
    def ci(self) -> tuple[float, float]:
        return self.bootstrap.lo, self.bootstrap.hi


@dataclass(frozen=True)
class MediationResult:
    """Linear natural-effects decomposition with bootstrap-over-parent CIs."""

    mediator: str
    n_pairs: int
    total_effect: float
    indirect_effect: float
    direct_effect: float
    proportion_mediated: float
    mediator_effect_a: float
    mediator_outcome_effect_b: float
    total_effect_bootstrap: BootstrapResult
    indirect_effect_bootstrap: BootstrapResult
    direct_effect_bootstrap: BootstrapResult
    proportion_mediated_bootstrap: BootstrapResult

    @property
    def total_effect_ci(self) -> tuple[float, float]:
        return self.total_effect_bootstrap.lo, self.total_effect_bootstrap.hi

    @property
    def indirect_effect_ci(self) -> tuple[float, float]:
        return self.indirect_effect_bootstrap.lo, self.indirect_effect_bootstrap.hi

    @property
    def direct_effect_ci(self) -> tuple[float, float]:
        return self.direct_effect_bootstrap.lo, self.direct_effect_bootstrap.hi

    @property
    def proportion_mediated_ci(self) -> tuple[float, float]:
        return self.proportion_mediated_bootstrap.lo, self.proportion_mediated_bootstrap.hi


@dataclass(frozen=True)
class _PairSummary:
    pair_id: str
    delta_m_targeted: float
    delta_m_random: float
    delta_q_targeted: float | None
    delta_q_random: float | None

    @property
    def mechanical_did(self) -> float:
        return self.delta_m_targeted - self.delta_m_random


def _state_scalar(rows: Sequence[BranchRow], observable: str) -> float:
    """Read the one candidate-state scalar without averaging branch copies."""

    if not rows:
        raise ValueError("candidate state has no branch rows")
    values: list[float] = []
    for row in rows:
        require_columns(row, (observable,))
        value = float(row[observable])
        if not math.isfinite(value):
            raise ValueError(f"mechanical observable {observable!r} must be finite")
        values.append(value)
    reference = values[0]
    if not all(math.isclose(value, reference, rel_tol=0.0, abs_tol=1.0e-12) for value in values[1:]):
        raise ValueError(
            f"mechanical observable {observable!r} must be constant across branches of one candidate state"
        )
    return reference


def _state_outcome_mean(rows: Sequence[BranchRow], endpoint: str) -> float:
    if not rows:
        raise ValueError("candidate state has no branch rows")
    outcomes: list[int] = []
    for row in rows:
        require_columns(row, (endpoint,))
        outcomes.append(as01(row[endpoint]))
    return float(sum(outcomes) / len(outcomes))


def _group_candidate_rows(
    table: BranchTable,
    observable: str,
) -> dict[str, dict[str, dict[str, list[BranchRow]]]]:
    """Group only the four required candidate states, retaining branch blocks."""

    grouped: dict[str, dict[str, dict[str, list[BranchRow]]]] = {}
    for row in table:
        require_columns(row, (PAIR_ID_COL, ARM_FAMILY_COL, ARM_SIGN_COL, observable))
        family = row[ARM_FAMILY_COL]
        if family not in REQUIRED_FAMILIES:
            continue
        sign = row[ARM_SIGN_COL]
        if sign not in SIGNS:
            raise ValueError(f"arm_sign must be one of {SIGNS}, got {sign!r}")
        pair_id = row[PAIR_ID_COL]
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be a non-empty string")
        grouped.setdefault(pair_id, {}).setdefault(str(family), {}).setdefault(str(sign), []).append(row)
    if not grouped:
        raise ValueError("no targeted/random candidate rows found")
    return grouped


def _pair_summaries(
    table: BranchTable,
    observable: str,
    *,
    endpoint: str | None = None,
) -> list[_PairSummary]:
    grouped = _group_candidate_rows(table, observable)
    summaries: list[_PairSummary] = []
    for pair_id in sorted(grouped):
        by_family = grouped[pair_id]
        missing = [
            f"{family}{sign}"
            for family in REQUIRED_FAMILIES
            for sign in SIGNS
            if family not in by_family or sign not in by_family[family]
        ]
        if missing:
            raise ValueError(f"pair {pair_id!r} is missing required candidate state(s): {', '.join(missing)}")
        m_targeted = _state_scalar(by_family["targeted"]["+"], observable) - _state_scalar(
            by_family["targeted"]["-"], observable
        )
        m_random = _state_scalar(by_family["random"]["+"], observable) - _state_scalar(
            by_family["random"]["-"], observable
        )
        if endpoint is None:
            q_targeted = q_random = None
        else:
            q_targeted = _state_outcome_mean(by_family["targeted"]["+"], endpoint) - _state_outcome_mean(
                by_family["targeted"]["-"], endpoint
            )
            q_random = _state_outcome_mean(by_family["random"]["+"], endpoint) - _state_outcome_mean(
                by_family["random"]["-"], endpoint
            )
        summaries.append(
            _PairSummary(
                pair_id=pair_id,
                delta_m_targeted=m_targeted,
                delta_m_random=m_random,
                delta_q_targeted=q_targeted,
                delta_q_random=q_random,
            )
        )
    return summaries


def _continuous_summary_table(summaries: Sequence[_PairSummary]) -> list[Mapping[str, object]]:
    return [{PAIR_ID_COL: summary.pair_id, "value": summary.mechanical_did} for summary in summaries]


def _mean_summary_value(table: BranchTable) -> float:
    if not table:
        raise ValueError("no parent summaries found")
    values = [float(row["value"]) for row in table]
    return float(np.mean(values))


def matched_mechanical_contrast(
    table: BranchTable,
    observable: str,
    *,
    seed: int = 0,
    n_boot: int = 2_000,
    alpha: float = 0.05,
) -> ContrastEstimate:
    """Estimate the parent-level targeted-minus-random mechanical DiD.

    Candidate-state mechanical values are checked for equality across their
    branch rows and then copied once into a collapsed parent table.  The
    bootstrap therefore resamples parent blocks, never raw branches.
    """

    summaries = _pair_summaries(table, observable)
    collapsed = _continuous_summary_table(summaries)
    bootstrap = bootstrap_pairs(collapsed, _mean_summary_value, n_boot=n_boot, alpha=alpha, seed=seed)
    deltas = tuple(float(summary.mechanical_did) for summary in summaries)
    normal = paired_diff_normal_approx(list(deltas), alpha=alpha) if len(deltas) >= 2 else None
    return ContrastEstimate(
        observable=observable,
        point=bootstrap.point,
        n_pairs=len(summaries),
        pair_deltas=deltas,
        bootstrap=bootstrap,
        normal_approx=normal,
    )


def _mediation_rows(summaries: Sequence[_PairSummary]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for summary in summaries:
        if summary.delta_q_targeted is None or summary.delta_q_random is None:
            raise ValueError("mediation summaries require a binary endpoint")
        rows.append(
            {
                PAIR_ID_COL: summary.pair_id,
                "E": 1.0,
                "delta_m": summary.delta_m_targeted,
                "delta_q": summary.delta_q_targeted,
            }
        )
        rows.append(
            {
                PAIR_ID_COL: summary.pair_id,
                "E": 0.0,
                "delta_m": summary.delta_m_random,
                "delta_q": summary.delta_q_random,
            }
        )
    return rows


@dataclass(frozen=True)
class _MediationFit:
    total: float
    indirect: float
    direct: float
    proportion: float
    a: float
    b: float


def _ols_coefficients(design: np.ndarray, response: np.ndarray, *, require_full_rank: bool) -> np.ndarray:
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if require_full_rank and rank < design.shape[1]:
        raise ValueError("mediation model is not identifiable: design matrix lacks independent parent variation")
    return coefficients


def _fit_mediation(rows: BranchTable, *, require_full_rank: bool = True) -> _MediationFit:
    if len(rows) < 4:
        raise ValueError("mediation requires at least two parent pairs (targeted and random rows per parent)")
    treatment = np.asarray([float(row["E"]) for row in rows], dtype=np.float64)
    mediator = np.asarray([float(row["delta_m"]) for row in rows], dtype=np.float64)
    outcome = np.asarray([float(row["delta_q"]) for row in rows], dtype=np.float64)
    if not (np.all(np.isfinite(treatment)) and np.all(np.isfinite(mediator)) and np.all(np.isfinite(outcome))):
        raise ValueError("mediation rows must be finite")
    intercept = np.ones_like(treatment)
    a = float(_ols_coefficients(np.column_stack((intercept, treatment)), mediator, require_full_rank=True)[1])
    total = float(_ols_coefficients(np.column_stack((intercept, treatment)), outcome, require_full_rank=True)[1])
    direct, b = _ols_coefficients(
        np.column_stack((intercept, treatment, mediator)), outcome, require_full_rank=require_full_rank
    )[1:]
    direct = float(direct)
    b = float(b)
    indirect = a * b
    proportion = indirect / total if abs(total) > 1.0e-15 else float("nan")
    return _MediationFit(total=total, indirect=indirect, direct=direct, proportion=proportion, a=a, b=b)


def _mediation_component(component: str):
    def statistic(rows: BranchTable) -> float:
        # A degenerate bootstrap draw is theoretically possible only when it
        # selects one parent repeatedly.  ``lstsq`` then supplies a deterministic
        # pseudoinverse estimate instead of aborting an otherwise valid CI.
        return float(getattr(_fit_mediation(rows, require_full_rank=False), component))

    return statistic


def mediation_decomposition(
    table: BranchTable,
    mediator: str,
    *,
    seed: int,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    endpoint: str = "q",
) -> MediationResult:
    """Fit the frozen linear mediator/outcome models and bootstrap parents.

    Per parent, two rows enter the model: targeted ``E=1`` and matched-random
    ``E=0``.  Their within-family signed contrasts are the mediator and fresh
    binary-outcome endpoints.  Thus the mediator slope is exactly the matched
    ``D_M`` estimand, and the outcome model gives direct ``c'`` plus indirect
    ``a*b`` effects under the stated sequential-ignorability assumption.
    """

    summaries = _pair_summaries(table, mediator, endpoint=endpoint)
    rows = _mediation_rows(summaries)
    fit = _fit_mediation(rows, require_full_rank=True)
    total_bootstrap = bootstrap_pairs(rows, _mediation_component("total"), n_boot=n_boot, alpha=alpha, seed=seed)
    indirect_bootstrap = bootstrap_pairs(
        rows, _mediation_component("indirect"), n_boot=n_boot, alpha=alpha, seed=seed
    )
    direct_bootstrap = bootstrap_pairs(rows, _mediation_component("direct"), n_boot=n_boot, alpha=alpha, seed=seed)
    proportion_bootstrap = bootstrap_pairs(
        rows, _mediation_component("proportion"), n_boot=n_boot, alpha=alpha, seed=seed
    )
    return MediationResult(
        mediator=mediator,
        n_pairs=len(summaries),
        total_effect=fit.total,
        indirect_effect=fit.indirect,
        direct_effect=fit.direct,
        proportion_mediated=fit.proportion,
        mediator_effect_a=fit.a,
        mediator_outcome_effect_b=fit.b,
        total_effect_bootstrap=total_bootstrap,
        indirect_effect_bootstrap=indirect_bootstrap,
        direct_effect_bootstrap=direct_bootstrap,
        proportion_mediated_bootstrap=proportion_bootstrap,
    )
