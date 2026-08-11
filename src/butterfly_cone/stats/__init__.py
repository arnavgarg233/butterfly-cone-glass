"""Gate 0 estimation & analysis (PLAN_v2.1 §5, §21).

Pure analysis code over result tables: no simulation, no allocation logic.
See ``README.md`` in this directory for the estimand <-> plan-section map
and the exchangeability-unit argument.
"""

from . import calibration, estimands, intervals, replication, shuffle_null
from .estimands import (
    ATEEstimate,
    ContrastResult,
    PairArmCounts,
    PairContrast,
    collect_pair_arm_counts,
    estimate_ate,
    family_contrast,
    mean_delta,
    negative_control_contrast,
    paired_contrasts,
    secondary_contrast_incumbent,
    secondary_contrast_softness,
)
from .intervals import (
    BootstrapResult,
    NormalApproxResult,
    bootstrap_pairs,
    independent_diff_ci,
    jeffreys_interval,
    paired_diff_normal_approx,
)
from .replication import (
    AgreementReport,
    CohortSplit,
    agreement_report,
    filter_table_by_pairs,
    reestimate_halves,
    run_variants,
    split_cohort,
)
from .calibration import (
    IsotonicCalibrator,
    ReliabilityBin,
    auc_rank,
    ece,
    isotonic_recalibrate,
    mce,
    reliability_diagram,
)
from .shuffle_null import (
    ShuffleNullResult,
    derive_extreme_arms,
    permute_field_scores,
    shuffle_null_test,
)

__all__ = [
    "calibration",
    "estimands",
    "intervals",
    "replication",
    "shuffle_null",
    "ATEEstimate",
    "ContrastResult",
    "PairArmCounts",
    "PairContrast",
    "collect_pair_arm_counts",
    "estimate_ate",
    "family_contrast",
    "mean_delta",
    "negative_control_contrast",
    "paired_contrasts",
    "secondary_contrast_incumbent",
    "secondary_contrast_softness",
    "BootstrapResult",
    "NormalApproxResult",
    "bootstrap_pairs",
    "independent_diff_ci",
    "jeffreys_interval",
    "paired_diff_normal_approx",
    "AgreementReport",
    "CohortSplit",
    "agreement_report",
    "filter_table_by_pairs",
    "reestimate_halves",
    "run_variants",
    "split_cohort",
    "IsotonicCalibrator",
    "ReliabilityBin",
    "auc_rank",
    "ece",
    "isotonic_recalibrate",
    "mce",
    "reliability_diagram",
    "ShuffleNullResult",
    "derive_extreme_arms",
    "permute_field_scores",
    "shuffle_null_test",
]
