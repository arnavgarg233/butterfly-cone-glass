"""Causal-Gardner protocol-A quench-perturb-branch probe (ButterflyCone Wave-18).

Local delta-perturbation operators on a quenched deep configuration, a
matched-seed counterfactual branch runner, and the response / finite-size-
scaling analysis that emits the four discrimination axes and the
marginal-vs-defect verdict.  Everything here is orchestration and analysis
over the verified ButterflyCone stack (``branching``, ``engine``, ``events``,
``harness``, ``pilot``, ``stats``); this package adds no MD and no RCCE.
"""

from __future__ import annotations

from .operators import (
    OPERATORS,
    R_PERT_DEFAULT,
    PerturbationProvenance,
    dilation_tensor,
    o_kick,
    o_shell,
    o_strain,
    shear_tensor,
    stratified_sites,
)
from .response import (
    AxisSummary,
    DecisionThresholds,
    EnsembleTrajectory,
    NonSelfAveraging,
    Susceptibility,
    assert_matched_seeds,
    branch_divergence,
    cage_relative_divergence_field,
    chaos_length,
    decide,
    divergence_field,
    non_self_averaging,
    participation_ratio,
    r_d,
    susceptibility,
    total_divergence,
)

__all__ = [
    "OPERATORS",
    "R_PERT_DEFAULT",
    "PerturbationProvenance",
    "dilation_tensor",
    "o_kick",
    "o_shell",
    "o_strain",
    "shear_tensor",
    "stratified_sites",
    "AxisSummary",
    "DecisionThresholds",
    "EnsembleTrajectory",
    "NonSelfAveraging",
    "Susceptibility",
    "assert_matched_seeds",
    "branch_divergence",
    "cage_relative_divergence_field",
    "chaos_length",
    "decide",
    "divergence_field",
    "non_self_averaging",
    "participation_ratio",
    "r_d",
    "susceptibility",
    "total_divergence",
]
