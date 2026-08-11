"""Results aggregator: uniform effect-size records, publication tables, and
forest / heterogeneity data for the paper's Results assembly (PLAN_v2.1 §21).

The estimand modules in this package each return their own richly-typed
result object -- :class:`estimands.ATEEstimate`, :class:`varcomp.VarianceComponents`,
the kernel-recruitment summaries, and so on. This module is the *last* step
before a figure or a Results paragraph: it collapses any of those into one
uniform record shape (point / 95% CI / n, plus a standardized effect where
that is meaningful), then renders those records as (a) a publication table,
(b) forest-plot input tuples with a between-rider heterogeneity summary, and
(c) outcome-keyed headline sentences that *refuse to render* when the CI
still spans the null -- the module's built-in guard against over-claiming.

Design notes
------------
* NO plotting. ``forest_data`` returns the *numbers* a forest plot consumes;
  the figures themselves are downstream (``figs/``), never built here.
* NO randomness. Every function here is a pure, deterministic transform of
  already-estimated numbers -- the resampling lives in ``intervals`` /
  ``varcomp`` and hands its CI endpoints to this module. Calling any function
  twice on equal inputs yields byte-identical outputs.
* Anti-over-claim by construction. ``passes_threshold`` and ``headline_sentence``
  both use CI-based (not point-based) criteria: a claim "passes" only when the
  whole interval sits on the required side of the threshold / null, mirroring
  the exclude-the-null discipline the estimand modules already enforce.

Third-party dependencies: none beyond the standard library (numpy is available
but not required; the numerics here are exact pure-Python sums).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import NamedTuple

# Standard-normal 0.975 quantile (two-sided 95%). Hard-coded so this module
# stays self-contained and dependency-free; it is the same z used to turn a
# reported 95% CI half-width back into a standard error for the
# inverse-variance heterogeneity math below.
Z_975 = 1.959963984540054

GREATER = "greater"
LESS = "less"
_DIRECTIONS = (GREATER, LESS)


class OverClaimError(ValueError):
    """Raised when a headline would assert an effect whose CI still spans the null.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers
    keep working, while callers who want to distinguish "refused to render"
    from other bad input can catch this specifically.
    """


# ---------------------------------------------------------------------------
# The uniform effect-size record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectRecord:
    """One estimand collapsed to a uniform, figure-ready effect-size record.

    Numeric core (the five arguments the spec fixes for
    :func:`standardized_effect`):

    - ``point`` -- the point estimate (a risk difference Delta q / ATE, a
      variance fraction f_local, a kernel branching ratio, ...).
    - ``ci_low`` / ``ci_high`` -- the 95% CI endpoints from whichever CI
      method was appropriate for that estimand (pair bootstrap, cavity
      bootstrap, Jeffreys, normal approximation).
    - ``n`` -- the number of *exchangeable units* behind the estimate (pairs
      or cavities), never the branch count.
    - ``standardized`` -- a unit-free effect size where one is meaningful
      (see :func:`standardized_effect`); ``None`` otherwise.
    - ``base_rate`` -- the comparator/baseline rate used to standardize, kept
      for provenance; ``None`` when not applicable.

    Reporting metadata (used by the table / forest / headline renderers):

    - ``label`` -- the claim / outcome key, e.g. ``"f_local"`` or
      ``"ATE_targeted @ T=0.45"``.
    - ``threshold`` / ``threshold_direction`` -- an optional declared in advance
      bound and the side the estimate must clear.
    - ``quantity`` -- a grouping key naming the *underlying* quantity a
      record estimates. Riders that re-estimate the same quantity across
      temperatures or geometries share a ``quantity`` and are pooled into one
      heterogeneity subgroup by :func:`forest_data`; ``None`` marks a
      standalone claim that joins no subgroup.
    - ``null_value`` -- the null this effect is tested against (0 for a
      difference / fraction; could be 1 for a ratio). Both the CI-span guard
      and ``spans_null`` use it.
    """

    point: float
    ci_low: float
    ci_high: float
    n: int
    standardized: float | None = None
    base_rate: float | None = None
    label: str = ""
    threshold: float | None = None
    threshold_direction: str = GREATER
    quantity: str | None = None
    null_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", float(self.point))
        object.__setattr__(self, "ci_low", float(self.ci_low))
        object.__setattr__(self, "ci_high", float(self.ci_high))
        object.__setattr__(self, "null_value", float(self.null_value))
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"ci_low ({self.ci_low}) must not exceed ci_high ({self.ci_high})"
            )
        if int(self.n) != self.n or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")
        object.__setattr__(self, "n", int(self.n))
        if self.standardized is not None:
            object.__setattr__(self, "standardized", float(self.standardized))
        if self.base_rate is not None:
            base_rate = float(self.base_rate)
            if not 0.0 <= base_rate <= 1.0:
                raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
            object.__setattr__(self, "base_rate", base_rate)
        if self.threshold is not None:
            object.__setattr__(self, "threshold", float(self.threshold))
        if self.threshold_direction not in _DIRECTIONS:
            raise ValueError(
                f"threshold_direction must be one of {_DIRECTIONS}, got "
                f"{self.threshold_direction!r}"
            )

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)

    @property
    def width(self) -> float:
        """95% CI width -- twice the (approximate) standard error."""
        return self.ci_high - self.ci_low

    @property
    def standard_error(self) -> float:
        """Standard error implied by the reported 95% CI half-width."""
        return self.width / (2.0 * Z_975)

    def spans_null(self, null: float | None = None) -> bool:
        """True iff the CI contains the null (so no directional claim is safe)."""
        target = self.null_value if null is None else float(null)
        return self.ci_low <= target <= self.ci_high

    @property
    def passes_threshold(self) -> bool | None:
        """Whether the CI clears ``threshold`` on the required side.

        CI-based, not point-based: for ``"greater"`` the *lower* limit must
        exceed the threshold; for ``"less"`` the *upper* limit must fall below
        it. Returns ``None`` when no threshold was attached.
        """
        if self.threshold is None:
            return None
        if self.threshold_direction == GREATER:
            return self.ci_low > self.threshold
        return self.ci_high < self.threshold

    def as_dict(self) -> dict[str, object]:
        """Round-trip-safe plain-dict view (inverse of :meth:`from_dict`)."""
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "standardized": self.standardized,
            "base_rate": self.base_rate,
            "label": self.label,
            "threshold": self.threshold,
            "threshold_direction": self.threshold_direction,
            "quantity": self.quantity,
            "null_value": self.null_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EffectRecord":
        """Rebuild a record from :meth:`as_dict` output (exact round-trip)."""
        return cls(**dict(data))


def _standardized(point: float, base_rate: float | None) -> float | None:
    """Baseline-SD-standardized effect for a proportion contrast, else None.

    For a binary endpoint with comparator rate ``p0`` the natural unit-free
    effect size is the risk difference expressed in baseline standard-
    deviation units, ``point / sqrt(p0 (1 - p0))`` -- a Cohen's-d analogue
    that is directly comparable across the binary ATE / Delta q estimands.
    It is *not* meaningful when no base rate is supplied (e.g. the kernel
    branching-ratio estimand) or when the base rate is degenerate (0 or 1,
    zero baseline variance), and is reported as ``None`` in those cases.
    """
    if base_rate is None:
        return None
    if not 0.0 < base_rate < 1.0:
        return None
    sd = math.sqrt(base_rate * (1.0 - base_rate))
    if sd <= 0.0:
        return None
    return point / sd


def standardized_effect(
    estimate: float,
    ci_low: float,
    ci_high: float,
    n: int,
    base_rate: float | None = None,
    *,
    label: str = "",
    threshold: float | None = None,
    threshold_direction: str = GREATER,
    quantity: str | None = None,
    null_value: float = 0.0,
) -> EffectRecord:
    """Build the uniform effect-size record for one estimand.

    The first five parameters are the fixed numeric core: a point estimate,
    its 95% CI, the exchangeable-unit count, and an optional comparator
    ``base_rate`` used to compute the standardized effect. The keyword-only
    extras attach reporting metadata (claim label, declared in advance threshold,
    heterogeneity grouping key, tested null) so the same record can flow
    straight into :func:`results_table`, :func:`forest_data`, and
    :func:`headline_sentence` without a second wrapper type.

    Works unchanged across the binary ATE / Delta q estimands (pass the
    control-arm rate as ``base_rate`` to get a standardized effect) and the
    kernel / variance-fraction estimands (omit ``base_rate``; ``standardized``
    is then ``None``).
    """
    return EffectRecord(
        point=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        standardized=_standardized(float(estimate), base_rate),
        base_rate=base_rate,
        label=label,
        threshold=threshold,
        threshold_direction=threshold_direction,
        quantity=quantity,
        null_value=null_value,
    )


# ---------------------------------------------------------------------------
# Publication table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultsTable:
    """A publication table in two synchronized forms.

    - ``markdown`` -- a GitHub-flavored markdown table (header, alignment
      separator, one row per record) ready to paste into the manuscript.
    - ``rows`` -- the same content as a tuple of plain dicts for programmatic
      use (assertions, JSON, further processing); ``columns`` names the
      display columns in order.
    """

    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]
    markdown: str

    def as_dict(self) -> dict[str, object]:
        return {"columns": list(self.columns), "rows": [dict(r) for r in self.rows]}


_TABLE_COLUMNS = ("claim", "estimate", "95% CI", "n", "passes-threshold")


def _passes_cell(passes: bool | None) -> str:
    if passes is None:
        return "-"  # no threshold attached
    return "yes" if passes else "no"


def results_table(
    records: Sequence[EffectRecord], *, precision: int = 3
) -> ResultsTable:
    """Render effect-size records as a publication table.

    Columns are ``claim | estimate | 95% CI | n | passes-threshold``. Rows
    preserve input order (deterministic; this module never reorders the
    author's chosen claim sequence). The ``passes-threshold`` cell is
    ``yes`` / ``no`` for records carrying a threshold and an em dash for
    those without one. ``precision`` controls decimal places for the point
    and CI cells (``n`` is always an integer).
    """
    rows: list[dict[str, object]] = []
    for record in records:
        passes = record.passes_threshold
        rows.append(
            {
                "claim": record.label,
                "point": record.point,
                "ci_low": record.ci_low,
                "ci_high": record.ci_high,
                "n": record.n,
                "threshold": record.threshold,
                "threshold_direction": record.threshold_direction,
                "passes_threshold": passes,
            }
        )

    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(_TABLE_COLUMNS)) + " |"
    lines = [header, separator]
    for record, row in zip(records, rows):
        estimate_cell = f"{record.point:.{precision}f}"
        ci_cell = f"[{record.ci_low:.{precision}f}, {record.ci_high:.{precision}f}]"
        cells = (
            str(record.label),
            estimate_cell,
            ci_cell,
            str(record.n),
            _passes_cell(row["passes_threshold"]),
        )
        lines.append("| " + " | ".join(cells) + " |")

    return ResultsTable(
        columns=_TABLE_COLUMNS,
        rows=tuple(rows),
        markdown="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Forest data + heterogeneity
# ---------------------------------------------------------------------------


class ForestEntry(NamedTuple):
    """One forest-plot row: ``(label, point, ci_low, ci_high)``.

    A ``NamedTuple`` so it *is* a 4-tuple -- it unpacks directly into a
    plotting call while still supporting attribute access.
    """

    label: str
    point: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class HeterogeneitySummary:
    """Between-rider heterogeneity for records sharing one ``quantity``.

    Fixed-effect (inverse-variance) meta-analysis over the ``k`` estimates in
    the group, with the standard Cochran ``Q`` / Higgins-Thompson ``I^2`` and
    a DerSimonian-Laird ``tau^2``. ``pooled_point`` and its CI are the
    inverse-variance-weighted pooled estimate -- a convenient single number
    for the subgroup, reported alongside (never instead of) the individual
    riders.
    """

    quantity: str
    k: int
    q_stat: float
    df: int
    i_squared: float  # percent, clamped to [0, 100]
    tau_squared: float
    pooled_point: float
    pooled_ci_low: float
    pooled_ci_high: float


@dataclass(frozen=True)
class ForestData:
    """Forest-plot input: the per-claim entries plus heterogeneity subgroups."""

    entries: tuple[ForestEntry, ...]
    heterogeneity: tuple[HeterogeneitySummary, ...]


def _heterogeneity(quantity: str, group: Sequence[EffectRecord]) -> HeterogeneitySummary:
    ys = [r.point for r in group]
    ses = [r.standard_error for r in group]
    k = len(group)
    df = k - 1

    if all(se > 0.0 for se in ses):
        weights = [1.0 / (se * se) for se in ses]
    else:
        # Degenerate: at least one zero-width CI. Fall back to equal weights
        # so the summary stays finite instead of dividing by zero.
        weights = [1.0] * k

    total_weight = sum(weights)
    pooled_point = sum(w * y for w, y in zip(weights, ys)) / total_weight
    q_stat = sum(w * (y - pooled_point) ** 2 for w, y in zip(weights, ys))
    i_squared = max(0.0, (q_stat - df) / q_stat) * 100.0 if q_stat > 0.0 else 0.0

    # DerSimonian-Laird tau^2.
    c = total_weight - sum(w * w for w in weights) / total_weight
    tau_squared = max(0.0, (q_stat - df) / c) if c > 0.0 else 0.0

    pooled_se = math.sqrt(1.0 / total_weight)
    return HeterogeneitySummary(
        quantity=quantity,
        k=k,
        q_stat=q_stat,
        df=df,
        i_squared=i_squared,
        tau_squared=tau_squared,
        pooled_point=pooled_point,
        pooled_ci_low=pooled_point - Z_975 * pooled_se,
        pooled_ci_high=pooled_point + Z_975 * pooled_se,
    )


def forest_data(
    records: Sequence[EffectRecord], *, order: str = "input"
) -> ForestData:
    """Assemble forest-plot tuples and a between-rider heterogeneity summary.

    ``entries`` are ``(label, point, ci_low, ci_high)`` tuples -- data only,
    no plotting. Their order is controlled by ``order``:

    - ``"input"`` (default) preserves the caller's record order;
    - ``"point"`` sorts by descending point estimate (ties broken by label);
    - ``"label"`` sorts by label ascending.

    ``heterogeneity`` holds one :class:`HeterogeneitySummary` per ``quantity``
    group that has at least two records -- the riders re-estimating the same
    underlying quantity across temperatures / geometries. Records with
    ``quantity is None``, or the sole member of a quantity, are standalone
    claims and contribute no subgroup. Summaries are ordered by ``quantity``
    name for determinism.
    """
    if order == "input":
        ordered = list(records)
    elif order == "point":
        ordered = sorted(records, key=lambda r: (-r.point, r.label))
    elif order == "label":
        ordered = sorted(records, key=lambda r: r.label)
    else:
        raise ValueError(f"unknown order {order!r}; expected 'input', 'point', or 'label'")

    entries = tuple(
        ForestEntry(r.label, r.point, r.ci_low, r.ci_high) for r in ordered
    )

    groups: dict[str, list[EffectRecord]] = {}
    for record in records:
        if record.quantity is None:
            continue
        groups.setdefault(record.quantity, []).append(record)

    heterogeneity = tuple(
        _heterogeneity(quantity, group)
        for quantity, group in sorted(groups.items())
        if len(group) >= 2
    )
    return ForestData(entries=entries, heterogeneity=heterogeneity)


# ---------------------------------------------------------------------------
# Headline sentence (with the over-claim guard)
# ---------------------------------------------------------------------------


def headline_sentence(record: EffectRecord, template: str) -> str:
    """Fill an outcome-keyed reporting sentence from a record.

    ``template`` is a ``str.format`` template drawing on these named fields::

        claim / label        the record's label
        point / estimate     the point estimate
        ci_low / lo          lower 95% CI endpoint
        ci_high / hi         upper 95% CI endpoint
        n                    exchangeable-unit count
        pct / pct_low / pct_high   the same three values times 100
        standardized         the standardized effect (may be None)

    e.g. ``"{pct:.0f}% of local relaxation is causally structural, 95% CI "``
    ``"[{ci_low:.3f}, {ci_high:.3f}]"``.

    Over-claim guard: if the 95% CI still contains the record's null
    (``record.spans_null()``), the sentence would assert an effect the data
    have not established, so this raises :class:`OverClaimError` instead of
    rendering it. A wholly-negative interval (CI entirely below the null) is
    a legitimate directional finding and renders normally.
    """
    if record.spans_null():
        raise OverClaimError(
            f"refusing to render headline for {record.label!r}: 95% CI "
            f"[{record.ci_low}, {record.ci_high}] spans the null "
            f"({record.null_value})"
        )
    fields: dict[str, object] = {
        "claim": record.label,
        "label": record.label,
        "point": record.point,
        "estimate": record.point,
        "ci_low": record.ci_low,
        "lo": record.ci_low,
        "ci_high": record.ci_high,
        "hi": record.ci_high,
        "n": record.n,
        "pct": record.point * 100.0,
        "pct_low": record.ci_low * 100.0,
        "pct_high": record.ci_high * 100.0,
        "standardized": record.standardized,
    }
    return template.format(**fields)


__all__ = [
    "GREATER",
    "LESS",
    "Z_975",
    "OverClaimError",
    "EffectRecord",
    "ForestData",
    "ForestEntry",
    "HeterogeneitySummary",
    "ResultsTable",
    "forest_data",
    "headline_sentence",
    "results_table",
    "standardized_effect",
]
