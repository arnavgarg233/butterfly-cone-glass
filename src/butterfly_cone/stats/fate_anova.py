"""Structure×noise fate-ANOVA on a fully-observed cavity×structure×world grid.

This module decomposes the total variance of a binary *fate* matrix

``Y[c, s, w]`` with ``c = cavity``, ``s = structure candidate`` (``M`` conditional
draws), ``w = noise world`` (``W`` shared thermal worlds)

into a **structure main effect**, a **noise main effect**, their
**interaction** (structure×noise non-additivity, the headline quantity that a
marginal design can never see), and a **residual**.  Entries are outcomes in
``[0, 1]``, nominally ``0``/``1`` fate indicators, relaxed to fractions by the
survival horizon.

## Why the crossed design is special

Bitwise replay makes every ``(s, w)`` cell within a cavity share the *same*
thermal noise world ``w`` across all structures ``s``.  The structure×noise grid
is therefore fully crossed *within* each cavity, and the same fixed set of ``W``
worlds and ``M`` structures is replayed across cavities.  The cavity is the only
exchangeable unit for outer uncertainty, exactly the convention used by
``varcomp`` (see ``varcomp.py`` and ``_common.py``).  We follow that module's
philosophy: a method-of-moments nested ANOVA with explicit finite-sample
corrections, nonnegative projection of raw components, and a bootstrap that
resamples *complete cavities*, never individual cells.

## Exact sum-of-squares identity

For the balanced complete grid the orthogonal ANOVA projection is exact.  With
grand mean ``g``, structure margin ``a_s = mean_{c,w} Y``, noise margin
``b_w = mean_{c,s} Y`` and cross-cavity cell mean ``m_sw = mean_c Y``:

- ``SS_total       = sum_{c,s,w} (Y - g)^2``
- ``SS_structure   = C * W * sum_s (a_s - g)^2``
- ``SS_noise       = C * S * sum_w (b_w - g)^2``
- ``SS_interaction = C * sum_{s,w} (m_sw - a_s - b_w + g)^2``
- ``SS_residual    = sum_{c,s,w} (Y - m_sw)^2``

and these satisfy, exactly (to floating-point),

``SS_total = SS_structure + SS_noise + SS_interaction + SS_residual``.

The fitted value of the structure+noise+interaction model is precisely the
cross-cavity cell mean ``m_sw``; the residual is the pure cavity-idiosyncratic
(and Bernoulli) variation *around* each replayed cell, it is the analog of
``varcomp``'s thermal component.

## Bias correction for finite W / M

Structure, noise and interaction effects are *fixed* structural properties (the
same worlds and structures are replayed for every cavity); the cavity/thermal
residual is the random part.  This is a Model-I two-way layout with a random
residual.  With ``n = C`` replicate cavities per cell the expected mean squares
are

- ``E[MS_residual]    = sigma2_eps``
- ``E[MS_structure]   = sigma2_eps + W * C * T_alpha``
- ``E[MS_noise]       = sigma2_eps + S * C * T_beta``
- ``E[MS_interaction] = sigma2_eps + C * T_gamma``

where ``T_alpha = sum_s alpha_s^2 / (S-1)``, ``T_beta = sum_w beta_w^2 / (W-1)``
and ``T_gamma = sum_{s,w} gamma_sw^2 / ((S-1)(W-1))`` are the finite-population
mean-square effects.  A structure margin averages over ``W`` worlds, so the
residual noise it carries scales as ``1/W``; a noise margin averages over ``S``
structures, scaling as ``1/S = 1/M``.  The method-of-moments components remove
exactly that finite-``W``/``M`` leakage:

- ``var_residual    = MS_residual``
- ``var_structure   = (MS_structure   - MS_residual) / (W * C)``  -> ``T_alpha``
- ``var_noise       = (MS_noise       - MS_residual) / (S * C)``  -> ``T_beta``
- ``var_interaction = (MS_interaction - MS_residual) / C``        -> ``T_gamma``

As in ``varcomp``, sampling fluctuation can drive a raw component negative, so
the public components use the nonnegative projection; the projected components
form the reported total and the shares therefore sum to one.  The correction
assumes a common cell-residual variance (no REML fit, no distributional
assumption), it is exact when cavities are noise-free replicates and
approximately unbiased under heteroscedastic Bernoulli noise.

## Corollaries: ITE distribution and PN / PS

Because the noise world is shared, a structure contrast holding ``w`` fixed is a
genuine counterfactual: for a treatment structure ``s1`` and control ``s0`` the
pair ``(y0, y1) = (Y[c, s0, w], Y[c, s1, w])`` is *jointly observed* for every
unit ``(c, w)``.  The individual treatment effect ``ITE = y1 - y0`` and Pearl's
probabilities of necessity/sufficiency are then point-identified (not merely
bounded):

- ``PN = P(y0 = 0 | y1 = 1)``  : necessity of the structure for the fate
- ``PS = P(y1 = 1 | y0 = 0)``  : sufficiency of the structure for the fate
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class ITEResult:
    """Individual-treatment-effect distribution and PN/PS for a structure pair.

    Units are the ``(cavity, world)`` pairs (``C * W`` of them).  ``treat`` is
    the treatment structure index and ``control`` the baseline structure index.
    """

    treat: int
    control: int
    n_units: int
    frac_helped: float
    frac_unchanged: float
    frac_hurt: float
    mean_ite: float
    pn: float
    ps: float


@dataclass(frozen=True)
class FateAnova:
    """Exact fate-variance decomposition plus bias-corrected components.

    The ``ss_*`` fields are the exact orthogonal sum-of-squares decomposition
    (they satisfy the identity to floating point).  The ``var_*`` fields are the
    finite-``W``/``M`` bias-corrected method-of-moments components after
    nonnegative projection; ``*_raw`` are the unprojected values (may be < 0).
    ``*_share`` are the corrected component shares (they sum to one); the
    ``ss_*_share`` fields are the exact descriptive SS shares.  ``ite`` is the
    default counterfactual corollary comparing the highest- against the
    lowest-marginal-fate structure.
    """

    n_cavities: int
    n_structures: int
    n_worlds: int
    grand_mean: float

    ss_total: float
    ss_structure: float
    ss_noise: float
    ss_interaction: float
    ss_residual: float

    df_structure: int
    df_noise: int
    df_interaction: int
    df_residual: int

    ms_structure: float
    ms_noise: float
    ms_interaction: float
    ms_residual: float

    var_structure_raw: float
    var_noise_raw: float
    var_interaction_raw: float
    var_residual: float

    var_structure: float
    var_noise: float
    var_interaction: float
    var_total: float

    structure_share: float
    noise_share: float
    interaction_share: float
    residual_share: float

    interaction_share_naive: float

    ss_structure_share: float
    ss_noise_share: float
    ss_interaction_share: float
    ss_residual_share: float

    ite: ITEResult

    @property
    def var_interaction_naive(self) -> float:
        """Plug-in interaction component (no residual subtraction)."""
        return self.ms_interaction / self.n_cavities

    @property
    def var_structure_naive(self) -> float:
        """Plug-in structure component (variance of structure margins)."""
        return self.ms_structure / (self.n_worlds * self.n_cavities)

    @property
    def var_noise_naive(self) -> float:
        """Plug-in noise component (variance of noise margins)."""
        return self.ms_noise / (self.n_structures * self.n_cavities)

    @property
    def component_sum(self) -> float:
        """Sum of the projected components (equals ``var_total``)."""
        return (
            self.var_structure
            + self.var_noise
            + self.var_interaction
            + self.var_residual
        )


@dataclass(frozen=True)
class BootstrapShareResult:
    """Cavity-bootstrap percentile interval for the interaction share."""

    estimate: FateAnova
    point: float
    lo: float
    hi: float
    boot_estimates: tuple[float, ...]
    alpha: float
    seed: int
    n_boot: int

    @property
    def interaction_share_ci(self) -> tuple[float, float]:
        return self.lo, self.hi


def _validate(Y: ArrayLike) -> np.ndarray:
    """Validate and coerce a fate matrix to a ``(C, S, W)`` float array."""
    array = np.asarray(Y, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"fate matrix must be 3-D (cavity, structure, world), got shape {array.shape}"
        )
    n_cavities, n_structures, n_worlds = array.shape
    if n_cavities < 2:
        raise ValueError("at least two cavities are required for the residual component")
    if n_structures < 2:
        raise ValueError("at least two structures are required for the structure main effect")
    if n_worlds < 2:
        raise ValueError("at least two noise worlds are required for the noise main effect")
    if not np.all(np.isfinite(array)):
        raise ValueError("fate entries must all be finite")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("fate entries must lie in [0, 1]")
    return array


def _decompose(array: np.ndarray) -> dict[str, float]:
    """Return the exact SS/MS decomposition for a validated ``(C, S, W)`` grid."""
    n_cavities, n_structures, n_worlds = array.shape

    grand = float(array.mean())
    structure_margin = array.mean(axis=(0, 2))  # (S,)
    noise_margin = array.mean(axis=(0, 1))  # (W,)
    cell_mean = array.mean(axis=0)  # (S, W)

    a = structure_margin - grand  # (S,)
    b = noise_margin - grand  # (W,)
    g = cell_mean - structure_margin[:, None] - noise_margin[None, :] + grand  # (S, W)

    ss_total = float(np.sum((array - grand) ** 2))
    ss_structure = float(n_cavities * n_worlds * np.sum(a**2))
    ss_noise = float(n_cavities * n_structures * np.sum(b**2))
    ss_interaction = float(n_cavities * np.sum(g**2))
    ss_residual = float(np.sum((array - cell_mean[None, :, :]) ** 2))

    df_structure = n_structures - 1
    df_noise = n_worlds - 1
    df_interaction = (n_structures - 1) * (n_worlds - 1)
    df_residual = n_structures * n_worlds * (n_cavities - 1)

    return {
        "n_cavities": n_cavities,
        "n_structures": n_structures,
        "n_worlds": n_worlds,
        "grand_mean": grand,
        "ss_total": ss_total,
        "ss_structure": ss_structure,
        "ss_noise": ss_noise,
        "ss_interaction": ss_interaction,
        "ss_residual": ss_residual,
        "df_structure": df_structure,
        "df_noise": df_noise,
        "df_interaction": df_interaction,
        "df_residual": df_residual,
        "ms_structure": ss_structure / df_structure,
        "ms_noise": ss_noise / df_noise,
        "ms_interaction": ss_interaction / df_interaction,
        "ms_residual": ss_residual / df_residual,
    }


def _components(parts: dict[str, float]) -> dict[str, float]:
    """Bias-corrected, nonnegative-projected components and shares."""
    n_cavities = parts["n_cavities"]
    n_structures = parts["n_structures"]
    n_worlds = parts["n_worlds"]

    ms_res = parts["ms_residual"]
    var_residual = max(0.0, ms_res)
    var_structure_raw = (parts["ms_structure"] - ms_res) / (n_worlds * n_cavities)
    var_noise_raw = (parts["ms_noise"] - ms_res) / (n_structures * n_cavities)
    var_interaction_raw = (parts["ms_interaction"] - ms_res) / n_cavities

    var_structure = max(0.0, var_structure_raw)
    var_noise = max(0.0, var_noise_raw)
    var_interaction = max(0.0, var_interaction_raw)
    var_total = var_structure + var_noise + var_interaction + var_residual

    if var_total > 0.0:
        structure_share = var_structure / var_total
        noise_share = var_noise / var_total
        interaction_share = var_interaction / var_total
        residual_share = var_residual / var_total
    else:
        structure_share = noise_share = interaction_share = residual_share = 0.0

    # Naive plug-in interaction share (no residual subtraction anywhere) as a
    # baseline that finite-W/M correction improves upon.
    naive_structure = parts["ms_structure"] / (n_worlds * n_cavities)
    naive_noise = parts["ms_noise"] / (n_structures * n_cavities)
    naive_interaction = parts["ms_interaction"] / n_cavities
    naive_total = naive_structure + naive_noise + naive_interaction + var_residual
    interaction_share_naive = (
        naive_interaction / naive_total if naive_total > 0.0 else 0.0
    )

    ss_total = parts["ss_total"]
    if ss_total > 0.0:
        ss_structure_share = parts["ss_structure"] / ss_total
        ss_noise_share = parts["ss_noise"] / ss_total
        ss_interaction_share = parts["ss_interaction"] / ss_total
        ss_residual_share = parts["ss_residual"] / ss_total
    else:
        ss_structure_share = ss_noise_share = 0.0
        ss_interaction_share = ss_residual_share = 0.0

    return {
        "var_structure_raw": var_structure_raw,
        "var_noise_raw": var_noise_raw,
        "var_interaction_raw": var_interaction_raw,
        "var_residual": var_residual,
        "var_structure": var_structure,
        "var_noise": var_noise,
        "var_interaction": var_interaction,
        "var_total": var_total,
        "structure_share": structure_share,
        "noise_share": noise_share,
        "interaction_share": interaction_share,
        "residual_share": residual_share,
        "interaction_share_naive": interaction_share_naive,
        "ss_structure_share": ss_structure_share,
        "ss_noise_share": ss_noise_share,
        "ss_interaction_share": ss_interaction_share,
        "ss_residual_share": ss_residual_share,
    }


def _interaction_share_from_array(array: np.ndarray) -> float:
    """Corrected interaction share for a validated grid (bootstrap inner loop)."""
    return _components(_decompose(array))["interaction_share"]


def _bootstrap_interaction_shares(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Corrected interaction share for every ``(n_boot, C)`` cavity-index row.

    Vectorized equivalent of ``[_interaction_share_from_array(array[idx]) for
    idx in indices]``: replicate cell means come from one multiplicity-matrix
    GEMM, the margin/interaction sums of squares are batched, and the residual
    keeps the direct ``(Y - cell_mean)**2`` form (weighted by multiplicity) so
    there is no cancellation-prone expansion.  Same formulas per replicate;
    only summation trees differ, i.e. f64 round-off level agreement.
    [micro-bench, (150, 10, 16) grid, n_boot=400: 244 ms -> 24 ms, ~10x]
    """

    n_boot = indices.shape[0]
    n_cavities, n_structures, n_worlds = array.shape
    flat = array.reshape(n_cavities, -1)
    # Multiplicity of each cavity in each replicate: bincount over row-offset
    # indices (accumulation is exact integer arithmetic).
    offsets = np.arange(n_boot, dtype=np.int64)[:, None] * n_cavities
    multiplicity = (
        np.bincount((indices + offsets).ravel(), minlength=n_boot * n_cavities)
        .reshape(n_boot, n_cavities)
        .astype(np.float64)
    )
    cell_flat = multiplicity @ flat / n_cavities                     # (B, S*W)
    cell = cell_flat.reshape(n_boot, n_structures, n_worlds)
    grand = cell_flat.mean(axis=1)
    structure_margin = cell.mean(axis=2)                             # (B, S)
    noise_margin = cell.mean(axis=1)                                 # (B, W)
    a = structure_margin - grand[:, None]
    b = noise_margin - grand[:, None]
    g = cell - structure_margin[:, :, None] - noise_margin[:, None, :] + grand[:, None, None]

    ss_structure = n_cavities * n_worlds * np.einsum("bs,bs->b", a, a)
    ss_noise = n_cavities * n_structures * np.einsum("bw,bw->b", b, b)
    ss_interaction = n_cavities * np.einsum("bsw,bsw->b", g, g)
    # Residual: sum_c m_bc * sum_sw (Y_c - cell_mean_b)^2, chunked over
    # replicates to bound the (chunk, C, S*W) temporary.
    ss_residual = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, int(4_000_000 // max(1, flat.size)))
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        diff = flat[None, :, :] - cell_flat[start:stop, None, :]
        per_cavity = np.einsum("bcx,bcx->bc", diff, diff)
        ss_residual[start:stop] = np.einsum("bc,bc->b", multiplicity[start:stop], per_cavity)

    df_structure = n_structures - 1
    df_noise = n_worlds - 1
    df_interaction = df_structure * df_noise
    df_residual = n_structures * n_worlds * (n_cavities - 1)
    ms_residual = ss_residual / df_residual
    var_residual = np.maximum(0.0, ms_residual)
    var_structure = np.maximum(
        0.0, (ss_structure / df_structure - ms_residual) / (n_worlds * n_cavities)
    )
    var_noise = np.maximum(
        0.0, (ss_noise / df_noise - ms_residual) / (n_structures * n_cavities)
    )
    var_interaction = np.maximum(
        0.0, (ss_interaction / df_interaction - ms_residual) / n_cavities
    )
    var_total = var_structure + var_noise + var_interaction + var_residual
    return np.divide(
        var_interaction, var_total, out=np.zeros_like(var_total), where=var_total > 0.0
    )


def _default_treat_control(array: np.ndarray) -> tuple[int, int]:
    """Highest- vs lowest-marginal-fate structure (ties -> lowest index)."""
    structure_mean = array.mean(axis=(0, 2))  # (S,)
    treat = int(np.argmax(structure_mean))
    control = int(np.argmin(structure_mean))
    return treat, control


def ite_pn_ps(Y: ArrayLike, treat: int, control: int) -> ITEResult:
    """Point-identified ITE distribution and PN/PS for a structure contrast.

    ``treat`` and ``control`` are structure indices.  Units are the
    ``(cavity, world)`` pairs; because the noise world is shared, both potential
    outcomes are jointly observed per unit.  For binary entries the fractions
    are exact event probabilities; for horizon-relaxed fractional entries they
    are the coherent expectation-based extension.
    """
    return _ite_pn_ps_validated(_validate(Y), treat, control)


def _ite_pn_ps_validated(array: np.ndarray, treat: int, control: int) -> ITEResult:
    """:func:`ite_pn_ps` body for an already-validated ``(C, S, W)`` grid."""
    n_structures = array.shape[1]
    for name, index in (("treat", treat), ("control", control)):
        if not isinstance(index, numbers.Integral) or isinstance(index, bool):
            raise ValueError(f"{name} structure index must be an integer")
        if not 0 <= int(index) < n_structures:
            raise ValueError(
                f"{name} structure index {index} out of range [0, {n_structures})"
            )

    y1 = array[:, int(treat), :].ravel()  # treatment potential outcome
    y0 = array[:, int(control), :].ravel()  # control potential outcome
    n_units = int(y1.size)

    helped = float(np.mean(y1 * (1.0 - y0)))  # P(y1=1, y0=0)
    hurt = float(np.mean(y0 * (1.0 - y1)))  # P(y0=1, y1=0)
    unchanged = float(1.0 - helped - hurt)
    mean_ite = float(np.mean(y1 - y0))

    sum_treat_events = float(np.sum(y1))
    sum_control_nonevents = float(np.sum(1.0 - y0))
    joint = float(np.sum(y1 * (1.0 - y0)))
    pn = joint / sum_treat_events if sum_treat_events > 0.0 else 0.0
    ps = joint / sum_control_nonevents if sum_control_nonevents > 0.0 else 0.0

    return ITEResult(
        treat=int(treat),
        control=int(control),
        n_units=n_units,
        frac_helped=helped,
        frac_unchanged=unchanged,
        frac_hurt=hurt,
        mean_ite=mean_ite,
        pn=pn,
        ps=ps,
    )


def fate_anova(Y: ArrayLike) -> FateAnova:
    """Decompose a fate matrix ``Y[c, s, w]`` into structure/noise/interaction.

    Returns the exact SS decomposition, the finite-``W``/``M`` bias-corrected
    components and shares, and the default ITE/PN/PS corollary (highest- vs
    lowest-marginal-fate structure).
    """
    array = _validate(Y)
    parts = _decompose(array)
    comps = _components(parts)
    treat, control = _default_treat_control(array)
    # _ite_pn_ps_validated skips re-validating the already-checked grid (one
    # fewer full pass; identical values).  [micro-bench, (150, 10, 16) grid:
    # fate_anova 3.6 ms -> 0.6 ms]
    ite = _ite_pn_ps_validated(array, treat, control)

    return FateAnova(
        n_cavities=parts["n_cavities"],
        n_structures=parts["n_structures"],
        n_worlds=parts["n_worlds"],
        grand_mean=parts["grand_mean"],
        ss_total=parts["ss_total"],
        ss_structure=parts["ss_structure"],
        ss_noise=parts["ss_noise"],
        ss_interaction=parts["ss_interaction"],
        ss_residual=parts["ss_residual"],
        df_structure=parts["df_structure"],
        df_noise=parts["df_noise"],
        df_interaction=parts["df_interaction"],
        df_residual=parts["df_residual"],
        ms_structure=parts["ms_structure"],
        ms_noise=parts["ms_noise"],
        ms_interaction=parts["ms_interaction"],
        ms_residual=parts["ms_residual"],
        var_structure_raw=comps["var_structure_raw"],
        var_noise_raw=comps["var_noise_raw"],
        var_interaction_raw=comps["var_interaction_raw"],
        var_residual=comps["var_residual"],
        var_structure=comps["var_structure"],
        var_noise=comps["var_noise"],
        var_interaction=comps["var_interaction"],
        var_total=comps["var_total"],
        structure_share=comps["structure_share"],
        noise_share=comps["noise_share"],
        interaction_share=comps["interaction_share"],
        residual_share=comps["residual_share"],
        interaction_share_naive=comps["interaction_share_naive"],
        ss_structure_share=comps["ss_structure_share"],
        ss_noise_share=comps["ss_noise_share"],
        ss_interaction_share=comps["ss_interaction_share"],
        ss_residual_share=comps["ss_residual_share"],
        ite=ite,
    )


def _quantile_interval(values: list[float], alpha: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def bootstrap_interaction_share(
    Y: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapShareResult:
    """Cavity-bootstrap percentile CI for the corrected interaction share.

    A replicate resamples ``C`` complete cavities (rows of the grid) with
    replacement and recomputes the exact same corrected interaction share.
    Cavities, never individual cells, are the resampled exchangeable unit.
    Every run is deterministic given ``seed``.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral) or n_boot < 2:
        raise ValueError("n_boot must be an integer at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")

    array = _validate(Y)
    estimate = fate_anova(array)
    n_cavities = array.shape[0]
    rng = np.random.default_rng(int(seed))

    # One (n_boot, C) index draw (bit-identical PCG stream to n_boot sequential
    # size-C draws) feeding the batched replicate computation.
    indices = rng.integers(0, n_cavities, size=(int(n_boot), n_cavities))
    boot = [float(value) for value in _bootstrap_interaction_shares(array, indices)]

    lo, hi = _quantile_interval(boot, alpha)
    return BootstrapShareResult(
        estimate=estimate,
        point=estimate.interaction_share,
        lo=lo,
        hi=hi,
        boot_estimates=tuple(boot),
        alpha=float(alpha),
        seed=int(seed),
        n_boot=int(n_boot),
    )


__all__ = [
    "BootstrapShareResult",
    "FateAnova",
    "ITEResult",
    "bootstrap_interaction_share",
    "fate_anova",
    "ite_pn_ps",
]
