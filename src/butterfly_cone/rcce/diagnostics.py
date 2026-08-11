"""Convergence diagnostics and a generic per-sample scalar-channel API.

Everything here operates on plain scalar channels (one float per recorded
sweep) so a future learned-field score plugs in unchanged: register a callable,
and R-hat / ESS / initialisation-dependence machinery treats it identically to
the built-in energy and overlap channels.

Definitions:
- ``split_rhat`` is the standard split Gelman-Rubin potential-scale-reduction.
- ``integrated_autocorrelation_time`` uses Sokal automatic windowing; ESS is
  ``n_draws / tau_int``.
- Core overlap is IDENTITY-FREE (cell/site occupancy), because swap moves make
  particle-identity overlap meaningless (see PRIOR_LOCAL_MACHINERY.md gotchas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, Sequence

import torch

from butterfly_cone.engine.potential import minimum_image

from .cavity import CandidateState, ParentState

# ---------------------------------------------------------------------------
# Identity-free core overlap
# ---------------------------------------------------------------------------


def cell_occupancy_overlap(
    positions: torch.Tensor,
    reference_positions: torch.Tensor,
    box: torch.Tensor,
    *,
    cage_scale: float = 0.3,
) -> float:
    """Identity-free site-occupancy overlap with the parent configuration.

    For every reference (parent) site we ask whether *any* current particle sits
    within ``cage_scale`` (minimum image).  The overlap is the fraction of
    occupied sites.  This is invariant under diameter swaps and under any
    relabelling of particles, unlike an index-matched self-overlap.

    A perfectly retained cage gives ``1``; a fully decorrelated region gives the
    small random-occupancy background.
    """

    if cage_scale <= 0.0:
        raise ValueError("cage_scale must be positive")
    if reference_positions.shape[0] == 0:
        return 0.0
    # (n_ref, n_now, 3) minimum-image displacements.
    delta = reference_positions[:, None, :] - positions[None, :, :]
    delta = minimum_image(delta, box)
    squared = delta.square().sum(dim=2)
    occupied = (squared < cage_scale * cage_scale).any(dim=1)
    return float(occupied.to(torch.float64).mean())


# ---------------------------------------------------------------------------
# Generic scalar-channel API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelContext:
    """Everything a scalar channel may read for one recorded sweep."""

    positions: torch.Tensor
    diameters: torch.Tensor
    velocities: torch.Tensor
    box: torch.Tensor
    active_mask: torch.Tensor
    core_mask: torch.Tensor
    shell_mask: torch.Tensor
    parent_positions: torch.Tensor
    parent_core_positions: torch.Tensor
    active_potential_energy: float


ChannelFn = Callable[[ChannelContext], float]


@dataclass
class ChannelRegistry:
    """Named scalar channels evaluated once per recorded sweep.

    Accepts arbitrary user callables so a learned-field score is a first-class
    channel.  Built-in channels cover active potential energy and identity-free
    core overlap.
    """

    cage_scale: float = 0.3
    _channels: dict[str, ChannelFn] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._channels:
            self.register("active_potential_energy", lambda ctx: ctx.active_potential_energy)
            self.register("core_overlap", self._core_overlap)

    def _core_overlap(self, ctx: ChannelContext) -> float:
        return cell_occupancy_overlap(
            ctx.positions[ctx.active_mask],
            ctx.parent_core_positions,
            ctx.box,
            cage_scale=self.cage_scale,
        )

    def register(self, name: str, func: ChannelFn) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("channel name must be a non-empty string")
        if not callable(func):
            raise TypeError("channel must be callable")
        self._channels[name] = func

    @property
    def names(self) -> list[str]:
        return list(self._channels)

    def evaluate(self, ctx: ChannelContext) -> dict[str, float]:
        return {name: float(func(ctx)) for name, func in self._channels.items()}


# ---------------------------------------------------------------------------
# Split R-hat
# ---------------------------------------------------------------------------


def _as_2d(chains: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    if isinstance(chains, torch.Tensor):
        tensor = chains.detach().to(device="cpu", dtype=torch.float64)
    else:
        rows = [torch.as_tensor(row, dtype=torch.float64).flatten() for row in chains]
        if not rows:
            raise ValueError("chains must contain at least one chain")
        tensor = torch.stack(rows)
    if tensor.ndim != 2:
        raise ValueError("chains must be a 2D (n_chains, n_draws) array")
    return tensor


def split_rhat(chains: Sequence[Sequence[float]] | torch.Tensor) -> float:
    """Standard split Gelman-Rubin potential scale reduction factor.

    Each of the ``m`` input chains is split in half to expose within-chain
    non-stationarity, giving ``2m`` sequences of length ``n//2``.
    """

    tensor = _as_2d(chains)
    m, n = tensor.shape
    half = n // 2
    if half < 2 or m < 1:
        return float("nan")
    split = torch.cat((tensor[:, :half], tensor[:, half : 2 * half]), dim=0)
    n_draws = half
    chain_means = split.mean(dim=1)
    chain_vars = split.var(dim=1, unbiased=True)
    grand_mean = chain_means.mean()
    between = n_draws * chain_means.sub(grand_mean).square().sum() / (split.shape[0] - 1)
    within = chain_vars.mean()
    if float(within) <= 0.0:
        # 0/0 is not evidence of convergence: an identically constant order
        # parameter can be a completely caged/stuck sampler. Treat it as an
        # undefined diagnostic that fails the finite R-hat gate.
        return float("inf")
    var_plus = (n_draws - 1) / n_draws * within + between / n_draws
    return float(torch.sqrt(var_plus / within))


# ---------------------------------------------------------------------------
# Integrated autocorrelation time and ESS (Sokal automatic windowing)
# ---------------------------------------------------------------------------


def autocorrelation(x: Sequence[float] | torch.Tensor) -> torch.Tensor:
    series = torch.as_tensor(x, dtype=torch.float64).detach().cpu().flatten()
    n = series.numel()
    if n < 2:
        return torch.ones(1, dtype=torch.float64)
    centered = series - series.mean()
    variance = centered.dot(centered) / n
    if float(variance) <= 0.0:
        return torch.full((n,), float("nan"), dtype=torch.float64)
    # FFT convolution makes long synthetic and pilot chains O(n log n), not
    # O(n^2). Divide by the number of overlapping terms for each lag.
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size)
    covariance = torch.fft.irfft(spectrum.conj() * spectrum, n=fft_size)[:n]
    covariance = covariance / torch.arange(n, 0, -1, dtype=torch.float64)
    return covariance / covariance[0]


def integrated_autocorrelation_time(
    x: Sequence[float] | torch.Tensor,
    *,
    c: float = 5.0,
) -> float:
    """Geyer initial-positive/monotone-pair estimate of ``tau_int >= 1``.

    The retained ``c`` argument is API-compatible with the earlier Sokal
    implementation; pair truncation is less prone to accumulating the noisy
    long-lag tail in the relatively short cavity traces.
    """

    rho = autocorrelation(x)
    n = rho.numel()
    if not bool(torch.all(torch.isfinite(rho))):
        return float("inf")
    if c <= 0.0:
        raise ValueError("autocorrelation window control must be positive")
    tau = 1.0
    previous_pair = float("inf")
    for lag in range(1, n - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair <= 0.0:
            break
        pair = min(pair, previous_pair)
        tau += 2.0 * pair
        previous_pair = pair
    return max(float(tau), 1.0)


def effective_sample_size(x: Sequence[float] | torch.Tensor, *, c: float = 5.0) -> float:
    """Effective sample size ``n / tau_int`` for one chain/channel."""

    series = torch.as_tensor(x, dtype=torch.float64).flatten()
    n = int(series.numel())
    if n < 2:
        return float(n)
    tau = integrated_autocorrelation_time(series, c=c)
    return min(float(n), n / tau)


def ess_per_chain(chains: Sequence[Sequence[float]] | torch.Tensor, *, c: float = 5.0) -> list[float]:
    tensor = _as_2d(chains)
    return [effective_sample_size(tensor[row], c=c) for row in range(tensor.shape[0])]


# ---------------------------------------------------------------------------
# Initialisation-dependence check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyInitializationDependence:
    mean: float
    z_score: float
    ess: float
    flagged: bool


@dataclass(frozen=True)
class InitializationDependence:
    channel: str
    pooled_mean: float
    per_family_mean: dict[str, float]
    per_family_z: dict[str, float]
    flagged: bool
    threshold: float
    per_family_ess: dict[str, float]

    @property
    def persistent(self) -> bool:
        return self.flagged

    @property
    def by_family(self) -> dict[str, FamilyInitializationDependence]:
        return {
            family: FamilyInitializationDependence(
                mean=mean,
                z_score=self.per_family_z[family],
                ess=self.per_family_ess[family],
                flagged=abs(self.per_family_z[family]) > self.threshold,
            )
            for family, mean in self.per_family_mean.items()
        }


def initialization_dependence(
    per_family_samples: Mapping[str, Sequence[float]],
    *,
    channel: str = "scalar",
    z_threshold: float = 3.0,
) -> InitializationDependence:
    """Flag persistent init dependence: family mean vs pooled, in SEM units.

    Each family's posterior mean is compared to the pooled mean; the spread is
    measured in standard errors of that family's own mean.  Any family beyond
    ``z_threshold`` flags persistent dependence.
    """

    pooled: list[float] = []
    family_mean: dict[str, float] = {}
    family_arrays: dict[str, torch.Tensor] = {}
    for family, samples in per_family_samples.items():
        arr = torch.as_tensor(samples, dtype=torch.float64).flatten()
        family_arrays[family] = arr
        pooled.extend(arr.tolist())
        family_mean[family] = float(arr.mean()) if arr.numel() else float("nan")
    if len(family_arrays) < 2 or any(array.numel() < 2 for array in family_arrays.values()):
        raise ValueError("initialization dependence requires at least two families with two samples each")
    pooled_tensor = torch.as_tensor(pooled, dtype=torch.float64)
    pooled_mean = float(pooled_tensor.mean()) if pooled_tensor.numel() else float("nan")
    pooled_variance = float(pooled_tensor.var(unbiased=True))
    family_ess = {
        family: effective_sample_size(array)
        for family, array in family_arrays.items()
    }
    pooled_ess = sum(family_ess.values())

    family_z: dict[str, float] = {}
    flagged = False
    for family, arr in family_arrays.items():
        n = arr.numel()
        if n < 2:
            family_z[family] = float("nan")
            continue
        family_variance = float(arr.var(unbiased=True))
        sem = math.sqrt(
            family_variance / max(family_ess[family], 1.0)
            + pooled_variance / max(pooled_ess, 1.0)
        )
        if sem <= 0.0:
            z = 0.0 if abs(family_mean[family] - pooled_mean) < 1e-12 else float("inf")
        else:
            z = (family_mean[family] - pooled_mean) / sem
        family_z[family] = z
        if abs(z) > z_threshold:
            flagged = True
    return InitializationDependence(
        channel=channel,
        pooled_mean=pooled_mean,
        per_family_mean=family_mean,
        per_family_z=family_z,
        flagged=flagged,
        threshold=z_threshold,
        per_family_ess=family_ess,
    )


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@dataclass
class CostAccounting:
    """Candidate-generation cost, in engine work units, for one chain."""

    md_steps: int = 0
    swap_attempts: int = 0
    swap_sweeps: int = 0
    relax_steps: int = 0
    rejected_segments: int = 0
    recorded_samples: int = 0

    def add(self, other: "CostAccounting") -> None:
        self.md_steps += other.md_steps
        self.swap_attempts += other.swap_attempts
        self.swap_sweeps += other.swap_sweeps
        self.relax_steps += other.relax_steps
        self.rejected_segments += other.rejected_segments
        self.recorded_samples += other.recorded_samples

    def cost_per_decorrelated_sample(self, ess: float) -> dict[str, float]:
        """MD steps, swap attempts and relaxation steps per decorrelated sample."""

        divisor = ess if ess and ess > 0.0 else float("nan")
        return {
            "md_steps_per_sample": self.md_steps / divisor,
            "swap_attempts_per_sample": self.swap_attempts / divisor,
            "relax_steps_per_sample": self.relax_steps / divisor,
            "ess": float(ess),
        }

    def to_dict(self) -> dict[str, int]:
        return {
            "md_steps": self.md_steps,
            "swap_attempts": self.swap_attempts,
            "swap_sweeps": self.swap_sweeps,
            "relax_steps": self.relax_steps,
            "rejected_segments": self.rejected_segments,
            "recorded_samples": self.recorded_samples,
        }


# ---------------------------------------------------------------------------
# Candidate-oriented scalar API used by RCCE chains and the pilot
# ---------------------------------------------------------------------------


def core_parent_overlap(
    candidate: CandidateState,
    parent: ParentState,
    *,
    cage_scale: float = 0.3,
) -> float:
    """Collective core-reference occupancy by any current buffer particle.

    Particle labels never enter the comparison.  This makes the observable
    exactly invariant to diameter-label swaps and repairs the self-overlap bug
    documented in ``PRIOR_LOCAL_MACHINERY.md``.
    """

    if candidate.provenance.parent_id != parent.parent_id:
        raise ValueError("candidate and overlap parent IDs differ")
    if candidate.core_indices.numel() == 0:
        raise ValueError("core overlap is undefined for an empty core")
    return cell_occupancy_overlap(
        candidate.positions[candidate.buffer_indices],
        parent.positions[candidate.core_indices],
        candidate.box,
        cage_scale=cage_scale,
    )


CandidateChannel = Callable[[CandidateState], float]


class ScalarChannelRegistry:
    """Named candidate-to-scalar functions with no channel-specific branches."""

    def __init__(self) -> None:
        self._channels: dict[str, CandidateChannel] = {}

    def register(self, name: str, function: CandidateChannel) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("channel name must be a non-empty string")
        if name in self._channels:
            raise ValueError(f"channel {name!r} is already registered")
        if not callable(function):
            raise TypeError("channel must be callable")
        self._channels[name] = function

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._channels)

    def evaluate(self, candidate: CandidateState) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, function in self._channels.items():
            try:
                value = float(function(candidate))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"channel {name!r} must return a finite scalar") from error
            if not math.isfinite(value):
                raise ValueError(f"channel {name!r} must return a finite scalar")
            result[name] = value
        return result


def _candidate_active_energy(candidate: CandidateState) -> float:
    recorded = candidate.observables.get("active_potential_energy")
    if recorded is not None:
        return float(recorded)
    # A fallback keeps externally constructed candidates compatible with the
    # same channel API; sampler-produced candidates use the cached value.
    from .sampler import conditional_potential

    return float(conditional_potential(candidate.to_system()).energy)


def default_channel_registry(
    parent: ParentState,
    *,
    cage_scale: float = 0.3,
) -> ScalarChannelRegistry:
    registry = ScalarChannelRegistry()
    registry.register("active_potential_energy", _candidate_active_energy)
    registry.register(
        "core_parent_overlap",
        lambda candidate: core_parent_overlap(candidate, parent, cage_scale=cage_scale),
    )
    return registry


def evaluate_samples(
    samples_by_chain: Mapping[str, Sequence[CandidateState]],
    registry: ScalarChannelRegistry,
) -> dict[str, dict[str, list[float]]]:
    traces = {
        name: {str(chain_id): [] for chain_id in samples_by_chain}
        for name in registry.names
    }
    for chain_id, samples in samples_by_chain.items():
        for sample in samples:
            values = registry.evaluate(sample)
            for name, value in values.items():
                traces[name][str(chain_id)].append(value)
    return traces


@dataclass(frozen=True)
class ScalarChannelDiagnostics:
    split_rhat: float
    iat_by_chain: dict[str, float]
    ess_by_chain: dict[str, float]
    stuck: bool

    @property
    def min_ess(self) -> float:
        return min(self.ess_by_chain.values()) if self.ess_by_chain else 0.0


def diagnose_scalar_channel(
    chains: Mapping[str, Sequence[float]],
) -> ScalarChannelDiagnostics:
    if len(chains) < 2:
        raise ValueError("split-Rhat requires at least two chains")
    lengths = [len(values) for values in chains.values()]
    if not lengths or min(lengths) < 4:
        raise ValueError("each chain needs at least four samples")
    common_length = min(lengths)
    truncated = [list(values)[:common_length] for values in chains.values()]
    iat = {
        str(chain_id): integrated_autocorrelation_time(values)
        for chain_id, values in chains.items()
    }
    ess = {
        str(chain_id): effective_sample_size(values)
        for chain_id, values in chains.items()
    }
    stuck = any(
        len(values) > 0
        and float(torch.as_tensor(values, dtype=torch.float64).var(unbiased=False)) == 0.0
        for values in chains.values()
    )
    return ScalarChannelDiagnostics(
        split_rhat=split_rhat(truncated),
        iat_by_chain=iat,
        ess_by_chain=ess,
        stuck=stuck,
    )


@dataclass(frozen=True)
class CandidateGenerationCost:
    decorrelated_samples: float
    md_steps_per_sample: float
    swap_sweeps_per_sample: float
    swap_attempts_per_sample: float
    relaxation_steps_per_sample: float

    def to_dict(self) -> dict[str, float]:
        return {
            "decorrelated_samples": self.decorrelated_samples,
            "md_steps_per_sample": self.md_steps_per_sample,
            "swap_sweeps_per_sample": self.swap_sweeps_per_sample,
            "swap_attempts_per_sample": self.swap_attempts_per_sample,
            "relaxation_steps_per_sample": self.relaxation_steps_per_sample,
        }


def candidate_generation_cost(
    costs: Sequence[object],
    ess_by_channel: Mapping[str, Mapping[str, float]],
) -> CandidateGenerationCost:
    """Aggregate executed work over the most conservative total channel ESS."""

    if not costs:
        raise ValueError("cost accounting requires at least one chain cost")
    channel_totals = [sum(float(value) for value in per_chain.values()) for per_chain in ess_by_channel.values()]
    positive_totals = [value for value in channel_totals if math.isfinite(value) and value > 0.0]
    if not positive_totals:
        return CandidateGenerationCost(
            decorrelated_samples=0.0,
            md_steps_per_sample=float("inf"),
            swap_sweeps_per_sample=float("inf"),
            swap_attempts_per_sample=float("inf"),
            relaxation_steps_per_sample=float("inf"),
        )
    # Zero-ESS stuck channels invalidate convergence separately. Retain a
    # clearly provisional work/ESS ratio from the least-effective non-stuck
    # channel so failed pilots still expose their computational scale.
    decorrelated = min(positive_totals)

    def total(primary: str, fallback: str | None = None) -> float:
        return float(
            sum(
                getattr(cost, primary, getattr(cost, fallback, 0) if fallback is not None else 0)
                for cost in costs
            )
        )

    return CandidateGenerationCost(
        decorrelated_samples=decorrelated,
        md_steps_per_sample=total("md_steps") / decorrelated,
        swap_sweeps_per_sample=total("swap_sweeps") / decorrelated,
        swap_attempts_per_sample=total("swap_attempts") / decorrelated,
        relaxation_steps_per_sample=total("relaxation_steps", "relax_steps") / decorrelated,
    )
