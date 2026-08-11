"""Patterned multi-site RCCE composition.

This module is deliberately an orchestration layer.  It owns no molecular
dynamics kernel and no sampler: every site is selected with
``select_cavity`` and sampled by the existing :class:`RCCEChain` (or by a
caller-supplied candidate library in tests and downstream replay workflows).

The important state transition is sequential.  A site is selected against the
configuration that exists immediately before that site, only its buffer is
spliced back, and that resulting configuration is the parent seen by the next
site.  The final emitted parent is fully active because RCCE's frozen exterior
is a sampling device, not a branch-phase constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
import io
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from butterfly_cone.engine.potential import minimum_image
from butterfly_cone.engine.system import ParticleSystem

from .cavity import (
    CandidateProvenance,
    CandidateState,
    CavitySelection,
    CavitySpec,
    ParentState,
    select_cavity,
)
from .diagnostics import diagnose_scalar_channel
from .sampler import (
    ChainInitError,
    InitFamily,
    RCCEChain,
    RCCEConfig,
    RCCESeeds,
    SamplerCost,
)


Sign = str
FieldScoreFn = Callable[..., float]
ProtectedMaskFn = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

_SIGNS = {"+", "-", "0"}
_OVERLAP_ALIASES = {
    "auto": "auto",
    "non_overlapping": "non_overlapping",
    "non-overlapping": "non_overlapping",
    "sparse": "non_overlapping",
    "overlapping": "overlapping",
    "overlap": "overlapping",
    "dense": "overlapping",
    "sequential": "overlapping",
}


def _box_tuple(box: Sequence[float] | torch.Tensor) -> tuple[float, float, float]:
    # CPU before float64: MPS has no float64, and as_tensor casts on the source
    # device first, so an MPS box tensor must be moved off-device before the cast.
    values = torch.as_tensor(box).detach().cpu().to(torch.float64).flatten().tolist()
    if len(values) != 3 or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("box must contain three positive finite lengths")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _box_tensor(
    box: Sequence[float] | torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(box, device=device, dtype=dtype).reshape(3)


def _wrapped_center(center: Sequence[float], box: tuple[float, float, float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in center)
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("each center must contain three finite coordinates")
    return tuple(value % length for value, length in zip(values, box, strict=True))  # type: ignore[return-value]


def _minimum_image_float(delta: float, length: float) -> float:
    return delta - length * round(delta / length)


def _center_distance(left: Sequence[float], right: Sequence[float], box: Sequence[float]) -> float:
    delta = [
        _minimum_image_float(float(a) - float(b), float(length))
        for a, b, length in zip(left, right, box, strict=True)
    ]
    return math.sqrt(sum(value * value for value in delta))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_direction(
    axis: int | str | Sequence[float] | None,
    direction: Sequence[float] | None,
) -> tuple[tuple[float, float, float], str]:
    if axis is not None and direction is not None:
        raise ValueError("provide either axis or direction, not both")
    if direction is None:
        value: int | str | Sequence[float] = 0 if axis is None else axis
        if isinstance(value, str):
            names = {"x": 0, "y": 1, "z": 2}
            if value.lower() not in names:
                raise ValueError("axis must be 0, 1, 2, x, y, or z")
            index = names[value.lower()]
        elif isinstance(value, int) and not isinstance(value, bool):
            index = value
        else:
            raise ValueError("axis must be an integer or x/y/z")
        if index not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        unit = [0.0, 0.0, 0.0]
        unit[index] = 1.0
        return tuple(unit), f"axis-{index}"
    raw = tuple(float(value) for value in direction)
    if len(raw) != 3 or any(not math.isfinite(value) for value in raw):
        raise ValueError("direction must contain three finite coordinates")
    norm = math.sqrt(sum(value * value for value in raw))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    unit = tuple(value / norm for value in raw)
    return unit, "vector-" + ",".join(f"{value:g}" for value in unit)


def _validate_site_count(k: int, name: str = "k") -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(k)


@dataclass(frozen=True)
class StencilTemplate:
    """Ordered PBC-aware site centers and their default sign pattern."""

    centers: tuple[tuple[float, float, float], ...]
    default_signs: tuple[Sign, ...]
    geometry: Mapping[str, Any]
    box: tuple[float, float, float]
    name: str = "stencil"
    _protected_region_mask_fn: ProtectedMaskFn | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        box = _box_tuple(self.box)
        centers = tuple(_wrapped_center(center, box) for center in self.centers)
        signs = tuple(str(sign) for sign in self.default_signs)
        if not centers:
            raise ValueError("stencil template must contain at least one center")
        if len(signs) != len(centers):
            raise ValueError("default_signs must have one entry per center")
        if any(sign not in _SIGNS for sign in signs):
            raise ValueError("target signs must be '+', '-', or '0'")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("template name must be non-empty")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "default_signs", signs)
        object.__setattr__(self, "geometry", _jsonable(dict(self.geometry)))

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    @property
    def site_count(self) -> int:
        return self.n_sites

    @property
    def signs(self) -> tuple[Sign, ...]:
        return self.default_signs

    @property
    def geometry_record(self) -> Mapping[str, Any]:
        return self.geometry

    @property
    def protected_region_mask(self) -> ProtectedMaskFn | None:
        """Callable returning a boolean protected/unprotected particle mask."""

        return self._protected_region_mask_fn

    @property
    def protected_mask(self) -> ProtectedMaskFn | None:
        return self._protected_region_mask_fn

    @property
    def protected_region_mask_fn(self) -> ProtectedMaskFn | None:
        return self._protected_region_mask_fn

    def particle_protected_mask(
        self,
        positions: torch.Tensor,
        box: Sequence[float] | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self._protected_region_mask_fn is None:
            return None
        return self._protected_region_mask_fn(positions, None if box is None else torch.as_tensor(box))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "box": list(self.box),
            "centers": [list(center) for center in self.centers],
            "default_signs": list(self.default_signs),
            "geometry": _jsonable(self.geometry),
            "has_protected_region_mask": self._protected_region_mask_fn is not None,
        }


def line_template(
    box: Sequence[float] | torch.Tensor,
    k: int,
    axis: int | str | Sequence[float] | None = None,
    *,
    direction: Sequence[float] | None = None,
    spacing: float | None = None,
    span: float | None = None,
) -> StencilTemplate:
    """Return ``k`` ordered collinear sites centered in an orthorhombic box.

    With neither ``spacing`` nor ``span``, the line uses the natural periodic
    spacing ``extent / k``.  Supplying one control is exact: ``span`` is the
    distance from the first to last unwrapped center and ``spacing`` is the
    adjacent-center distance.  The centers are wrapped only after their
    ordered unwrapped construction.
    """

    clean_box = _box_tuple(box)
    count = _validate_site_count(k)
    unit, direction_label = _resolve_direction(axis, direction)
    if spacing is not None and (not math.isfinite(float(spacing)) or float(spacing) <= 0.0):
        raise ValueError("spacing must be positive and finite")
    if span is not None and (not math.isfinite(float(span)) or float(span) < 0.0):
        raise ValueError("span must be non-negative and finite")
    if spacing is not None and span is not None and count > 1:
        expected = (count - 1) * float(spacing)
        if not math.isclose(expected, float(span), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("spacing and span disagree: span must equal (k - 1) * spacing")
    if count == 1:
        clean_spacing = 0.0
        clean_span = 0.0
    else:
        if isinstance(axis, (int, str)) or (axis is None and direction is None):
            axis_index = 0 if axis is None else ({"x": 0, "y": 1, "z": 2}.get(axis.lower(), -1) if isinstance(axis, str) else axis)
            if axis_index not in (0, 1, 2):
                axis_index = max(range(3), key=lambda index: abs(unit[index]))
            extent = clean_box[axis_index]
        else:
            extent_candidates = [
                clean_box[index] / abs(unit[index])
                for index in range(3)
                if abs(unit[index]) > 1e-15
            ]
            extent = min(extent_candidates)
        if spacing is None and span is None:
            clean_spacing = extent / count
            clean_span = clean_spacing * (count - 1)
        elif spacing is None:
            clean_span = float(span)
            clean_spacing = clean_span / (count - 1)
        else:
            clean_spacing = float(spacing)
            clean_span = clean_spacing * (count - 1)
    midpoint = tuple(length / 2.0 for length in clean_box)
    centers = []
    for index in range(count):
        offset = -0.5 * clean_span + clean_spacing * index
        centers.append(
            _wrapped_center(
                tuple(midpoint[axis_index] + offset * unit[axis_index] for axis_index in range(3)),
                clean_box,
            )
        )
    return StencilTemplate(
        centers=tuple(centers),
        default_signs=("+",) * count,
        box=clean_box,
        name="line",
        geometry={
            "kind": "line",
            "direction": list(unit),
            "direction_label": direction_label,
            "spacing": clean_spacing,
            "span": clean_span,
            "periodic_extent": extent if count > 1 else 0.0,
        },
    )


def _normal_vector(normal: int | str | Sequence[float]) -> tuple[float, float, float]:
    vector, _ = _resolve_direction(normal if isinstance(normal, (int, str)) else None, None if isinstance(normal, (int, str)) else normal)
    return vector


def _plane_point(
    box: tuple[float, float, float],
    normal: tuple[float, float, float],
    plane_position: float | Sequence[float],
) -> tuple[float, float, float]:
    if isinstance(plane_position, (int, float)) and not isinstance(plane_position, bool):
        value = float(plane_position)
        axis = max(range(3), key=lambda index: abs(normal[index]))
        if sum(abs(normal[index]) for index in range(3) if index != axis) < 1e-12:
            point = [length / 2.0 for length in box]
            point[axis] = value
            return _wrapped_center(point, box)
        return _wrapped_center(tuple(box[index] / 2.0 + value * normal[index] for index in range(3)), box)
    point = tuple(float(value) for value in plane_position)
    if len(point) != 3:
        raise ValueError("plane_position must be a scalar or a three-coordinate point")
    return _wrapped_center(point, box)


def wall_template(
    box: Sequence[float] | torch.Tensor,
    normal: int | str | Sequence[float],
    thickness_sites: int,
    plane_position: float | Sequence[float],
    *,
    site_spacing: float | None = None,
) -> StencilTemplate:
    """Return an ordered planar slab and a PBC-aware protected half-space.

    ``thickness_sites`` is the number of RCCE cavities in the normal
    direction.  The default spacing is deliberately compact; callers that
    want a thicker contiguous wall can set ``site_spacing`` explicitly.
    The protected side is the non-negative signed side of the plane normal.
    """

    clean_box = _box_tuple(box)
    count = _validate_site_count(thickness_sites, "thickness_sites")
    unit = _normal_vector(normal)
    point = _plane_point(clean_box, unit, plane_position)
    if site_spacing is None:
        clean_spacing = min(clean_box) / max(4.0 * count, 4.0)
    else:
        clean_spacing = float(site_spacing)
        if not math.isfinite(clean_spacing) or clean_spacing <= 0.0:
            raise ValueError("site_spacing must be positive and finite")
    center = torch.tensor(point, dtype=torch.float64)
    normal_tensor = torch.tensor(unit, dtype=torch.float64)
    centers = []
    for index in range(count):
        offset = (index - 0.5 * (count - 1)) * clean_spacing
        centers.append(_wrapped_center((center + offset * normal_tensor).tolist(), clean_box))

    def protected_mask(
        positions: torch.Tensor,
        runtime_box: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(positions)
        target_box = _box_tensor(clean_box if runtime_box is None else runtime_box, device=values.device, dtype=values.dtype)
        point_tensor = torch.as_tensor(point, device=values.device, dtype=values.dtype)
        normal_value = torch.as_tensor(unit, device=values.device, dtype=values.dtype)
        signed = minimum_image(values - point_tensor, target_box) @ normal_value
        return signed >= 0.0

    return StencilTemplate(
        centers=tuple(centers),
        default_signs=("-",) * count,
        box=clean_box,
        name="wall",
        geometry={
            "kind": "wall",
            "normal": list(unit),
            "plane_position": list(point),
            "site_spacing": clean_spacing,
            "protected_side": ">=0 signed distance",
        },
        _protected_region_mask_fn=protected_mask,
    )


def _plane_axes(plane: str | int | Sequence[int]) -> tuple[int, int]:
    if isinstance(plane, str):
        names = {"x": 0, "y": 1, "z": 2}
        raw = tuple(names.get(char.lower(), -1) for char in plane)
        if len(raw) != 2 or any(index not in (0, 1, 2) for index in raw) or raw[0] == raw[1]:
            raise ValueError("plane must be one of xy, xz, or yz")
        return raw  # type: ignore[return-value]
    if isinstance(plane, int) and not isinstance(plane, bool):
        if plane not in (0, 1, 2):
            raise ValueError("plane normal axis must be 0, 1, or 2")
        remaining = tuple(index for index in range(3) if index != plane)
        return remaining  # type: ignore[return-value]
    raw = tuple(int(index) for index in plane)
    if len(raw) != 2 or any(index not in (0, 1, 2) for index in raw) or raw[0] == raw[1]:
        raise ValueError("plane must contain two distinct axes")
    return raw  # type: ignore[return-value]


def ring_template(
    box: Sequence[float] | torch.Tensor,
    plane: str | int | Sequence[int],
    radius: float,
    k: int,
) -> StencilTemplate:
    """Return ``k`` ordered centers on a closed periodic loop."""

    clean_box = _box_tuple(box)
    count = _validate_site_count(k)
    if count < 3:
        raise ValueError("ring requires at least three sites")
    clean_radius = float(radius)
    if not math.isfinite(clean_radius) or clean_radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    axes = _plane_axes(plane)
    if clean_radius >= 0.5 * min(clean_box[index] for index in axes):
        raise ValueError("ring radius must be smaller than half the shortest in-plane box length")
    midpoint = [length / 2.0 for length in clean_box]
    centers = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        point = midpoint.copy()
        point[axes[0]] += clean_radius * math.cos(angle)
        point[axes[1]] += clean_radius * math.sin(angle)
        centers.append(_wrapped_center(point, clean_box))

    def protected_mask(
        positions: torch.Tensor,
        runtime_box: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(positions)
        target_box = _box_tensor(clean_box if runtime_box is None else runtime_box, device=values.device, dtype=values.dtype)
        center_tensor = torch.as_tensor(midpoint, device=values.device, dtype=values.dtype)
        delta = minimum_image(values - center_tensor, target_box)
        radial_square = delta[..., axes[0]].square() + delta[..., axes[1]].square()
        return radial_square <= clean_radius * clean_radius

    return StencilTemplate(
        centers=tuple(centers),
        default_signs=("+",) * count,
        box=clean_box,
        name="ring",
        geometry={
            "kind": "ring",
            "plane": list(axes),
            "radius": clean_radius,
            "protected_region": f"in-plane radial distance <= {clean_radius:g}",
        },
        _protected_region_mask_fn=protected_mask,
    )


@dataclass(frozen=True)
class StencilSpec:
    """Complete frozen stencil protocol and provenance configuration."""

    template: StencilTemplate
    cavity_specs: tuple[CavitySpec, ...]
    target_signs: tuple[Sign, ...] | None
    config: RCCEConfig
    burn_in_sweeps: int = 0
    production_sweeps: int = 4
    exact_core_composition: bool = False
    overlap_regime: str = "auto"
    sample_interval: int = 1
    rhat_max: float = 1.05
    ess_min: float = 50.0
    drift_tolerance: float = 0.05
    field_score_fn: FieldScoreFn | None = field(default=None, repr=False, compare=False)
    field_scores: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.template, StencilTemplate):
            raise TypeError("template must be a StencilTemplate")
        cavities = tuple(self.cavity_specs)
        if len(cavities) != self.template.n_sites:
            raise ValueError("cavity_specs must contain one CavitySpec per template site")
        if any(not isinstance(spec, CavitySpec) for spec in cavities):
            raise TypeError("cavity_specs must contain CavitySpec values")
        for index, (center, cavity) in enumerate(zip(self.template.centers, cavities, strict=True)):
            if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(center, cavity.center, strict=True)):
                raise ValueError(f"cavity_specs[{index}] center does not match template center")
        signs = self.template.default_signs if self.target_signs is None else tuple(str(sign) for sign in self.target_signs)
        if len(signs) != self.template.n_sites or any(sign not in _SIGNS for sign in signs):
            raise ValueError("target_signs must contain one of '+', '-', or '0' per site")
        if isinstance(self.burn_in_sweeps, bool) or not isinstance(self.burn_in_sweeps, int) or self.burn_in_sweeps < 0:
            raise ValueError("burn_in_sweeps must be a non-negative integer")
        if isinstance(self.production_sweeps, bool) or not isinstance(self.production_sweeps, int) or self.production_sweeps <= 0:
            raise ValueError("production_sweeps must be a positive integer")
        if isinstance(self.sample_interval, bool) or not isinstance(self.sample_interval, int) or self.sample_interval <= 0:
            raise ValueError("sample_interval must be a positive integer")
        if self.production_sweeps // self.sample_interval < 4:
            raise ValueError("production protocol must record at least four samples per chain")
        regime = _OVERLAP_ALIASES.get(str(self.overlap_regime).lower())
        if regime is None:
            raise ValueError("overlap_regime must be non_overlapping or overlapping")
        if regime == "auto":
            regime = "overlapping" if self.template.name in {"wall", "ring"} else "non_overlapping"
        for name, value in (("rhat_max", self.rhat_max), ("ess_min", self.ess_min), ("drift_tolerance", self.drift_tolerance)):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(self, "cavity_specs", cavities)
        object.__setattr__(self, "target_signs", signs)
        object.__setattr__(self, "overlap_regime", regime)
        effective_config = replace(self.config, exact_core_composition=bool(self.exact_core_composition))
        object.__setattr__(self, "config", effective_config)
        self.validate_geometry(self.template.box)

    @property
    def n_sites(self) -> int:
        return self.template.n_sites

    @property
    def cavities(self) -> tuple[CavitySpec, ...]:
        return self.cavity_specs

    @property
    def effective_config(self) -> RCCEConfig:
        return self.config

    def validate_geometry(self, box: Sequence[float] | torch.Tensor) -> None:
        clean_box = _box_tuple(box)
        shortest_half = 0.5 * min(clean_box)
        if any(cavity.buffer_radius > shortest_half + 1e-12 for cavity in self.cavity_specs):
            raise ValueError(
                "stencil is infeasible: every buffer_radius must not exceed half the shortest box length"
            )
        if self.overlap_regime != "non_overlapping":
            return
        for left_index in range(self.n_sites):
            for right_index in range(left_index + 1, self.n_sites):
                left = self.cavity_specs[left_index]
                right = self.cavity_specs[right_index]
                separation = _center_distance(left.center, right.center, clean_box)
                required = left.buffer_radius + right.buffer_radius
                if separation + 1e-9 < required:
                    raise ValueError(
                        "non-overlapping stencil is infeasible: center spacing "
                        f"{separation:.6g} is smaller than buffer spacing requirement "
                        f"{required:.6g} for sites {left_index} and {right_index}"
                    )

    def to_dict(self) -> dict[str, Any]:
        scorer = self.field_score_fn
        if scorer is not None:
            scorer_name = getattr(scorer, "__qualname__", getattr(scorer, "__name__", type(scorer).__name__))
        else:
            scorer_name = None
        return {
            "template": self.template.to_dict(),
            "cavity_specs": [spec.to_dict() for spec in self.cavity_specs],
            "target_signs": list(self.target_signs or ()),
            "rcce_config": asdict(self.config),
            "burn_in_sweeps": self.burn_in_sweeps,
            "production_sweeps": self.production_sweeps,
            "sample_interval": self.sample_interval,
            "exact_core_composition": self.exact_core_composition,
            "overlap_regime": self.overlap_regime,
            "rhat_max": self.rhat_max,
            "ess_min": self.ess_min,
            "drift_tolerance": self.drift_tolerance,
            "field_score_function": scorer_name,
            "field_scores": _jsonable(self.field_scores),
        }

    to_provenance = to_dict


@dataclass(frozen=True)
class StencilDiagnostics:
    """R-hat/ESS summary for one site's recorded candidate channels."""

    channels: Mapping[str, Mapping[str, Any]]
    max_split_rhat: float
    min_ess: float
    rhat_pass: bool
    ess_pass: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": _jsonable(self.channels),
            "max_split_rhat": self.max_split_rhat,
            "min_ess": self.min_ess,
            "rhat_pass": self.rhat_pass,
            "ess_pass": self.ess_pass,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EarlierCoreDrift:
    """Post-write re-score of all earlier cores."""

    per_site_delta: Mapping[int, float]
    max_abs_delta: float
    tolerance: float
    available: bool
    flagged: bool

    @property
    def drift_flag(self) -> bool:
        return self.flagged

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_site_delta": {str(key): float(value) for key, value in self.per_site_delta.items()},
            "max_abs_delta": self.max_abs_delta,
            "tolerance": self.tolerance,
            "available": self.available,
            "flagged": self.flagged,
        }


@dataclass(frozen=True)
class StencilSiteRecord:
    """Immutable audit record for one conditional site edit."""

    site_index: int
    cavity_spec: CavitySpec
    selection: CavitySelection
    buffer_mask: torch.Tensor
    buffer_indices: torch.Tensor
    core_mask: torch.Tensor
    core_indices: torch.Tensor
    frozen_exterior_indices: torch.Tensor
    frozen_exterior_positions: torch.Tensor
    post_site_positions: torch.Tensor
    post_site_velocities: torch.Tensor
    post_site_diameters: torch.Tensor
    post_site_unwrapped_positions: torch.Tensor
    candidate_provenance: CandidateProvenance
    selected_sign: Sign
    field_score: float
    diagnostics: StencilDiagnostics
    earlier_core_drift: EarlierCoreDrift
    displacement_dose: float
    energy_dose: float
    candidate: CandidateState | None = field(default=None, repr=False, compare=False)

    @property
    def provenance(self) -> CandidateProvenance:
        return self.candidate_provenance

    @property
    def sign(self) -> Sign:
        return self.selected_sign

    @property
    def drift_flag(self) -> bool:
        return self.earlier_core_drift.flagged

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_index": self.site_index,
            "cavity_spec": self.cavity_spec.to_dict(),
            "buffer_indices": self.buffer_indices.detach().cpu().tolist(),
            "core_indices": self.core_indices.detach().cpu().tolist(),
            "n_buffer": int(self.buffer_indices.numel()),
            "n_core": int(self.core_indices.numel()),
            "selected_sign": self.selected_sign,
            "field_score": self.field_score,
            "candidate_provenance": self.candidate_provenance.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "earlier_core_drift": self.earlier_core_drift.to_dict(),
            "displacement_dose": self.displacement_dose,
            "energy_dose": self.energy_dose,
        }


@dataclass(frozen=True)
class StencilSiteLibrary:
    """Candidate samples for one site, grouped by initialization family."""

    site_index: int
    selection: CavitySelection
    samples_by_chain: Mapping[str, tuple[CandidateState, ...]]
    cost: Mapping[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> tuple[CandidateState, ...]:
        return tuple(sample for chain in self.samples_by_chain.values() for sample in chain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_index": self.site_index,
            "n_chains": len(self.samples_by_chain),
            "samples_per_chain": {str(key): len(value) for key, value in self.samples_by_chain.items()},
            "cost": _jsonable(self.cost),
        }


@dataclass(frozen=True)
class StencilCandidateLibrary:
    """All per-site production samples retained for arm matching/replay."""

    sites: tuple[StencilSiteLibrary, ...]

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    def site(self, index: int) -> StencilSiteLibrary:
        return self.sites[index]

    def to_dict(self) -> dict[str, Any]:
        return {"n_sites": self.n_sites, "sites": [site.to_dict() for site in self.sites]}


@dataclass(frozen=True)
class PatternedParent:
    """A fully active patterned parent and its site-level provenance."""

    system: ParticleSystem
    stencil_spec: StencilSpec
    site_records: tuple[StencilSiteRecord, ...]
    stencil_sha256: str
    candidate_library: StencilCandidateLibrary | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not bool(torch.all(self.system.active_mask)):
            raise ValueError("PatternedParent must emit an all-True active_mask")
        if len(self.site_records) != self.stencil_spec.n_sites:
            raise ValueError("one site record is required per stencil site")

    @property
    def positions(self) -> torch.Tensor:
        return self.system.positions

    @property
    def velocities(self) -> torch.Tensor:
        return self.system.velocities

    @property
    def diameters(self) -> torch.Tensor:
        return self.system.diameters

    @property
    def box(self) -> torch.Tensor:
        return self.system.box

    @property
    def active_mask(self) -> torch.Tensor:
        return self.system.active_mask

    @property
    def unwrapped_positions(self) -> torch.Tensor:
        return self.system.unwrapped_positions

    @property
    def records(self) -> tuple[StencilSiteRecord, ...]:
        return self.site_records

    @property
    def parent(self) -> ParticleSystem:
        return self.system

    @property
    def device(self) -> torch.device:
        return self.system.device

    @property
    def dtype(self) -> torch.dtype:
        return self.system.dtype

    @property
    def n_particles(self) -> int:
        return self.system.n_particles

    def to_system(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ParticleSystem:
        target_device = self.system.device if device is None else torch.device(device)
        target_dtype = self.system.dtype if dtype is None else dtype
        return ParticleSystem(
            positions=self.positions.detach().to(device=target_device, dtype=target_dtype).clone(),
            velocities=self.velocities.detach().to(device=target_device, dtype=target_dtype).clone(),
            diameters=self.diameters.detach().to(device=target_device, dtype=target_dtype).clone(),
            box=self.box.detach().to(device=target_device, dtype=target_dtype).clone(),
            active_mask=self.active_mask.detach().to(device=target_device, dtype=torch.bool).clone(),
            unwrapped_positions=self.unwrapped_positions.detach().to(device=target_device, dtype=target_dtype).clone(),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.to_system().state_dict()

    @property
    def total_displacement_dose(self) -> float:
        return float(sum(record.displacement_dose for record in self.site_records))

    @property
    def total_energy_dose(self) -> float:
        return float(sum(record.energy_dose for record in self.site_records))

    @property
    def interference_drift_flag(self) -> bool:
        return any(record.earlier_core_drift.flagged for record in self.site_records)

    @property
    def protected_mask(self) -> torch.Tensor | None:
        return self.stencil_spec.template.particle_protected_mask(self.positions, self.box)

    @property
    def protected_indices(self) -> torch.Tensor | None:
        mask = self.protected_mask
        return None if mask is None else torch.nonzero(mask, as_tuple=False).flatten()

    @property
    def unprotected_indices(self) -> torch.Tensor | None:
        mask = self.protected_mask
        return None if mask is None else torch.nonzero(~mask, as_tuple=False).flatten()

    def to_dict(self) -> dict[str, Any]:
        mask = self.protected_mask
        protected = None
        if mask is not None:
            protected = {
                "n_protected": int(mask.sum()),
                "n_unprotected": int((~mask).sum()),
                "protected_indices": torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist(),
                "unprotected_indices": torch.nonzero(~mask, as_tuple=False).flatten().cpu().tolist(),
            }
        return {
            "stencil_sha256": self.stencil_sha256,
            "stencil_spec": self.stencil_spec.to_dict(),
            "sites": [record.to_dict() for record in self.site_records],
            "protected_region": protected,
            "total_displacement_dose": self.total_displacement_dose,
            "total_energy_dose": self.total_energy_dose,
            "interference_drift_flag": self.interference_drift_flag,
        }


def _state_payload(system: ParticleSystem) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in system.state_dict().items()}


def _state_sha256(system: ParticleSystem) -> str:
    """Hash the same complete CPU state payload used for branch identity."""

    buffer = io.BytesIO()
    torch.save(_state_payload(system), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _parent_system(
    parent: ParticleSystem | ParentState,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[ParticleSystem, str]:
    if isinstance(parent, ParentState):
        return parent.to_system(device=device, dtype=dtype), parent.parent_id
    if not isinstance(parent, ParticleSystem):
        raise TypeError("parent must be a ParticleSystem or ParentState")
    system = parent.clone()
    target_device = torch.device(device)
    system = ParticleSystem(
        positions=system.positions.to(device=target_device, dtype=dtype),
        velocities=system.velocities.to(device=target_device, dtype=dtype),
        diameters=system.diameters.to(device=target_device, dtype=dtype),
        box=system.box.to(device=target_device, dtype=dtype),
        active_mask=system.active_mask.to(device=target_device, dtype=torch.bool),
        unwrapped_positions=system.unwrapped_positions.to(device=target_device, dtype=dtype),
    )
    return system, f"parent-{_state_sha256(system)}"


class _SeedSource:
    def __init__(self, seed: Any) -> None:
        if hasattr(seed, "seed_for") and callable(seed.seed_for):
            self._external = seed
            self._seed = None
        else:
            self._external = None
            if isinstance(seed, bool):
                raise ValueError("seeds must not be boolean")
            if isinstance(seed, int):
                self._seed = str(seed)
            elif isinstance(seed, Mapping):
                self._seed = repr(sorted((str(key), repr(value)) for key, value in seed.items()))
            elif isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
                self._seed = repr(tuple(seed))
            else:
                self._seed = repr(seed)

    def seed_for(self, domain: str, index: int) -> int:
        if self._external is not None:
            return int(self._external.seed_for(domain, index))
        message = b"\0".join((str(self._seed).encode(), str(domain).encode(), str(int(index)).encode()))
        return int.from_bytes(hashlib.sha256(message).digest(), byteorder="big")


class _ChoiceRNG:
    """Small adapter for Python, NumPy-like, and torch random sources."""

    def __init__(self, value: Any, *, fallback_seed: int) -> None:
        if value is None:
            self._python = random.Random(int(fallback_seed) % (2**63 - 1))
            self._external = None
        elif isinstance(value, bool):
            raise ValueError("rng must not be boolean")
        elif isinstance(value, int):
            self._python = random.Random(int(value) % (2**63 - 1))
            self._external = None
        elif isinstance(value, random.Random):
            self._python = value
            self._external = None
        else:
            self._python = None
            self._external = value

    def choice(self, count: int) -> int:
        if count <= 0:
            raise ValueError("cannot choose from an empty candidate set")
        if self._python is not None:
            return int(self._python.randrange(count))
        if isinstance(self._external, torch.Generator):
            return int(torch.randint(count, (1,), generator=self._external, device="cpu"))
        if hasattr(self._external, "integers"):
            return int(self._external.integers(0, count))
        if hasattr(self._external, "randrange"):
            return int(self._external.randrange(count))
        raise TypeError("rng must provide integers(), randrange(), or be a torch.Generator")

    def permutation(self, values: Sequence[Any]) -> tuple[Any, ...]:
        result = list(values)
        if self._python is not None:
            self._python.shuffle(result)
            return tuple(result)
        if isinstance(self._external, torch.Generator):
            order = torch.randperm(len(result), generator=self._external, device="cpu").tolist()
            return tuple(result[index] for index in order)
        if hasattr(self._external, "permutation"):
            order = self._external.permutation(len(result))
            return tuple(result[int(index)] for index in order)
        if hasattr(self._external, "shuffle"):
            self._external.shuffle(result)
            return tuple(result)
        raise TypeError("rng must provide permutation(), shuffle(), or be a torch.Generator")


def _rng_seed(seed_source: _SeedSource, value: Any) -> int:
    if value is None:
        return seed_source.seed_for("stencil.selection", 0)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return seed_source.seed_for("stencil.selection", 0)


def _raw_site_library(raw: Any, site_index: int) -> Any:
    if isinstance(raw, StencilCandidateLibrary):
        return raw.site(site_index)
    if isinstance(raw, Mapping):
        if site_index in raw:
            return raw[site_index]
        key = str(site_index)
        if key in raw:
            return raw[key]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return raw[site_index]
    raise TypeError("candidate_libraries must contain one site library per stencil site")


def _normalise_site_library(
    raw: Any,
    *,
    site_index: int,
    selection: CavitySelection,
) -> StencilSiteLibrary:
    if isinstance(raw, StencilSiteLibrary):
        samples_by_chain = raw.samples_by_chain
        cost = raw.cost
    elif isinstance(raw, Mapping):
        samples_by_chain = raw
        cost = {}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(raw)
        if not all(isinstance(value, CandidateState) for value in values):
            raise TypeError("flat candidate libraries must contain CandidateState values")
        samples_by_chain = {"library": values}
        cost = {}
    else:
        raise TypeError("each site candidate library must be a mapping or CandidateState sequence")
    clean: dict[str, tuple[CandidateState, ...]] = {}
    for chain_id, values in samples_by_chain.items():
        if isinstance(values, CandidateState):
            values = (values,)
        samples = tuple(values)
        if not samples or not all(isinstance(value, CandidateState) for value in samples):
            raise ValueError(f"candidate chain {chain_id!r} must contain CandidateState values")
        for candidate in samples:
            if not torch.equal(candidate.buffer_mask, selection.buffer_mask.cpu()):
                raise ValueError(
                    f"candidate library site {site_index} has a buffer membership different from current selection"
                )
        clean[str(chain_id)] = samples
    if not clean:
        raise ValueError(f"candidate library site {site_index} is empty")
    return StencilSiteLibrary(
        site_index=site_index,
        selection=selection,
        samples_by_chain=clean,
        cost=_jsonable(cost),
    )


def _site_score_table(field_scores: Any, site_index: int) -> Sequence[float] | None:
    if field_scores is None or callable(field_scores):
        return None
    if isinstance(field_scores, Mapping):
        if site_index in field_scores:
            value = field_scores[site_index]
        elif str(site_index) in field_scores:
            value = field_scores[str(site_index)]
        else:
            return None
    elif isinstance(field_scores, Sequence) and not isinstance(field_scores, (str, bytes)):
        if not field_scores:
            return None
        first = field_scores[0]
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            value = field_scores
        else:
            value = field_scores[site_index]
    else:
        return None
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TypeError("field score tables must contain numeric sequences") from error
    if any(not math.isfinite(item) for item in result):
        raise ValueError("field scores must be finite")
    return result


def _callable_score(
    function: FieldScoreFn,
    candidate: CandidateState,
    *,
    site_index: int,
    selection: CavitySelection | None = None,
    system: ParticleSystem | None = None,
) -> float:
    """Call a field scorer using one of the documented small signatures."""

    try:
        signature = inspect.signature(function)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        names = [parameter.name.lower() for parameter in positional]
    except (TypeError, ValueError):
        names = []
    if names and names[0] in {"system", "state", "particle_system"} and system is not None:
        if len(names) >= 3:
            value = function(system, selection, site_index)
        elif len(names) == 2:
            value = function(system, site_index)
        else:
            value = function(system)
    elif names and names[0] in {"positions", "coords", "coordinates"}:
        value = function(candidate.positions)
    elif len(names) >= 2:
        value = function(candidate, site_index)
    else:
        value = function(candidate)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("field scorer must return finite scalar scores")
    return result


def _candidate_score(
    candidate: CandidateState,
    *,
    site_index: int,
    flat_index: int,
    field_scores: Any,
    spec: StencilSpec,
    selection: CavitySelection | None = None,
    system: ParticleSystem | None = None,
) -> float:
    table = _site_score_table(field_scores, site_index)
    if table is not None and len(table) > 1:
        if flat_index >= len(table):
            raise ValueError(
                f"field score table for site {site_index} has {len(table)} values, "
                f"but candidate library has at least {flat_index + 1} samples"
            )
        return float(table[flat_index])
    function = field_scores if callable(field_scores) else spec.field_score_fn
    if function is not None:
        return _callable_score(function, candidate, site_index=site_index, selection=selection, system=system)
    recorded = candidate.observables.get("field_score")
    if recorded is not None:
        result = float(recorded)
    else:
        # A frozen field model is normally supplied by the campaign.  Keeping
        # active energy as an explicit fallback makes primitive-only replay
        # useful while the provenance still records the selected scalar.
        result = float(candidate.observables.get("active_potential_energy", 0.0))
    if not math.isfinite(result):
        raise ValueError("candidate field scores must be finite")
    return result


def _flatten_scored(
    site_library: StencilSiteLibrary,
    *,
    site_index: int,
    field_scores: Any,
    spec: StencilSpec,
    system: ParticleSystem | None = None,
) -> list[tuple[CandidateState, float, int]]:
    result: list[tuple[CandidateState, float, int]] = []
    flat_index = 0
    for samples in site_library.samples_by_chain.values():
        for candidate in samples:
            score = _candidate_score(
                candidate,
                site_index=site_index,
                flat_index=flat_index,
                field_scores=field_scores,
                spec=spec,
                selection=site_library.selection,
                system=system,
            )
            result.append((candidate, score, flat_index))
            flat_index += 1
    return result


def _candidate_dose(
    candidate: CandidateState,
    *,
    system: ParticleSystem | None = None,
) -> tuple[float, float]:
    recorded_displacement = candidate.observables.get("displacement_dose")
    if recorded_displacement is not None:
        displacement = float(recorded_displacement)
    elif system is not None:
        indices = candidate.buffer_indices.to(system.device)
        candidate_positions = candidate.positions.to(device=system.device, dtype=system.dtype)
        current_positions = system.positions[indices]
        box = candidate.box.to(device=system.device, dtype=system.dtype)
        delta = minimum_image(candidate_positions[indices] - current_positions, box)
        displacement = float(torch.linalg.vector_norm(delta, dim=1).square().mean().sqrt())
    else:
        displacement = 0.0
    recorded_energy = candidate.observables.get("energy_dose")
    if recorded_energy is not None:
        energy = float(recorded_energy)
    else:
        energy = abs(float(candidate.observables.get("active_potential_energy", 0.0)))
    if not math.isfinite(displacement) or not math.isfinite(energy):
        raise ValueError("candidate dose values must be finite")
    return displacement, energy


def _pick_candidate(
    scored: Sequence[tuple[CandidateState, float, int]],
    sign: Sign,
    *,
    rng: _ChoiceRNG,
    reference: CandidateState | None = None,
    system: ParticleSystem | None = None,
) -> tuple[CandidateState, float]:
    if not scored:
        raise ValueError("candidate library is empty")
    if sign not in _SIGNS:
        raise ValueError("target sign must be '+', '-', or '0'")
    if sign == "+":
        best_score = max(item[1] for item in scored)
        for candidate, score, _ in scored:
            if score == best_score:
                return candidate, score
        raise AssertionError("unreachable")
    if sign == "-":
        best_score = min(item[1] for item in scored)
        for candidate, score, _ in scored:
            if score == best_score:
                return candidate, score
        raise AssertionError("unreachable")

    # Random edits exclude both field extremes whenever an interior candidate
    # exists.  If a targeted reference is available, choose the closest dose
    # among those interior candidates, then break exact ties with the supplied
    # RNG.  This preserves the mechanical dose while removing field ordering.
    minimum = min(item[1] for item in scored)
    maximum = max(item[1] for item in scored)
    interior = [item for item in scored if item[1] > minimum and item[1] < maximum]
    pool = interior or list(scored)
    if reference is None:
        chosen = pool[rng.choice(len(pool))]
        return chosen[0], chosen[1]
    reference_dose = _candidate_dose(reference, system=system)
    distances = []
    for item in pool:
        dose = _candidate_dose(item[0], system=system)
        distances.append((abs(dose[0] - reference_dose[0]) + abs(dose[1] - reference_dose[1]), item))
    best_distance = min(distance for distance, _ in distances)
    ties = [item for distance, item in distances if math.isclose(distance, best_distance, rel_tol=0.0, abs_tol=1e-12)]
    chosen = ties[rng.choice(len(ties))]
    return chosen[0], chosen[1]


def select_candidate(
    candidates: Sequence[CandidateState],
    target_sign: Sign,
    *,
    field_scores: Sequence[float] | None = None,
    rng: Any = None,
    reference: CandidateState | None = None,
) -> CandidateState:
    """Select a candidate by field sign; public helper for planted libraries."""

    if field_scores is not None and len(field_scores) != len(candidates):
        raise ValueError("field_scores must have one value per candidate")
    scores = [
        float(field_scores[index]) if field_scores is not None else float(candidate.observables.get("field_score", 0.0))
        for index, candidate in enumerate(candidates)
    ]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("field scores must be finite")
    adapter = _ChoiceRNG(rng, fallback_seed=0)
    scored = [(candidate, score, index) for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True))]
    return _pick_candidate(scored, target_sign, rng=adapter, reference=reference)[0]


def _diagnose_site(
    site_library: StencilSiteLibrary,
    *,
    site_index: int,
    spec: StencilSpec,
    field_scores: Any,
) -> StencilDiagnostics:
    channels: dict[str, dict[str, Any]] = {}
    energy_chains: dict[str, list[float]] = {}
    score_chains: dict[str, list[float]] = {}
    flat_index = 0
    have_score = False
    for chain_id, samples in site_library.samples_by_chain.items():
        energy_chains[str(chain_id)] = [
            float(sample.observables.get("active_potential_energy", 0.0)) for sample in samples
        ]
        chain_scores: list[float] = []
        for sample in samples:
            try:
                score = _candidate_score(
                    sample,
                    site_index=site_index,
                    flat_index=flat_index,
                    field_scores=field_scores,
                    spec=spec,
                    selection=site_library.selection,
                )
                chain_scores.append(score)
                have_score = True
            except (TypeError, ValueError):
                chain_scores.append(float("nan"))
            flat_index += 1
        score_chains[str(chain_id)] = chain_scores

    for name, traces in (("active_potential_energy", energy_chains), ("field_score", score_chains)):
        if name == "field_score" and not have_score:
            continue
        if len(traces) < 2 or any(len(values) < 4 for values in traces.values()):
            continue
        if any(not math.isfinite(value) for values in traces.values() for value in values):
            continue
        diagnostic = diagnose_scalar_channel(traces)
        channels[name] = {
            "split_rhat": diagnostic.split_rhat,
            "iat_by_chain": diagnostic.iat_by_chain,
            "ess_by_chain": diagnostic.ess_by_chain,
            "min_ess": diagnostic.min_ess,
            "stuck": diagnostic.stuck,
        }

    if channels:
        finite_rhats = [float(values["split_rhat"]) for values in channels.values()]
        max_rhat = max(finite_rhats) if finite_rhats else float("inf")
        min_ess = min(float(values["min_ess"]) for values in channels.values())
    else:
        max_rhat = float("inf")
        min_ess = 0.0
    rhat_pass = math.isfinite(max_rhat) and max_rhat <= spec.rhat_max
    ess_pass = math.isfinite(min_ess) and min_ess >= spec.ess_min
    return StencilDiagnostics(
        channels=channels,
        max_split_rhat=max_rhat,
        min_ess=min_ess,
        rhat_pass=rhat_pass,
        ess_pass=ess_pass,
        passed=rhat_pass and ess_pass,
    )


def _generate_site_library(
    system: ParticleSystem,
    *,
    selection: CavitySelection,
    site_index: int,
    parent_id: str,
    spec: StencilSpec,
    seed_source: _SeedSource,
    device: torch.device | str,
    dtype: torch.dtype,
) -> StencilSiteLibrary:
    """Run the existing four RCCE initialization families for one site."""

    parent_state = ParentState.capture(system, parent_id=parent_id)
    samples_by_chain: dict[str, tuple[CandidateState, ...]] = {}
    costs: dict[str, Any] = {}
    failures: list[str] = []
    families = tuple(InitFamily)
    for family_index, family in enumerate(families):
        samples: list[CandidateState] | None = None
        final_cost: SamplerCost | None = None
        for attempt in range(2):
            chain_index = site_index * len(families) + family_index + attempt * 10_000
            seeds = RCCESeeds.allocate(
                seed_source,
                chain_index=chain_index,
                domain_prefix=f"stencil.site-{site_index}.{family.value}",
            )
            try:
                chain = RCCEChain(
                    parent_state,
                    selection,
                    chain_id=f"site-{site_index}-{family.value}" + ("" if attempt == 0 else f"-retry{attempt}"),
                    init_family=family,
                    config=spec.config,
                    seeds=seeds,
                    device=device,
                    dtype=dtype,
                )
                samples = chain.run(
                    burn_in_sweeps=spec.burn_in_sweeps,
                    production_sweeps=spec.production_sweeps,
                    sample_interval=spec.sample_interval,
                )
                final_cost = chain.cost
                break
            except (ChainInitError, RuntimeError) as error:
                failures.append(f"{family.value} attempt {attempt}: {error}")
        if samples is None or final_cost is None:
            detail = "; ".join(failures[-2:])
            raise RuntimeError(f"RCCE site {site_index} failed to produce {family.value} candidates: {detail}")
        if not samples:
            raise RuntimeError(f"RCCE site {site_index} produced no samples for {family.value}")
        samples_by_chain[family.value] = tuple(samples)
        costs[family.value] = final_cost.to_dict()
    return StencilSiteLibrary(
        site_index=site_index,
        selection=selection,
        samples_by_chain=samples_by_chain,
        cost=costs,
    )


def _splice_buffer(
    system: ParticleSystem,
    candidate: CandidateState,
    selection: CavitySelection,
) -> None:
    """Copy only candidate buffer degrees of freedom into the growing parent."""

    if not torch.equal(candidate.buffer_mask, selection.buffer_mask.cpu()):
        raise ValueError("candidate buffer membership differs from the current cavity selection")
    indices = selection.buffer_indices.to(system.device)
    candidate_indices = candidate.buffer_indices.to(device=system.device)
    candidate_positions = candidate.positions.to(device=system.device, dtype=system.dtype)
    candidate_velocities = candidate.velocities.to(device=system.device, dtype=system.dtype)
    candidate_diameters = candidate.diameters.to(device=system.device, dtype=system.dtype)
    candidate_unwrapped = candidate.unwrapped_positions.to(device=system.device, dtype=system.dtype)
    if not torch.equal(indices, candidate_indices):
        raise ValueError("candidate and current cavity buffer indices differ")
    system.positions[indices] = candidate_positions[indices]
    system.velocities[indices] = candidate_velocities[indices]
    system.diameters[indices] = candidate_diameters[indices]
    system.unwrapped_positions[indices] = candidate_unwrapped[indices]


def _rescore_snapshot(
    system: ParticleSystem,
    *,
    selection: CavitySelection,
    provenance: CandidateProvenance,
    function: FieldScoreFn | None,
    site_index: int,
) -> float | None:
    if function is None:
        return None
    snapshot = CandidateState.capture(
        system,
        selection=selection,
        provenance=provenance,
        observables={},
    )
    return _callable_score(
        function,
        snapshot,
        site_index=site_index,
        selection=selection,
        system=system,
    )


def _drift_after_write(
    system: ParticleSystem,
    *,
    prior: Sequence[tuple[int, CavitySelection, CandidateProvenance, float]],
    function: FieldScoreFn | None,
    tolerance: float,
) -> EarlierCoreDrift:
    if not prior or function is None:
        return EarlierCoreDrift(
            per_site_delta={},
            max_abs_delta=0.0,
            tolerance=tolerance,
            available=False,
            flagged=False,
        )
    deltas: dict[int, float] = {}
    for site_index, selection, provenance, baseline in prior:
        current = _rescore_snapshot(
            system,
            selection=selection,
            provenance=provenance,
            function=function,
            site_index=site_index,
        )
        if current is not None:
            deltas[site_index] = float(current - baseline)
    max_abs = max((abs(value) for value in deltas.values()), default=0.0)
    return EarlierCoreDrift(
        per_site_delta=deltas,
        max_abs_delta=max_abs,
        tolerance=tolerance,
        available=bool(deltas),
        flagged=max_abs > tolerance,
    )


def compose_stencil(
    parent: ParticleSystem | ParentState,
    stencil_spec: StencilSpec,
    *,
    seeds: Any,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    candidate_libraries: Any = None,
    candidate_library: Any = None,
    field_scores: Any = None,
    rng: Any = None,
    reference_candidates: Sequence[CandidateState] | None = None,
) -> PatternedParent:
    """Compose an ordered stencil by sequential conditional RCCE edits.

    ``candidate_libraries`` is an optional replay/test seam.  When omitted,
    each site runs the real RCCE initialization families and production
    sweeps.  A supplied library must contain the same parent-time buffer
    membership as the current sequential selection, so it cannot accidentally
    bypass the frozen-exterior contract.
    """

    if not isinstance(stencil_spec, StencilSpec):
        raise TypeError("stencil_spec must be a StencilSpec")
    target_device = torch.device(device)
    if target_device.type == "mps" and dtype is torch.float64:
        raise ValueError("MPS does not support float64; use float32")
    current, parent_id = _parent_system(parent, device=target_device, dtype=dtype)
    if not torch.allclose(
        current.box.detach().cpu().to(torch.float64),
        torch.as_tensor(stencil_spec.template.box, dtype=torch.float64),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("stencil template box does not match parent box")
    stencil_spec.validate_geometry(current.box)
    if reference_candidates is not None and len(reference_candidates) != stencil_spec.n_sites:
        raise ValueError("reference_candidates must contain one candidate per stencil site")
    effective_scores = stencil_spec.field_scores if field_scores is None else field_scores
    drift_function = effective_scores if callable(effective_scores) else stencil_spec.field_score_fn
    seed_source = _SeedSource(seeds)
    choice_rng = _ChoiceRNG(rng, fallback_seed=_rng_seed(seed_source, rng))
    libraries: list[StencilSiteLibrary] = []
    records: list[StencilSiteRecord] = []
    prior_core_contexts: list[tuple[int, CavitySelection, CandidateProvenance, float]] = []

    for site_index, cavity_spec in enumerate(stencil_spec.cavity_specs):
        # This selection is intentionally repeated against the growing system;
        # it is the physical conditional-conditioning boundary for site k.
        selection = select_cavity(current, cavity_spec)
        frozen_exterior_positions = current.positions[selection.exterior_indices.to(current.device)].detach().cpu().clone()
        if candidate_libraries is None and candidate_library is None:
            site_library = _generate_site_library(
                current,
                selection=selection,
                site_index=site_index,
                parent_id=parent_id,
                spec=stencil_spec,
                seed_source=seed_source,
                device=target_device,
                dtype=dtype,
            )
        else:
            raw = _raw_site_library(
                candidate_libraries if candidate_libraries is not None else candidate_library,
                site_index,
            )
            site_library = _normalise_site_library(raw, site_index=site_index, selection=selection)
        libraries.append(site_library)

        scored = _flatten_scored(
            site_library,
            site_index=site_index,
            field_scores=effective_scores,
            spec=stencil_spec,
            system=current,
        )
        reference = None if reference_candidates is None else reference_candidates[site_index]
        candidate, score = _pick_candidate(
            scored,
            stencil_spec.target_signs[site_index],
            rng=choice_rng,
            reference=reference,
            system=current,
        )
        displacement_dose, energy_dose = _candidate_dose(candidate, system=current)
        diagnostics = _diagnose_site(
            site_library,
            site_index=site_index,
            spec=stencil_spec,
            field_scores=effective_scores,
        )
        _splice_buffer(current, candidate, selection)
        current.active_mask = torch.ones(current.n_particles, device=current.device, dtype=torch.bool)
        drift = _drift_after_write(
            current,
            prior=prior_core_contexts,
            function=drift_function,
            tolerance=stencil_spec.drift_tolerance,
        )
        post_positions = current.positions.detach().cpu().clone()
        post_velocities = current.velocities.detach().cpu().clone()
        post_diameters = current.diameters.detach().cpu().clone()
        post_unwrapped = current.unwrapped_positions.detach().cpu().clone()
        record = StencilSiteRecord(
            site_index=site_index,
            cavity_spec=cavity_spec,
            selection=selection,
            buffer_mask=selection.buffer_mask.detach().cpu().clone(),
            buffer_indices=selection.buffer_indices.detach().cpu().clone(),
            core_mask=selection.core_mask.detach().cpu().clone(),
            core_indices=selection.core_indices.detach().cpu().clone(),
            frozen_exterior_indices=selection.exterior_indices.detach().cpu().clone(),
            frozen_exterior_positions=frozen_exterior_positions,
            post_site_positions=post_positions,
            post_site_velocities=post_velocities,
            post_site_diameters=post_diameters,
            post_site_unwrapped_positions=post_unwrapped,
            candidate_provenance=candidate.provenance,
            selected_sign=stencil_spec.target_signs[site_index],
            field_score=score,
            diagnostics=diagnostics,
            earlier_core_drift=drift,
            displacement_dose=displacement_dose,
            energy_dose=energy_dose,
            candidate=candidate,
        )
        records.append(record)
        if drift_function is not None:
            baseline = _rescore_snapshot(
                current,
                selection=selection,
                provenance=candidate.provenance,
                function=drift_function,
                site_index=site_index,
            )
            if baseline is not None:
                prior_core_contexts.append((site_index, selection, candidate.provenance, baseline))

    current.active_mask = torch.ones(current.n_particles, device=current.device, dtype=torch.bool)
    patterned = PatternedParent(
        system=current,
        stencil_spec=stencil_spec,
        site_records=tuple(records),
        stencil_sha256=_state_sha256(current),
        candidate_library=StencilCandidateLibrary(tuple(libraries)),
    )
    return patterned


def _base_box_value(base: ParticleSystem | ParentState) -> tuple[float, float, float]:
    if isinstance(base, ParentState):
        return _box_tuple(base.box)
    if isinstance(base, ParticleSystem):
        return _box_tuple(base.box)
    raise TypeError("base must be a ParticleSystem or ParentState")


def _default_arm_spec(
    base: ParticleSystem | ParentState,
    template: StencilTemplate,
    *,
    cavity_specs: Sequence[CavitySpec] | None,
    config: RCCEConfig | None,
    burn_in_sweeps: int,
    production_sweeps: int,
    sample_interval: int,
    exact_core_composition: bool,
    overlap_regime: str,
    rhat_max: float,
    ess_min: float,
    drift_tolerance: float,
    core_radius: float,
    buffer_radius: float,
    target_signs: Sequence[Sign] | None,
    field_scores: Any,
    field_score_fn: FieldScoreFn | None,
    stencil_spec: StencilSpec | None,
) -> StencilSpec:
    if stencil_spec is not None:
        if stencil_spec.template != template:
            raise ValueError("stencil_spec template differs from the supplied template")
        signs = tuple(stencil_spec.target_signs or ()) if target_signs is None else tuple(target_signs)
        return replace(
            stencil_spec,
            target_signs=signs,
            field_scores=field_scores if field_scores is not None else stencil_spec.field_scores,
            field_score_fn=field_score_fn if field_score_fn is not None else stencil_spec.field_score_fn,
        )
    if cavity_specs is None:
        if not math.isfinite(float(core_radius)) or not math.isfinite(float(buffer_radius)):
            raise ValueError("core_radius and buffer_radius must be finite")
        cavities = tuple(
            CavitySpec(center=center, core_radius=core_radius, buffer_radius=buffer_radius)
            for center in template.centers
        )
    else:
        cavities = tuple(cavity_specs)
    effective_config = config or RCCEConfig(temperature=0.15)
    # Touch the base here so an arm constructor fails early for malformed
    # parents even when the candidate library is supplied and no chain runs.
    _base_box_value(base)
    return StencilSpec(
        template=template,
        cavity_specs=cavities,
        target_signs=template.default_signs if target_signs is None else tuple(target_signs),
        config=effective_config,
        burn_in_sweeps=burn_in_sweeps,
        production_sweeps=production_sweeps,
        exact_core_composition=exact_core_composition,
        overlap_regime=overlap_regime,
        sample_interval=sample_interval,
        rhat_max=rhat_max,
        ess_min=ess_min,
        drift_tolerance=drift_tolerance,
        field_score_fn=field_score_fn,
        field_scores=field_scores,
    )


def _prepare_library_for_arm(
    base: ParticleSystem | ParentState,
    spec: StencilSpec,
    raw: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> StencilCandidateLibrary:
    system, _ = _parent_system(base, device=device, dtype=dtype)
    sites: list[StencilSiteLibrary] = []
    for site_index, cavity_spec in enumerate(spec.cavity_specs):
        selection = select_cavity(system, cavity_spec)
        sites.append(
            _normalise_site_library(
                _raw_site_library(raw, site_index),
                site_index=site_index,
                selection=selection,
            )
        )
    return StencilCandidateLibrary(tuple(sites))


def _reference_candidates(
    base: ParticleSystem | ParentState,
    spec: StencilSpec,
    library: StencilCandidateLibrary,
    *,
    field_scores: Any,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[CandidateState, ...]:
    system, _ = _parent_system(base, device=device, dtype=dtype)
    references: list[CandidateState] = []
    adapter = _ChoiceRNG(0, fallback_seed=0)
    for site_index, site_library in enumerate(library.sites):
        scored = _flatten_scored(
            site_library,
            site_index=site_index,
            field_scores=field_scores,
            spec=spec,
            system=system,
        )
        candidate, _ = _pick_candidate(
            scored,
            spec.target_signs[site_index],
            rng=adapter,
            system=system,
        )
        references.append(candidate)
    return tuple(references)


def targeted_arm(
    base: ParticleSystem | ParentState,
    template: StencilTemplate,
    *,
    field_scores: Any = None,
    cavity_specs: Sequence[CavitySpec] | None = None,
    config: RCCEConfig | None = None,
    burn_in_sweeps: int = 0,
    production_sweeps: int = 4,
    sample_interval: int = 1,
    exact_core_composition: bool = False,
    overlap_regime: str = "auto",
    rhat_max: float = 1.05,
    ess_min: float = 50.0,
    drift_tolerance: float = 0.05,
    core_radius: float = 0.75,
    buffer_radius: float = 1.5,
    target_signs: Sequence[Sign] | None = None,
    seeds: Any = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    candidate_libraries: Any = None,
    candidate_library: Any = None,
    stencil_spec: StencilSpec | None = None,
) -> PatternedParent:
    """Build the intended field-sign arm from one base parent."""

    spec = _default_arm_spec(
        base,
        template,
        cavity_specs=cavity_specs,
        config=config,
        burn_in_sweeps=burn_in_sweeps,
        production_sweeps=production_sweeps,
        sample_interval=sample_interval,
        exact_core_composition=exact_core_composition,
        overlap_regime=overlap_regime,
        rhat_max=rhat_max,
        ess_min=ess_min,
        drift_tolerance=drift_tolerance,
        core_radius=core_radius,
        buffer_radius=buffer_radius,
        target_signs=target_signs,
        field_scores=field_scores,
        field_score_fn=field_scores if callable(field_scores) else None,
        stencil_spec=stencil_spec,
    )
    return compose_stencil(
        base,
        spec,
        seeds=seeds,
        device=device,
        dtype=dtype,
        candidate_libraries=candidate_libraries,
        candidate_library=candidate_library,
        field_scores=field_scores,
    )


def random_edit_arm(
    base: ParticleSystem | ParentState,
    template: StencilTemplate,
    *,
    field_scores: Any = None,
    rng: Any = None,
    cavity_specs: Sequence[CavitySpec] | None = None,
    config: RCCEConfig | None = None,
    burn_in_sweeps: int = 0,
    production_sweeps: int = 4,
    sample_interval: int = 1,
    exact_core_composition: bool = False,
    overlap_regime: str = "auto",
    rhat_max: float = 1.05,
    ess_min: float = 50.0,
    drift_tolerance: float = 0.05,
    core_radius: float = 0.75,
    buffer_radius: float = 1.5,
    seeds: Any = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    candidate_libraries: Any = None,
    candidate_library: Any = None,
    stencil_spec: StencilSpec | None = None,
) -> PatternedParent:
    """Build the field-score-matched random/null arm."""

    target_spec = _default_arm_spec(
        base,
        template,
        cavity_specs=cavity_specs,
        config=config,
        burn_in_sweeps=burn_in_sweeps,
        production_sweeps=production_sweeps,
        sample_interval=sample_interval,
        exact_core_composition=exact_core_composition,
        overlap_regime=overlap_regime,
        rhat_max=rhat_max,
        ess_min=ess_min,
        drift_tolerance=drift_tolerance,
        core_radius=core_radius,
        buffer_radius=buffer_radius,
        target_signs=None,
        field_scores=field_scores,
        field_score_fn=field_scores if callable(field_scores) else None,
        stencil_spec=stencil_spec,
    )
    raw_library = candidate_libraries if candidate_libraries is not None else candidate_library
    if raw_library is None:
        targeted = targeted_arm(
            base,
            template,
            field_scores=field_scores,
            cavity_specs=target_spec.cavity_specs,
            config=target_spec.config,
            burn_in_sweeps=target_spec.burn_in_sweeps,
            production_sweeps=target_spec.production_sweeps,
            sample_interval=target_spec.sample_interval,
            exact_core_composition=target_spec.exact_core_composition,
            overlap_regime=target_spec.overlap_regime,
            rhat_max=target_spec.rhat_max,
            ess_min=target_spec.ess_min,
            drift_tolerance=target_spec.drift_tolerance,
            seeds=seeds,
            device=device,
            dtype=dtype,
        )
        library = targeted.candidate_library
        if library is None:  # pragma: no cover - PatternedParent always retains it
            raise RuntimeError("targeted arm did not retain a candidate library")
    else:
        library = _prepare_library_for_arm(base, target_spec, raw_library, device=device, dtype=dtype)
    references = _reference_candidates(
        base,
        target_spec,
        library,
        field_scores=field_scores,
        device=device,
        dtype=dtype,
    )
    random_spec = replace(target_spec, target_signs=("0",) * target_spec.n_sites)
    return compose_stencil(
        base,
        random_spec,
        seeds=seeds,
        device=device,
        dtype=dtype,
        candidate_libraries=library,
        field_scores=field_scores,
        rng=rng,
        reference_candidates=references,
    )


def shuffled_arm(
    base: ParticleSystem | ParentState,
    template: StencilTemplate,
    *,
    field_scores: Any = None,
    rng: Any = None,
    cavity_specs: Sequence[CavitySpec] | None = None,
    config: RCCEConfig | None = None,
    burn_in_sweeps: int = 0,
    production_sweeps: int = 4,
    sample_interval: int = 1,
    exact_core_composition: bool = False,
    overlap_regime: str = "auto",
    rhat_max: float = 1.05,
    ess_min: float = 50.0,
    drift_tolerance: float = 0.05,
    core_radius: float = 0.75,
    buffer_radius: float = 1.5,
    seeds: Any = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    candidate_libraries: Any = None,
    candidate_library: Any = None,
    stencil_spec: StencilSpec | None = None,
) -> PatternedParent:
    """Build the sign-shuffled null arm with the same site libraries."""

    target_spec = _default_arm_spec(
        base,
        template,
        cavity_specs=cavity_specs,
        config=config,
        burn_in_sweeps=burn_in_sweeps,
        production_sweeps=production_sweeps,
        sample_interval=sample_interval,
        exact_core_composition=exact_core_composition,
        overlap_regime=overlap_regime,
        rhat_max=rhat_max,
        ess_min=ess_min,
        drift_tolerance=drift_tolerance,
        core_radius=core_radius,
        buffer_radius=buffer_radius,
        target_signs=None,
        field_scores=field_scores,
        field_score_fn=field_scores if callable(field_scores) else None,
        stencil_spec=stencil_spec,
    )
    raw_library = candidate_libraries if candidate_libraries is not None else candidate_library
    if raw_library is None:
        targeted = targeted_arm(
            base,
            template,
            field_scores=field_scores,
            cavity_specs=target_spec.cavity_specs,
            config=target_spec.config,
            burn_in_sweeps=target_spec.burn_in_sweeps,
            production_sweeps=target_spec.production_sweeps,
            sample_interval=target_spec.sample_interval,
            exact_core_composition=target_spec.exact_core_composition,
            overlap_regime=target_spec.overlap_regime,
            rhat_max=target_spec.rhat_max,
            ess_min=target_spec.ess_min,
            drift_tolerance=target_spec.drift_tolerance,
            seeds=seeds,
            device=device,
            dtype=dtype,
        )
        library = targeted.candidate_library
        if library is None:  # pragma: no cover
            raise RuntimeError("targeted arm did not retain a candidate library")
    else:
        library = _prepare_library_for_arm(base, target_spec, raw_library, device=device, dtype=dtype)
    seed_source = _SeedSource(seeds)
    adapter = _ChoiceRNG(rng, fallback_seed=_rng_seed(seed_source, rng))
    shuffled_signs = adapter.permutation(target_spec.target_signs)
    shuffled_spec = replace(target_spec, target_signs=tuple(shuffled_signs))
    return compose_stencil(
        base,
        shuffled_spec,
        seeds=seeds,
        device=device,
        dtype=dtype,
        candidate_libraries=library,
        field_scores=field_scores,
        rng=rng,
    )


__all__ = [
    "EarlierCoreDrift",
    "PatternedParent",
    "StencilCandidateLibrary",
    "StencilDiagnostics",
    "StencilSiteLibrary",
    "StencilSiteRecord",
    "StencilSpec",
    "StencilTemplate",
    "compose_stencil",
    "line_template",
    "random_edit_arm",
    "ring_template",
    "select_candidate",
    "shuffled_arm",
    "targeted_arm",
    "wall_template",
]
