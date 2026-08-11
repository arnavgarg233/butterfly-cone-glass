"""Frozen parent-time slab geometry: a confined film with pinned amorphous walls.

The confinement protocol is the published random-pinning construction, applied
to a slab rather than to a random subset or a sphere.  The box is split along
one axis into a mobile fluid film of thickness ``thickness`` and a wall region
either side; every particle whose *parent* position lies in the wall region is
frozen.  Because the wall labels are drawn from an already equilibrated parent,
the mobile film is in thermal equilibrium **by construction** and needs no
separate equilibration run.  See Cammarota and Biroli, PNAS 109, 8850 (2012)
and Ozawa, Kob, Ikeda and Miyazaki, J. Chem. Phys. 141, 224503 (2014); the
extension of pinning to wall and cavity geometries is standard.

As in :mod:`butterfly_cone.rcce.cavity`, membership is fixed once at parent time and
those labels remain the active labels even if centers later drift across the
geometric boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from butterfly_cone.engine.potential import minimum_image
from butterfly_cone.engine.system import ParticleSystem


@dataclass(frozen=True)
class SlabSpec:
    """A mobile film of given thickness, normal to one axis, in a periodic box.

    ``center`` is the film midplane coordinate along ``axis``.  ``interface``
    is the width of the mobile sub-layer adjacent to each wall, used to
    separate interfacial from mid-film particles when reading anisotropy; it is
    a label only and does not affect which particles are frozen.
    """

    axis: int
    center: float
    thickness: float
    interface: float

    def __post_init__(self) -> None:
        axis = int(self.axis)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        center = float(self.center)
        thickness = float(self.thickness)
        interface = float(self.interface)
        for name, value in (
            ("center", center),
            ("thickness", thickness),
            ("interface", interface),
        ):
            if not math.isfinite(value):
                raise ValueError(f"slab {name} must be finite")
        if thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if interface < 0.0 or interface > 0.5 * thickness:
            raise ValueError("interface must satisfy 0 <= interface <= thickness / 2")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "interface", interface)

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "center": self.center,
            "thickness": self.thickness,
            "interface": self.interface,
        }


@dataclass(frozen=True)
class SlabSelection:
    """Parent-time label sets for one confined film."""

    spec: SlabSpec
    parent_offsets: torch.Tensor
    mobile_mask: torch.Tensor
    wall_mask: torch.Tensor
    interface_mask: torch.Tensor
    midfilm_mask: torch.Tensor
    mobile_indices: torch.Tensor
    wall_indices: torch.Tensor
    interface_indices: torch.Tensor
    midfilm_indices: torch.Tensor

    @property
    def n_mobile(self) -> int:
        return int(self.mobile_indices.numel())

    @property
    def n_wall(self) -> int:
        return int(self.wall_indices.numel())

    @property
    def mobile_fraction(self) -> float:
        total = int(self.mobile_mask.numel())
        return float(self.n_mobile) / float(total) if total else 0.0


def signed_offset_from_midplane(
    positions: torch.Tensor,
    axis: int,
    center: float,
    box: torch.Tensor,
) -> torch.Tensor:
    """Minimum-image signed offsets from the film midplane along ``axis``.

    The full three-vector is folded so the periodic convention matches
    :func:`butterfly_cone.rcce.cavity.minimum_image_from_center`, then the requested
    component is returned.
    """

    center_vector = torch.zeros(3, device=positions.device, dtype=positions.dtype)
    center_vector[axis] = float(center)
    return minimum_image(positions - center_vector, box)[:, axis]


def select_slab(system: ParticleSystem, spec: SlabSpec) -> SlabSelection:
    """Label mobile film, pinned wall, and the two mobile sub-layers.

    A film thicker than the box along ``axis`` is rejected: the wall region
    would be empty, so there would be no confinement, and the periodic image
    of the film would touch itself.
    """

    box_length = float(system.box[spec.axis].detach().cpu())
    if spec.thickness >= box_length:
        raise ValueError("thickness must be smaller than the box length along axis")

    offsets = signed_offset_from_midplane(system.positions, spec.axis, spec.center, system.box)
    distance = offsets.abs()
    half = 0.5 * spec.thickness

    mobile = distance < half
    wall = ~mobile
    interface = mobile & (distance >= half - spec.interface)
    midfilm = mobile & ~interface

    def indices(mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask, as_tuple=False).flatten()

    return SlabSelection(
        spec=spec,
        parent_offsets=offsets.detach().clone(),
        mobile_mask=mobile.detach().clone(),
        wall_mask=wall.detach().clone(),
        interface_mask=interface.detach().clone(),
        midfilm_mask=midfilm.detach().clone(),
        mobile_indices=indices(mobile),
        wall_indices=indices(wall),
        interface_indices=indices(interface),
        midfilm_indices=indices(midfilm),
    )


def resolve_divergence_components(
    divergence_field: torch.Tensor,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-particle displacement field into normal and in-plane parts.

    ``divergence_field`` has shape ``(..., n_particles, 3)``.  Returns the
    absolute normal component and the in-plane norm, in that order, so the
    caller can test whether the cone is anisotropic under confinement.  The two
    are reported separately rather than as a ratio so that a zero in-plane
    response does not produce a divide-by-zero.
    """

    if divergence_field.shape[-1] != 3:
        raise ValueError("divergence_field must have a trailing axis of size 3")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    normal = divergence_field[..., axis].abs()
    in_plane_axes = [index for index in (0, 1, 2) if index != axis]
    in_plane = torch.linalg.vector_norm(divergence_field[..., in_plane_axes], dim=-1)
    return normal, in_plane


def anisotropy_ratio(divergence_field: torch.Tensor, axis: int) -> float:
    """Calibrated in-plane versus normal anisotropy: exactly ``1`` when isotropic.

    Built from **second** moments, so the isotropic normalisation is the clean
    ``sqrt(2)`` (the in-plane subspace has two dimensions and the normal one).
    Do not build this from means of the component norms: for an isotropic
    Gaussian field ``E|d_normal|`` is half-normal and ``E|d_in_plane|`` is
    Rayleigh, whose ratio is ``pi / 2``, so a ``sqrt(2)`` normalisation there
    leaves a spurious offset of ``(pi / 2) / sqrt(2) = 1.1107``.

    Returns greater than one when divergence is preferentially in-plane.
    """

    normal, in_plane = resolve_divergence_components(divergence_field, axis)
    normal_second = float(normal.square().mean())
    in_plane_second = float(in_plane.square().mean())
    if normal_second <= 0.0:
        return float("nan")
    return math.sqrt(in_plane_second / (2.0 * normal_second))
