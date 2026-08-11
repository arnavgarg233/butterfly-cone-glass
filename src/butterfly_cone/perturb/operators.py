"""Local delta-perturbation operators for the causal-Gardner protocol-A probe.

Each operator acts on a *copy* of a quenched parent :class:`ParticleSystem` at a
site center ``c`` with a size ``delta`` measured in units of the mean diameter
(reduced units, sigma-bar = 1).  All operators conserve, exactly:

* particle number ``N``;
* the diameter multiset (composition) -- diameters are never touched;
* the periodic box;
* the exterior -- every particle outside the perturbed set keeps its position
  bitwise unchanged.

The perturbation is applied to the parent *before* any branch is launched; the
branch phase then relaxes the whole box with fresh momenta.  Each operator
returns the perturbed system plus a :class:`PerturbationProvenance` record.

Determinism / device independence: every random draw is taken on a CPU float64
generator built from the harness-issued integer seed (matching the engine's
``system``/``integrate`` convention) and only then cast to the system device and
dtype.  This makes an operator's output reproducible across cpu and mps.

The delta=0 contract is the crux of the whole probe: at ``delta == 0`` every
operator returns a system that is *bitwise identical* to its input, so a
matched-seed branch ensemble grown from it reproduces the unperturbed ensemble
exactly (zero divergence).  This is achieved by (a) multiplying the displacement
by ``delta`` so it is exactly zero, and (b) writing back only the perturbed
particles' positions, leaving the exterior untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from butterfly_cone.engine.potential import analytic_potential, minimum_image
from butterfly_cone.engine.system import ParticleSystem, make_generator

R_PERT_DEFAULT = 2.5  # primary O_shell core scale, in units of sigma-bar
_TORCH_SEED_MODULUS = 2**63 - 1


def _seeded_generator(seed: int) -> torch.Generator:
    """CPU generator from a harness seed, projected into PyTorch's seed range.

    Harness-issued seeds are full SHA-256 integers; PyTorch's ``manual_seed``
    only accepts int64.  The projection is deterministic, matching
    ``branching.ensemble.torch_seed``, so the same issued seed reproduces the
    same draw.
    """

    return make_generator(int(seed) % _TORCH_SEED_MODULUS)


@dataclass(frozen=True)
class PerturbationProvenance:
    """Immutable record of one applied perturbation.

    ``delta_u`` is the injected potential energy (E_after - E_before) evaluated
    with the canonical engine potential; ``rms_displacement`` is the achieved RMS
    3D displacement over the perturbed set; ``com_shift`` is the magnitude of the
    net center-of-mass displacement of the perturbed set (approximately zero for
    O_shell by construction, delta/N-scale for O_kick).
    """

    operator: str
    delta: float
    site: tuple[float, float, float]
    seed: int | None
    r_pert: float | None
    n_perturbed: int
    delta_u: float
    rms_displacement: float
    com_shift: float
    strain_tensor: tuple[tuple[float, float, float], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operator": self.operator,
            "delta": self.delta,
            "site": list(self.site),
            "seed": self.seed,
            "r_pert": self.r_pert,
            "n_perturbed": self.n_perturbed,
            "delta_u": self.delta_u,
            "rms_displacement": self.rms_displacement,
            "com_shift": self.com_shift,
        }
        if self.strain_tensor is not None:
            payload["strain_tensor"] = [list(row) for row in self.strain_tensor]
        return payload


def _as_center(center: Sequence[float] | torch.Tensor, system: ParticleSystem) -> torch.Tensor:
    tensor = torch.as_tensor(center, device=system.device, dtype=system.dtype).reshape(-1)
    if tensor.numel() != 3:
        raise ValueError("site center must be a length-3 coordinate")
    return tensor


def _potential_energy(system: ParticleSystem) -> float:
    result = analytic_potential(
        system.positions, system.diameters, system.box, active_mask=system.active_mask
    )
    return float(result.energy)


def _apply(system: ParticleSystem, displacement: torch.Tensor, mask: torch.Tensor) -> ParticleSystem:
    """Return a copy with ``displacement`` applied only to the ``mask`` set.

    Exterior particles keep their positions bitwise; perturbed particles are
    re-wrapped into the primary cell.  Unwrapped positions get the raw (zero
    outside the mask) displacement added, so the exterior is exact there too.
    """

    perturbed = system.clone()
    moved = torch.remainder(system.positions + displacement, system.box)
    perturbed.positions = torch.where(mask[:, None], moved, system.positions)
    perturbed.unwrapped_positions = system.unwrapped_positions + displacement
    return perturbed


def _rms_displacement(displacement: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return 0.0
    # CPU-cast before float64: MPS has no float64. These are scalar diagnostics
    # off the hot path, so the host round-trip is free and more accurate.
    squared = displacement[mask].cpu().to(torch.float64).square().sum().item()
    return float(math.sqrt(squared / count))


def _com_shift(displacement: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return 0.0
    net = displacement[mask].cpu().to(torch.float64).sum(dim=0)
    return float(torch.linalg.vector_norm(net).item() / count)


def o_kick(
    system: ParticleSystem,
    center: Sequence[float] | torch.Tensor,
    delta: float,
    seed: int,
) -> tuple[ParticleSystem, PerturbationProvenance]:
    """Displace the single particle nearest ``center`` by ``delta * u_hat``.

    ``u_hat`` is a random unit vector drawn from ``seed``.  This is the most
    localized operator -- the defect-scale probe.
    """

    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    center_t = _as_center(center, system)
    energy_before = _potential_energy(system)

    relative = minimum_image(system.positions - center_t, system.box)
    distance = torch.linalg.vector_norm(relative, dim=1)
    index = int(torch.argmin(distance).item())

    generator = _seeded_generator(seed)
    raw = torch.randn(3, generator=generator, dtype=torch.float64)
    unit = raw / torch.linalg.vector_norm(raw)
    displacement = torch.zeros_like(system.positions)
    displacement[index] = (unit * float(delta)).to(device=system.device, dtype=system.dtype)

    mask = torch.zeros(system.n_particles, device=system.device, dtype=torch.bool)
    mask[index] = True
    perturbed = _apply(system, displacement, mask)
    energy_after = _potential_energy(perturbed)

    provenance = PerturbationProvenance(
        operator="O_kick",
        delta=float(delta),
        site=tuple(float(v) for v in center_t.tolist()),
        seed=int(seed),
        r_pert=None,
        n_perturbed=1,
        delta_u=energy_after - energy_before,
        rms_displacement=_rms_displacement(displacement, mask),
        com_shift=_com_shift(displacement, mask),
    )
    return perturbed, provenance


def o_shell(
    system: ParticleSystem,
    center: Sequence[float] | torch.Tensor,
    delta: float,
    r_pert: float = R_PERT_DEFAULT,
    *,
    seed: int,
) -> tuple[ParticleSystem, PerturbationProvenance]:
    """Gaussian displace every particle within ``r_pert`` of ``center``.

    Per-component standard deviation is ``delta / sqrt(3)`` so the nominal RMS 3D
    displacement equals ``delta``; the shell-mean displacement is then subtracted
    so the local center of mass is conserved.  This is the PRIMARY operator.
    """

    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if r_pert <= 0.0:
        raise ValueError("r_pert must be positive")
    center_t = _as_center(center, system)
    energy_before = _potential_energy(system)

    relative = minimum_image(system.positions - center_t, system.box)
    distance = torch.linalg.vector_norm(relative, dim=1)
    mask = distance < float(r_pert)
    count = int(mask.sum().item())

    displacement = torch.zeros_like(system.positions)
    if count > 0:
        generator = _seeded_generator(seed)
        std = float(delta) / math.sqrt(3.0)
        draws = torch.randn((count, 3), generator=generator, dtype=torch.float64) * std
        # Subtract the shell mean so the perturbed set's COM is conserved.
        draws = draws - draws.mean(dim=0, keepdim=True)
        displacement[mask] = draws.to(device=system.device, dtype=system.dtype)

    perturbed = _apply(system, displacement, mask)
    energy_after = _potential_energy(perturbed)

    provenance = PerturbationProvenance(
        operator="O_shell",
        delta=float(delta),
        site=tuple(float(v) for v in center_t.tolist()),
        seed=int(seed),
        r_pert=float(r_pert),
        n_perturbed=count,
        delta_u=energy_after - energy_before,
        rms_displacement=_rms_displacement(displacement, mask),
        com_shift=_com_shift(displacement, mask),
    )
    return perturbed, provenance


def shear_tensor(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A canonical traceless symmetric pure-shear tensor (xy plane)."""

    tensor = torch.zeros((3, 3), dtype=dtype)
    tensor[0, 1] = 1.0
    tensor[1, 0] = 1.0
    return tensor


def dilation_tensor(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The isotropic part -- a local compression/dilation tensor (identity)."""

    return torch.eye(3, dtype=dtype)


def o_strain(
    system: ParticleSystem,
    center: Sequence[float] | torch.Tensor,
    delta: float,
    r_pert: float = R_PERT_DEFAULT,
    *,
    strain: torch.Tensor | None = None,
) -> tuple[ParticleSystem, PerturbationProvenance]:
    """Affine deformation ``r -> r + delta * E @ (r - c)`` within ``r_pert``.

    ``strain`` (``E``) defaults to the canonical pure-shear tensor.  Deterministic
    (no seed): its delta=0 baseline and delta-scaling are the least float32-fragile.
    """

    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if r_pert <= 0.0:
        raise ValueError("r_pert must be positive")
    center_t = _as_center(center, system)
    matrix = shear_tensor(system.dtype) if strain is None else torch.as_tensor(
        strain, device=system.device, dtype=system.dtype
    )
    if matrix.shape != (3, 3):
        raise ValueError("strain tensor must have shape (3, 3)")
    matrix = matrix.to(device=system.device, dtype=system.dtype)
    energy_before = _potential_energy(system)

    relative = minimum_image(system.positions - center_t, system.box)
    distance = torch.linalg.vector_norm(relative, dim=1)
    mask = distance < float(r_pert)

    affine = float(delta) * (relative @ matrix.T)
    displacement = torch.where(mask[:, None], affine, torch.zeros_like(affine))
    perturbed = _apply(system, displacement, mask)
    energy_after = _potential_energy(perturbed)

    provenance = PerturbationProvenance(
        operator="O_strain",
        delta=float(delta),
        site=tuple(float(v) for v in center_t.tolist()),
        seed=None,
        r_pert=float(r_pert),
        n_perturbed=int(mask.sum().item()),
        delta_u=energy_after - energy_before,
        rms_displacement=_rms_displacement(displacement, mask),
        com_shift=_com_shift(displacement, mask),
        strain_tensor=tuple(tuple(float(v) for v in row) for row in matrix.to(torch.float64).tolist()),
    )
    return perturbed, provenance


def stratified_sites(
    box: torch.Tensor | Sequence[float],
    k: int,
    min_sep: float,
    *,
    seed: int,
    max_attempts: int = 10000,
) -> torch.Tensor:
    """Return ``k`` PBC-aware site centers with pairwise min-image separation >= ``min_sep``.

    Centers are drawn uniformly in the box and greedily accepted when their
    minimum-image distance to every already-accepted center is at least
    ``min_sep``.  Deterministic given ``seed``.  Raises if ``k`` well-separated
    centers cannot be placed within ``max_attempts`` draws (the caller then knows
    the box is too small for the requested separation).
    """

    # Sampling is done on CPU in float64; move the box off any accelerator first
    # (MPS has no float64, so an on-device dtype cast would raise).
    box_t = torch.as_tensor(box).detach().cpu().to(torch.float64).reshape(-1)
    if box_t.numel() != 3 or bool(torch.any(box_t <= 0)):
        raise ValueError("box must contain three positive lengths")
    if k <= 0:
        raise ValueError("k must be positive")
    if min_sep < 0.0:
        raise ValueError("min_sep must be nonnegative")
    generator = _seeded_generator(seed)
    accepted: list[torch.Tensor] = []
    attempts = 0
    while len(accepted) < k and attempts < max_attempts:
        attempts += 1
        candidate = torch.rand(3, generator=generator, dtype=torch.float64) * box_t
        if accepted:
            stack = torch.stack(accepted)
            disp = minimum_image(candidate[None, :] - stack, box_t)
            if float(torch.linalg.vector_norm(disp, dim=1).min()) < min_sep:
                continue
        accepted.append(candidate)
    if len(accepted) < k:
        raise ValueError(
            f"could only place {len(accepted)} of {k} sites at min_sep={min_sep} "
            f"in box {box_t.tolist()} within {max_attempts} attempts"
        )
    return torch.stack(accepted)


OPERATORS = {"O_kick": o_kick, "O_shell": o_shell, "O_strain": o_strain}
