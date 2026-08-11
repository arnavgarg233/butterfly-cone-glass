"""Wolf-truncated BKS silica: two-species deterministic force engine.

Model
-----
Short-range BKS silica (van Beest, Kramer & van Santen, PRL 64, 1955 (1990))
with the Coulomb part replaced by the charge-neutralized Wolf truncation used
by Carre, Berthier, Horbach, Ispas & Kob, "Amorphous silica modeled with
truncated and screened Coulomb interactions", J. Chem. Phys. 127, 114512
(2007) (arXiv:0707.0319).  The pair energy between ions of species a, b is

    phi_ab(r) = q_a q_b e^2 [ 1/r - 1/r_c + (r - r_c)/r_c^2 ]     (r < r_c)
              + A_ab exp(-B_ab r) - C_ab / r^6 - S_ab             (r < r_sr)

where the Wolf term (their Eq. (4)) and its first derivative both vanish at
r_c, and the Buckingham part is truncated and *shifted* at r_sr (S_ab is the
Buckingham value at r_sr), exactly as in the paper: "the short range part of
the potential was truncated and shifted at 5.5 A" and r_c = 10.17 A is their
recommended Coulomb cutoff.  Charge neutrality of the truncated sphere is
completed by the constant Wolf self term (alpha -> 0 limit of Wolf et al.,
J. Chem. Phys. 110, 8254 (1999))

    E_self = - e^2/(2 r_c) * sum_i q_i^2 ,

which is configuration independent and therefore exerts no force.

Constants verified against the sources on 2026-07-18 (web search + full-text
PDFs of arXiv:0707.0319 and the BKS parameter literature):

    q_Si = +2.4 e,  q_O = -1.2 e                         (BKS convention)
    Si-O: A = 18003.7572 eV, B = 4.87318 1/A, C = 133.5381 eV A^6
    O-O : A =  1388.7730 eV, B = 2.76000 1/A, C = 175.0000 eV A^6
    Si-Si: no Buckingham term (A = B = C = 0), Coulomb only
    r_sr = 5.5 A (Horbach & Kob, PRB 60, 3169 (1999); Carre et al. Sec. II.A)
    r_c  = 10.17 A (Carre et al., Sec. II.B and III)
    masses: m_Si = 28.086 u, m_O = 15.9994 u (Horbach & Kob, Sec. II)
    density: N = 8016 ions in L = 48.37 A, i.e. 2.37 g/cm^3 (Horbach & Kob)

Note on the citation trail: the task sheet lists EPL 82, 17001 (2008) next to
arXiv:0707.0319, but those are two different papers; arXiv:0707.0319 is the
Wolf-truncation study (J. Chem. Phys. 127, 114512 (2007)) and is the one
implemented here.  EPL 82, 17001 is the later CHIK re-parametrization.

Small-r guard
-------------
The raw BKS pair energy diverges to -infinity as r -> 0 (the -C/r^6 term
overwhelms the finite exponential), with a barrier top near 1.19 A (Si-O) /
1.62 A (O-O).  At melt temperatures (6000 K) rare collisions can cross the
barrier and fuse ions.  Following standard practice for high-T BKS work
(e.g. Saika-Voivod, Sciortino & Poole, PRE 63, 011202 (2000) add a
short-range regularization), the full pair interaction is continued below
the radius of maximum repulsive force r_lo with the strictly repulsive C^1
continuation

    phi_guard(r) = c1 / r^12 + c2,   c1 = -phi'(r_lo) r_lo^13 / 12 > 0,

matched in value and slope at r_lo.  r_lo is computed deterministically at
model construction (~1.26 A for Si-O, ~1.79 A for O-O with the production
cutoffs) and sits far below the physical first-neighbor distances (1.60 A
Si-O bond, 2.59 A O-O), so equilibrium structure is unaffected.  The guard
can be disabled (``guard=False``) to recover the literal published model.

Units
-----
BKS conventional units: length in Angstrom, energy in eV, mass in atomic
mass units (u), charge in units of e with e^2/(4 pi eps0) = 14.399645 eV A,
temperature in Kelvin through k_B = 8.617333262e-5 eV/K.  The derived
internal time unit is t* = A sqrt(u/eV) = 10.180506 fs; helpers convert
femtoseconds/picoseconds to internal units (1 ps = 98.22685 t*).

Determinism doctrine
--------------------
All reductions are fixed-order: pair lists are lexicographic (i < j), forces
are accumulated with the repository's sorted-neighbor segment sum
(:func:`butterfly_cone.engine.potential._deterministic_particle_sum`).  No scatter_add,
index_add, index_put_(accumulate=True), bincount, or atomic accumulation
appears anywhere in this module.  Twin trajectories stay bit-identical until
explicitly perturbed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

# Reuse the exact minimum-image convention, canonical i<j pair builder, cell
# lists, and (crucially) the fixed-order segment-sum reduction of the flagship
# engine so BKS forces obey the same bitwise-determinism contract.
from .neighbors import cell_list_pairs
from .potential import _deterministic_particle_sum, all_pairs, minimum_image

# --- species encoding ------------------------------------------------------
SPECIES_SI = 0
SPECIES_O = 1

# --- verified physical constants (see module docstring for provenance) -----
Q_SI = 2.4           # e   (van Beest-Kramer-van Santen 1990)
Q_O = -1.2           # e
COULOMB_K = 14.399645  # eV*A  = e^2 / (4 pi eps0)
MASS_SI = 28.086     # u   (Horbach & Kob, PRB 60, 3169 (1999))
MASS_O = 15.9994     # u
KB_EV_PER_K = 8.617333262e-5  # eV/K

# Buckingham tables indexed by kind k = species_i + species_j:
#   k=0 Si-Si (Coulomb only), k=1 Si-O, k=2 O-O.
_BUCK_A = (0.0, 18003.7572, 1388.7730)   # eV
_BUCK_B = (0.0, 4.87318, 2.76000)        # 1/A
_BUCK_C = (0.0, 133.5381, 175.0000)      # eV*A^6
_QQ = (Q_SI * Q_SI, Q_SI * Q_O, Q_O * Q_O)  # e^2

WOLF_CUTOFF = 10.17         # A (Carre et al. 2007, recommended r_c)
SHORT_RANGE_CUTOFF = 5.5    # A (Horbach-Kob truncate-and-shift radius)

# Horbach-Kob box: N = 8016 ions in L = 48.37 A <-> 2.37 g/cm^3.
NUMBER_DENSITY = 8016.0 / 48.37**3  # ions / A^3 (= 0.070833...)

# Internal time unit t* = A sqrt(u/eV) expressed in fs:
# sqrt(1.66053906660e-27 kg * (1e-10 m)^2 / 1.602176634e-19 J) = 10.180506 fs.
FS_PER_TIME_UNIT = 10.180505710774743
PS_PER_TIME_UNIT = FS_PER_TIME_UNIT * 1.0e-3


def fs_to_internal(dt_fs: float) -> float:
    """Convert a femtosecond interval to internal time units."""

    return float(dt_fs) / FS_PER_TIME_UNIT


def ps_to_internal(dt_ps: float) -> float:
    """Convert a picosecond interval to internal time units."""

    return float(dt_ps) / PS_PER_TIME_UNIT


@dataclass(frozen=True)
class BKSResult:
    """Energy/force/virial bundle mirroring ``PotentialResult``."""

    energy: torch.Tensor          # total: pair sum + Wolf self term
    forces: torch.Tensor          # (N, 3)
    virial: torch.Tensor          # sum of pair virials -r dphi/dr
    self_energy: torch.Tensor     # constant Wolf charge-neutralization term
    pair_energies: torch.Tensor   # per interacting pair
    pair_virials: torch.Tensor
    pair_indices: torch.Tensor    # (2, P) lexicographic i<j


class BKSPotential:
    """Wolf-truncated BKS pair potential with deterministic evaluation.

    ``wolf_cutoff``/``short_range_cutoff`` default to the published values;
    unit tests use reduced cutoffs so that tiny periodic boxes satisfy the
    minimum-image requirement (cutoff < L/2).
    """

    def __init__(
        self,
        *,
        wolf_cutoff: float = WOLF_CUTOFF,
        short_range_cutoff: float = SHORT_RANGE_CUTOFF,
        guard: bool = True,
    ) -> None:
        if wolf_cutoff <= 0.0 or short_range_cutoff <= 0.0:
            raise ValueError("cutoffs must be positive")
        self.wolf_cutoff = float(wolf_cutoff)
        self.short_range_cutoff = float(short_range_cutoff)
        self.cutoff = max(self.wolf_cutoff, self.short_range_cutoff)
        # Truncate-and-shift constants: S_k = A e^{-B r_sr} - C / r_sr^6.
        self._shift = tuple(
            _BUCK_A[k] * math.exp(-_BUCK_B[k] * self.short_range_cutoff)
            - _BUCK_C[k] / self.short_range_cutoff**6
            for k in range(3)
        )
        self.guard_enabled = bool(guard)
        # Guard parameters per kind: (r_lo, c1, c2); r_lo = 0 disables.
        self._guard = tuple(self._compute_guard(k) for k in range(3))
        self._tables: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = {}

    # -- scalar float64 pair functions (guard fitting + reference values) ---

    def _phi_scalar(self, r: float, kind: int) -> float:
        """Full un-guarded pair energy at radius ``r`` (float64 scalar path)."""

        value = 0.0
        rc = self.wolf_cutoff
        if r < rc:
            value += _QQ[kind] * COULOMB_K * (1.0 / r - 1.0 / rc + (r - rc) / rc**2)
        rs = self.short_range_cutoff
        if r < rs:
            value += _BUCK_A[kind] * math.exp(-_BUCK_B[kind] * r) - _BUCK_C[kind] / r**6 - self._shift[kind]
        return value

    def _dphi_scalar(self, r: float, kind: int) -> float:
        derivative = 0.0
        rc = self.wolf_cutoff
        if r < rc:
            derivative += _QQ[kind] * COULOMB_K * (-1.0 / r**2 + 1.0 / rc**2)
        rs = self.short_range_cutoff
        if r < rs:
            derivative += (
                -_BUCK_A[kind] * _BUCK_B[kind] * math.exp(-_BUCK_B[kind] * r)
                + 6.0 * _BUCK_C[kind] / r**7
            )
        return derivative

    def _compute_guard(self, kind: int) -> tuple[float, float, float]:
        """Fit the C^1 repulsive continuation below the max-force radius.

        Deterministic: a fixed 4001-point float64 grid scan (no stochastic
        optimizer), identical on every construction with equal parameters.
        """

        if not self.guard_enabled or _BUCK_C[kind] <= 0.0:
            return (0.0, 0.0, 0.0)
        low = 0.6
        high = min(2.9, 0.9 * self.short_range_cutoff, 0.9 * self.wolf_cutoff)
        if high <= low + 0.1:
            return (0.0, 0.0, 0.0)
        n_grid = 4001
        best_r, best_d = low, math.inf
        for index in range(n_grid):
            r = low + (high - low) * index / (n_grid - 1)
            d = self._dphi_scalar(r, kind)
            if d < best_d:
                best_d, best_r = d, r
        if best_d >= 0.0:
            # No repulsive wall inside the scan window (pathological cutoffs):
            # the guard cannot be anchored, so leave the pair un-guarded.
            return (0.0, 0.0, 0.0)
        r_lo = best_r
        c1 = -best_d * r_lo**13 / 12.0
        c2 = self._phi_scalar(r_lo, kind) - c1 / r_lo**12
        return (r_lo, c1, c2)

    def guard_parameters(self, kind: int) -> tuple[float, float, float]:
        """Return ``(r_lo, c1, c2)`` for kind ``k`` (r_lo = 0 means no guard)."""

        return self._guard[kind]

    # -- tensor tables ------------------------------------------------------

    def _table(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        key = (device, dtype)
        cached = self._tables.get(key)
        if cached is not None:
            return cached

        def const(values: tuple[float, ...]) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=dtype)

        table = {
            "A": const(_BUCK_A),
            "AB": const(tuple(a * b for a, b in zip(_BUCK_A, _BUCK_B))),
            "B": const(_BUCK_B),
            "C": const(_BUCK_C),
            "shift": const(self._shift),
            "qqk": const(tuple(q * COULOMB_K for q in _QQ)),
            "guard_r": const(tuple(g[0] for g in self._guard)),
            "guard_c1": const(tuple(g[1] for g in self._guard)),
            "guard_c2": const(tuple(g[2] for g in self._guard)),
        }
        self._tables[key] = table
        return table

    # -- vectorized pair kernel ---------------------------------------------

    def pair_interaction(
        self,
        radius: torch.Tensor,
        kind: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Elementwise pair energy and radial derivative dphi/dr.

        ``kind`` is the int64 species-pair index ``species_i + species_j``.
        Purely elementwise gather/where arithmetic: deterministic by
        construction.
        """

        if radius.shape != kind.shape:
            raise ValueError("radius and kind must have identical shapes")
        table = self._table(radius.device, radius.dtype)
        a = table["A"][kind]
        ab = table["AB"][kind]
        b = table["B"][kind]
        c = table["C"][kind]
        shift = table["shift"][kind]
        qqk = table["qqk"][kind]
        r = torch.clamp(radius, min=torch.finfo(radius.dtype).eps)
        inv_r = r.reciprocal()
        rc = self.wolf_cutoff
        rs = self.short_range_cutoff

        # Wolf shifted-force Coulomb (Carre et al. 2007, Eq. (4)): value and
        # slope both vanish at r_c, so there is no cutoff discontinuity.
        coulomb = qqk * (inv_r - 1.0 / rc + (r - rc) / rc**2)
        coulomb_d = qqk * (1.0 / rc**2 - inv_r.square())
        in_coulomb = r < rc
        zero = torch.zeros_like(r)
        value = torch.where(in_coulomb, coulomb, zero)
        derivative = torch.where(in_coulomb, coulomb_d, zero)

        # Truncated-and-shifted Buckingham (value-continuous at r_sr; the
        # small residual force step at 5.5 A is part of the published model).
        exponential = torch.exp(-b * r)
        inv_r6 = inv_r.square().square() * inv_r.square()
        buckingham = a * exponential - c * inv_r6 - shift
        buckingham_d = -ab * exponential + 6.0 * c * inv_r6 * inv_r
        in_short = r < rs
        value = value + torch.where(in_short, buckingham, zero)
        derivative = derivative + torch.where(in_short, buckingham_d, zero)

        # Strictly repulsive C^1 continuation below the max-force radius.
        guard_r = table["guard_r"][kind]
        guard_c1 = table["guard_c1"][kind]
        guard_c2 = table["guard_c2"][kind]
        in_guard = r < guard_r
        inv_r12 = inv_r6.square()
        value = torch.where(in_guard, guard_c1 * inv_r12 + guard_c2, value)
        derivative = torch.where(in_guard, -12.0 * guard_c1 * inv_r12 * inv_r, derivative)
        return value, derivative

    # -- totals -------------------------------------------------------------

    def self_energy(
        self,
        species: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Constant Wolf self term -e^2/(2 r_c) sum_i q_i^2 (no force)."""

        if species.ndim != 1:
            raise ValueError("species must be a 1D tensor")
        target_dtype = species.dtype if species.is_floating_point() else (dtype or torch.float64)
        n_si = (species == SPECIES_SI).sum()
        n_o = (species == SPECIES_O).sum()
        charge_square = n_si.to(target_dtype) * (Q_SI * Q_SI) + n_o.to(target_dtype) * (Q_O * Q_O)
        scale = -COULOMB_K / (2.0 * self.wolf_cutoff)
        return charge_square * scale

    def evaluate(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
        box: torch.Tensor,
        *,
        pairs: torch.Tensor | None = None,
    ) -> BKSResult:
        """Total energy, analytic forces, and per-pair virials.

        Pairs are canonicalized to the ordered set with r < cutoff so a
        Verlet candidate list and the dense O(N^2) reference reduce over the
        identical ordered pair sequence and agree bitwise.
        """

        n_particles = int(positions.shape[0])
        if positions.shape != (n_particles, 3):
            raise ValueError("positions must have shape (N, 3)")
        if species.shape != (n_particles,) or species.is_floating_point():
            raise ValueError("species must be an integer tensor of shape (N,)")
        if pairs is None:
            pairs = all_pairs(n_particles, positions.device)
        if pairs.shape[0] != 2:
            raise ValueError("pairs must have shape (2, P)")
        i, j = pairs[0], pairs[1]
        displacement = minimum_image(positions[i] - positions[j], box)
        radius = torch.linalg.vector_norm(displacement, dim=1)
        interacting = radius < self.cutoff
        pairs = pairs[:, interacting]
        i, j = pairs[0], pairs[1]
        displacement = displacement[interacting]
        radius = radius[interacting]
        kind = species[i] + species[j]
        pair_energy, derivative = self.pair_interaction(radius, kind)
        safe_radius = torch.clamp(radius, min=torch.finfo(radius.dtype).eps)
        force_i = -(derivative / safe_radius)[:, None] * displacement
        pair_virial = -derivative * radius

        contribution_indices = torch.cat((i, j))
        contribution_values = torch.cat((force_i, -force_i), dim=0)
        forces = _deterministic_particle_sum(contribution_indices, contribution_values, n_particles)
        self_term = self.self_energy(species, dtype=positions.dtype).to(
            device=positions.device, dtype=positions.dtype
        )
        return BKSResult(
            energy=pair_energy.sum() + self_term,
            forces=forces,
            virial=pair_virial.sum(),
            self_energy=self_term,
            pair_energies=pair_energy,
            pair_virials=pair_virial,
            pair_indices=pairs,
        )


def bks_autograd_forces(
    potential: BKSPotential,
    positions: torch.Tensor,
    species: torch.Tensor,
    box: torch.Tensor,
    *,
    pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Test-only force path obtained as ``-grad(total energy)``."""

    differentiable = positions.detach().clone().requires_grad_(True)
    energy = potential.evaluate(differentiable, species, box, pairs=pairs).energy
    forces = -torch.autograd.grad(energy, differentiable, create_graph=False)[0]
    return forces.detach()


# --------------------------------------------------------------------------
# System state
# --------------------------------------------------------------------------


def _cpu_random(
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    normal: bool = False,
) -> torch.Tensor:
    """CPU float64 draws from the caller-owned generator (engine convention)."""

    if str(generator.device) != "cpu":
        raise ValueError("ButterflyCone generators must be CPU generators")
    draw = torch.randn if normal else torch.rand
    return draw(shape, generator=generator, device="cpu", dtype=torch.float64)


@dataclass
class SilicaSystem:
    """Tensor state for an SiO2 ionic system in a periodic cubic box."""

    positions: torch.Tensor
    velocities: torch.Tensor
    species: torch.Tensor        # int64, 0 = Si, 1 = O
    masses: torch.Tensor         # u
    box: torch.Tensor
    unwrapped_positions: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.positions.shape[0])
        if self.positions.shape != (n, 3):
            raise ValueError("positions must have shape (N, 3)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must match positions")
        if self.unwrapped_positions.shape != self.positions.shape:
            raise ValueError("unwrapped_positions must match positions")
        if self.species.shape != (n,) or self.species.is_floating_point():
            raise ValueError("species must be an integer tensor of shape (N,)")
        if self.masses.shape != (n,) or bool(torch.any(self.masses <= 0)):
            raise ValueError("masses must be positive with shape (N,)")
        if self.box.shape != (3,) or bool(torch.any(self.box <= 0)):
            raise ValueError("box must contain three positive lengths")
        tensors = (self.velocities, self.species, self.masses, self.box, self.unwrapped_positions)
        if any(tensor.device != self.positions.device for tensor in tensors):
            raise ValueError("all state tensors must share a device")

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    def clone(self) -> "SilicaSystem":
        return SilicaSystem(
            positions=self.positions.detach().clone(),
            velocities=self.velocities.detach().clone(),
            species=self.species.detach().clone(),
            masses=self.masses.detach().clone(),
            box=self.box.detach().clone(),
            unwrapped_positions=self.unwrapped_positions.detach().clone(),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        clone = self.clone()
        return {
            "positions": clone.positions,
            "velocities": clone.velocities,
            "species": clone.species,
            "masses": clone.masses,
            "box": clone.box,
            "unwrapped_positions": clone.unwrapped_positions,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> "SilicaSystem":
        target = state["positions"].device if device is None else torch.device(device)
        return cls(**{name: tensor.detach().clone().to(target) for name, tensor in state.items()})


def species_masses(
    species: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map the 0/1 species labels to (m_Si, m_O) in atomic mass units."""

    mass_si = torch.tensor(MASS_SI, device=species.device, dtype=dtype)
    mass_o = torch.tensor(MASS_O, device=species.device, dtype=dtype)
    return torch.where(species == SPECIES_SI, mass_si, mass_o)


def _lattice_positions(n_atoms: int, box: torch.Tensor) -> torch.Tensor:
    """Cell centers of the smallest enclosing cubic grid (engine convention)."""

    side = math.ceil(n_atoms ** (1.0 / 3.0))
    while side**3 < n_atoms:
        side += 1
    axis = (torch.arange(side, device=box.device, dtype=box.dtype) + 0.5) / side
    grid = torch.cartesian_prod(axis, axis, axis)
    if grid.ndim == 1:
        grid = grid.reshape(-1, 3)
    return grid[:n_atoms] * box


def make_silica_system(
    n_units: int,
    *,
    generator: torch.Generator,
    number_density: float = NUMBER_DENSITY,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SilicaSystem:
    """Construct a reproducible zero-velocity SiO2 configuration.

    ``n_units`` formula units give N = 3 n_units ions with exact 2:1
    stoichiometry (N_Si = n_units silicons, N_O = 2 n_units oxygens), placed
    on a cubic lattice at the Horbach-Kob number density with the species
    labels deterministically shuffled through the caller-owned CPU generator
    (argsort of uniform draws; no backend-global RNG).
    """

    if n_units <= 0:
        raise ValueError("n_units must be positive")
    if number_density <= 0.0:
        raise ValueError("number_density must be positive")
    n_atoms = 3 * n_units
    length = (n_atoms / number_density) ** (1.0 / 3.0)
    box = torch.full((3,), length, device=device, dtype=dtype)
    positions = _lattice_positions(n_atoms, box)
    ordered = torch.cat(
        (
            torch.full((n_units,), SPECIES_SI, dtype=torch.int64),
            torch.full((2 * n_units,), SPECIES_O, dtype=torch.int64),
        )
    )
    shuffle = torch.argsort(_cpu_random((n_atoms,), generator))
    species = ordered[shuffle].to(device=torch.device(device))
    masses = species_masses(species, dtype=dtype)
    return SilicaSystem(
        positions=positions,
        velocities=torch.zeros_like(positions),
        species=species,
        masses=masses,
        box=box,
        unwrapped_positions=positions.clone(),
    )


def silica_maxwell_velocities(
    system: SilicaSystem,
    temperature_kelvin: float,
    generator: torch.Generator,
    *,
    remove_com: bool = True,
) -> torch.Tensor:
    """Maxwell-Boltzmann velocities at T (Kelvin) with per-species masses."""

    if temperature_kelvin < 0.0:
        raise ValueError("temperature must be nonnegative")
    kt = KB_EV_PER_K * float(temperature_kelvin)
    normal = _cpu_random((system.n_atoms, 3), generator, normal=True).to(
        device=system.device, dtype=system.dtype
    )
    sigma = torch.sqrt(torch.as_tensor(kt, device=system.device, dtype=system.dtype) / system.masses)
    velocities = normal * sigma[:, None]
    if remove_com:
        total_mass = system.masses.sum()
        momentum = (system.masses[:, None] * velocities).sum(dim=0)
        velocities = velocities - momentum / total_mass
    return velocities


# --------------------------------------------------------------------------
# Neighbor list
# --------------------------------------------------------------------------


@dataclass
class BKSNeighborList:
    """Verlet list over the BKS interaction range (deterministic cell lists)."""

    potential: BKSPotential
    skin: float
    pair_indices: torch.Tensor
    reference_positions: torch.Tensor
    list_radius: float
    rebuild_count: int = 1

    @classmethod
    def from_system(
        cls,
        system: SilicaSystem,
        potential: BKSPotential,
        *,
        skin: float = 0.5,
    ) -> "BKSNeighborList":
        if skin <= 0.0:
            raise ValueError("skin must be positive")
        list_radius = potential.cutoff + float(skin)
        pairs = cell_list_pairs(system.positions, system.box, list_radius)
        return cls(
            potential=potential,
            skin=float(skin),
            pair_indices=pairs,
            reference_positions=system.positions.detach().clone(),
            list_radius=list_radius,
        )

    def needs_rebuild(self, positions: torch.Tensor, box: torch.Tensor) -> bool:
        displacement = minimum_image(positions - self.reference_positions, box)
        maximum = torch.linalg.vector_norm(displacement, dim=1).max()
        return bool(maximum > 0.5 * self.skin)

    def update(self, system: SilicaSystem) -> bool:
        if not self.needs_rebuild(system.positions, system.box):
            return False
        self.pair_indices = cell_list_pairs(system.positions, system.box, self.list_radius)
        self.reference_positions = system.positions.detach().clone()
        self.rebuild_count += 1
        return True

    def evaluate(self, system: SilicaSystem) -> BKSResult:
        self.update(system)
        return self.potential.evaluate(
            system.positions,
            system.species,
            system.box,
            pairs=self.pair_indices,
        )


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------


class SilicaBussiThermostat:
    """Bussi-Donadio-Parrinello stochastic velocity rescaling with masses.

    Temperature is in Kelvin; the kinetic energy is 0.5 sum_i m_i v_i^2 in
    eV.  Mirrors :class:`butterfly_cone.engine.integrate.BussiThermostat` (which is
    unit-mass) with the mass-weighted kinetic energy.
    """

    def __init__(self, temperature_kelvin: float, tau: float, generator: torch.Generator) -> None:
        if temperature_kelvin <= 0.0 or tau <= 0.0:
            raise ValueError("temperature and tau must be positive")
        if str(generator.device) != "cpu":
            raise ValueError("ButterflyCone generators must be CPU generators")
        self.temperature_kelvin = float(temperature_kelvin)
        self.tau = float(tau)
        self.generator = generator
        self.last_alpha = 1.0
        self.heat = 0.0

    def apply(self, system: SilicaSystem, dt: float) -> float:
        n_atoms = system.n_atoms
        ndof = 3 * n_atoms - (3 if n_atoms > 1 else 0)
        if ndof <= 0:
            return 1.0
        kinetic_before = 0.5 * (system.masses * system.velocities.square().sum(dim=1)).sum()
        if float(kinetic_before) <= 0.0:
            raise ValueError("Bussi rescaling requires nonzero kinetic energy")
        randoms = _cpu_random((ndof,), self.generator, normal=True).to(system.device, system.dtype)
        gaussian = randoms[0]
        chi_square = randoms[1:].square().sum()
        c = math.exp(-float(dt) / self.tau)
        target_kinetic = 0.5 * ndof * KB_EV_PER_K * self.temperature_kelvin
        ratio = torch.as_tensor(target_kinetic, device=system.device, dtype=system.dtype) / kinetic_before
        alpha_squared = (
            c
            + (1.0 - c) * ratio * (chi_square + gaussian.square()) / ndof
            + 2.0 * gaussian * torch.sqrt(c * (1.0 - c) * ratio / ndof)
        )
        alpha = torch.sqrt(torch.clamp(alpha_squared, min=0.0))
        sign_threshold = gaussian + torch.sqrt(
            torch.as_tensor(c / (1.0 - c) * ndof, device=system.device, dtype=system.dtype) / ratio
        )
        alpha = torch.where(sign_threshold < 0.0, -alpha, alpha)
        system.velocities = system.velocities * alpha
        kinetic_after = 0.5 * (system.masses * system.velocities.square().sum(dim=1)).sum()
        self.last_alpha = float(alpha)
        self.heat += float(kinetic_after - kinetic_before)
        return self.last_alpha

    def state_dict(self) -> dict[str, Any]:
        return {
            "temperature_kelvin": self.temperature_kelvin,
            "tau": self.tau,
            "generator_state": self.generator.get_state().detach().clone(),
            "last_alpha": self.last_alpha,
            "heat": self.heat,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SilicaBussiThermostat":
        generator = torch.Generator(device="cpu")
        generator.set_state(state["generator_state"].detach().clone().cpu())
        thermostat = cls(float(state["temperature_kelvin"]), float(state["tau"]), generator)
        thermostat.last_alpha = float(state["last_alpha"])
        thermostat.heat = float(state["heat"])
        return thermostat


class SilicaIntegrator:
    """Velocity-Verlet with per-species masses and optional Bussi NVT.

    ``dt`` is in internal time units (use :func:`fs_to_internal`).  The
    update order and torch operations exactly mirror
    :class:`butterfly_cone.engine.integrate.MDIntegrator` so twin trajectories on the
    same device stay bit-identical until explicitly perturbed.
    """

    def __init__(
        self,
        system: SilicaSystem,
        potential: BKSPotential,
        *,
        dt: float,
        skin: float = 0.5,
        neighbor_list: BKSNeighborList | None = None,
        thermostat: SilicaBussiThermostat | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.system = system
        self.potential = potential
        self.dt = float(dt)
        self.neighbor_list = (
            BKSNeighborList.from_system(system, potential, skin=skin)
            if neighbor_list is None
            else neighbor_list
        )
        self.thermostat = thermostat
        self.step_count = 0
        result = self.neighbor_list.evaluate(system)
        self.forces = result.forces
        self.potential_energy = result.energy
        self.virial = result.virial

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        inverse_mass = self.system.masses.reciprocal()[:, None]
        for _ in range(steps):
            half_velocity = self.system.velocities + 0.5 * self.dt * self.forces * inverse_mass
            displacement = self.dt * half_velocity
            self.system.unwrapped_positions = self.system.unwrapped_positions + displacement
            self.system.positions = torch.remainder(self.system.positions + displacement, self.system.box)
            result = self.neighbor_list.evaluate(self.system)
            self.system.velocities = half_velocity + 0.5 * self.dt * result.forces * inverse_mass
            self.forces = result.forces
            self.potential_energy = result.energy
            self.virial = result.virial
            if self.thermostat is not None:
                self.thermostat.apply(self.system, self.dt)
            self.step_count += 1

    def kinetic_energy(self) -> torch.Tensor:
        return 0.5 * (self.system.masses * self.system.velocities.square().sum(dim=1)).sum()

    def total_energy(self) -> torch.Tensor:
        return self.potential_energy + self.kinetic_energy()

    def instantaneous_temperature(self) -> float:
        """Kinetic temperature in Kelvin (3N - 3 degrees of freedom)."""

        n_atoms = self.system.n_atoms
        ndof = 3 * n_atoms - (3 if n_atoms > 1 else 0)
        if ndof <= 0:
            return 0.0
        return float(2.0 * self.kinetic_energy() / (ndof * KB_EV_PER_K))


def capped_descent(
    system: SilicaSystem,
    potential: BKSPotential,
    *,
    steps: int = 500,
    max_displacement: float = 0.01,
    skin: float = 0.5,
    force_tolerance: float = 1.0e-3,
) -> dict[str, float]:
    """Deterministic force-aligned capped minimization (lattice preparation).

    Mirrors :func:`butterfly_cone.engine.system.relax_overlaps`: displacement per step
    is the force direction capped at ``max_displacement`` Angstrom, which is
    robust to the very large forces of a freshly assigned lattice.
    """

    if steps < 0 or max_displacement <= 0.0 or force_tolerance < 0.0:
        raise ValueError("invalid descent controls")
    neighbors = BKSNeighborList.from_system(system, potential, skin=skin)
    result = neighbors.evaluate(system)
    initial_energy = float(result.energy)
    completed = 0
    for _ in range(steps):
        force_norm = torch.linalg.vector_norm(result.forces, dim=1, keepdim=True)
        maximum_force = float(force_norm.max())
        if maximum_force <= force_tolerance:
            break
        scale = torch.clamp(
            max_displacement / torch.clamp(force_norm, min=torch.finfo(system.dtype).eps),
            max=1.0,
        )
        displacement = result.forces * scale
        system.positions = torch.remainder(system.positions + displacement, system.box)
        system.unwrapped_positions = system.unwrapped_positions + displacement
        result = neighbors.evaluate(system)
        completed += 1
    return {
        "steps_completed": float(completed),
        "initial_energy": initial_energy,
        "final_energy": float(result.energy),
        "final_max_force": float(torch.linalg.vector_norm(result.forces, dim=1).max()),
    }
