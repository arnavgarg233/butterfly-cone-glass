#!/usr/bin/env python3
"""Debye--Waller identity for the butterfly-cone saturation plateau.

The claim (docs/amplify/wave32/insane-additions.md, Addition 3): the
delta-independent sub-cage divergence plateau ``D_sat`` is not merely "small",
it is the value forced by *complete Gaussian decorrelation inside intact cages*.
If the perturbed and unperturbed branches both relax to independent draws from
the same intra-cage displacement distribution, then per particle

    D_sat / N  =  E| dr_pert - dr_unpert |  =  c * u_DW ,

where ``u_DW = sqrt(<u^2>)`` is the root-mean-square cage displacement about the
cage centre (the Debye--Waller amplitude, a *length* in sigma), and

    c = 2*sqrt(2/pi) * sqrt(2) / sqrt(3)  ~= 1.30294

is the first-absolute-moment of the difference of two independent isotropic 3-D
Gaussian cage displacements, normalised by the single-branch RMS amplitude
(chi_3 statistics of ``dr_A - dr_B`` with the ``divergence_power = 1`` first-power
norm the pipeline uses).  Derivation, with dr ~ N(0, sigma^2 I_3) per component
so <u^2> = 3 sigma^2:

    dr_A - dr_B ~ N(0, 2 sigma^2 I_3)
    E|dr_A - dr_B| = sqrt(2) sigma * E[chi_3] = sqrt(2) sigma * 2 sqrt(2/pi)
                   = [2 sqrt(2/pi) sqrt(2) / sqrt(3)] * sqrt(3) sigma
                   = c * u_DW .

This module is *pure re-analysis* of already-persisted trajectories: it reads the
unperturbed Gardner branch stores (``runs/gardner/<run>--c*-unpert/branches/*``),
measures ``u_DW`` from the intrinsic single-branch cage rattle, and tests
``D_sat/N == c * u_DW`` against the landed ``D_sat/N`` (``gardner_r0.json``).  It
adds no MD.  It imports the response stack (``butterfly_cone.perturb.response``) read-only
to reduce a per-particle field to a total the same way the cone pipeline does.

Both outcomes are findings.  A hit universalises the plateau: butterfly
saturation in any glass = 1.30 * its Debye--Waller amplitude, and ``u_DW`` is
directly measurable by elastic neutron/X-ray scattering, so the chaos plateau
becomes experimentally portable.  A miss (the plateau sits above ``c * u_DW``)
means a genuine third length scale between vibration and cage -- more
interesting, and it motivates the ``D_sat(T)`` curve.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Read-only reuse of the response stack: identical (T,N)->(T,) total-divergence
# reduction the cone pipeline applies to D_sat, so our per-particle numbers are
# compared to the landed number on exactly the same footing.
from butterfly_cone.perturb.response import total_divergence


# ---------------------------------------------------------------------------
# The parameter-free chi_3 constant
# ---------------------------------------------------------------------------


def chi3_constant() -> float:
    """``c = 2 sqrt(2/pi) sqrt(2) / sqrt(3)`` -- the DW identity prefactor.

    First-absolute-moment of ``dr_A - dr_B`` (difference of two independent 3-D
    isotropic Gaussian cage displacements) divided by the single-branch RMS
    amplitude ``sqrt(<u^2>)``.  ``~= 1.302940``.
    """

    return 2.0 * math.sqrt(2.0 / math.pi) * math.sqrt(2.0) / math.sqrt(3.0)


CHI3 = chi3_constant()


# ---------------------------------------------------------------------------
# Minimum-image helper (bit-identical ops to response.branch_divergence)
# ---------------------------------------------------------------------------


def _minimum_image(delta: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Wrap displacement vectors to the nearest periodic image.

    Uses the same ``x - rint(x/L)*L`` elementwise operations as
    ``butterfly_cone.perturb.response.branch_divergence`` so distances are constant-for-
    constant with the pipeline.  ``box=None`` returns ``delta`` unchanged (open
    boundaries / already-unwrapped intra-cage displacements).
    """

    if box is None:
        return delta
    box_arr = np.asarray(box, dtype=float)
    images = delta / box_arr
    np.rint(images, out=images)
    images *= box_arr
    return delta - images


# ---------------------------------------------------------------------------
# u_DW from an intrinsic branch ensemble (the Debye--Waller amplitude)
# ---------------------------------------------------------------------------


def mean_squared_cage_displacement(
    positions: np.ndarray,
    box: np.ndarray | None = None,
    *,
    ddof: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """``<u^2>`` about the branch-mean cage centre for one plateau frame.

    ``positions`` has shape ``(B, N, 3)`` -- B independent branches sharing a
    parent, at one (plateau) frame.  The cage centre of particle i is the mean
    of its B branch positions; ``u^2_i`` is the (``ddof``-corrected) variance of
    the branch positions about that centre summed over x,y,z.  Returns
    ``(<u^2> over particles, per-particle u^2 (N,), cage_centre (N,3))``.

    ``ddof=1`` (default) is the unbiased population-variance estimator: the
    centre is fitted from the same B branches, consuming one degree of freedom.
    """

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape (B, N, 3)")
    n_branches = pos.shape[0]
    if n_branches - ddof <= 0:
        raise ValueError("need more branches than ddof to estimate a cage variance")
    centre = pos.mean(axis=0)
    dev = _minimum_image(pos - centre[None, :, :], box)
    sq = np.square(dev).sum(axis=-1)  # (B, N) squared amplitude per branch/particle
    per_particle = sq.sum(axis=0) / (n_branches - ddof)  # (N,)
    return float(per_particle.mean()), per_particle, centre


def cage_msd_curve(
    positions: np.ndarray,
    box: np.ndarray | None = None,
    *,
    ddof: int = 1,
) -> np.ndarray:
    """Per-frame ``<u^2>(t)`` about the per-frame branch-mean cage centre.

    ``positions`` has shape ``(T, B, N, 3)``.  Recomputing the centre per frame
    removes any slow drift of the cage within the window.  Returns ``(T,)``.
    """

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 4 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape (T, B, N, 3)")
    return np.array(
        [mean_squared_cage_displacement(pos[t], box, ddof=ddof)[0] for t in range(pos.shape[0])],
        dtype=float,
    )


def msd_relative_to_reference(
    positions: np.ndarray,
    reference: np.ndarray,
    box: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame ``<|r(t) - r_ref|^2>`` over branches and particles, shape ``(T,)``.

    With ``reference`` = the shared parent, a fully decorrelated cage gives
    ``MSD_plateau -> 2 <u^2>`` (the parent is itself a draw from the cage), so
    ``u_DW = sqrt(MSD_plateau / 2)`` -- an independent cross-check of the
    branch-variance route.
    """

    pos = np.asarray(positions, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pos.ndim != 4 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape (T, B, N, 3)")
    dev = _minimum_image(pos - ref[None, None, :, :], box)
    return np.square(dev).sum(axis=-1).mean(axis=(1, 2))


def plateau_mean(curve: Sequence[float] | np.ndarray, frac: float = 0.5) -> float:
    """Mean of the final ``frac`` fraction of a curve (its plateau value)."""

    arr = np.asarray(list(curve), dtype=float)
    if arr.size == 0:
        raise ValueError("empty curve")
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    start = min(arr.size - 1, int(math.floor(arr.size * (1.0 - frac))))
    return float(arr[start:].mean())


def u_dw_from_mean_squared(u2: float) -> float:
    """``u_DW = sqrt(<u^2>)`` -- guards against tiny negative round-off."""

    return math.sqrt(max(float(u2), 0.0))


def u_dw_from_msd_plateau(msd_plateau: float) -> float:
    """``u_DW = sqrt(MSD_plateau / 2)`` for MSD measured against a cage draw."""

    return math.sqrt(max(float(msd_plateau), 0.0) / 2.0)


def u2_from_self_overlap(q_plateau: float, a: float) -> float:
    """Invert the Gaussian-window self-overlap plateau to ``<u^2>``.

    For a Gaussian overlap kernel ``w(r) = exp(-r^2 / 2a^2)`` and isotropic
    Gaussian cage displacements, ``Q = (1 + <u^2>/(3 a^2))^{-3/2}``, hence
    ``<u^2> = 3 a^2 (Q^{-2/3} - 1)``.  (The small-displacement heuristic
    ``<u^2> ~= -2 a^2 ln Q`` agrees to leading order as ``Q -> 1``.)
    """

    if not 0.0 < q_plateau <= 1.0:
        raise ValueError("q_plateau must be in (0, 1]")
    if a <= 0.0:
        raise ValueError("a must be positive")
    return 3.0 * a * a * (q_plateau ** (-2.0 / 3.0) - 1.0)


def pairwise_branch_divergence_per_particle(
    positions: np.ndarray,
    box: np.ndarray | None = None,
) -> float:
    """Direct ``E|r_A - r_B|`` over unordered branch pairs, mean over particles.

    ``positions`` has shape ``(B, N, 3)`` at one plateau frame.  This reproduces
    the pipeline's first-power divergence-per-particle *without any perturbation*
    -- the intrinsic cage-decorrelation plateau -- and should match the landed
    (perturbed) ``D_sat/N`` if saturation is complete intra-cage decorrelation.
    The per-particle field is reduced with ``response.total_divergence`` /N so it
    is constant-for-constant with the cone pipeline's ``D_sat``.
    """

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape (B, N, 3)")
    n_branches, n_particles, _ = pos.shape
    if n_branches < 2:
        raise ValueError("need at least two branches for a pairwise divergence")
    field = np.zeros(n_particles, dtype=float)  # per-particle sum of pair magnitudes
    n_pairs = 0
    for a in range(n_branches - 1):
        diff = _minimum_image(pos[a + 1 :] - pos[a][None, :, :], box)  # (B-a-1, N, 3)
        field += np.linalg.norm(diff, axis=-1).sum(axis=0)
        n_pairs += n_branches - a - 1
    field /= n_pairs
    return float(total_divergence(field)) / n_particles


# ---------------------------------------------------------------------------
# The identity test
# ---------------------------------------------------------------------------


def dw_identity(
    u_dw: float,
    d_sat_over_n: float,
    *,
    c: float | None = None,
    tol: float = 0.10,
) -> dict[str, Any]:
    """Test ``D_sat/N == c * u_DW``.

    Returns the predicted plateau ``c * u_DW``, the empirical prefactor
    ``D_sat/N / u_DW`` (== ``c`` iff the cages are exactly Gaussian), the ratio
    ``measured / predicted``, its relative error, and whether the identity holds
    within ``tol`` (default 10%).
    """

    c_val = CHI3 if c is None else float(c)
    u = float(u_dw)
    measured = float(d_sat_over_n)
    predicted = c_val * u
    empirical_c = measured / u if u > 0.0 else float("nan")
    ratio = measured / predicted if predicted > 0.0 else float("nan")
    rel_error = abs(ratio - 1.0) if math.isfinite(ratio) else float("nan")
    holds = bool(math.isfinite(rel_error) and rel_error <= tol)
    return {
        "c": c_val,
        "u_DW": u,
        "measured_D_sat_over_N": measured,
        "predicted_D_sat_over_N": predicted,
        "empirical_c": empirical_c,
        "ratio_measured_over_predicted": ratio,
        "rel_error": rel_error,
        "tol": float(tol),
        "holds": holds,
    }


# ---------------------------------------------------------------------------
# Disk drivers (real persisted Gardner branch stores)
# ---------------------------------------------------------------------------


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("loading persisted .pt branch stores requires torch") from exc
    return torch


def load_run_positions(config_dir: str | Path) -> dict[str, Any]:
    """Load the persisted unperturbed branch ensemble of one config directory.

    Expects ``<config_dir>/parent_state.pt`` and
    ``<config_dir>/branches/*/trajectory.pt`` as written by the Gardner probe.
    Returns ``{positions: (T,B,N,3), box: (3,), parent: (N,3), times, n_branches}``
    using unwrapped positions (the engine's dynamics convention).
    """

    torch = _require_torch()
    root = Path(config_dir)
    parent_state = torch.load(root / "parent_state.pt", map_location="cpu", weights_only=False)
    box = parent_state["box"].detach().cpu().numpy().astype(float)
    parent = parent_state["unwrapped_positions"].detach().cpu().numpy().astype(float)
    branch_dirs = sorted((root / "branches").glob("*"))
    if not branch_dirs:
        raise FileNotFoundError(f"no branches under {root / 'branches'}")
    frames: list[np.ndarray] = []
    steps: np.ndarray | None = None
    for bdir in branch_dirs:
        traj = torch.load(bdir / "trajectory.pt", map_location="cpu", weights_only=False)
        frames.append(traj["unwrapped_positions"].detach().cpu().numpy().astype(float))
        if steps is None:
            steps = traj["steps"].detach().cpu().numpy().astype(float)
    positions = np.stack(frames, axis=1)  # (T, B, N, 3)
    return {
        "positions": positions,
        "box": box,
        "parent": parent,
        "steps": steps,
        "n_branches": positions.shape[1],
    }


def measure_u_dw_for_run(
    config_dir: str | Path,
    *,
    plateau_frac: float = 0.5,
    ddof: int = 1,
    with_pairwise: bool = True,
) -> dict[str, Any]:
    """Measure ``u_DW`` (and cross-checks) for one persisted config directory."""

    loaded = load_run_positions(config_dir)
    positions = loaded["positions"]
    box = loaded["box"]
    parent = loaded["parent"]
    n_frames = positions.shape[0]

    u2_curve = cage_msd_curve(positions, box, ddof=ddof)
    msd_curve = msd_relative_to_reference(positions, parent, box)
    u2_plateau = plateau_mean(u2_curve, plateau_frac)
    msd_plateau = plateau_mean(msd_curve, plateau_frac)

    u_dw = u_dw_from_mean_squared(u2_plateau)
    u_dw_msd = u_dw_from_msd_plateau(msd_plateau)

    result: dict[str, Any] = {
        "config_dir": str(config_dir),
        "n_branches": loaded["n_branches"],
        "n_frames": n_frames,
        "plateau_frac": plateau_frac,
        "ddof": ddof,
        "u2_cage_plateau": u2_plateau,
        "u_DW": u_dw,
        "msd_rel_parent_plateau": msd_plateau,
        "u_DW_from_msd": u_dw_msd,
        "u2_cage_curve": [float(x) for x in u2_curve],
    }
    if with_pairwise:
        start = min(n_frames - 1, int(math.floor(n_frames * (1.0 - plateau_frac))))
        pairwise = [
            pairwise_branch_divergence_per_particle(positions[t], box)
            for t in range(start, n_frames)
        ]
        result["pairwise_divergence_per_particle_plateau"] = float(np.mean(pairwise))
    return result


def landed_d_sat_over_n(gardner_r0_json: str | Path) -> tuple[float, int]:
    """Read the landed ``(D_sat/N, N)`` from a ``gardner_r0.json`` file."""

    data = json.loads(Path(gardner_r0_json).read_text(encoding="utf-8"))
    d_sat = float(data["pooled"]["D_sat"]["mean"])
    ensembles = data.get("ensembles") or []
    n = int(ensembles[0]["N"]) if ensembles else int(data["by_N"] and next(iter(data["by_N"])))
    return d_sat / n, n


def analyze_dw_identity(
    config_dirs: Iterable[str | Path],
    gardner_r0_json: str | Path,
    *,
    plateau_frac: float = 0.5,
    ddof: int = 1,
    tol: float = 0.10,
) -> dict[str, Any]:
    """Full DW-identity analysis over one or more persisted config directories."""

    per_config = [
        measure_u_dw_for_run(cdir, plateau_frac=plateau_frac, ddof=ddof)
        for cdir in config_dirs
    ]
    if not per_config:
        raise ValueError("no config directories supplied")
    u2_mean = float(np.mean([r["u2_cage_plateau"] for r in per_config]))
    u_dw = u_dw_from_mean_squared(u2_mean)
    pairwise_vals = [
        r["pairwise_divergence_per_particle_plateau"]
        for r in per_config
        if "pairwise_divergence_per_particle_plateau" in r
    ]
    d_sat_over_n, n = landed_d_sat_over_n(gardner_r0_json)
    identity = dw_identity(u_dw, d_sat_over_n, tol=tol)
    return {
        "schema_version": 1,
        "n_configs": len(per_config),
        "N": n,
        "u2_cage_plateau_mean": u2_mean,
        "u_DW": u_dw,
        "landed_D_sat_over_N": d_sat_over_n,
        "pairwise_divergence_per_particle": (
            float(np.mean(pairwise_vals)) if pairwise_vals else None
        ),
        "identity": identity,
        "per_config": per_config,
    }


def _default_config_dirs(run_dir: Path) -> list[Path]:
    """Auto-discover ``<run>--c*-unpert`` dirs that carry loadable branches."""

    parent = run_dir.parent
    stem = run_dir.name
    dirs: list[Path] = []
    for cand in sorted(parent.glob(f"{stem}--c*-unpert")):
        if (cand / "parent_state.pt").is_file() and any((cand / "branches").glob("*")):
            dirs.append(cand)
    return dirs


def render_markdown(report: dict[str, Any]) -> str:
    ident = report["identity"]
    lines = [
        "# Debye--Waller identity: D_sat/N = c * u_DW",
        "",
        f"- configs analysed: {report['n_configs']} (N = {report['N']})",
        f"- chi_3 constant c = {ident['c']:.6f}",
        f"- measured u_DW (RMS cage amplitude) = {report['u_DW']:.6f} sigma",
        f"- landed D_sat/N = {report['landed_D_sat_over_N']:.6f} sigma",
        f"- predicted c * u_DW = {ident['predicted_D_sat_over_N']:.6f} sigma",
        f"- empirical prefactor D_sat/N / u_DW = {ident['empirical_c']:.4f} (Gaussian: {ident['c']:.4f})",
        f"- ratio measured/predicted = {ident['ratio_measured_over_predicted']:.4f} "
        f"(rel error {ident['rel_error'] * 100:.1f}%, tol {ident['tol'] * 100:.0f}%)",
    ]
    if report.get("pairwise_divergence_per_particle") is not None:
        lines.append(
            f"- cross-check: intrinsic pairwise cage divergence/particle = "
            f"{report['pairwise_divergence_per_particle']:.6f} sigma "
            f"(vs landed {report['landed_D_sat_over_N']:.6f})"
        )
    verdict = "HOLDS" if ident["holds"] else "MISSES (third length scale)"
    lines += ["", f"**Identity {verdict}** within {ident['tol'] * 100:.0f}%."]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/gardner/gardner-T0075-fss"),
        help="Gardner FSS run dir holding gardner_r0.json (default: T0075-fss)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        action="append",
        default=None,
        help="explicit '--c*-unpert' branch directory (repeatable); "
        "default: auto-discover next to --run-dir",
    )
    parser.add_argument(
        "--gardner-json",
        type=Path,
        default=None,
        help="path to gardner_r0.json (default: <run-dir>/gardner_r0.json)",
    )
    parser.add_argument("--plateau-frac", type=float, default=0.5)
    parser.add_argument("--ddof", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("runs/dw_identity/dw_identity.json"))
    parser.add_argument("--markdown", type=Path, default=Path("runs/dw_identity/dw_identity.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gardner_json = args.gardner_json or (args.run_dir / "gardner_r0.json")
    config_dirs = args.config_dir or _default_config_dirs(args.run_dir)
    if not config_dirs:
        print(f"no unperturbed branch directories found for {args.run_dir}", file=sys.stderr)
        return 2

    report = analyze_dw_identity(
        config_dirs,
        gardner_json,
        plateau_frac=args.plateau_frac,
        ddof=args.ddof,
        tol=args.tol,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
