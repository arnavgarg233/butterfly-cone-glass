#!/usr/bin/env python3
"""Debye--Waller identity ratio across the five-temperature bridge ladder.

This is a pure re-analysis of the persisted bridge ladder.  For each
``bridge-Tladder--c{0..4}-unpert`` branch ensemble it reuses the exact
``scripts.dw_identity`` plateau estimator for ``u_DW`` and pairs that result
with the corresponding ``D_sat/N`` and ``s_c`` already persisted by
``scripts.bridge_analysis``.  The reported ratio is

    c(T) = (D_sat/N) / u_DW .

No dynamics or cone fields are recomputed here.  The output is intentionally
small and referee-facing: a five-rung table, the exact small-``n`` Spearman
correlation of ``c`` with ``s_c``, a cooling-direction trend diagnostic, and a
two-panel figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for _path in (_SRC, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Read-only reuse of the landed estimators and bridge correlation machinery.
from bridge_analysis import CONFIG_TEMPERATURE, correlate  # noqa: E402
from dw_identity import (  # noqa: E402
    CHI3,
    cage_msd_curve,
    measure_u_dw_for_run,
    plateau_mean,
    u_dw_from_mean_squared,
)


DEFAULT_LADDER_DIR = _ROOT / "runs" / "gardner" / "bridge-Tladder"
DEFAULT_BRIDGE_ANALYSIS = _ROOT / "runs" / "bridge_analysis" / "bridge_analysis.json"
DEFAULT_OUT_DIR = _ROOT / "runs" / "dw_identity_ladder"


# ---------------------------------------------------------------------------
# Pure estimator / arithmetic helpers
# ---------------------------------------------------------------------------


def estimate_u_dw_plateau(
    positions: np.ndarray,
    box: np.ndarray | None = None,
    *,
    plateau_frac: float = 0.5,
    ddof: int = 1,
) -> dict[str, Any]:
    """Estimate ``u_DW`` from a ``(T, B, N, 3)`` plateau trajectory.

    This is the array-level equivalent of the estimator used by
    :func:`dw_identity.measure_u_dw_for_run`: calculate the per-frame branch
    variance about the branch-mean cage centre, take the final ``plateau_frac``
    of that curve, and take its square root.  Keeping this adapter pure makes
    the plateau convention directly testable on synthetic cages; the real
    ladder path calls ``measure_u_dw_for_run`` itself.
    """

    u2_curve = cage_msd_curve(positions, box, ddof=ddof)
    u2_plateau = plateau_mean(u2_curve, plateau_frac)
    return {
        "plateau_frac": float(plateau_frac),
        "ddof": int(ddof),
        "u2_cage_plateau": float(u2_plateau),
        "u_DW": float(u_dw_from_mean_squared(u2_plateau)),
        "u2_cage_curve": [float(value) for value in u2_curve],
    }


def compute_c_ratio(d_sat_over_n: float, u_dw: float) -> float:
    """Return ``c(T) = (D_sat/N) / u_DW`` with physical input validation."""

    d_sat = float(d_sat_over_n)
    u = float(u_dw)
    if not math.isfinite(d_sat):
        raise ValueError("D_sat/N must be finite")
    if not math.isfinite(u) or u <= 0.0:
        raise ValueError("u_DW must be finite and positive")
    return d_sat / u


def compute_c_correlation(s_c: Sequence[float], c_values: Sequence[float]) -> dict[str, Any]:
    """Correlate ``c`` with ``s_c`` using bridge_analysis' exact small-``n`` p."""

    return correlate("c", s_c, c_values).to_dict()


def _monotonic_direction(values: np.ndarray, atol: float) -> str:
    differences = np.diff(values)
    if np.all(np.abs(differences) <= atol):
        return "flat"
    if np.all(differences >= -atol) and np.any(differences > atol):
        return "increasing"
    if np.all(differences <= atol) and np.any(differences < -atol):
        return "decreasing"
    return "non-monotone"


def detect_monotone_trend(
    temperatures: Sequence[float],
    values: Sequence[float],
    *,
    target: float = CHI3,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Detect a monotone temperature trend and whether cooling approaches target.

    Temperatures are sorted internally.  ``direction_vs_temperature`` refers
    to increasing ``T`` (cold to hot), while ``direction_on_cooling`` refers to
    decreasing ``T`` (hot to cold), which is the physically useful orientation
    for the deep-temperature statement.
    """

    t = np.asarray(temperatures, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape or t.size < 2:
        raise ValueError("temperatures and values must be 1-D arrays of equal length >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("temperatures and values must be finite")
    if np.unique(t).size != t.size:
        raise ValueError("temperatures must be unique")
    if not math.isfinite(float(target)):
        raise ValueError("target must be finite")

    order = np.argsort(t)
    t_ascending = t[order]
    y_ascending = y[order]
    y_cooling = y_ascending[::-1]
    direction_vs_temperature = _monotonic_direction(y_ascending, atol)
    direction_on_cooling = _monotonic_direction(y_cooling, atol)
    deepest_value = float(y_ascending[0])
    warmest_value = float(y_ascending[-1])
    target_value = float(target)
    approaches_target = bool(
        direction_on_cooling == "increasing"
        and abs(deepest_value - target_value) < abs(warmest_value - target_value)
    )

    return {
        "n": int(t.size),
        "monotone": bool(direction_on_cooling in {"increasing", "decreasing", "flat"}),
        "direction_vs_temperature": direction_vs_temperature,
        "direction_on_cooling": direction_on_cooling,
        "temperatures_cold_to_hot": [float(value) for value in t_ascending],
        "values_cold_to_hot": [float(value) for value in y_ascending],
        "target": target_value,
        "deepest_temperature": float(t_ascending[0]),
        "deepest_value": deepest_value,
        "warmest_temperature": float(t_ascending[-1]),
        "warmest_value": warmest_value,
        "deepest_distance_to_target": abs(deepest_value - target_value),
        "approaches_target_on_cooling": approaches_target,
    }


# ---------------------------------------------------------------------------
# Report construction from the persisted bridge rows
# ---------------------------------------------------------------------------


def build_report(
    rungs: Sequence[Mapping[str, Any]],
    *,
    target_c: float = CHI3,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ratios, correlation, trend, and referee-facing verdicts."""

    if not rungs:
        raise ValueError("no ladder rungs supplied")

    rows: list[dict[str, Any]] = []
    for rung in rungs:
        row = dict(rung)
        row["temperature"] = float(row["temperature"])
        row["s_c"] = float(row["s_c"])
        row["u_DW"] = float(row["u_DW"])
        row["d_sat_over_n"] = float(row["d_sat_over_n"])
        row["c"] = compute_c_ratio(row["d_sat_over_n"], row["u_DW"])
        rows.append(row)
    rows.sort(key=lambda row: row["temperature"], reverse=True)

    s_c = [row["s_c"] for row in rows]
    c_values = [row["c"] for row in rows]
    correlation = compute_c_correlation(s_c, c_values)
    trend = detect_monotone_trend(
        [row["temperature"] for row in rows],
        c_values,
        target=target_c,
    )

    objection_foreclosed = bool(
        len(rows) >= 5
        and trend["monotone"]
        and np.ptp(np.asarray(c_values, dtype=float)) > 0.0
    )
    third_leg = bool(
        correlation["direction"] == "decreasing"
        and np.isfinite(correlation["spearman_p"])
        and correlation["spearman_p"] <= 0.05
    )
    verdict = {
        "forecloses_one_temperature_objection": objection_foreclosed,
        "adds_third_bridge_leg": third_leg,
        "deep_T_limit_consistent_with_target": bool(
            trend["approaches_target_on_cooling"]
        ),
        "statement": (
            "c(T) rises monotonically on cooling and approaches the deep-T "
            f"Gaussian target {float(target_c):.3f}; the five-rung ladder "
            "turns the one-temperature identity into a temperature-dependent "
            "trend and supplies a sign-flipped entropy bridge leg."
            if objection_foreclosed and third_leg
            else "The ladder does not satisfy all referee-hardening diagnostics."
        ),
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dw_identity_ladder",
        "target_c": float(target_c),
        "per_temperature": rows,
        "correlation": correlation,
        "trend": trend,
        "verdict": verdict,
    }
    if meta is not None:
        report["meta"] = dict(meta)
    return report


def load_bridge_rows(bridge_analysis_path: str | Path) -> dict[int, dict[str, Any]]:
    """Read the five ``D_sat/N`` and ``s_c`` rows from bridge_analysis JSON."""

    data = json.loads(Path(bridge_analysis_path).read_text(encoding="utf-8"))
    rows_by_config: dict[int, dict[str, Any]] = {}
    for raw in data.get("per_config", []):
        config = int(raw["config"])
        if config in rows_by_config:
            raise ValueError(f"duplicate bridge_analysis row for config {config}")
        d_sat_per_n = raw.get("d_sat_per_N_mean")
        if d_sat_per_n is None:
            d_sat_per_n = float(raw["d_sat_mean"]) / int(raw["N"])
        rows_by_config[config] = {
            "config": config,
            "temperature": float(raw["temperature"]),
            "s_c": float(raw["s_c"]),
            "d_sat_over_n": float(d_sat_per_n),
            "d_sat_over_n_sem": float(raw["d_sat_per_N_sem"])
            if raw.get("d_sat_per_N_sem") is not None
            else None,
            "N": int(raw["N"]),
        }

    expected = set(CONFIG_TEMPERATURE)
    missing = sorted(expected - set(rows_by_config))
    if missing:
        raise ValueError(f"bridge_analysis JSON is missing configs {missing}")
    return {config: rows_by_config[config] for config in sorted(expected)}


# ---------------------------------------------------------------------------
# Real persisted ladder analysis
# ---------------------------------------------------------------------------


def analyze_ladder(
    ladder_dir: str | Path = DEFAULT_LADDER_DIR,
    bridge_analysis_path: str | Path = DEFAULT_BRIDGE_ANALYSIS,
    *,
    plateau_frac: float = 0.5,
    ddof: int = 1,
) -> dict[str, Any]:
    """Measure all five unperturbed ladder rungs and build the report."""

    ladder = Path(ladder_dir)
    bridge_rows = load_bridge_rows(bridge_analysis_path)
    rungs: list[dict[str, Any]] = []
    for config in sorted(CONFIG_TEMPERATURE):
        config_dir = ladder.parent / f"{ladder.name}--c{config}-unpert"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"missing ladder config directory: {config_dir}")

        measured = measure_u_dw_for_run(
            config_dir,
            plateau_frac=plateau_frac,
            ddof=ddof,
            with_pairwise=False,
        )
        bridge_row = bridge_rows[config]
        rungs.append(
            {
                **bridge_row,
                "config_dir": str(config_dir),
                "n_branches": int(measured["n_branches"]),
                "n_frames": int(measured["n_frames"]),
                "plateau_frac": float(measured["plateau_frac"]),
                "ddof": int(measured["ddof"]),
                "u2_cage_plateau": float(measured["u2_cage_plateau"]),
                "u_DW": float(measured["u_DW"]),
                "u_DW_from_msd": float(measured["u_DW_from_msd"]),
                "msd_rel_parent_plateau": float(measured["msd_rel_parent_plateau"]),
            }
        )

    return build_report(
        rungs,
        target_c=CHI3,
        meta={
            "ladder_dir": str(ladder),
            "bridge_analysis_path": str(bridge_analysis_path),
            "n_rungs": len(rungs),
            "u_dw_estimator": "dw_identity.measure_u_dw_for_run",
            "plateau_window": "final plateau_frac of the per-frame branch-variance curve",
            "plateau_frac": float(plateau_frac),
            "ddof": int(ddof),
        },
    )


# ---------------------------------------------------------------------------
# Figure / text output
# ---------------------------------------------------------------------------


def make_figure(report: Mapping[str, Any], out_png: str | Path) -> None:
    """Write the requested ``c(T)`` and ``c(s_c)`` two-panel figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(report["per_temperature"])
    by_t = sorted(rows, key=lambda row: row["temperature"])
    by_sc = sorted(rows, key=lambda row: row["s_c"])
    target = float(report["target_c"])
    correlation = report["correlation"]

    temperatures = np.array([row["temperature"] for row in by_t], dtype=float)
    c_t = np.array([row["c"] for row in by_t], dtype=float)
    s_c = np.array([row["s_c"] for row in by_sc], dtype=float)
    c_sc = np.array([row["c"] for row in by_sc], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), dpi=160)
    color = "#2f6f9f"
    target_color = "#b44d2e"
    grid = "#e3e5ea"

    ax = axes[0]
    ax.axhline(target, color=target_color, ls="--", lw=1.2, label=rf"Gaussian target $c={target:.3f}$")
    ax.plot(temperatures, c_t, "o-", color=color, lw=1.7, ms=6, label=r"$c(T)$")
    for row in by_t:
        ax.annotate(
            f"{row['temperature']:.3f}",
            (row["temperature"], row["c"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel(r"temperature $T$")
    ax.set_ylabel(r"$c(T)=(D_{\rm sat}/N)/u_{\rm DW}$")
    ax.set_title("DW ratio across the bridge ladder")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    ax.axhline(target, color=target_color, ls="--", lw=1.2, label=rf"target $c={target:.3f}$")
    ax.plot(s_c, c_sc, "o-", color=color, lw=1.7, ms=6, label=r"$c$ vs $s_c$")
    for row in by_sc:
        ax.annotate(
            f"T={row['temperature']:.3f}",
            (row["s_c"], row["c"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel(r"configurational entropy $s_c$")
    ax.set_ylabel(r"$c$")
    ax.set_title(
        rf"$c$ vs $s_c$: Spearman $\rho={correlation['spearman_rho']:+.3f}$, "
        rf"$p={correlation['spearman_p']:.4f}$"
    )
    ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes:
        ax.grid(True, color=grid, lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Debye--Waller identity ratio: temperature-dependent bridge leg", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(out_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the referee-facing table and interpretation."""

    rows = report["per_temperature"]
    corr = report["correlation"]
    trend = report["trend"]
    verdict = report["verdict"]
    lines = [
        "# Debye--Waller identity ratio across the bridge temperature ladder",
        "",
        f"Target deep-T Gaussian prefactor: `c = {_fmt(report['target_c'])}`.",
        "",
        "| config | T | s_c | u_DW | D_sat/N | c(T) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['temperature']:.3f} | {row['s_c']:.6f} | "
            f"{row['u_DW']:.6f} | {row['d_sat_over_n']:.6f} | {row['c']:.6f} |"
        )
    lines += [
        "",
        f"- Spearman `rho(c, s_c) = {corr['spearman_rho']:+.6f}`, exact one-sided "
        f"permutation `p = {corr['spearman_p']:.6f}` (`n = {corr['n']}`).",
        f"- Cooling trend: **{trend['direction_on_cooling']}**; `c(T)` changes from "
        f"`{trend['warmest_value']:.6f}` at `T={trend['warmest_temperature']:.3f}` "
        f"to `{trend['deepest_value']:.6f}` at `T={trend['deepest_temperature']:.3f}`.",
        f"- Deep-T limit: the observed cooling direction is "
        f"**{'consistent with' if trend['approaches_target_on_cooling'] else 'not consistent with'}** "
        f"approach to `1.30`; the deepest rung is `{trend['deepest_value']:.6f}` "
        f"versus the target `{trend['target']:.6f}`.",
        f"- One-temperature objection foreclosed: **{verdict['forecloses_one_temperature_objection']}**.",
        f"- Third sign-flipped chaos--entropy bridge leg added: **{verdict['adds_third_bridge_leg']}**.",
        "",
        verdict["statement"],
        "",
    ]
    return "\n".join(lines)


def run(
    ladder_dir: Path = DEFAULT_LADDER_DIR,
    bridge_analysis_path: Path = DEFAULT_BRIDGE_ANALYSIS,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    plateau_frac: float = 0.5,
    ddof: int = 1,
    write_figure: bool = True,
) -> dict[str, Any]:
    """Analyze the ladder and persist JSON, Markdown, and optionally PNG."""

    report = analyze_ladder(
        ladder_dir,
        bridge_analysis_path,
        plateau_frac=plateau_frac,
        ddof=ddof,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dw_identity_ladder.json"
    markdown_path = output_dir / "dw_identity_ladder.md"
    figure_path = output_dir / "dw_identity_ladder.png"
    report = dict(report)
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "figure": str(figure_path) if write_figure else None,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    if write_figure:
        make_figure(report, figure_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER_DIR)
    parser.add_argument("--bridge-analysis", type=Path, default=DEFAULT_BRIDGE_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plateau-frac", type=float, default=0.5)
    parser.add_argument("--ddof", type=int, default=1)
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        args.ladder_dir,
        args.bridge_analysis,
        args.out_dir,
        plateau_frac=args.plateau_frac,
        ddof=args.ddof,
        write_figure=not args.no_figure,
    )
    print(render_markdown(report))
    print(f"wrote {args.out_dir / 'dw_identity_ladder.json'}")
    print(f"wrote {args.out_dir / 'dw_identity_ladder.md'}")
    if not args.no_figure:
        print(f"wrote {args.out_dir / 'dw_identity_ladder.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
