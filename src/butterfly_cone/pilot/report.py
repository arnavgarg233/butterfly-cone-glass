"""Aggregation, figures, and Markdown reporting for bulk-pilot run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from .dynamics import TauAlphaResult, extract_tau_alpha


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    metrics: dict[str, Any]
    curves: dict[str, np.ndarray] | None


def load_artifact(path: Path | str) -> RunArtifact:
    """Load one completed harness run and its optional numerical curve sidecar."""

    run_path = Path(path)
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    curve_path = run_path / "curves.npz"
    if not curve_path.exists():
        return RunArtifact(run_path, metrics, None)
    with np.load(curve_path, allow_pickle=False) as data:
        curves = {key: np.asarray(data[key]) for key in data.files}
    return RunArtifact(run_path, metrics, curves)


def _finite(values: Iterable[float | None]) -> np.ndarray:
    values_array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return values_array[np.isfinite(values_array)]


def _temperature_key(temperature: float) -> str:
    return f"{temperature:.3f}"


def _record_role(artifact: RunArtifact, role: str) -> bool:
    return str(artifact.metrics.get("role", "")) == role


def _curve_aggregate(records: list[RunArtifact]) -> dict[str, Any] | None:
    dynamic = [record for record in records if record.curves is not None and record.metrics.get("production_performed")]
    if not dynamic:
        return None
    first = dynamic[0].curves
    assert first is not None
    lag_times = first["lag_times"]
    for record in dynamic[1:]:
        assert record.curves is not None
        if not np.array_equal(record.curves["lag_times"], lag_times):
            raise ValueError("cannot aggregate bulk-pilot runs with different lag grids")
    fs = np.mean(np.stack([record.curves["fs"] for record in dynamic if record.curves is not None]), axis=0)
    msd = np.mean(np.stack([record.curves["msd"] for record in dynamic if record.curves is not None]), axis=0)
    overlap = np.mean(np.stack([record.curves["overlap"] for record in dynamic if record.curves is not None]), axis=0)
    n_particles = int(dynamic[0].metrics["n_particles"])
    q_samples = []
    for lag_index in range(len(lag_times)):
        values = []
        for record in dynamic:
            assert record.curves is not None
            field = record.curves["q_samples"][lag_index]
            values.append(field[np.isfinite(field)])
        merged = np.concatenate(values) if values else np.empty(0, dtype=float)
        q_samples.append(merged)
    chi4 = np.array(
        [n_particles * np.var(values, ddof=1) if len(values) > 1 else np.nan for values in q_samples],
        dtype=float,
    )
    valid_peak = np.flatnonzero(np.isfinite(chi4))
    if len(valid_peak):
        peak_index = int(valid_peak[np.argmax(chi4[valid_peak])])
        chi4_peak = float(chi4[peak_index])
        chi4_peak_time = float(lag_times[peak_index])
    else:
        chi4_peak = None
        chi4_peak_time = None
    tau = extract_tau_alpha(lag_times, fs)
    return {
        "lag_times": lag_times,
        "fs": fs,
        "msd": msd,
        "overlap": overlap,
        "chi4": chi4,
        "tau": tau,
        "chi4_peak": chi4_peak,
        "chi4_peak_time": chi4_peak_time,
        "n_dynamic_records": len(dynamic),
    }


def aggregate_temperature(records: list[RunArtifact]) -> dict[str, dict[str, Any]]:
    """Aggregate a homogeneous role cohort into one entry per temperature."""

    grouped: dict[str, list[RunArtifact]] = {}
    for record in records:
        temperature = float(record.metrics["temperature"])
        grouped.setdefault(_temperature_key(temperature), []).append(record)
    output: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items(), key=lambda item: float(item[0]), reverse=True):
        equilibrated = [bool(record.metrics["equilibration"]["passed"]) for record in group]
        structural = [bool(record.metrics["structure"]["structural_pass"]) for record in group]
        crystal = [bool(record.metrics["structure"]["crystallization_pass"]) for record in group]
        demix = [bool(record.metrics["structure"]["demixing_pass"]) for record in group]
        low_k = [bool(record.metrics["structure"]["low_k_pass"]) for record in group]
        plateaus = [record.metrics.get("production", {}).get("cage_plateau", {}) for record in group]
        plateau_heights = _finite(entry.get("height") for entry in plateaus)
        plateau_present = [bool(entry.get("present")) for entry in plateaus if entry]
        event_by_horizon: dict[str, np.ndarray] = {}
        for horizon in ("10.0", "50.0", "250.0"):
            event_by_horizon[horizon] = _finite(
                record.metrics.get("production", {}).get("event_proxy", {}).get(horizon)
                for record in group
            )
        q6_std = _finite(record.metrics["structure"].get("q6_std") for record in group)
        q6_means = _finite(record.metrics["structure"].get("q6_mean") for record in group)
        demix_z = _finite(record.metrics["structure"].get("demixing_z_score") for record in group)
        aggregate = _curve_aggregate(group)
        output[key] = {
            "temperature": float(key),
            "records": group,
            "n_records": len(group),
            "equilibration_passed": int(sum(equilibrated)),
            "structural_passed": int(sum(structural)),
            "crystal_passed": int(sum(crystal)),
            "demix_passed": int(sum(demix)),
            "low_k_passed": int(sum(low_k)),
            "plateau_present": int(sum(plateau_present)),
            "plateau_height_median": float(np.median(plateau_heights)) if len(plateau_heights) else None,
            "event_mean": {
                horizon: (float(np.mean(values)) if len(values) else None)
                for horizon, values in event_by_horizon.items()
            },
            "q6_mean_median": float(np.median(q6_means)) if len(q6_means) else None,
            "q6_std_median": float(np.median(q6_std)) if len(q6_std) else None,
            "demixing_abs_z_max": float(np.max(np.abs(demix_z))) if len(demix_z) else None,
            "curves": aggregate,
        }
    return output


def _format_tau(result: TauAlphaResult | None) -> str:
    if result is None:
        return "-"
    if result.crossed and result.value is not None:
        return f"{result.value:.2f}"
    if result.lower_bound is not None:
        return f"> {result.lower_bound:.0f}"
    return "not resolved"


def _format_number(value: float | None, precision: int = 3) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.{precision}g}"


def _yes_no(passed: int, total: int) -> str:
    return f"{passed}/{total} pass"


def _render_temperature_table(aggregates: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, aggregate in aggregates.items():
        curves = aggregate["curves"]
        tau = _format_tau(None if curves is None else curves["tau"])
        if curves is None or curves["chi4_peak"] is None:
            chi4 = "-"
        else:
            chi4 = f"{curves['chi4_peak']:.3g} @ {curves['chi4_peak_time']:.2g}"
        plateau = _format_number(aggregate["plateau_height_median"])
        events = "/".join(_format_number(aggregate["event_mean"][horizon]) for horizon in ("10.0", "50.0", "250.0"))
        rows.append(
            "| "
            + " | ".join(
                (
                    f"{aggregate['temperature']:.3f}",
                    _yes_no(aggregate["equilibration_passed"], aggregate["n_records"]),
                    f"C {_yes_no(aggregate['crystal_passed'], aggregate['n_records'])}; D {_yes_no(aggregate['demix_passed'], aggregate['n_records'])}",
                    tau,
                    chi4,
                    plateau,
                    events,
                )
            )
            + " |"
        )
    header = "| T | Equilibration | Crystal / demix | τ_α | χ₄ peak @ t | Cage plateau | Event proxy H=10/50/250 |"
    divider = "|---:|:---:|:---:|---:|---:|---:|---:|"
    return "\n".join((header, divider, *rows))


def _render_source_table(records: list[RunArtifact]) -> str:
    rows = []
    for record in sorted(
        records,
        key=lambda value: (
            str(value.metrics.get("source_label", "")),
            -float(value.metrics["temperature"]),
            int(value.metrics["replica"]),
        ),
    ):
        metrics = record.metrics
        stationarity = metrics["equilibration"]
        structure = metrics["structure"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(metrics.get("source_label", "")),
                    f"{float(metrics['temperature']):.3f}",
                    str(metrics["replica"]),
                    "PASS" if stationarity["passed"] else "FAIL",
                    f"{stationarity['absolute_drift']:.3g} ≤ {stationarity['threshold']:.3g}",
                    "PASS" if structure["structural_pass"] else "FAIL",
                    str(record.path.name),
                )
            )
            + " |"
        )
    return "\n".join(
        (
            "| Archive | T | Replica | U/N drift | Drift comparison | Structural integrity | Run ID |",
            "|:---|---:|---:|:---:|:---:|:---:|:---|",
            *rows,
        )
    )


def _criterion_summary(primary: dict[str, dict[str, Any]]) -> str:
    fragments = []
    for key, aggregate in primary.items():
        curves = aggregate["curves"]
        tau = _format_tau(None if curves is None else curves["tau"])
        heterogeneity = _format_number(None if curves is None else curves["chi4_peak"])
        events = _format_number(aggregate["event_mean"]["50.0"])
        fragments.append(f"T={key}: τ_α {tau}, χ₄* {heterogeneity}, proxy(H=50) {events}")
    detail = "; ".join(fragments) or "no primary dynamics records"
    return (
        "Criteria 1–2: cage-plateau and χ₄ evidence are shown in the table; the temperature-resolved "
        f"dynamics read {detail}. Criteria 3–4: the table and per-file audit distinguish structural and "
        "stationarity passes from any failures rather than assuming inherited equilibration. Criterion 5: a "
        "missing 1/e crossing is reported as a finite-window lower bound, not extrapolated. Criterion 6: "
        "the H=10/50/250 values are a pre-freeze sustained cage-relative displacement proxy only; they are "
        "not the Phase 2 event definition. Criterion 7: q6 width and the absence/presence of structural "
        "pathologies quantify local structural diversity, while a learned candidate-field distribution does "
        "not yet exist in Phase 1a and remains unresolved. Criterion 8 (cavity feasibility) is intentionally "
        "outside this report, so this evidence does not select a Gate 0 temperature."
    )


def render_bulk_pilot_report(
    artifacts: list[RunArtifact],
    *,
    batch_metadata: dict[str, Any],
) -> str:
    """Render the merged Decision-1 evidence report without making a decision."""

    primary_records = [artifact for artifact in artifacts if _record_role(artifact, "primary")]
    deep_records = [artifact for artifact in artifacts if _record_role(artifact, "deep_revalidation")]
    primary = aggregate_temperature(primary_records)
    deep = aggregate_temperature(deep_records)
    protocol = batch_metadata["protocol"]
    device = batch_metadata["device"]
    mps = batch_metadata["mps_available"]
    return f"""# Bulk temperature pilot: ButterflyCone Phase 1a

## Scope and execution record

This report is evidence for Decision 1 only. It does **not** choose a Gate 0 temperature. The primary dynamics cohort is the four-replica `configs_N1500.npz` ladder at T=0.150, 0.130, 0.108, and 0.090; T=0.075 receives re-equilibration and structural checks only. The independent `configs_deep_N1500.npz` states at T=0.108, 0.090, and 0.075 are revalidation-only checks, avoiding a second non-independent dynamics cohort in the primary τ/χ₄ curves.

- Actual device: `{device}`; MPS available at launch: `{mps}`.
- Wall clock: {batch_metadata['wall_clock_seconds']:.1f} s; total MD steps: {batch_metadata['total_steps']:,}.
- Dynamics: velocity-Verlet with Bussi NVT throughout, `dt={protocol['dt']}`, `τ_thermostat={protocol['thermostat_tau']}`. The 0.5 coupling time (100 steps) is deliberately gentle relative to the integration step while guarding float32 energy drift; therefore production is NVT, not NVE.
- Re-equilibration: {protocol['reequilibration_cycles']} blocks × {protocol['reequilibration_md_steps']} NVT steps, followed by {protocol['swap_attempts_per_cycle']} seeded diameter-swap attempts per block. Physical production has swaps disabled and redrawn, harness-seeded Maxwell–Boltzmann velocities.
- Stationarity gate, specified before inspection: `|mean(U/N)_second − mean(U/N)_first| ≤ max(0.0025, 2·SE_Δ)`. Structural gate: q6 high-tail / distribution comparison to the stored T=0.150 baseline, no **positive** diameter-neighbour correlation above 4 shuffled-null SD (negative unlike-size packing is retained as a diagnostic), and finite low-k S(k) with max < 10.

## Primary temperature table

{_render_temperature_table(primary)}

`τ_α` is the first 1/e crossing of time-origin-averaged F_s(k=7.1,t); `> value` means no crossing in the executed window. `χ₄=N Var(Q̂)` combines valid time origins from all primary replicas. Plateau is the median local minimum-slope MSD value. Event entries are mean fractions at H=10/50/250 and are explicitly pre-freeze diagnostics.

## Per-file revalidation audit

{_render_source_table(primary_records + deep_records)}

## Decision-1 evidence against criteria 1–7

{_criterion_summary(primary)}

## Figures

- `figs/bulk_pilot/fs_family.png`
- `figs/bulk_pilot/chi4_family.png`
- `figs/bulk_pilot/msd_family.png`
- `figs/bulk_pilot/tau_vs_inverse_temperature.png`

The hollow warm markers in the τ plot are the prior-project Langevin values (`dt=0.005`, production γ=0.4), included only for qualitative shape comparison; they are not pooled with the Newtonian/Bussi estimates.
"""


def _atomic_replace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: Path | str, artifacts: list[RunArtifact], *, batch_metadata: dict[str, Any]) -> Path:
    destination = Path(path)
    _atomic_replace(destination, render_bulk_pilot_report(artifacts, batch_metadata=batch_metadata))
    return destination


def write_figures(output_directory: Path | str, artifacts: list[RunArtifact]) -> list[Path]:
    """Write the four requested temperature-family figures with a headless backend."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/butterfly_cone-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/butterfly_cone-xdg-cache")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    primary = aggregate_temperature([artifact for artifact in artifacts if _record_role(artifact, "primary")])
    ordered = list(primary.items())
    color_map = plt.get_cmap("viridis")
    colors = {key: color_map(index / max(1, len(ordered) - 1)) for index, (key, _) in enumerate(ordered)}
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for key, aggregate in ordered:
        curves = aggregate["curves"]
        if curves is not None:
            mask = curves["lag_times"] > 0.0
            axis.semilogx(curves["lag_times"][mask], curves["fs"][mask], label=f"T={key}", color=colors[key])
    axis.axhline(np.exp(-1.0), color="0.4", linewidth=1.0, linestyle="--", label="1/e")
    axis.set(xlabel="time", ylabel="F_s(k=7.1, t)", title="Self-intermediate scattering")
    axis.legend(fontsize=8)
    path = destination / "fs_family.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for key, aggregate in ordered:
        curves = aggregate["curves"]
        if curves is not None:
            mask = (curves["lag_times"] > 0.0) & np.isfinite(curves["chi4"])
            axis.semilogx(curves["lag_times"][mask], curves["chi4"][mask], label=f"T={key}", color=colors[key])
    axis.set(xlabel="time", ylabel="χ₄(t)", title="Time-origin four-point susceptibility")
    axis.legend(fontsize=8)
    path = destination / "chi4_family.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for key, aggregate in ordered:
        curves = aggregate["curves"]
        if curves is not None:
            mask = (curves["lag_times"] > 0.0) & (curves["msd"] > 0.0)
            axis.loglog(curves["lag_times"][mask], curves["msd"][mask], label=f"T={key}", color=colors[key])
    axis.set(xlabel="time", ylabel="MSD", title="Mean-squared displacement")
    axis.legend(fontsize=8)
    path = destination / "msd_family.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    x_values, y_values = [], []
    lower_x, lower_y = [], []
    for key, aggregate in ordered:
        curves = aggregate["curves"]
        if curves is None:
            continue
        tau = curves["tau"]
        if tau.crossed and tau.value is not None:
            x_values.append(1.0 / aggregate["temperature"])
            y_values.append(tau.value)
        elif tau.lower_bound is not None:
            lower_x.append(1.0 / aggregate["temperature"])
            lower_y.append(tau.lower_bound)
    if x_values:
        axis.semilogy(x_values, y_values, "o-", label="Newtonian + Bussi (this pilot)")
    if lower_x:
        axis.semilogy(lower_x, lower_y, "^", fillstyle="none", label="this pilot lower bound")
    old_temperature = np.array([0.50, 0.35, 0.25, 0.19, 0.15])
    old_tau = np.array([0.806, 1.232, 2.000, 5.314, 9.432])
    axis.semilogy(1.0 / old_temperature, old_tau, "o", markerfacecolor="none", color="0.35", label="Langevin, prior project")
    axis.set(xlabel="1 / T", ylabel="τ_α", title="Relaxation-time shape comparison")
    axis.legend(fontsize=8)
    path = destination / "tau_vs_inverse_temperature.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)
    return outputs
