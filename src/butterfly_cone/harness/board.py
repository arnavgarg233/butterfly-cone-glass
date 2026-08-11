"""Plain-text status view over published ButterflyCone run manifests and metrics."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .storage import project_root


def _duration(manifest: dict[str, Any]) -> str:
    start, end = manifest.get("start_time"), manifest.get("end_time")
    if not start:
        return "-"
    if not end:
        return "running"
    try:
        started = datetime.fromisoformat(start.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "?"
    seconds = max(0, int((finished - started).total_seconds()))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _metrics(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "-"
    if not isinstance(data, dict):
        return "-"
    items = [
        f"{key}={value}"
        for key, value in sorted(data.items())
        if isinstance(value, (str, int, float, bool)) and not isinstance(value, complex)
    ]
    return ", ".join(items[:4]) or "-"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    format_row = lambda row: "  ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    )
    return "\n".join((format_row(headers), format_row(["-" * width for width in widths]), *(format_row(row) for row in rows)))


def render_status_board(*, root: Path | str | None = None) -> str:
    """Return a compact table without mutating the run archive."""
    runs = project_root(root) / "runs"
    rows: list[list[str]] = []
    for manifest_path in sorted(runs.glob("*/*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        rows.append(
            [
                str(manifest.get("phase", manifest_path.parent.parent.name)),
                str(manifest.get("run_id", manifest_path.parent.name)),
                str(manifest.get("status", "unknown")),
                str(manifest.get("start_time", "-")),
                _duration(manifest),
                _metrics(manifest_path.parent / "metrics.json"),
            ]
        )
    if not rows:
        return "No runs found."
    return _table(["PHASE", "RUN ID", "STATUS", "STARTED", "DURATION", "METRICS"], rows)
