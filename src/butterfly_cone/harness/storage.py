"""Small filesystem primitives shared by the append-only harness modules."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

try:  # ButterflyCone runs on POSIX systems; retain an import-safe fallback for tooling.
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]


def project_root(root: Path | str | None = None) -> Path:
    """Return an explicit root, BUTTERFLY_CONE_ROOT, or this source tree's repository root."""
    if root is not None:
        # Preserve the caller's spelling (notably /var versus /private/var on
        # macOS) so returned run paths compare equal to explicit input paths.
        return Path(root).expanduser()
    configured = os.environ.get("BUTTERFLY_CONE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@contextmanager
def advisory_lock(lock_path: Path) -> Iterator[None]:
    """Hold a process-wide advisory lock for one short critical section."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one fsynced JSONL record using O_APPEND for concurrent writers."""
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):  # pragma: no cover - unusual partial-write guard
            raise OSError("could not append complete JSONL record")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_file(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_new_bytes(destination: Path, payload: bytes) -> Path:
    """Publish a new file atomically, refusing to replace an existing file.

    ``link`` is the no-replace publication operation: unlike ``rename``, it
    fails atomically when another process has already published the result.
    """
    temporary = _temporary_file(destination, payload)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def replace_bytes(destination: Path, payload: bytes) -> Path:
    """Atomically replace internal control metadata after fully writing it."""
    temporary = _temporary_file(destination, payload)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
