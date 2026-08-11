"""Run-directory lifecycle management for the declared in advance ButterflyCone phases."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable
from uuid import uuid4

from .config import ExperimentConfig
from .seeds import SeedAllocator
from .storage import (
    advisory_lock,
    append_jsonl,
    project_root,
    replace_bytes,
    utc_timestamp,
    write_new_bytes,
)


class RunExistsError(FileExistsError):
    """Raised when a requested run ID already has a directory."""


class ResultExistsError(FileExistsError):
    """Raised when a run attempts to replace a published result."""


class RunFinalizedError(RuntimeError):
    """Raised when code attempts to finalise an already finalised run."""


def atomic_write(path: Path | str, payload: bytes | str) -> Path:
    """Atomically publish a new result file without replacing an old one."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not isinstance(data, bytes):
        raise TypeError("atomic_write payload must be bytes or text")
    return write_new_bytes(Path(path), data)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:10]}"


class RunManager:
    """Own one ``runs/<phase>/<run_id>`` directory and its public records."""

    _RESERVED_NAMES = {"config.yaml", "manifest.json", "log.txt"}

    def __init__(
        self,
        *,
        root: Path,
        phase: str,
        run_id: str,
        project_salt: str,
    ) -> None:
        self.root = root
        self.phase = phase
        self.run_id = run_id
        self.project_salt = project_salt

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        config: ExperimentConfig,
        root: Path | str | None = None,
        run_id: str | None = None,
        device: str | None = None,
        project_salt: str = "butterfly_cone",
    ) -> "RunManager":
        if not isinstance(phase, str) or not phase or Path(phase).name != phase:
            raise ValueError("phase must be one simple directory name")
        if config.phase != phase:
            raise ValueError("config.phase must match the run phase")
        resolved_root = project_root(root)
        identifier = run_id or _new_run_id()
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier:
            raise ValueError("run_id must be one simple directory name")
        run_path = resolved_root / "runs" / phase / identifier
        try:
            run_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunExistsError(f"run directory already exists: {run_path}") from error

        config.freeze()
        config.dump_yaml(run_path / "config.yaml")
        manifest = {
            "run_id": identifier,
            "phase": phase,
            "status": "running",
            "start_time": utc_timestamp(),
            "end_time": None,
            "config_hash": config.frozen_hash,
            "seeds_used": [],
            "git_commit": _git_commit(resolved_root),
            "host": socket.gethostname(),
            "device": device if device is not None else os.environ.get("BUTTERFLY_CONE_DEVICE", "unknown"),
            "interpreter": {"executable": sys.executable, "version": sys.version},
        }
        write_new_bytes(
            run_path / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
        )
        write_new_bytes(run_path / "log.txt", b"")
        return cls(
            root=resolved_root,
            phase=phase,
            run_id=identifier,
            project_salt=project_salt,
        )

    @property
    def path(self) -> Path:
        return self.root / "runs" / self.phase / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def _manifest_lock_path(self) -> Path:
        return self.path / ".manifest.lock"

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _update_manifest(
        self, update: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        with advisory_lock(self._manifest_lock_path):
            manifest = self._read_manifest()
            update(manifest)
            replace_bytes(
                self.manifest_path,
                (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
            )
        return manifest

    def seed_for(self, domain: str, index: int) -> int:
        """Allocate a seed, log it project-wide, and include it in this manifest."""
        allocator = SeedAllocator(
            project_salt=self.project_salt, root=self.root, run_id=self.run_id
        )
        allocation = allocator.allocate(domain, index)

        def add_seed(manifest: dict[str, Any]) -> None:
            manifest["seeds_used"].append(allocation.to_dict())

        self._update_manifest(add_seed)
        return allocation.seed

    def _result_path(self, name: str | Path) -> Path:
        relative = Path(name)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.name in self._RESERVED_NAMES
        ):
            raise ValueError("result name must be a non-reserved relative path")
        return self.path / relative

    def write_bytes(self, name: str | Path, payload: bytes) -> Path:
        destination = self._result_path(name)
        try:
            return atomic_write(destination, payload)
        except FileExistsError as error:
            raise ResultExistsError(f"result already exists: {destination}") from error

    def write_text(self, name: str | Path, text: str) -> Path:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_json(self, name: str | Path, value: Any) -> Path:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        return self.write_text(name, text + "\n")

    def log(self, message: str) -> None:
        if not isinstance(message, str):
            raise TypeError("log message must be a string")
        append_jsonl(self.path / "log.txt", {"timestamp": utc_timestamp(), "message": message})

    def finish(self, status: str = "completed") -> dict[str, Any]:
        if status not in {"completed", "failed", "stopped"}:
            raise ValueError("final status must be completed, failed, or stopped")

        def finalise(manifest: dict[str, Any]) -> None:
            if manifest.get("end_time") is not None:
                raise RunFinalizedError(f"run {self.run_id!r} is already finalised")
            manifest["status"] = status
            manifest["end_time"] = utc_timestamp()

        return self._update_manifest(finalise)
