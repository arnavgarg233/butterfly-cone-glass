"""The project-level, append-only advance-declaration decision ledger.

Records written by :class:`DecisionLedger` form a SHA-256 hash chain.  The
module-level helpers also operate on any JSONL ledger path, which makes the
one-time migration and external anchoring functions useful for existing
ledger files without changing their callers' storage layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import canonical_json
from .storage import advisory_lock, project_root, replace_bytes, utc_timestamp


GENESIS_HASH = "0" * 64
_DECLARED_KINDS = frozenset({"decision", "prediction"})


class LedgerChainError(RuntimeError):
    """Raised when a ledger cannot be safely extended or migrated."""

    def __init__(self, path: Path | str, index: int | None) -> None:
        self.path = Path(path)
        self.first_broken_index = index
        location = "unknown" if index is None else str(index)
        super().__init__(f"ledger chain verification failed at index {location}: {self.path}")


class DecisionRecordedError(RuntimeError):
    """Raised when code attempts to revise an already recorded decision."""


class _LedgerFormatError(ValueError):
    def __init__(self, index: int, message: str) -> None:
        self.index = index
        super().__init__(message)


def _lock_path(path: Path) -> Path:
    return path.parent / f".{path.stem}.lock"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            raise _LedgerFormatError(index, "blank JSONL record")
        try:
            record = json.loads(line)
        except (TypeError, ValueError) as error:
            raise _LedgerFormatError(index, "invalid JSONL record") from error
        if not isinstance(record, dict):
            raise _LedgerFormatError(index, "JSONL record must be an object")
        records.append(record)
    return records


def _record_hash(record: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in record.items() if key != "record_hash"}
    return hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()


def _with_hashes(record: dict[str, Any], prev_hash: str) -> dict[str, Any]:
    hashed = dict(record)
    hashed["prev_hash"] = prev_hash
    hashed.pop("record_hash", None)
    hashed["record_hash"] = _record_hash(hashed)
    return hashed


def _verify_records(records: list[dict[str, Any]]) -> tuple[bool, int | None]:
    previous_hash = GENESIS_HASH
    for index, record in enumerate(records):
        if "prev_hash" not in record or "record_hash" not in record:
            return False, index
        if record["prev_hash"] != previous_hash:
            return False, index
        try:
            expected_hash = _record_hash(record)
        except (TypeError, ValueError):
            return False, index
        if record["record_hash"] != expected_hash:
            return False, index
        previous_hash = record["record_hash"]
    return True, None


def _verified_head(records: list[dict[str, Any]], path: Path) -> str:
    ok, first_broken_index = _verify_records(records)
    if not ok:
        raise LedgerChainError(path, first_broken_index)
    return records[-1]["record_hash"] if records else GENESIS_HASH


def _json_value(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be JSON serializable") from error


def _atomic_append(path: Path, record: dict[str, Any]) -> None:
    existing = path.read_bytes() if path.exists() else b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\n"
    payload = existing + canonical_json(record).encode("utf-8") + b"\n"
    replace_bytes(path, payload)


def verify_chain(path: Path | str) -> tuple[bool, int | None]:
    """Recompute a JSONL hash chain and report its first broken index.

    An empty or missing ledger is a valid empty chain.  Legacy records without
    hash fields return ``(False, 0)`` until ``migrate_chain`` is called.
    """
    ledger_path = Path(path)
    try:
        records = _read_records(ledger_path)
    except _LedgerFormatError as error:
        return False, error.index
    except (OSError, UnicodeError):
        return False, 0
    return _verify_records(records)


def migrate_chain(path: Path | str) -> Path:
    """Back-fill hashes in legacy order, creating a new genesis anchor.

    This is an explicit one-time genesis re-anchor, not a claim that the
    unhashed history was tamper-evident before migration.  A valid hashed
    chain is left unchanged; a partially hashed or broken chain is refused.
    """
    ledger_path = Path(path)
    with advisory_lock(_lock_path(ledger_path)):
        records = _read_records(ledger_path)
        if not records:
            return ledger_path

        hashed = [
            "prev_hash" in record and "record_hash" in record for record in records
        ]
        if all(hashed):
            _verified_head(records, ledger_path)
            return ledger_path
        if any(hashed):
            raise LedgerChainError(ledger_path, hashed.index(False))

        migrated: list[dict[str, Any]] = []
        previous_hash = GENESIS_HASH
        for record in records:
            migrated_record = _with_hashes(record, previous_hash)
            migrated.append(migrated_record)
            previous_hash = migrated_record["record_hash"]
        payload = b"".join(
            canonical_json(record).encode("utf-8") + b"\n" for record in migrated
        )
        replace_bytes(ledger_path, payload)
    return ledger_path


def anchor_head(
    path: Path | str,
    timestamp: str | None = None,
    *,
    timestamp_field: str | None = None,
) -> dict[str, Any]:
    """Return a publishable head anchor using a caller-supplied timestamp.

    ``timestamp`` is accepted positionally or by keyword.  The
    ``timestamp_field`` keyword is an equivalent descriptive alias for
    callers that name arguments after the returned field.  No clock is read
    here, so identical inputs produce identical anchors.
    """
    if timestamp is not None and timestamp_field is not None:
        raise TypeError("supply timestamp or timestamp_field, not both")
    supplied_timestamp = timestamp if timestamp is not None else timestamp_field
    if not isinstance(supplied_timestamp, str) or not supplied_timestamp:
        raise ValueError("a non-empty caller-supplied timestamp is required")

    ledger_path = Path(path)
    records = _read_records(ledger_path)
    head_hash = _verified_head(records, ledger_path)
    return {
        "head_hash": head_hash,
        "index": len(records) - 1,
        "timestamp_field": supplied_timestamp,
    }


class DecisionLedger:
    def __init__(self, *, root: Path | str | None = None) -> None:
        self.root = project_root(root)

    @property
    def path(self) -> Path:
        return self.root / "runs" / "decision_ledger.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.root / "runs" / ".decision_ledger.lock"

    def record_decision(
        self,
        key: str,
        value: Any,
        rationale: str,
        *,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Record a frozen decision once, atomically relative to other writers."""
        if not isinstance(key, str) or not key:
            raise ValueError("decision key must be a non-empty string")
        if not isinstance(rationale, str) or not rationale:
            raise ValueError("decision rationale must be a non-empty string")
        canonical_value = _json_value(value, label="decision value")
        if timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("decision timestamp must be a string")

        with advisory_lock(self.lock_path):
            records = _read_records(self.path)
            keys = {record["key"] for record in records if "key" in record}
            if key in keys:
                raise DecisionRecordedError(f"decision {key!r} is already frozen")
            record = {
                "key": key,
                "value": canonical_value,
                "rationale": rationale,
                "timestamp": utc_timestamp() if timestamp is None else timestamp,
            }
            previous_hash = _verified_head(records, self.path)
            record = _with_hashes(record, previous_hash)
            _atomic_append(self.path, record)
        return record

    def freeze(self, payload: Any, *, kind: str) -> str:
        """Append a deterministic frozen decision or prospective prediction.

        Any timestamp needed for reproducibility belongs in ``payload`` and is
        supplied by the caller; this method never reads the clock.
        """
        if not isinstance(kind, str) or kind not in _DECLARED_KINDS:
            raise ValueError("kind must be one of {'decision', 'prediction'}")
        canonical_payload = _json_value(payload, label="frozen payload")

        with advisory_lock(self.lock_path):
            records = _read_records(self.path)
            previous_hash = _verified_head(records, self.path)
            record = _with_hashes(
                {"kind": kind, "payload": canonical_payload}, previous_hash
            )
            _atomic_append(self.path, record)
        return record["record_hash"]
