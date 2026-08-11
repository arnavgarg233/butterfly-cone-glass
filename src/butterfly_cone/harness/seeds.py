"""Deterministic, domain-separated RNG allocation with an audit ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from .storage import append_jsonl, project_root, utc_timestamp


def derive_seed(project_salt: str, domain: str, index: int) -> int:
    """Return the SHA-256 integer for one named RNG consumer.

    The project salt and a NUL-delimited domain are part of every digest input;
    callers never allocate from a shared arithmetic range.
    """
    if not isinstance(project_salt, str) or not project_salt:
        raise ValueError("project_salt must be a non-empty string")
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    message = b"\0".join((project_salt.encode(), domain.encode(), str(index).encode()))
    return int.from_bytes(hashlib.sha256(message).digest(), byteorder="big")


@dataclass(frozen=True)
class SeedAllocation:
    seed: int
    domain: str
    index: int
    timestamp: str
    run_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SeedAllocator:
    """Issue domain-hashed seeds and append each issuance to ``seed_ledger``."""

    def __init__(
        self,
        *,
        project_salt: str,
        root: Path | str | None = None,
        run_id: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        # Validate the salt even before the first allocation.
        derive_seed(project_salt, "validation", 0)
        self.project_salt = project_salt
        self.root = project_root(root)
        self.run_id = run_id

    @property
    def ledger_path(self) -> Path:
        return self.root / "runs" / "seed_ledger.jsonl"

    def allocate(self, domain: str, index: int) -> SeedAllocation:
        allocation = SeedAllocation(
            seed=derive_seed(self.project_salt, domain, index),
            domain=domain,
            index=index,
            timestamp=utc_timestamp(),
            run_id=self.run_id,
        )
        append_jsonl(self.ledger_path, allocation.to_dict())
        return allocation

    def seed_for(self, domain: str, index: int) -> int:
        """Allocate and record the seed for one explicit consumer/index pair."""
        return self.allocate(domain, index).seed
