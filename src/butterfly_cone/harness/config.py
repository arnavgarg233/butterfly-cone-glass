"""Dataclass configuration with canonical, freezeable experiment inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .storage import write_new_bytes

try:
    import yaml
except ImportError:  # The JSON fallback remains valid YAML 1.2.
    yaml = None  # type: ignore[assignment]


class FrozenConfigError(RuntimeError):
    """Raised when code tries to change a frozen configuration."""


class FrozenDict(dict[str, Any]):
    """A recursively produced mapping which rejects every mutating operation."""

    @staticmethod
    def _raise(*_: Any, **__: Any) -> None:
        raise FrozenConfigError("configuration is frozen")

    __setitem__ = _raise
    __delitem__ = _raise
    clear = _raise
    pop = _raise
    popitem = _raise
    setdefault = _raise
    update = _raise
    __ior__ = _raise


def _normalise(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normalise(item) for item in value), key=repr)
    return value


def canonical_json(value: Any) -> str:
    """Stable JSON used as the sole source of a semantic configuration hash."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass
class ExperimentConfig:
    """The intentionally small schema shared by every ButterflyCone phase.

    ``values`` holds phase-specific, YAML-native settings.  Keeping those
    settings under one mapping makes the immutable experiment input explicit
    without prescribing engine-specific fields in the harness.
    """

    phase: str
    values: Mapping[str, Any] = field(default_factory=dict)
    _is_frozen: bool = field(default=False, init=False, repr=False)
    _frozen_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("phase must be a non-empty string")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        object.__setattr__(self, "values", dict(self.values))

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and getattr(self, "_is_frozen", False):
            raise FrozenConfigError("configuration is frozen")
        object.__setattr__(self, name, value)

    @property
    def payload(self) -> dict[str, Any]:
        return {"phase": self.phase, "values": _normalise(self.values)}

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.payload)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def frozen_hash(self) -> str | None:
        return self._frozen_hash

    def freeze(self) -> str:
        """Freeze all configuration fields and retain the audit hash."""
        if not self._is_frozen:
            frozen_hash = self.config_hash
            object.__setattr__(self, "values", _freeze(self.values))
            object.__setattr__(self, "_frozen_hash", frozen_hash)
            object.__setattr__(self, "_is_frozen", True)
        return self._frozen_hash  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload
        payload.update({"frozen": self.is_frozen, "frozen_hash": self.frozen_hash})
        return payload

    def dump_yaml(self, path: Path | str) -> Path:
        """Atomically create a YAML config file; never silently replace one."""
        payload = self.to_dict()
        if yaml is not None:
            text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
        else:
            text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        return write_new_bytes(Path(path), text.encode("utf-8"))


def load_config(path: Path | str) -> ExperimentConfig:
    """Load a dumped ButterflyCone config and validate any recorded frozen hash."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("config YAML must contain a mapping")
    config = ExperimentConfig(phase=raw["phase"], values=raw.get("values", {}))
    if raw.get("frozen", False):
        actual_hash = config.freeze()
        expected_hash = raw.get("frozen_hash")
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError("frozen config hash does not match config contents")
    return config
