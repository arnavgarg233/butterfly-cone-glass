"""Structural contract: the tree describes itself and ships no banned term."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

EXCLUDED_PARTS = frozenset(
    {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache", "runs"}
)

TOP_LEVEL = {
    ".github",
    ".gitignore",
    "LICENSE",
    "MANIFEST.sha256",
    "README.md",
    "configs",
    "data",
    "pyproject.toml",
    "reproduce.sh",
    "results",
    "scripts",
    "src",
    "tests",
    "uv.lock",
}

#: Terms the author does not use in writing. No shipped file may contain one,
#: and no shipped path may be named after one.
BANNED = re.compile(r"prereg|pre-regist|\bseal", re.IGNORECASE)

TEXT_SUFFIXES = {".py", ".md", ".sh", ".json", ".toml", ".yml", ".yaml", ".txt", ".cff", ".lock"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shipped() -> list[str]:
    names = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(REPO)
        if EXCLUDED_PARTS.intersection(rel.parts):
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        names.append(str(rel))
    return sorted(names)


def test_top_level_is_the_house_layout() -> None:
    present = {
        path.name
        for path in REPO.iterdir()
        if path.name not in EXCLUDED_PARTS and not path.name.endswith(".egg-info")
    }
    assert present == TOP_LEVEL


def test_manifest_lists_and_pins_every_shipped_file() -> None:
    manifest = {}
    for line in (REPO / "MANIFEST.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            manifest[name] = digest
    present = {name for name in shipped() if name != "MANIFEST.sha256"}
    assert present - set(manifest) == set(), "shipped files missing from MANIFEST.sha256"
    assert set(manifest) - present == set(), "MANIFEST.sha256 lists files that are absent"
    assert [name for name, d in manifest.items() if sha256_file(REPO / name) != d] == []


def test_no_shipped_path_or_text_uses_a_banned_term() -> None:
    offenders = []
    for name in shipped():
        if BANNED.search(name):
            offenders.append(name)
            continue
        path = REPO / name
        if path.suffix not in TEXT_SUFFIXES:
            continue
        if name in ("tests/test_repository_contract.py", "MANIFEST.sha256", "uv.lock"):
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if BANNED.search(line):
                offenders.append(f"{name}:{number}: {line.strip()[:80]}")
    assert offenders == []


def test_no_internal_project_code_is_exposed() -> None:
    code = re.compile(r"\bp0\d{2}\b", re.IGNORECASE)
    offenders = []
    for name in shipped():
        path = REPO / name
        if path.suffix not in TEXT_SUFFIXES or name == "tests/test_repository_contract.py":
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if code.search(line):
                offenders.append(f"{name}:{number}")
    assert offenders == []


def test_every_result_artifact_the_claim_surface_names_is_present() -> None:
    frozen = json.loads((REPO / "configs/expected_values.json").read_text())
    for record in frozen["values"].values():
        for token in record["source"].split():
            if token.startswith("results/"):
                assert (REPO / token).is_file(), token


def test_reproduce_entry_point_is_executable_and_fails_closed() -> None:
    script = REPO / "reproduce.sh"
    assert script.stat().st_mode & 0o111
    text = script.read_text()
    assert "set -euo pipefail" in text
