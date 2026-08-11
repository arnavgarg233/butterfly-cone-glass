#!/usr/bin/env bash
# Recompute or re-read every load-bearing number in the butterfly-cone paper,
# or exit nonzero. Nothing here is simulated at production scale: the heavy
# cone integration and equilibration need GPU hours and multi-GB parent
# archives that this repository does not ship. What runs here is the whole
# evidence path that turns the shipped artifacts into the paper's numbers, plus
# a CPU end-to-end self-test of the exact production code path.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/butterfly-cone-replay.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT

export SOURCE_DATE_EPOCH=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$ROOT"

# 1. The pinned environment, exactly as locked.
uv sync --locked
uv lock --check

# 2. Every frozen load-bearing value: 83 recomputed from the stored
#    measurements in results/, 18 re-read from a named pointer. Fails closed on
#    any mismatch, any missing artifact and any value that is measured but not
#    frozen.
uv run --frozen --no-sync python scripts/verify_claims.py \
  --output "$TMP_ROOT/claim_verification.json"

# 3. Regression tests: the slab selector and the second-moment anisotropy
#    estimator, the claim surface, and the repository contract.
uv run --frozen --no-sync python -m pytest -q

# 4. Every shipped byte against MANIFEST.sha256.
shasum -a 256 -c MANIFEST.sha256 >/dev/null

# 5. CPU end-to-end self-test of the production twin-cone path: bit-identical
#    twins, one localized kick, deterministic NVE, cone read-out. Small system,
#    short horizon, no archive needed. The device is passed explicitly, so the
#    self-test runs on cpu/float64 on any machine and never asks for a Metal
#    allocator; the campaign CLI still defaults to mps for production callers.
uv run --frozen --no-sync python scripts/gardner_cone_campaign.py \
  --smoke --device cpu --out "$TMP_ROOT/cone_smoke" >/dev/null

# 6. The working tree must be clean, so the replay describes the shipped bytes.
git diff --check
test -z "$(git status --porcelain --untracked-files=all)"

printf '%s\n' "butterfly cone replay: PASS"
printf '%s\n' "frozen load-bearing values: 101"
printf '%s\n' "recomputed from stored measurements: 83"
printf '%s\n' "re-read from a named pointer: 18"
printf '%s\n' "twin pairs behind the confinement result: 40"
printf '%s\n' "glass formers behind the ceiling identity: 6"
