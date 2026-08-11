"""The frozen claim surface must recompute and must fail closed."""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import verify_claims  # noqa: E402


def frozen() -> dict:
    return json.loads((REPO / "configs/expected_values.json").read_text())


def test_every_frozen_value_recomputes() -> None:
    measured = verify_claims.measure()
    values = frozen()["values"]
    assert set(measured) == set(values)
    for key, record in values.items():
        want, got = record["value"], measured[key]
        if isinstance(want, bool):
            assert bool(got) == want, key
        elif isinstance(want, (int, float)):
            assert abs(float(got) - float(want)) <= 1e-12 * max(1.0, abs(float(want))), key
        else:
            assert got == want, key


def test_every_frozen_value_names_its_kind_and_source() -> None:
    for key, record in frozen()["values"].items():
        assert record["kind"] in {"recomputed", "reread"}, key
        assert record["source"], key


def test_verifier_exits_nonzero_on_a_broken_value(tmp_path) -> None:
    payload = frozen()
    key = "confinement.c_ratio.bulk"
    payload["values"][key]["value"] = payload["values"][key]["value"] + 1.0
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload))
    assert verify_claims.main(["--expected", str(broken)]) == 1


def test_verifier_exits_zero_on_the_shipped_surface() -> None:
    assert verify_claims.main([]) == 0


def test_ceiling_identity_holds_across_the_films() -> None:
    values = {key: record["value"] for key, record in frozen()["values"].items()}
    # The ceiling ratio is preserved under confinement: the worst film sits
    # inside the stated bound and the films do not order with thickness.
    assert values["confinement.ceiling.worst_deviation_percent"] < 3.6
    assert values["confinement.ceiling.worst_film"] == 0.45
    # The cone shape is not preserved: the two wider films are indistinguishable
    # from bulk and the two narrower ones are not.
    assert values["confinement.anisotropy_z.0.70"] < 1.0
    assert values["confinement.anisotropy_z.0.50"] < 1.0
    assert values["confinement.anisotropy_z.0.45"] > 3.0
    assert values["confinement.anisotropy_z.0.35"] > 5.0
    # The matched-kick bulk control separates the boundary from the truncation.
    assert abs(values["confinement.anisotropy_z.matched_clip_control"]) < 1.0


def test_rate_follows_the_entropy_axis_not_the_stiffness_axis() -> None:
    values = {key: record["value"] for key, record in frozen()["values"].items()}
    assert values["entropy_axis.pooled.r_lambda_vs_proxy"] > 0.9
    assert values["entropy_axis.pooled.r_proxy_vs_stiffness"] < 0.0
    assert values["entropy_axis.pooled.r_lambda_vs_stiffness"] < 0.0
    assert values["entropy_axis.pooled.n_states"] == 12
    assert values["entropy_axis.pooled.n_models"] == 6
