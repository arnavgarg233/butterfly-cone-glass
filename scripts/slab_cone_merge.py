#!/usr/bin/env python3
"""Merge sharded slab-cone runs into one artifact, carrying the controls forward.

The first merge dropped the per-shard ``controls`` block, which made the time
axis of ``divergence_curve`` unreconstructable from the merged file alone (a
figure builder had to reach back into the shards to recover ``stride``).  This
merge asserts every shard shares identical controls and then persists them, so
the merged artifact is self-describing.

It also records the kick-containment audit per film.  ``o_shell`` selects its
shell by distance in the full periodic box and knows nothing about the slab, so
for a film thinner than the shell diameter it would displace frozen wall
particles; the campaign now discards those, and ``n_kick_in_wall_discarded``
records how many were dropped so the correction is auditable after the fact.

Run:
    ./.venv/bin/python scripts/slab_cone_merge.py runs/slab_cone/shards2 \
        --out runs/slab_cone/slab_cone_merged.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from slab_cone_campaign import summarise  # noqa: E402

# keys whose value must agree across shards for the merge to be meaningful
SHARED_CONTROL_KEYS = ("dt", "horizon_steps", "horizon_time", "temperature", "deltas", "r_pert", "interface")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.shard_dir) / "*.json")))
    if not paths:
        raise SystemExit(f"no shards found in {args.shard_dir}")

    pairs: list[dict] = []
    controls: dict | None = None
    parents: set[str] = set()
    for path in paths:
        blob = json.load(open(path))
        shard_controls = {k: blob["controls"][k] for k in SHARED_CONTROL_KEYS if k in blob.get("controls", {})}
        if controls is None:
            controls = shard_controls
        elif shard_controls != controls:
            raise SystemExit(f"shard {path} controls disagree:\n  {shard_controls}\n  {controls}")
        parents.add(blob.get("parent", "unknown"))
        pairs += blob["pairs"]

    if len(parents) != 1:
        raise SystemExit(f"shards use different parents: {sorted(parents)}")

    table = summarise(pairs)

    # self-describing time axis for divergence_curve
    n_frames = len(pairs[0]["divergence_curve"])
    stride = controls["horizon_steps"] / (n_frames - 1)
    time_axis = [i * stride * controls["dt"] for i in range(n_frames)]

    # kick-containment audit: how many shell hits landed on the pinned wall and
    # were discarded, per film
    audit = {}
    for row in table:
        key = "bulk" if row["geometry"] == "bulk_control" else f"{row['thickness_fraction_of_box']:.2f}"
        group = [p for p in pairs if p["thickness_fraction_of_box"] == row["thickness_fraction_of_box"]]
        audit[key] = {
            "n_perturbed_raw": sorted({p.get("n_perturbed") for p in group}),
            "n_perturbed_mobile": sorted({p.get("n_perturbed_mobile") for p in group}),
            "n_kick_in_wall_discarded": sorted({p.get("n_kick_in_wall_discarded") for p in group}),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "parent": sorted(parents)[0],
                "controls": controls,
                "time_axis": time_axis,
                "kick_containment_audit": audit,
                "n_pairs": len(pairs),
                "summary": table,
                "pairs": pairs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(paths)} shards, {len(pairs)} pairs -> {out}")
    print(f"controls: {controls}")
    print("kick containment (raw -> mobile, discarded):")
    for key, rec in audit.items():
        print(f"  {key:>5}: {rec['n_perturbed_raw']} -> {rec['n_perturbed_mobile']}, "
              f"discarded {rec['n_kick_in_wall_discarded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
