"""Attach partition provenance to the v3 E8 rows, in place.

The August pipeline ran E8 as a separate chunk and ``merge_results.py`` stitched
it onto the main store, copying the client histogram across from the matching
E1 row. The v3 suite writes every group into one store, so that step never runs
and the downstream audit gate rejects E8 for having no ``client_class_hist``.

This performs the same backfill with the same guard: the histogram is copied
only after the E8 row's realized client sizes are confirmed identical to the E1
row's, so "same partition" stays a checked claim rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def backfill(store: dict) -> list[str]:
    e1 = store["E1_alpha_sweep"]
    touched = []
    for key, record in store["E8_baselines"].items():
        if key.startswith("local_only"):
            continue
        if "client_class_hist" in record:
            continue
        fields = dict(part.split("=", 1) for part in key.split("|") if "=" in part)
        ref_key = f"alpha={fields['alpha']}|seed={fields['seed']}"
        ref = e1.get(ref_key)
        if ref is None:
            raise ValueError(f"missing E1 partition reference for {key}")
        if record.get("client_sizes") != ref.get("client_sizes"):
            raise ValueError(
                f"E8/E1 client-size mismatch for {key}: "
                f"{record.get('client_sizes')} vs {ref.get('client_sizes')}")
        record["client_class_hist"] = ref["client_class_hist"]
        record["partition_reference"] = f"E1_alpha_sweep/{ref_key}"
        record["trainable_params"] = record["upload_bytes_per_client_per_round"] // 4
        touched.append(key)
    return touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, type=Path)
    args = ap.parse_args()

    store = json.loads(args.store.read_text(encoding="utf8"))
    touched = backfill(store)
    args.store.write_text(json.dumps(store, indent=1), encoding="utf8")
    print(f"partition provenance attached to {len(touched)} E8 rows")
    for key in touched:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
