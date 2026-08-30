"""Safely merge corrected OmniMed-FL result stores.

Diagnostic warm-started files are deliberately rejected. The output is suitable
for ``omnimed_make_tables.py`` only after all inputs pass protocol checks. The
optional ``--extra`` stores support disjoint GPU chunks such as E5--E7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MAIN = {
    "E1_alpha_sweep", "E2_client_sweep", "E3_anticollapse",
    "E3b_early_abort", "E4_warmstart", "E5_fusion_seeds", "E6_cost",
    "E7_rag",
}


def load(path: Path) -> dict:
    if "diagnostic" in str(path).lower() or "warmstarted" in str(path).lower():
        raise ValueError(f"refusing diagnostic result file: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"refusing non-final result file: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def _check_meta(reference: dict, candidate: dict, path: Path) -> None:
    for field in ("tier", "n_train", "n_val"):
        if reference.get("_meta", {}).get(field) != candidate.get("_meta", {}).get(field):
            raise ValueError(f"metadata mismatch for {field} in {path}")


def merge(main_path: Path, e8_path: Path, extra_paths: list[Path] | None = None) -> dict:
    main = load(main_path)
    e8 = load(e8_path)
    extra_paths = extra_paths or []

    if "E8_baselines" not in e8:
        raise ValueError("E8 result store is incomplete")

    out = dict(main)
    merged_from = [main_path.name]
    for path in extra_paths:
        extra = load(path)
        _check_meta(main, extra, path)
        for section in EXPECTED_MAIN & set(extra):
            if section not in out:
                out[section] = extra[section]
                continue
            if not isinstance(out[section], dict) or not isinstance(extra[section], dict):
                raise ValueError(f"cannot record-merge non-dict section {section}")
            for key, value in extra[section].items():
                if key in out[section] and out[section][key] != value:
                    raise ValueError(f"conflicting record {section}/{key} in {path}")
                out[section][key] = value
        merged_from.append(path.name)

    if not EXPECTED_MAIN.issubset(out):
        missing = sorted(EXPECTED_MAIN - set(out))
        raise ValueError(f"main result stores are incomplete after merging: {missing}")

    # The live store uses this historical internal key for the E4 pooled-data
    # oracle checkpoint. Rename it in the publication artifact so it cannot be
    # mistaken for the initialization of operational E1--E3/E5--E8 runs.
    if "_warmstart" in out:
        out["_e4_pooled_oracle_training"] = out.pop("_warmstart")

    _check_meta(main, e8, e8_path)
    if "E8_baselines" in out:
        raise ValueError("main store unexpectedly contains E8; refusing overwrite")
    # E8's live launcher predates client-histogram instrumentation. Its
    # partitions are nevertheless the same deterministic split used by E1 for
    # the same alpha/seed. Verify the client sizes and attach that provenance so
    # the matched-algorithm table remains auditable.
    e8_records = json.loads(json.dumps(e8["E8_baselines"]))
    for key, record in e8_records.items():
        if key.startswith("local_only"):
            continue
        fields = dict(part.split("=", 1) for part in key.split("|") if "=" in part)
        ref_key = f"alpha={fields['alpha']}|seed={fields['seed']}"
        ref = out["E1_alpha_sweep"].get(ref_key)
        if ref is None:
            raise ValueError(f"missing E1 partition reference for {key}")
        if record.get("client_sizes") != ref.get("client_sizes"):
            raise ValueError(f"E8/E1 client-size mismatch for {key}")
        record["client_class_hist"] = ref["client_class_hist"]
        record["partition_reference"] = f"E1_alpha_sweep/{ref_key}"
        record["trainable_params"] = record["upload_bytes_per_client_per_round"] // 4
    out["E8_baselines"] = e8_records
    merged_from.append(e8_path.name)
    out.setdefault("_meta", {}).update({
        "data_protocol": "controlled_v2_real4_synthetic_covid_template_text",
        "data_cache_sha256": "4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba",
        "image_source_counts": {
            "public_radiographs": 2400,
            "synthetic_covid": 600,
        },
        "text_source_counts": {"synthetic_class_conditioned_templates": 3000},
        "fl_initialization": "public_pretrained_encoders_random_task_heads",
        "training_precision": "fp32_tensors_no_amp",
        "deterministic_algorithms_enforced": False,
        "runtime_environment": "shared_gpu_server_with_uncontrolled_contention",
        "executed_model_source_sha256": "ce473f4bca58f8920d7c22b55b3e0dd28a2de227049f4ad77141659468cbf227",
        "experiment_runner_sha256": "1805c5bafb5f4889ecab87fe16e3e16788d5e0d1c7d205f19c88f81555f420e4",
        "chunk_wrapper_sha256": "644e4049c2e22d52a88136e763b9cdbb5a75f6d178076a2b7696592b985a4fce",
        "chunking": "disjoint_complete_record_stores_with_conflict_checked_merge",
        "merged_from": merged_from,
        "e8_gpu": e8.get("_meta", {}).get("gpu"),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, type=Path)
    ap.add_argument("--e8", required=True, type=Path)
    ap.add_argument("--extra", action="append", default=[], type=Path,
                    help="additional disjoint/partial main result store; repeatable")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    merged = merge(args.main, args.e8, args.extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=1), encoding="utf8")
    tmp.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
