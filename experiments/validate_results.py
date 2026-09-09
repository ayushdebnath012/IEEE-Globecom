"""Fail-fast validation for the corrected OmniMed-FL reviewer results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_COUNTS = {
    "E1_alpha_sweep": 10,
    "E2_client_sweep": 8,
    "E3_anticollapse": 16,
    "E3b_early_abort": 4,
    "E4_warmstart": 4,
    "E5_fusion_seeds": 16,
    "E6_cost": 3,
    "E7_rag": 1,
    "E8_baselines": 7,
}

EXPECTED_KEYS = {
    "E1_alpha_sweep": {
        f"alpha={alpha}|seed={seed}"
        for alpha in (0.1, 0.3, 0.5, 1.0, 5.0) for seed in (0, 1)
    },
    "E2_client_sweep": {
        f"K={clients}|seed={seed}"
        for clients in (3, 5, 10, 20) for seed in (0, 1)
    },
    "E3_anticollapse": {
        f"{variant}|alpha={alpha}|seed={seed}"
        for variant in ("full", "no_balanced", "no_diversity", "neither")
        for alpha in (0.1, 1.0) for seed in (0, 1)
    },
    "E3b_early_abort": {
        f"early_abort={setting}|seed={seed}"
        for setting in (True, False) for seed in (0, 1)
    },
    "E4_warmstart": {
        f"warm_start={setting}|seed={seed}"
        for setting in (True, False) for seed in (0, 1)
    },
    "E5_fusion_seeds": {
        f"{fusion}|seed={seed}"
        for fusion in ("concat", "attention", "gated", "clip", "flamingo",
                       "blip2", "coca", "unified_io")
        for seed in (0, 1)
    },
    "E6_cost": {"Fed-LLM", "Fed-ViT", "Fed-VLM"},
    "E7_rag": {"heldout"},
    "E8_baselines": {
        *(f"{algo}|alpha=0.1|seed={seed}"
          for algo in ("fedavg", "fedprox", "scaffold") for seed in (0, 1)),
        "local_only|alpha=0.1|seed=0",
    },
}

FEDERATED_EXPERIMENTS = {
    "E1_alpha_sweep", "E2_client_sweep", "E3_anticollapse",
    "E4_warmstart", "E6_cost", "E8_baselines",
}


def check_number(value, where: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite number at {where}: {value!r}")


def validate(data: dict) -> None:
    meta = data.get("_meta", {})
    protocol = meta.get("data_protocol")
    expected_counts = dict(EXPECTED_COUNTS)
    expected_keys = {name: set(keys) for name, keys in EXPECTED_KEYS.items()}
    if protocol == "controlled_v3_real5_template_text":
        # In the frozen v3 store, ``_warmstart.concat`` is the recorded
        # pooled-data training run that produced the diagnostic E4 checkpoint.
        # Preserve the immutable store and validate that record in place.
        pooled = data.get("_warmstart", {}).get("concat")
        if not isinstance(pooled, dict):
            raise ValueError("v3 pooled-oracle training record is missing")
        for field in ("f1", "accuracy", "diversity", "epoch",
                      "wall_seconds", "trainable_params"):
            check_number(pooled.get(field), f"_warmstart/concat/{field}")
        expected_counts["E8_baselines"] = 14
        expected_keys["E8_baselines"] = {
            *(f"{algo}|alpha={alpha}|seed={seed}"
              for algo in ("fedavg", "fedprox", "scaffold")
              for alpha in (0.1, 1.0) for seed in (0, 1)),
            *(f"local_only|alpha={alpha}|seed=0" for alpha in (0.1, 1.0)),
        }
    else:
        if "_warmstart" in data:
            raise ValueError(
                "ambiguous historical _warmstart key remains in publication artifact")
        if "_e4_pooled_oracle_training" not in data:
            raise ValueError("E4 pooled-oracle training provenance is missing")
    for exp, expected in expected_counts.items():
        got = len(data.get(exp, {}))
        if got != expected:
            raise ValueError(f"{exp}: expected {expected} records, found {got}")
        got_keys = set(data[exp])
        if got_keys != expected_keys[exp]:
            missing = sorted(expected_keys[exp] - got_keys)
            extra = sorted(got_keys - expected_keys[exp])
            raise ValueError(f"{exp}: wrong configuration keys; missing={missing}, extra={extra}")

    for exp, records in data.items():
        if exp.startswith("_") or not isinstance(records, dict):
            continue
        for key, rec in records.items():
            if not isinstance(rec, dict) or "error" in rec:
                raise ValueError(f"invalid record {exp}/{key}: {rec!r}")
            if "f1" in rec:
                check_number(rec["f1"], f"{exp}/{key}/f1")
                if not 0 <= rec["f1"] <= 1:
                    raise ValueError(f"F1 outside [0,1] at {exp}/{key}")

            if exp == "E8_baselines" and key.startswith("local_only"):
                values = rec.get("per_client_f1", [])
                best_values = rec.get("per_client_best_epoch_f1", [])
                if len(values) != 5:
                    raise ValueError(f"{exp}/{key}: expected five local client scores")
                if len(best_values) != 5:
                    raise ValueError(f"{exp}/{key}: expected five local best-epoch diagnostics")
                for i, value in enumerate(values):
                    check_number(value, f"{exp}/{key}/per_client_f1/{i}")
                    check_number(best_values[i],
                                 f"{exp}/{key}/per_client_best_epoch_f1/{i}")
                    if value > best_values[i] + 1e-12:
                        raise ValueError(f"{exp}/{key}: final local F1 exceeds its recorded best")
                if rec.get("epochs_per_client") != 24:
                    raise ValueError(f"{exp}/{key}: local-only budget is not 24 epochs")
                if rec.get("early_abort") is not False:
                    raise ValueError(f"{exp}/{key}: local-only early abort must be disabled")

            if exp not in FEDERATED_EXPERIMENTS or key.startswith("local_only"):
                continue
            hist = rec.get("history", {})
            rounds = hist.get("round_f1", [])
            if len(rounds) != 8:
                raise ValueError(f"{exp}/{key}: expected 8 completed rounds")
            for i, value in enumerate(rounds):
                check_number(value, f"{exp}/{key}/round_f1/{i}")

            sizes = rec.get("client_sizes", [])
            hists = rec.get("client_class_hist")
            if hists is None:
                raise ValueError(f"{exp}/{key}: missing client class histograms")
            if len(sizes) != len(hists):
                raise ValueError(f"{exp}/{key}: client histogram length mismatch")
            if any(sum(h) != n for h, n in zip(hists, sizes)):
                raise ValueError(f"{exp}/{key}: client histogram totals mismatch")

            if "upload_bytes_per_client_per_round" in rec:
                expected_comm = (2 * len(sizes) * 8 *
                                 rec["upload_bytes_per_client_per_round"])
                if rec.get("total_comm_bytes") != expected_comm:
                    raise ValueError(f"{exp}/{key}: communication formula mismatch")

    if meta.get("n_train") != 2400 or meta.get("n_val") != 600:
        raise ValueError("unexpected train/validation sizes")
    if protocol == "controlled_v3_real5_template_text":
        # Real-only images. Every class is a public radiograph, so no synthetic
        # image count may appear and the cache hash is whatever that build
        # produced -- it only has to be recorded.
        if meta.get("image_source_counts") != {"public_radiographs": 3000}:
            raise ValueError("v3 store must source all 3,000 images publicly")
        if not meta.get("data_cache_sha256"):
            raise ValueError("v3 store is missing its data-cache hash")
    elif protocol == "controlled_v2_real4_synthetic_covid_template_text":
        if meta.get("data_cache_sha256") != (
                "4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba"):
            raise ValueError("unexpected controlled data-cache hash")
        if meta.get("image_source_counts") != {
                "public_radiographs": 2400, "synthetic_covid": 600}:
            raise ValueError("unexpected image-source provenance")
    else:
        raise ValueError("result store is not marked as a known data protocol")
    if meta.get("text_source_counts") != {
            "synthetic_class_conditioned_templates": 3000}:
        raise ValueError("unexpected text-source provenance")
    if meta.get("fl_initialization") != "public_pretrained_encoders_random_task_heads":
        raise ValueError("unexpected federated initialization protocol")
    if meta.get("training_precision") != "fp32_tensors_no_amp":
        raise ValueError("unexpected training precision")
    if meta.get("deterministic_algorithms_enforced") is not False:
        raise ValueError("GPU determinism provenance is missing or incorrect")
    if meta.get("executed_model_source_sha256") != (
            "ce473f4bca58f8920d7c22b55b3e0dd28a2de227049f4ad77141659468cbf227"):
        raise ValueError("unexpected model-source hash")
    if protocol != "controlled_v3_real5_template_text":
        if meta.get("experiment_runner_sha256") != (
                "1805c5bafb5f4889ecab87fe16e3e16788d5e0d1c7d205f19c88f81555f420e4"):
            raise ValueError("unexpected experiment-runner hash")
        if meta.get("chunk_wrapper_sha256") != (
                "644e4049c2e22d52a88136e763b9cdbb5a75f6d178076a2b7696592b985a4fce"):
            raise ValueError("unexpected chunk-wrapper hash")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v3_chain(results_path: Path, data: dict) -> None:
    """Validate the frozen v3 store against its merged provenance sidecar."""
    if data.get("_meta", {}).get("data_protocol") != (
            "controlled_v3_real5_template_text"):
        return

    merged_path = results_path.with_name("merged_v3.json")
    if not merged_path.is_file():
        raise ValueError(f"v3 provenance sidecar is missing: {merged_path}")
    merged = json.loads(merged_path.read_text(encoding="utf8"))
    provenance = merged.get("_meta", {})

    expected_files = {
        "legacy_sha256": results_path,
        "core_runner_sha256": results_path.parent / "omnimed_experiments_v3.py",
        "reviewer_runner_sha256": results_path.parent.parent / "reviewer_completion.py",
        "pfin_helper_sha256": results_path.parent.parent / "pfin_matched.py",
        "retrieval_runner_sha256": (
            results_path.parent.parent / "rag_retrieval_corrected.py"),
        "corrected_retrieval_sha256": results_path.with_name("rag_v3.json"),
    }
    for field, path in expected_files.items():
        if not path.is_file():
            raise ValueError(f"v3 provenance file is missing: {path}")
        actual = file_sha256(path)
        if provenance.get(field) != actual:
            raise ValueError(
                f"v3 provenance mismatch for {field}: "
                f"recorded={provenance.get(field)!r}, actual={actual!r}")

    if provenance.get("base_model_sha256") != data["_meta"].get(
            "executed_model_source_sha256"):
        raise ValueError("v3 base-model hash does not match the result store")
    if provenance.get("data_cache_sha256") != data["_meta"].get(
            "data_cache_sha256"):
        raise ValueError("v3 data-cache hash does not match the result store")
    for flag in ("legacy_meta_and_protocol_validated",
                 "new_record_protocols_validated",
                 "corrected_retrieval_protocol_validated",
                 "grid_model_state_invariants_validated"):
        if provenance.get(flag) is not True:
            raise ValueError(f"v3 merged provenance flag is not true: {flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    data = json.loads(args.results.read_text(encoding="utf8"))
    validate(data)
    validate_v3_chain(args.results, data)
    print("validated", args.results)


if __name__ == "__main__":
    main()
