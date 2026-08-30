"""Fail-fast validation for the corrected OmniMed-FL reviewer results."""

from __future__ import annotations

import argparse
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
    if "_warmstart" in data:
        raise ValueError("ambiguous historical _warmstart key remains in publication artifact")
    if "_e4_pooled_oracle_training" not in data:
        raise ValueError("E4 pooled-oracle training provenance is missing")
    for exp, expected in EXPECTED_COUNTS.items():
        got = len(data.get(exp, {}))
        if got != expected:
            raise ValueError(f"{exp}: expected {expected} records, found {got}")
        got_keys = set(data[exp])
        if got_keys != EXPECTED_KEYS[exp]:
            missing = sorted(EXPECTED_KEYS[exp] - got_keys)
            extra = sorted(got_keys - EXPECTED_KEYS[exp])
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

    meta = data.get("_meta", {})
    if meta.get("n_train") != 2400 or meta.get("n_val") != 600:
        raise ValueError("unexpected train/validation sizes")
    if meta.get("data_protocol") != "controlled_v2_real4_synthetic_covid_template_text":
        raise ValueError("result store is not marked as the controlled-v2 protocol")
    if meta.get("data_cache_sha256") != (
            "4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba"):
        raise ValueError("unexpected controlled data-cache hash")
    if meta.get("image_source_counts") != {
            "public_radiographs": 2400, "synthetic_covid": 600}:
        raise ValueError("unexpected image-source provenance")
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
    if meta.get("experiment_runner_sha256") != (
            "1805c5bafb5f4889ecab87fe16e3e16788d5e0d1c7d205f19c88f81555f420e4"):
        raise ValueError("unexpected experiment-runner hash")
    if meta.get("chunk_wrapper_sha256") != (
            "644e4049c2e22d52a88136e763b9cdbb5a75f6d178076a2b7696592b985a4fce"):
        raise ValueError("unexpected chunk-wrapper hash")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    data = json.loads(args.results.read_text(encoding="utf8"))
    validate(data)
    print("validated", args.results)


if __name__ == "__main__":
    main()
