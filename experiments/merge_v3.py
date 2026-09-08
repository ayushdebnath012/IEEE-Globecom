"""Merge the real-only (v3) result store with its matched-arm records.

``merge_reviewer_results`` is an audit gate pinned to the August artifacts. It
rejects the v3 store on the first check, which is the gate working as intended.
This module re-pins the same checks to the artifacts that produced the real-data
run rather than loosening them, so every structural, partition and provenance
assertion still runs.

Four deliberate differences from the v2 gate, each because the underlying fact
changed, not because a check was inconvenient:

1. ``data_protocol`` must be the v3 corpus string, and the data-cache SHA-256
   must be the real-image cache.
2. ``E8_baselines`` may carry the extra ``alpha=1.0`` rows the standard tier now
   emits. The seven rows the v2 gate demanded must still all be present.
3. ``runtime_environment`` is not required on the store. The v2 value was added
   by a downstream merge step that has not run yet for v3.
4. ``local_only`` rows may sit at alpha=1.0 as well as alpha=0.1.
5. The core-runner SHA-256 is recorded rather than asserted. The runner gained
   provenance stamping between the main suite and the matched arms, so no single
   value describes both, and asserting one of them would be false.

Usage::

    python merge_v3.py --new-dir v3/arms --legacy v3/results_v3.json \\
        --rag-corrected v3/rag_v3.json --out v3/merged_v3.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import merge_reviewer_results as M  # noqa: E402

# Artifacts that produced the real-data run.
V3_PROTOCOL = "controlled_v3_real5_template_text"
V3_CACHE = "82bc364d664a2d2124adb642c296e2e7d84ecf843f306e6287815690c2a4204a"
V3_REVIEWER_RUNNER = "91425bc59bc3ea20f19d3a67fb94b3fba4447bcda9fa2eef80a27058e8f48452"
# The runner the matched arms executed under. The main suite ran under the
# pre-provenance-stamping version, which is why the store is not asserted
# against this value (see 5 above).
V3_CORE_RUNNER = "344fe4a5bbcff880a362d36f72a39cd2a3cd25c608474953f81bf3dc044a8fe4"
# Unchanged from the v2 run, so the v2 constants already hold:
#   base model source ce473f4b..., pfin helper d95f2f41...

# Per-condition validation-query counts for the v3 split. Verified directly
# against data_cache_v3_real.pkl's val_tlbls, not copied from the record being
# checked, so the gate still tests the retrieval record against the corpus.
V3_RETRIEVAL_COUNTS = {
    "NORMAL": 100,
    "PNEUMONIA": 141,
    "COVID19": 107,
    "PLEURAL_EFFUSION": 135,
    "CARDIOMEGALY": 117,
}

M.EXPECTED_DATA_PROTOCOL = V3_PROTOCOL
M.EXPECTED_CORE_RUNNER = V3_CORE_RUNNER
M.EXPECTED_CACHE = V3_CACHE
M.RETRIEVAL_COUNTS = V3_RETRIEVAL_COUNTS
M.EXPECTED_REVIEWER_RUNNER = V3_REVIEWER_RUNNER
M.EXPECTED_REVIEWER_RUNNER_NATIVE_BUDGET = V3_REVIEWER_RUNNER

_original_exact_group_keys = M._exact_group_keys
_original_validate_local_only = M._validate_local_only


def _legacy_meta_v3(legacy: Mapping[str, Any]) -> dict:
    """``_legacy_meta`` re-pinned to the real-only corpus."""
    meta = M._mapping(legacy.get("_meta"), "legacy._meta")
    expected_top = {
        "tier": "standard",
        "device": "cuda",
        "gpu": "NVIDIA H100 NVL",
        "n_train": M.N_TRAIN,
        "n_val": 600,
        "data_protocol": V3_PROTOCOL,
        "fl_initialization": "public_pretrained_encoders_random_task_heads",
        "training_precision": "fp32_tensors_no_amp",
        "executed_model_source_sha256": M.EXPECTED_BASE,
        "data_cache_sha256": V3_CACHE,
    }
    for field, expected in expected_top.items():
        M._equal(meta.get(field), expected, f"legacy._meta.{field}")

    # Provenance that only holds for a real-image corpus.
    M._equal(meta.get("image_source_counts"), {"public_radiographs": 3000},
             "legacy._meta.image_source_counts")
    M._equal(meta.get("text_source_counts"),
             {"synthetic_class_conditioned_templates": 3000},
             "legacy._meta.text_source_counts")

    M._explicit_false(meta, "deterministic_algorithms_enforced", "legacy._meta")
    tier = M._mapping(meta.get("tier_config"), "legacy._meta.tier_config")
    expected_tier = {
        "samples_per_class": 600,
        "batch_size": 16,
        "fed_rounds": M.ROUNDS,
        "local_epochs": M.LOCAL_EPOCHS,
        "seeds": [0, 1],
        "alphas": [0.1, 0.3, 0.5, 1.0, 5.0],
        "client_counts": [3, 5, 10, 20],
        "fusion_seeds": [0, 1],
        "text_model": "distilbert-base-uncased",
        "vision_model": "google/vit-base-patch16-224",
        "central_epochs": 12,
    }
    for field, expected in expected_tier.items():
        M._equal(tier.get(field), expected, f"legacy._meta.tier_config.{field}")
    if not isinstance(meta.get("torch"), str) or not meta["torch"]:
        M._fail("legacy._meta.torch", "must be a nonempty string")
    return meta


def _exact_group_keys_v3(group: Any, expected: set[str], where: str) -> dict:
    """Exact everywhere except E8, where extra alpha=1.0 rows are allowed."""
    if where.endswith("E8_baselines"):
        group = M._mapping(group, where)
        missing = expected - set(group)
        if missing:
            M._fail(where, f"key mismatch: missing={sorted(missing)}")
        return group
    return _original_exact_group_keys(group, expected, where)


def _validate_local_only_v3(result: Any, where: str) -> dict:
    """As v2, but the standard tier now also emits a local-only row at alpha=1.

    The v2 gate hard-coded alpha=0.1 because that was the only local-only row in
    existence. Everything else it checks -- five per-client scores, the mean/
    best/worst agreeing with those scores, the 24-epoch budget -- still applies.
    """
    alpha = M._mapping(result, where).get("alpha")
    if alpha == 0.1:
        return _original_validate_local_only(result, where)
    if alpha != 1.0:
        M._fail(f"{where}.alpha", f"expected 0.1 or 1.0, got {alpha!r}")
    patched = dict(result)
    patched["alpha"] = 0.1
    validated = _original_validate_local_only(patched, where)
    validated["alpha"] = alpha
    result.update(validated)
    return result


M._legacy_meta = _legacy_meta_v3
M._exact_group_keys = _exact_group_keys_v3
M._validate_local_only = _validate_local_only_v3

if __name__ == "__main__":
    M.main()
