"""Strictly validate and merge the reviewer-completion experiment suite.

The merge is intentionally fail closed.  A JSON file is admitted only when its
experiment key, task, arguments, training contract, source hashes, metrics, and
task-specific result shape all agree.  The historical result store is checked
with the same care before any of its rows are reused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_BASE = "ce473f4bca58f8920d7c22b55b3e0dd28a2de227049f4ad77141659468cbf227"
EXPECTED_CACHE = "4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba"
EXPECTED_CORE_RUNNER = "1805c5bafb5f4889ecab87fe16e3e16788d5e0d1c7d205f19c88f81555f420e4"
EXPECTED_REVIEWER_RUNNER = "cdb9382800a3a21ee3665d0690e8591531c9c7ad8b3e9667427226b55dee4c9c"
# Revision adding only the --fedmme-epochs option; the matched-budget code
# path is unchanged, so records keep per-record hash pinning rather than a
# blanket allow-list.
EXPECTED_REVIEWER_RUNNER_NATIVE_BUDGET = (
    "cfd91f86a047006b84501567af206d9b6a976259106b44d27036b1cb014d04b6"
)
EXPECTED_PFIN_HELPER = "d95f2f41331fa6fe86fc2837a86148075ffbd5329db6b18b87e0cbd7c698a896"
EXPECTED_RAG_RUNNER = "53dd61d8beb164da6027e97b5e516954276e09bc63adfd4e02d2f617876d59da"

FUSIONS = ["concat", "attention", "gated", "clip", "flamingo", "blip2", "coca", "unified_io"]
PFIN_MODES = ["zero", "deterministic", "probabilistic", "probabilistic_uq"]
PARAMETER_FIELDS = {
    "seed",
    "alpha",
    "clients",
    "fusion",
    "init_mode",
    "pfin_mode",
    "deterministic",
}
N_TRAIN = 2400
N_CLASSES = 5
ROUNDS = 8
LOCAL_EPOCHS = 3
MIN_STANDARD_CLIENT_SHARD = 4
EXPECTED_MULTIMODAL_TRAINABLE_PARAMS = 153_935_621
RETRIEVAL_CONDITIONS = (
    "NORMAL",
    "PNEUMONIA",
    "COVID19",
    "PLEURAL_EFFUSION",
    "CARDIOMEGALY",
)
RETRIEVAL_COUNTS = {
    "NORMAL": 115,
    "PNEUMONIA": 121,
    "COVID19": 128,
    "PLEURAL_EFFUSION": 126,
    "CARDIOMEGALY": 110,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _fail(where: str, message: str) -> None:
    raise ValueError(f"{where}: {message}")


def _mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        _fail(where, f"expected an object, got {type(value).__name__}")
    return value


def _number(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f"expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(where, f"number is not finite: {value!r}")
    if minimum is not None and result < minimum:
        _fail(where, f"expected >= {minimum}, got {result}")
    return result


def _integer(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f"expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        _fail(where, f"expected >= {minimum}, got {value}")
    return value


def _probability(value: Any, where: str) -> float:
    result = _number(value, where)
    if not 0.0 <= result <= 1.0:
        _fail(where, f"metric is outside [0, 1]: {result}")
    return result


def _explicit_false(mapping: Mapping[str, Any], key: str, where: str) -> None:
    if key not in mapping or mapping[key] is not False:
        _fail(where, f"{key} must be present and exactly false")


def _equal(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        _fail(where, f"expected {expected!r}, got {actual!r}")


def _close(actual: Any, expected: Any, where: str) -> None:
    left = _number(actual, where)
    right = _number(expected, f"{where}.expected")
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        _fail(where, f"expected {right}, got {left}")


def finite_result_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_result_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_result_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(path, f"non-finite number {value!r}")


def expected_record_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}

    def add(task: str, key: str, **parameters: Any) -> None:
        if key in specs:
            raise AssertionError(f"duplicate internal manifest key: {key}")
        specs[key] = {"task": task, **parameters}

    for fusion in FUSIONS:
        for seed in (0, 1):
            add(
                "fusion",
                f"fusion={fusion}|alpha=0.1|K=5|seed={seed}",
                seed=seed,
                alpha=0.1,
                clients=5,
                fusion=fusion,
                init_mode=None,
                pfin_mode=None,
            )
    for alpha in (0.1, 5.0):
        for clients in (3, 10, 20):
            for seed in (0, 1):
                add(
                    "grid",
                    f"alpha={alpha}|K={clients}|seed={seed}",
                    seed=seed,
                    alpha=alpha,
                    clients=clients,
                    fusion=None,
                    init_mode=None,
                    pfin_mode=None,
                )
    for seed in (0, 1):
        add(
            "fedmme",
            f"fedmme_style|alpha=0.1|K=5|seed={seed}",
            seed=seed,
            alpha=0.1,
            clients=5,
            fusion=None,
            init_mode=None,
            pfin_mode=None,
            local_epochs=24,
        )
        add(
            "fedmme",
            f"fedmme_native|epochs=100|alpha=0.1|K=5|seed={seed}",
            seed=seed,
            alpha=0.1,
            clients=5,
            fusion=None,
            init_mode=None,
            pfin_mode=None,
            local_epochs=100,
            runner_sha256=EXPECTED_REVIEWER_RUNNER_NATIVE_BUDGET,
        )
        add(
            "init",
            f"init=random|alpha=1.0|K=5|seed={seed}",
            seed=seed,
            alpha=1.0,
            clients=5,
            fusion=None,
            init_mode="random",
            pfin_mode=None,
        )
        for mode in PFIN_MODES:
            add(
                "pfin",
                f"pfin={mode}|alpha=0.1|K=5|seed={seed}",
                seed=seed,
                alpha=0.1,
                clients=5,
                fusion=None,
                init_mode=None,
                pfin_mode=mode,
            )
    # 40 original reviewer-completion records plus the two native-budget FedMME
    # runs added to remove the local-epoch departure from the source method.
    if len(specs) != 42:
        raise AssertionError(f"internal reviewer manifest has {len(specs)} keys, not 42")
    return specs


def _validate_metric_bundle(result: Mapping[str, Any], where: str) -> None:
    for field in ("f1", "accuracy", "diversity"):
        _probability(result.get(field), f"{where}.{field}")


def _validate_partition(
    result: Mapping[str, Any], clients: int, where: str, *, require_histogram: bool
) -> tuple[list[int], list[list[int]] | None]:
    sizes = result.get("client_sizes")
    if not isinstance(sizes, list) or len(sizes) != clients:
        _fail(f"{where}.client_sizes", f"expected {clients} entries")
    normalized_sizes = [
        _integer(value, f"{where}.client_sizes[{index}]", minimum=0)
        for index, value in enumerate(sizes)
    ]
    if sum(normalized_sizes) != N_TRAIN:
        _fail(f"{where}.client_sizes", f"expected total {N_TRAIN}, got {sum(normalized_sizes)}")

    histogram = result.get("client_class_hist")
    if histogram is None and not require_histogram:
        return normalized_sizes, None
    if not isinstance(histogram, list) or len(histogram) != clients:
        _fail(f"{where}.client_class_hist", f"expected {clients} client rows")
    normalized_histogram: list[list[int]] = []
    for client_id, (row, size) in enumerate(zip(histogram, normalized_sizes)):
        if not isinstance(row, list) or len(row) != N_CLASSES:
            _fail(
                f"{where}.client_class_hist[{client_id}]",
                f"expected {N_CLASSES} class counts",
            )
        counts = [
            _integer(value, f"{where}.client_class_hist[{client_id}][{class_id}]", minimum=0)
            for class_id, value in enumerate(row)
        ]
        if sum(counts) != size:
            _fail(
                f"{where}.client_class_hist[{client_id}]",
                f"row total {sum(counts)} does not match shard size {size}",
            )
        normalized_histogram.append(counts)
    return normalized_sizes, normalized_histogram


def _active_client_metadata(
    result: dict,
    clients: int,
    where: str,
    *,
    skip_small_shards: bool,
) -> None:
    sizes = result["client_sizes"]
    if skip_small_shards:
        skipped = [index for index, size in enumerate(sizes) if size < MIN_STANDARD_CLIENT_SHARD]
        rule = f"derived from client_sizes; shards < {MIN_STANDARD_CLIENT_SHARD} do not locally train"
    else:
        skipped = [index for index, size in enumerate(sizes) if size <= 0]
        rule = "derived from client_sizes; every nonempty shard participates"
    active = clients - len(skipped)

    reported = all(
        field in result for field in ("nominal_clients", "active_clients", "skipped_client_ids")
    )
    if "nominal_clients" in result:
        _equal(result["nominal_clients"], clients, f"{where}.nominal_clients")
    if "active_clients" in result:
        _equal(result["active_clients"], active, f"{where}.active_clients")
    if "skipped_client_ids" in result:
        _equal(result["skipped_client_ids"], skipped, f"{where}.skipped_client_ids")

    result["nominal_clients"] = clients
    result["active_clients"] = active
    result["skipped_client_ids"] = skipped
    result["active_client_metadata_source"] = (
        "reported_and_validated" if reported else rule
    )
    result["active_client_rule"] = rule


def _validate_round_history(history: Any, where: str, *, pfin: bool = False) -> dict:
    history = _mapping(history, where)
    if pfin:
        metric_fields = (
            "round_full_text_f1",
            "round_forced_missing_text_f1",
            "round_full_text_accuracy",
            "round_forced_missing_text_accuracy",
            "round_full_text_diversity",
            "round_forced_missing_text_diversity",
        )
    else:
        metric_fields = ("round_f1", "round_acc", "round_div")
    for field in metric_fields:
        values = history.get(field)
        if not isinstance(values, list) or len(values) != ROUNDS:
            _fail(f"{where}.{field}", f"expected {ROUNDS} values")
        for index, value in enumerate(values):
            _probability(value, f"{where}.{field}[{index}]")
    for field in ("round_seconds", "round_peak_mib"):
        values = history.get(field)
        if not isinstance(values, list) or len(values) != ROUNDS:
            _fail(f"{where}.{field}", f"expected {ROUNDS} values")
        minimum = 0.0
        for index, value in enumerate(values):
            _number(value, f"{where}.{field}[{index}]", minimum=minimum)
    return history


def _validate_cost_fields(result: Mapping[str, Any], clients: int, rounds: int, where: str) -> None:
    upload = _integer(
        result.get("upload_bytes_per_client_per_round"),
        f"{where}.upload_bytes_per_client_per_round",
        minimum=1,
    )
    total = _integer(result.get("total_comm_bytes"), f"{where}.total_comm_bytes", minimum=1)
    expected_total = 2 * clients * rounds * upload
    if total != expected_total:
        _fail(f"{where}.total_comm_bytes", f"expected {expected_total}, got {total}")
    trainable = _integer(result.get("trainable_params"), f"{where}.trainable_params", minimum=1)
    if upload != 4 * trainable:
        _fail(
            f"{where}.upload_bytes_per_client_per_round",
            f"expected 4 * trainable_params = {4 * trainable}, got {upload}",
        )
    if "total_params" in result:
        total_parameters = _integer(result["total_params"], f"{where}.total_params", minimum=1)
        if total_parameters < trainable:
            _fail(f"{where}.total_params", "cannot be smaller than trainable_params")


def validate_standard_federated_result(result: Any, clients: int, where: str) -> dict:
    result = _mapping(result, where)
    _validate_metric_bundle(result, where)
    _validate_partition(result, clients, where, require_histogram=True)
    history = _validate_round_history(result.get("history"), f"{where}.history")
    _close(result["f1"], history["round_f1"][-1], f"{where}.f1_vs_history")
    _close(result["accuracy"], history["round_acc"][-1], f"{where}.accuracy_vs_history")
    _close(result["diversity"], history["round_div"][-1], f"{where}.diversity_vs_history")
    _validate_cost_fields(result, clients, ROUNDS, where)
    wall = _number(result.get("wall_seconds"), f"{where}.wall_seconds", minimum=0.0)
    if not math.isclose(wall, sum(history["round_seconds"]), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.wall_seconds", "does not equal the sum of round_seconds")
    peak = _number(result.get("peak_mib"), f"{where}.peak_mib", minimum=0.0)
    if not math.isclose(peak, max(history["round_peak_mib"]), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.peak_mib", "does not equal the maximum round_peak_mib")
    _active_client_metadata(result, clients, where, skip_small_shards=True)
    return result


def validate_pooled_early_abort_result(result: Any, where: str) -> dict:
    """Validate the centralized pooled-data control used only for E3b.

    This diagnostic is deliberately separate from the federated result schema:
    it reports the best validation epoch from one complete 12-epoch pooled
    training history and has no clients, rounds, or communication accounting.
    """

    result = _mapping(result, where)
    expected_result_fields = {
        "f1",
        "accuracy",
        "diversity",
        "epoch",
        "history",
        "wall_seconds",
        "min_diversity",
        "trainable_params",
    }
    if set(result) != expected_result_fields:
        _fail(
            where,
            "field mismatch for pooled early-abort control: "
            f"missing={sorted(expected_result_fields - set(result))} "
            f"extra={sorted(set(result) - expected_result_fields)}",
        )
    _validate_metric_bundle(result, where)
    epoch = _integer(result.get("epoch"), f"{where}.epoch", minimum=1)
    if epoch > 12:
        _fail(f"{where}.epoch", f"best epoch {epoch} is outside the 12-epoch schedule")

    history = _mapping(result.get("history"), f"{where}.history")
    expected_history_fields = {"val_f1", "val_acc", "diversity", "epoch_seconds"}
    if set(history) != expected_history_fields:
        _fail(
            f"{where}.history",
            "field mismatch: "
            f"missing={sorted(expected_history_fields - set(history))} "
            f"extra={sorted(set(history) - expected_history_fields)}",
        )
    for field in ("val_f1", "val_acc", "diversity"):
        values = history.get(field)
        if not isinstance(values, list) or len(values) != 12:
            _fail(f"{where}.history.{field}", "expected exactly 12 pooled-training epochs")
        for index, value in enumerate(values):
            _probability(value, f"{where}.history.{field}[{index}]")
    seconds = history.get("epoch_seconds")
    if not isinstance(seconds, list) or len(seconds) != 12:
        _fail(f"{where}.history.epoch_seconds", "expected exactly 12 epoch timings")
    for index, value in enumerate(seconds):
        if _number(value, f"{where}.history.epoch_seconds[{index}]", minimum=0.0) <= 0.0:
            _fail(f"{where}.history.epoch_seconds[{index}]", "must be positive")

    best_index = epoch - 1
    first_max_index = history["val_f1"].index(max(history["val_f1"]))
    if best_index != first_max_index:
        _fail(
            f"{where}.epoch",
            f"reported best epoch {epoch} does not match first maximum F1 epoch {first_max_index + 1}",
        )
    _close(result["f1"], history["val_f1"][best_index], f"{where}.f1_vs_best_epoch")
    _close(
        result["accuracy"],
        history["val_acc"][best_index],
        f"{where}.accuracy_vs_best_epoch",
    )
    _close(
        result["diversity"],
        history["diversity"][best_index],
        f"{where}.diversity_vs_best_epoch",
    )
    _close(
        result["min_diversity"],
        min(history["diversity"]),
        f"{where}.min_diversity_vs_history",
    )
    wall = _number(result.get("wall_seconds"), f"{where}.wall_seconds", minimum=0.0)
    if not math.isclose(wall, sum(seconds), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.wall_seconds", "does not equal the sum of 12 epoch timings")
    _equal(
        result.get("trainable_params"),
        EXPECTED_MULTIMODAL_TRAINABLE_PARAMS,
        f"{where}.trainable_params",
    )
    return result


def validate_fedmme_result(
    result: Any, clients: int, where: str, expected_epochs: int = 24
) -> dict:
    result = _mapping(result, where)
    _validate_metric_bundle(result, where)
    sizes, _ = _validate_partition(result, clients, where, require_histogram=False)
    if any(size <= 0 for size in sizes):
        _fail(f"{where}.client_sizes", "FedMME-style training requires nonempty shards")
    per_f1 = result.get("per_client_f1")
    if not isinstance(per_f1, list) or len(per_f1) != clients:
        _fail(f"{where}.per_client_f1", f"expected {clients} values")
    for index, value in enumerate(per_f1):
        _probability(value, f"{where}.per_client_f1[{index}]")
    per_seconds = result.get("per_client_seconds")
    if not isinstance(per_seconds, list) or len(per_seconds) != clients:
        _fail(f"{where}.per_client_seconds", f"expected {clients} values")
    for index, value in enumerate(per_seconds):
        _number(value, f"{where}.per_client_seconds[{index}]", minimum=0.0)
    _equal(result.get("local_epochs"), expected_epochs, f"{where}.local_epochs")
    _equal(
        result.get("ensemble"),
        "equal hard plurality; mean-softmax tie break",
        f"{where}.ensemble",
    )
    upload = _integer(result.get("upload_bytes_per_client"), f"{where}.upload_bytes_per_client", minimum=1)
    _equal(
        result.get("total_comm_bytes_upload_only"),
        clients * upload,
        f"{where}.total_comm_bytes_upload_only",
    )
    _equal(
        result.get("total_comm_bytes_bidirectional"),
        2 * clients * upload,
        f"{where}.total_comm_bytes_bidirectional",
    )
    wall = _number(result.get("wall_seconds"), f"{where}.wall_seconds", minimum=0.0)
    if not math.isclose(wall, sum(per_seconds), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.wall_seconds", "does not equal the sum of per_client_seconds")
    _number(result.get("peak_mib"), f"{where}.peak_mib", minimum=0.0)
    _active_client_metadata(result, clients, where, skip_small_shards=False)
    return result


def _validate_weight_rows(rows: Any, where: str, clients: int) -> list[list[float]]:
    if not isinstance(rows, list) or len(rows) != ROUNDS:
        _fail(where, f"expected {ROUNDS} rows")
    normalized: list[list[float]] = []
    for round_id, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != clients:
            _fail(f"{where}[{round_id}]", f"expected {clients} weights")
        weights = [
            _number(value, f"{where}[{round_id}][{client_id}]", minimum=0.0)
            for client_id, value in enumerate(row)
        ]
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            _fail(f"{where}[{round_id}]", f"weights sum to {sum(weights)}, not 1")
        normalized.append(weights)
    return normalized


def validate_pfin_result(result: Any, spec: Mapping[str, Any], where: str) -> dict:
    result = _mapping(result, where)
    clients = spec["clients"]
    mode = spec["pfin_mode"]
    _validate_metric_bundle(result, where)
    for field in (
        "primary_f1",
        "full_text_f1",
        "full_text_accuracy",
        "full_text_diversity",
        "forced_missing_text_f1",
        "forced_missing_text_accuracy",
        "forced_missing_text_diversity",
    ):
        _probability(result.get(field), f"{where}.{field}")
    _close(result["f1"], result["full_text_f1"], f"{where}.f1_vs_full_text_f1")
    _close(result["accuracy"], result["full_text_accuracy"], f"{where}.accuracy_vs_full_text")
    _close(result["diversity"], result["full_text_diversity"], f"{where}.diversity_vs_full_text")
    _close(result["primary_f1"], result["forced_missing_text_f1"], f"{where}.primary_f1")
    for bundle_name, prefix in (
        ("full_text_metrics", "full_text"),
        ("forced_missing_text_metrics", "forced_missing_text"),
    ):
        bundle = _mapping(result.get(bundle_name), f"{where}.{bundle_name}")
        _validate_metric_bundle(bundle, f"{where}.{bundle_name}")
        for field in ("f1", "accuracy", "diversity"):
            _close(
                bundle[field],
                result[f"{prefix}_{field}"],
                f"{where}.{bundle_name}.{field}_consistency",
            )

    _equal(result.get("mode"), mode, f"{where}.mode")
    if result.get("matched_adaptation") is not True:
        _fail(f"{where}.matched_adaptation", "must be exactly true")
    if not isinstance(result.get("method"), str) or not result["method"]:
        _fail(f"{where}.method", "must be a nonempty label")
    if not isinstance(result.get("fusion_substitution"), str) or not result["fusion_substitution"]:
        _fail(f"{where}.fusion_substitution", "must disclose the matched substitution")

    sizes, _ = _validate_partition(result, clients, where, require_histogram=True)
    if any(size <= 0 for size in sizes):
        _fail(f"{where}.client_sizes", "P-FIN stress-test shards must be nonempty")
    mask = result.get("client_modality_mask")
    if not isinstance(mask, list) or len(mask) != clients:
        _fail(f"{where}.client_modality_mask", f"expected {clients} entries")
    if any(value not in {"multimodal", "image_only"} for value in mask):
        _fail(f"{where}.client_modality_mask", "contains an unknown modality label")
    if mask.count("multimodal") != 3 or mask.count("image_only") != 2:
        _fail(f"{where}.client_modality_mask", "expected exactly 3 multimodal and 2 image-only clients")
    multimodal = [index for index, value in enumerate(mask) if value == "multimodal"]
    image_only = [index for index, value in enumerate(mask) if value == "image_only"]
    _equal(result.get("multimodal_clients"), multimodal, f"{where}.multimodal_clients")
    _equal(result.get("image_only_clients"), image_only, f"{where}.image_only_clients")

    history = _validate_round_history(result.get("history"), f"{where}.history", pfin=True)
    _close(result["full_text_f1"], history["round_full_text_f1"][-1], f"{where}.full_f1_vs_history")
    _close(
        result["forced_missing_text_f1"],
        history["round_forced_missing_text_f1"][-1],
        f"{where}.missing_f1_vs_history",
    )
    uncertainties = history.get("client_mean_uncertainty")
    if not isinstance(uncertainties, list) or len(uncertainties) != ROUNDS:
        _fail(f"{where}.history.client_mean_uncertainty", f"expected {ROUNDS} rows")
    for round_id, row in enumerate(uncertainties):
        if not isinstance(row, list) or len(row) != clients:
            _fail(f"{where}.history.client_mean_uncertainty[{round_id}]", f"expected {clients} values")
        for client_id, value in enumerate(row):
            _number(
                value,
                f"{where}.history.client_mean_uncertainty[{round_id}][{client_id}]",
                minimum=0.0,
            )
    aggregation = _validate_weight_rows(
        history.get("aggregation_weights"), f"{where}.history.aggregation_weights", clients
    )
    fin_aggregation = _validate_weight_rows(
        history.get("fin_aggregation_weights"),
        f"{where}.history.fin_aggregation_weights",
        clients,
    )
    data_weights = [size / sum(sizes) for size in sizes]
    for round_id, weights in enumerate(aggregation):
        if mode == "probabilistic_uq":
            logits = [-float(value) / 0.2 for value in uncertainties[round_id]]
            offset = max(logits)
            exponentials = [math.exp(value - offset) for value in logits]
            denominator = sum(exponentials)
            confidence_weights = [value / denominator for value in exponentials]
            expected_weights = [
                0.4 * data_weight + 0.6 * confidence_weight
                for data_weight, confidence_weight in zip(
                    data_weights, confidence_weights
                )
            ]
            normalization = sum(expected_weights)
            expected_weights = [value / normalization for value in expected_weights]
        else:
            expected_weights = data_weights
        for client_id, (actual, expected) in enumerate(zip(weights, expected_weights)):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                _fail(
                    f"{where}.history.aggregation_weights[{round_id}][{client_id}]",
                    f"does not match the recomputed {'Fed-UQ-Avg' if mode == 'probabilistic_uq' else 'FedAvg'} weight",
                )
    for round_id, weights in enumerate(fin_aggregation):
        expected_fin = [
            aggregation[round_id][client_id] if client_id in multimodal else 0.0
            for client_id in range(clients)
        ]
        normalization = sum(expected_fin)
        expected_fin = [value / normalization for value in expected_fin]
        for client_id, (actual, expected) in enumerate(zip(weights, expected_fin)):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                _fail(
                    f"{where}.history.fin_aggregation_weights[{round_id}][{client_id}]",
                    "does not match aggregation weights renormalized over paired clients",
                )

    _equal(result.get("alpha"), spec["alpha"], f"{where}.alpha")
    _equal(result.get("num_clients"), clients, f"{where}.num_clients")
    _equal(result.get("rounds"), ROUNDS, f"{where}.rounds")
    _equal(result.get("local_epochs"), LOCAL_EPOCHS, f"{where}.local_epochs")
    _equal(result.get("seed"), spec["seed"], f"{where}.seed")
    _equal(result.get("feature_dim"), 256, f"{where}.feature_dim")
    expected_beta = 0.5 if "probabilistic" in mode else None
    expected_rho = 0.6 if mode == "probabilistic_uq" else None
    expected_temperature = 0.2 if mode == "probabilistic_uq" else None
    _equal(result.get("beta"), expected_beta, f"{where}.beta")
    _equal(result.get("rho"), expected_rho, f"{where}.rho")
    _equal(result.get("temperature"), expected_temperature, f"{where}.temperature")
    _number(result.get("fin_loss_weight"), f"{where}.fin_loss_weight", minimum=0.0)

    upload = _integer(
        result.get("upload_bytes_per_client_per_round"),
        f"{where}.upload_bytes_per_client_per_round",
        minimum=1,
    )
    trainable = _integer(result.get("trainable_params"), f"{where}.trainable_params", minimum=1)
    _equal(upload, 4 * trainable, f"{where}.upload_bytes_per_client_per_round")
    total_parameters = _integer(result.get("total_params"), f"{where}.total_params", minimum=1)
    if total_parameters < trainable:
        _fail(f"{where}.total_params", "cannot be smaller than trainable_params")
    expected_uq_bytes = 4 * clients * ROUNDS if mode == "probabilistic_uq" else 0
    _equal(result.get("uq_scalar_bytes"), expected_uq_bytes, f"{where}.uq_scalar_bytes")
    expected_comm = 2 * clients * ROUNDS * upload + expected_uq_bytes
    _equal(result.get("total_comm_bytes"), expected_comm, f"{where}.total_comm_bytes")
    wall = _number(result.get("wall_seconds"), f"{where}.wall_seconds", minimum=0.0)
    if not math.isclose(wall, sum(history["round_seconds"]), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.wall_seconds", "does not equal the sum of round_seconds")
    peak = _number(result.get("peak_mib"), f"{where}.peak_mib", minimum=0.0)
    if not math.isclose(peak, max(history["round_peak_mib"]), rel_tol=1e-9, abs_tol=1e-6):
        _fail(f"{where}.peak_mib", "does not equal maximum round_peak_mib")
    if "peak_gpu_allocated_mib" in result:
        _close(result["peak_gpu_allocated_mib"], peak, f"{where}.peak_gpu_allocated_mib")
    _active_client_metadata(result, clients, where, skip_small_shards=False)
    return result


def _validate_new_protocol(payload: Mapping[str, Any], spec: Mapping[str, Any], where: str) -> None:
    parameters = _mapping(payload.get("parameters"), f"{where}.parameters")
    if set(parameters) != PARAMETER_FIELDS:
        _fail(
            f"{where}.parameters",
            f"field set mismatch: expected {sorted(PARAMETER_FIELDS)}, got {sorted(parameters)}",
        )
    for field in PARAMETER_FIELDS - {"deterministic"}:
        _equal(parameters[field], spec[field], f"{where}.parameters.{field}")
    _explicit_false(parameters, "deterministic", f"{where}.parameters")

    protocol = _mapping(payload.get("protocol"), f"{where}.protocol")
    task = spec["task"]
    expected_rounds = 1 if task == "fedmme" else ROUNDS
    # FedMME runs at either the matched 24-epoch budget or the source paper's
    # native 100; the manifest pins which one each record must carry.
    expected_local_epochs = (
        spec.get("local_epochs", 24) if task == "fedmme" else LOCAL_EPOCHS
    )
    expected_schedule = (
        "one upload per independently trained client model"
        if task == "fedmme"
        else "one download and upload per client per round"
    )
    expected_initialization = (
        "same DistilBERT/ViT configurations with random encoder weights and random heads"
        if task == "init"
        else "public encoders + random task heads"
    )
    expected_values = {
        "rounds": expected_rounds,
        "local_epochs": expected_local_epochs,
        "communication_schedule": expected_schedule,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "precision": "FP32",
        "initialization": expected_initialization,
        "data": "controlled_v2_real4_synthetic_covid_template_text",
    }
    for field, expected in expected_values.items():
        _equal(protocol.get(field), expected, f"{where}.protocol.{field}")


def validate_new_payload(payload: Any, spec: Mapping[str, Any], path: Path) -> dict:
    where = str(path)
    payload = _mapping(payload, where)
    _equal(payload.get("task"), spec["task"], f"{where}.task")
    expected_key = next(key for key, value in expected_record_specs().items() if value == spec)
    _equal(payload.get("key"), expected_key, f"{where}.key")
    _validate_new_protocol(payload, spec, where)

    provenance = _mapping(payload.get("provenance"), f"{where}.provenance")
    for field, expected in (
        ("base_model_sha256", EXPECTED_BASE),
        ("data_cache_sha256", EXPECTED_CACHE),
        ("core_runner_sha256", EXPECTED_CORE_RUNNER),
        (
            "reviewer_runner_sha256",
            spec.get("runner_sha256", EXPECTED_REVIEWER_RUNNER),
        ),
    ):
        _equal(provenance.get(field), expected, f"{where}.provenance.{field}")
    _explicit_false(
        provenance,
        "deterministic_algorithms_enforced",
        f"{where}.provenance",
    )
    if spec["task"] == "pfin":
        _equal(
            provenance.get("pfin_helper_sha256"),
            EXPECTED_PFIN_HELPER,
            f"{where}.provenance.pfin_helper_sha256",
        )
    if not isinstance(provenance.get("gpu"), str) or not provenance["gpu"]:
        _fail(f"{where}.provenance.gpu", "must be a nonempty string")
    if not isinstance(provenance.get("torch"), str) or not provenance["torch"]:
        _fail(f"{where}.provenance.torch", "must be a nonempty string")

    result = payload.get("result")
    task = spec["task"]
    if task in {"fusion", "grid", "init"}:
        result = validate_standard_federated_result(result, spec["clients"], f"{where}.result")
    elif task == "fedmme":
        result = validate_fedmme_result(
            result, spec["clients"], f"{where}.result", spec.get("local_epochs", 24)
        )
    elif task == "pfin":
        result = validate_pfin_result(result, spec, f"{where}.result")
    else:
        raise AssertionError(task)
    finite_result_tree(result, f"{where}.result")
    payload["result"] = result
    return payload


def load_new(directory: Path) -> dict[str, dict]:
    specs = expected_record_specs()
    records: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(str(path), f"cannot read valid JSON: {exc}")
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str):
            _fail(str(path), "missing string key")
        if key not in specs:
            _fail(str(path), f"unexpected reviewer key {key!r}")
        if key in records:
            _fail(str(path), f"duplicate reviewer key {key!r}")
        records[key] = validate_new_payload(payload, specs[key], path)

    missing = set(specs) - set(records)
    extra = set(records) - set(specs)
    if missing or extra or len(records) != len(specs):
        raise ValueError(
            "reviewer result-key mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)} count={len(records)}"
        )
    return records


def _legacy_meta(legacy: Mapping[str, Any]) -> dict:
    meta = _mapping(legacy.get("_meta"), "legacy._meta")
    expected_top = {
        "tier": "standard",
        "device": "cuda",
        "gpu": "NVIDIA H100 NVL",
        "n_train": N_TRAIN,
        "n_val": 600,
        "data_protocol": "controlled_v2_real4_synthetic_covid_template_text",
        "fl_initialization": "public_pretrained_encoders_random_task_heads",
        "training_precision": "fp32_tensors_no_amp",
        "runtime_environment": "shared_gpu_server_with_uncontrolled_contention",
        "executed_model_source_sha256": EXPECTED_BASE,
        "data_cache_sha256": EXPECTED_CACHE,
        "experiment_runner_sha256": EXPECTED_CORE_RUNNER,
    }
    for field, expected in expected_top.items():
        _equal(meta.get(field), expected, f"legacy._meta.{field}")
    _explicit_false(meta, "deterministic_algorithms_enforced", "legacy._meta")
    tier = _mapping(meta.get("tier_config"), "legacy._meta.tier_config")
    expected_tier = {
        "samples_per_class": 600,
        "batch_size": 16,
        "fed_rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "seeds": [0, 1],
        "alphas": [0.1, 0.3, 0.5, 1.0, 5.0],
        "client_counts": [3, 5, 10, 20],
        "fusion_seeds": [0, 1],
        "text_model": "distilbert-base-uncased",
        "vision_model": "google/vit-base-patch16-224",
        "central_epochs": 12,
    }
    for field, expected in expected_tier.items():
        _equal(tier.get(field), expected, f"legacy._meta.tier_config.{field}")
    if not isinstance(meta.get("torch"), str) or not meta["torch"]:
        _fail("legacy._meta.torch", "must be a nonempty string")
    return meta


def _exact_group_keys(group: Any, expected: set[str], where: str) -> dict:
    group = _mapping(group, where)
    if set(group) != expected:
        _fail(
            where,
            f"key mismatch: missing={sorted(expected - set(group))} extra={sorted(set(group) - expected)}",
        )
    return group


def _partition_object(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "client_sizes": result.get("client_sizes"),
        "client_class_hist": result.get("client_class_hist"),
    }


def _assert_partition_equal(left: Mapping[str, Any], right: Mapping[str, Any], where: str) -> None:
    if _partition_object(left) != _partition_object(right):
        _fail(where, "client sizes/class histograms differ despite a matched partition contract")


def _validate_local_only(result: Any, where: str) -> dict:
    result = _mapping(result, where)
    values = result.get("per_client_f1")
    best_values = result.get("per_client_best_epoch_f1")
    for field, sequence in (("per_client_f1", values), ("per_client_best_epoch_f1", best_values)):
        if not isinstance(sequence, list) or len(sequence) != 5:
            _fail(f"{where}.{field}", "expected five client values")
        for index, value in enumerate(sequence):
            _probability(value, f"{where}.{field}[{index}]")
    for field in ("mean_f1", "best_f1", "worst_f1"):
        _probability(result.get(field), f"{where}.{field}")
    _number(result.get("std_f1"), f"{where}.std_f1", minimum=0.0)
    _close(result["mean_f1"], mean(values), f"{where}.mean_f1")
    _close(result["best_f1"], max(values), f"{where}.best_f1")
    _close(result["worst_f1"], min(values), f"{where}.worst_f1")
    _equal(result.get("num_clients"), 5, f"{where}.num_clients")
    _equal(result.get("alpha"), 0.1, f"{where}.alpha")
    _equal(result.get("epochs_per_client"), 24, f"{where}.epochs_per_client")
    _explicit_false(result, "early_abort", where)
    result["nominal_clients"] = 5
    result["active_clients"] = 5
    result["skipped_client_ids"] = []
    result["active_client_metadata_source"] = "derived from num_clients; all local-only models trained"
    result["active_client_rule"] = "every local-only client model participates"
    return result


def load_corrected_retrieval(path: Path) -> dict:
    """Validate the leakage-free train-fit/validation-query retrieval record."""

    where = f"corrected retrieval {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read corrected retrieval record {path}: {exc}") from exc
    payload = _mapping(payload, where)
    finite_result_tree(payload, where)
    expected_top = {"schema_version", "task", "key", "protocol", "provenance", "result"}
    if set(payload) != expected_top:
        _fail(
            where,
            f"field mismatch: missing={sorted(expected_top - set(payload))} "
            f"extra={sorted(set(payload) - expected_top)}",
        )
    _equal(payload.get("schema_version"), 1, f"{where}.schema_version")
    _equal(
        payload.get("task"),
        "heldout_tfidf_faiss_retrieval",
        f"{where}.task",
    )
    _equal(
        payload.get("key"),
        "fit=train_only|index=train|query=validation|top_k=5",
        f"{where}.key",
    )

    protocol = _mapping(payload.get("protocol"), f"{where}.protocol")
    expected_protocol = {
        "fit_scope": "training notes only",
        "index_scope": "2,400 training notes",
        "query_scope": "600 validation notes transformed after fit",
        "labels_used_for": "evaluation only",
        "vectorizer": "TfidfVectorizer(max_features=4096)",
        "normalization": "L2",
        "search": "exact FAISS IndexFlatIP",
        "similarity_sd_ddof": 0,
    }
    if protocol != expected_protocol:
        _fail(f"{where}.protocol", f"expected {expected_protocol!r}, got {protocol!r}")

    provenance = _mapping(payload.get("provenance"), f"{where}.provenance")
    _equal(
        provenance.get("data_cache_sha256"),
        EXPECTED_CACHE,
        f"{where}.provenance.data_cache_sha256",
    )
    _equal(
        provenance.get("runner_sha256"),
        EXPECTED_RAG_RUNNER,
        f"{where}.provenance.runner_sha256",
    )
    for field in ("numpy", "scikit_learn", "faiss"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            _fail(f"{where}.provenance.{field}", "must be a nonempty string")

    result = _mapping(payload.get("result"), f"{where}.result")
    expected_result_fields = {
        "n_queries",
        "corpus_size",
        "top_k",
        "per_condition",
        "condition_macro_top1_accuracy",
        "condition_macro_precision_at_k",
        "condition_macro_mean_top1_similarity",
        "overall_top1_accuracy",
        "overall_mean_top1_similarity",
        "micro_top1_accuracy",
        "micro_precision_at_k",
        "micro_mean_top1_similarity",
    }
    if set(result) != expected_result_fields:
        _fail(
            f"{where}.result",
            f"field mismatch: missing={sorted(expected_result_fields - set(result))} "
            f"extra={sorted(set(result) - expected_result_fields)}",
        )
    _equal(result.get("n_queries"), 600, f"{where}.result.n_queries")
    _equal(result.get("corpus_size"), 2400, f"{where}.result.corpus_size")
    _equal(result.get("top_k"), 5, f"{where}.result.top_k")

    per_condition = _mapping(result.get("per_condition"), f"{where}.result.per_condition")
    if set(per_condition) != set(RETRIEVAL_CONDITIONS):
        _fail(
            f"{where}.result.per_condition",
            f"expected exactly {list(RETRIEVAL_CONDITIONS)!r}, got {list(per_condition)!r}",
        )
    expected_condition_fields = {
        "n",
        "top1_accuracy",
        "precision_at_k",
        "mean_top1_similarity",
        "std_top1_similarity",
    }
    for condition in RETRIEVAL_CONDITIONS:
        row = _mapping(
            per_condition[condition],
            f"{where}.result.per_condition.{condition}",
        )
        if set(row) != expected_condition_fields:
            _fail(
                f"{where}.result.per_condition.{condition}",
                f"expected fields {sorted(expected_condition_fields)}, got {sorted(row)}",
            )
        _equal(row.get("n"), RETRIEVAL_COUNTS[condition], f"{where}.{condition}.n")
        for field in ("top1_accuracy", "precision_at_k", "mean_top1_similarity"):
            _probability(row.get(field), f"{where}.{condition}.{field}")
        std = _number(
            row.get("std_top1_similarity"),
            f"{where}.{condition}.std_top1_similarity",
            minimum=0.0,
        )
        if std > 1.0:
            _fail(f"{where}.{condition}.std_top1_similarity", "must not exceed 1")

    rows = [per_condition[condition] for condition in RETRIEVAL_CONDITIONS]
    macro_top1 = mean(row["top1_accuracy"] for row in rows)
    macro_precision = mean(row["precision_at_k"] for row in rows)
    macro_similarity = mean(row["mean_top1_similarity"] for row in rows)
    _close(
        result.get("condition_macro_top1_accuracy"),
        macro_top1,
        f"{where}.result.condition_macro_top1_accuracy",
    )
    _close(
        result.get("condition_macro_precision_at_k"),
        macro_precision,
        f"{where}.result.condition_macro_precision_at_k",
    )
    _close(
        result.get("condition_macro_mean_top1_similarity"),
        macro_similarity,
        f"{where}.result.condition_macro_mean_top1_similarity",
    )
    _close(result.get("overall_top1_accuracy"), macro_top1, f"{where}.result.overall_top1_accuracy")
    _close(
        result.get("overall_mean_top1_similarity"),
        macro_similarity,
        f"{where}.result.overall_mean_top1_similarity",
    )
    count = sum(row["n"] for row in rows)
    if count != 600:
        _fail(f"{where}.result.per_condition", f"counts sum to {count}, not 600")
    weighted = lambda field: sum(row["n"] * row[field] for row in rows) / count
    _close(
        result.get("micro_top1_accuracy"),
        weighted("top1_accuracy"),
        f"{where}.result.micro_top1_accuracy",
    )
    _close(
        result.get("micro_precision_at_k"),
        weighted("precision_at_k"),
        f"{where}.result.micro_precision_at_k",
    )
    _close(
        result.get("micro_mean_top1_similarity"),
        weighted("mean_top1_similarity"),
        f"{where}.result.micro_mean_top1_similarity",
    )
    return payload


def validate_legacy(legacy: Any) -> dict:
    legacy = _mapping(legacy, "legacy")
    _legacy_meta(legacy)
    finite_result_tree(legacy, "legacy")

    e1_keys = {
        f"alpha={alpha}|seed={seed}"
        for alpha in (0.1, 0.3, 0.5, 1.0, 5.0)
        for seed in (0, 1)
    }
    e2_keys = {f"K={clients}|seed={seed}" for clients in (3, 5, 10, 20) for seed in (0, 1)}
    e3_keys = {
        f"{arm}|alpha={alpha}|seed={seed}"
        for arm in ("full", "no_balanced", "no_diversity", "neither")
        for alpha in (0.1, 1.0)
        for seed in (0, 1)
    }
    e3b_keys = {
        f"early_abort={enabled}|seed={seed}"
        for enabled in (True, False)
        for seed in (0, 1)
    }
    e4_keys = {f"warm_start={value}|seed={seed}" for value in (True, False) for seed in (0, 1)}
    e6_keys = {"Fed-LLM", "Fed-ViT", "Fed-VLM"}
    e8_keys = {
        f"{algorithm}|alpha=0.1|seed={seed}"
        for algorithm in ("fedavg", "fedprox", "scaffold")
        for seed in (0, 1)
    } | {"local_only|alpha=0.1|seed=0"}

    e1 = _exact_group_keys(legacy.get("E1_alpha_sweep"), e1_keys, "legacy.E1_alpha_sweep")
    e2 = _exact_group_keys(legacy.get("E2_client_sweep"), e2_keys, "legacy.E2_client_sweep")
    e3 = _exact_group_keys(legacy.get("E3_anticollapse"), e3_keys, "legacy.E3_anticollapse")
    e3b = _exact_group_keys(
        legacy.get("E3b_early_abort"), e3b_keys, "legacy.E3b_early_abort"
    )
    e4 = _exact_group_keys(legacy.get("E4_warmstart"), e4_keys, "legacy.E4_warmstart")
    e6 = _exact_group_keys(legacy.get("E6_cost"), e6_keys, "legacy.E6_cost")
    e8 = _exact_group_keys(legacy.get("E8_baselines"), e8_keys, "legacy.E8_baselines")
    for required_group in ("E7_rag",):
        _mapping(legacy.get(required_group), f"legacy.{required_group}")

    for key, result in e1.items():
        validate_standard_federated_result(result, 5, f"legacy.E1_alpha_sweep.{key}")
    for key, result in e2.items():
        clients = int(key.split("|")[0].split("=")[1])
        validate_standard_federated_result(result, clients, f"legacy.E2_client_sweep.{key}")
    for key, result in e3.items():
        validate_standard_federated_result(result, 5, f"legacy.E3_anticollapse.{key}")
    for key, result in e3b.items():
        validate_pooled_early_abort_result(result, f"legacy.E3b_early_abort.{key}")
    for key, result in e4.items():
        validate_standard_federated_result(result, 5, f"legacy.E4_warmstart.{key}")
    for key, result in e6.items():
        validate_standard_federated_result(result, 5, f"legacy.E6_cost.{key}")
    for key, result in e8.items():
        if key.startswith("local_only|"):
            _validate_local_only(result, f"legacy.E8_baselines.{key}")
        else:
            validate_standard_federated_result(result, 5, f"legacy.E8_baselines.{key}")
            algorithm = key.split("|")[0]
            _equal(result.get("algorithm"), algorithm, f"legacy.E8_baselines.{key}.algorithm")

    # These comparisons certify that rows later described as reruns really did
    # use the same realized shards, not merely the same nominal alpha and seed.
    for seed in (0, 1):
        for alpha in (0.1, 1.0):
            reference = e1[f"alpha={alpha}|seed={seed}"]
            for arm in ("full", "no_balanced", "no_diversity", "neither"):
                _assert_partition_equal(
                    reference,
                    e3[f"{arm}|alpha={alpha}|seed={seed}"],
                    f"legacy partition E1/E3 alpha={alpha} seed={seed} arm={arm}",
                )
        alpha_one = e1[f"alpha=1.0|seed={seed}"]
        _assert_partition_equal(
            alpha_one,
            e2[f"K=5|seed={seed}"],
            f"legacy partition E1/E2 alpha=1.0 seed={seed}",
        )
        for warm in (True, False):
            _assert_partition_equal(
                alpha_one,
                e4[f"warm_start={warm}|seed={seed}"],
                f"legacy partition E1/E4 alpha=1.0 seed={seed} warm={warm}",
            )
        alpha_point_one = e1[f"alpha=0.1|seed={seed}"]
        for algorithm in ("fedavg", "fedprox", "scaffold"):
            row = e8[f"{algorithm}|alpha=0.1|seed={seed}"]
            _assert_partition_equal(
                alpha_point_one,
                row,
                f"legacy partition E1/E8 alpha=0.1 seed={seed} algorithm={algorithm}",
            )
            _equal(
                row.get("partition_reference"),
                f"E1_alpha_sweep/alpha=0.1|seed={seed}",
                f"legacy.E8_baselines.{algorithm}.partition_reference",
            )
    for modality, row in e6.items():
        _assert_partition_equal(
            e1["alpha=1.0|seed=0"],
            row,
            f"legacy partition E1/E6 seed=0 modality={modality}",
        )
    return legacy


def _result(records: Mapping[str, dict], key: str) -> dict:
    return records[key]["result"]


def validate_cross_record_partitions(legacy: Mapping[str, Any], records: Mapping[str, dict]) -> None:
    for seed in (0, 1):
        reference = legacy["E1_alpha_sweep"][f"alpha=0.1|seed={seed}"]
        for fusion in FUSIONS:
            key = f"fusion={fusion}|alpha=0.1|K=5|seed={seed}"
            _assert_partition_equal(reference, _result(records, key), f"partition {key}")
        pfin_masks = []
        for mode in PFIN_MODES:
            key = f"pfin={mode}|alpha=0.1|K=5|seed={seed}"
            row = _result(records, key)
            _assert_partition_equal(reference, row, f"partition {key}")
            pfin_masks.append(row["client_modality_mask"])
        if any(mask != pfin_masks[0] for mask in pfin_masks[1:]):
            _fail(f"P-FIN seed={seed}", "modality masks differ across method arms")
        fedmme_key = f"fedmme_style|alpha=0.1|K=5|seed={seed}"
        if _result(records, fedmme_key)["client_sizes"] != reference["client_sizes"]:
            _fail(f"partition {fedmme_key}", "client sizes differ from the matched alpha=0.1 partition")

        init_key = f"init=random|alpha=1.0|K=5|seed={seed}"
        alpha_one = legacy["E1_alpha_sweep"][f"alpha=1.0|seed={seed}"]
        _assert_partition_equal(alpha_one, _result(records, init_key), f"partition {init_key}")


def _resolve_grid_source(
    legacy: Mapping[str, Any], records: Mapping[str, dict], alpha: float, clients: int, seed: int
) -> tuple[dict, dict[str, Any]]:
    new_key = f"alpha={alpha}|K={clients}|seed={seed}"
    if new_key in records:
        return _result(records, new_key), {
            "kind": "reviewer_completion",
            "record_key": new_key,
            "base_model_sha256": EXPECTED_BASE,
            "data_cache_sha256": EXPECTED_CACHE,
            "core_runner_sha256": EXPECTED_CORE_RUNNER,
            "reviewer_runner_sha256": EXPECTED_REVIEWER_RUNNER,
            "protocol_validated": True,
        }
    if clients == 5:
        legacy_key = f"alpha={alpha}|seed={seed}"
        row = legacy["E1_alpha_sweep"][legacy_key]
        return row, {
            "kind": "validated_legacy",
            "suite": "E1_alpha_sweep",
            "record_key": legacy_key,
            "selection_reason": "K=5 is resolved explicitly from the alpha-sweep record",
            "base_model_sha256": EXPECTED_BASE,
            "data_cache_sha256": EXPECTED_CACHE,
            "core_runner_sha256": EXPECTED_CORE_RUNNER,
            "legacy_meta_and_protocol_validated": True,
        }
    if alpha == 1.0 and clients in (3, 10, 20):
        legacy_key = f"K={clients}|seed={seed}"
        row = legacy["E2_client_sweep"][legacy_key]
        return row, {
            "kind": "validated_legacy",
            "suite": "E2_client_sweep",
            "record_key": legacy_key,
            "selection_reason": "alpha=1 client-count sweep",
            "base_model_sha256": EXPECTED_BASE,
            "data_cache_sha256": EXPECTED_CACHE,
            "core_runner_sha256": EXPECTED_CORE_RUNNER,
            "legacy_meta_and_protocol_validated": True,
        }
    raise KeyError((alpha, clients, seed))


def _assert_exact_duplicate_contract(rows: Mapping[str, Mapping[str, Any]], where: str) -> None:
    items = list(rows.items())
    reference_name, reference = items[0]
    exact_fields = (
        "client_sizes",
        "client_class_hist",
        "trainable_params",
        "upload_bytes_per_client_per_round",
        "total_comm_bytes",
    )
    for name, row in items[1:]:
        for field in exact_fields:
            if row.get(field) != reference.get(field):
                _fail(
                    where,
                    f"{name}.{field} differs from {reference_name}.{field}; not an exact matched rerun",
                )


def _duplicate_group_audit(
    *, alpha: float, seed: int, rows: Mapping[str, Mapping[str, Any]], note: str | None = None
) -> dict[str, Any]:
    where = f"rerun audit alpha={alpha} K=5 seed={seed}"
    _assert_exact_duplicate_contract(rows, where)
    partition = _partition_object(next(iter(rows.values())))
    summaries = {
        name: {
            "f1": row["f1"],
            "accuracy": row["accuracy"],
            "diversity": row["diversity"],
            "active_clients": row["active_clients"],
            "wall_seconds": row["wall_seconds"],
            "peak_mib": row["peak_mib"],
        }
        for name, row in rows.items()
    }
    pairwise = []
    for (left_name, left), (right_name, right) in itertools.combinations(rows.items(), 2):
        pairwise.append(
            {
                "left": left_name,
                "right": right_name,
                "absolute_f1_difference": abs(left["f1"] - right["f1"]),
                "absolute_accuracy_difference": abs(left["accuracy"] - right["accuracy"]),
                "absolute_diversity_difference": abs(left["diversity"] - right["diversity"]),
            }
        )
    f1_values = [row["f1"] for row in rows.values()]
    audit = {
        "configuration": {
            "alpha": alpha,
            "clients": 5,
            "seed": seed,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "fusion": "concat",
            "initialization": "public pretrained encoders + random task heads",
            "balanced_local_sampling": True,
            "diversity_regularizer": True,
            "precision": "FP32 without enforced deterministic algorithms",
            "base_model_sha256": EXPECTED_BASE,
            "data_cache_sha256": EXPECTED_CACHE,
            "core_runner_sha256": EXPECTED_CORE_RUNNER,
        },
        "exact_contract_fields_verified": [
            "realized client sizes and class histograms",
            "trainable parameter count",
            "per-client-per-round payload",
            "total bidirectional communication",
        ],
        "partition_sha256": canonical_sha256(partition),
        "records": summaries,
        "pairwise_absolute_differences": pairwise,
        "f1_min": min(f1_values),
        "f1_max": max(f1_values),
        "f1_span": max(f1_values) - min(f1_values),
        "interpretation": (
            "Independent exact-configuration reruns; outcome differences are retained as observed "
            "CUDA/non-bitwise-deterministic variability, not silently cherry-picked."
        ),
    }
    if note is not None:
        audit["scope_note"] = note
    return audit


def build_rerun_audit(legacy: Mapping[str, Any], records: Mapping[str, dict]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for seed in (0, 1):
        rows_point_one = {
            "legacy/E1_alpha_sweep": legacy["E1_alpha_sweep"][f"alpha=0.1|seed={seed}"],
            "legacy/E3_anticollapse/full": legacy["E3_anticollapse"][f"full|alpha=0.1|seed={seed}"],
            "legacy/E8_baselines/fedavg": legacy["E8_baselines"][f"fedavg|alpha=0.1|seed={seed}"],
            "reviewer/fusion/concat": _result(
                records, f"fusion=concat|alpha=0.1|K=5|seed={seed}"
            ),
        }
        audit[f"alpha=0.1|K=5|seed={seed}"] = _duplicate_group_audit(
            alpha=0.1, seed=seed, rows=rows_point_one
        )

        rows_one = {
            "legacy/E1_alpha_sweep": legacy["E1_alpha_sweep"][f"alpha=1.0|seed={seed}"],
            "legacy/E2_client_sweep/K=5": legacy["E2_client_sweep"][f"K=5|seed={seed}"],
            "legacy/E4_warmstart/cold": legacy["E4_warmstart"][
                f"warm_start=False|seed={seed}"
            ],
        }
        note = None
        if seed == 0:
            rows_one["legacy/E6_cost/Fed-VLM"] = legacy["E6_cost"]["Fed-VLM"]
        else:
            note = "E6 cost profiling contains only its declared seed-0 Fed-VLM run."
        audit[f"alpha=1.0|K=5|seed={seed}"] = _duplicate_group_audit(
            alpha=1.0, seed=seed, rows=rows_one, note=note
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--rag-corrected", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        legacy_raw = json.loads(args.legacy.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read legacy result store {args.legacy}: {exc}") from exc
    legacy = validate_legacy(copy.deepcopy(legacy_raw))
    corrected_retrieval = load_corrected_retrieval(args.rag_corrected)
    records = load_new(args.new_dir)
    validate_cross_record_partitions(legacy, records)

    grid: dict[str, dict] = {}
    for alpha in (0.1, 1.0, 5.0):
        for clients in (3, 5, 10, 20):
            for seed in (0, 1):
                key = f"alpha={alpha}|K={clients}|seed={seed}"
                result, source = _resolve_grid_source(legacy, records, alpha, clients, seed)
                row = copy.deepcopy(result)
                row["source"] = (
                    f"reviewer_completion:{source['record_key']}"
                    if source["kind"] == "reviewer_completion"
                    else f"validated_legacy:{source['suite']}/{source['record_key']}"
                )
                row["source_provenance"] = source
                grid[key] = row

    for key, row in grid.items():
        _equal(
            row.get("trainable_params"),
            EXPECTED_MULTIMODAL_TRAINABLE_PARAMS,
            f"merged.grid.{key}.trainable_params",
        )
        _equal(
            row.get("upload_bytes_per_client_per_round"),
            4 * EXPECTED_MULTIMODAL_TRAINABLE_PARAMS,
            f"merged.grid.{key}.upload_bytes_per_client_per_round",
        )

    fusion = {
        key: copy.deepcopy(payload["result"])
        for key, payload in records.items()
        if key.startswith("fusion=")
    }
    fedmme = {
        key: copy.deepcopy(payload["result"])
        for key, payload in records.items()
        if key.startswith("fedmme_style|")
    }
    fedmme_native = {
        key: copy.deepcopy(payload["result"])
        for key, payload in records.items()
        if key.startswith("fedmme_native|")
    }
    pfin = {
        key: copy.deepcopy(payload["result"])
        for key, payload in records.items()
        if key.startswith("pfin=")
    }
    random_init = {
        key: copy.deepcopy(payload["result"])
        for key, payload in records.items()
        if key.startswith("init=random|")
    }

    rerun = build_rerun_audit(legacy, records)
    merged = {
        "_meta": {
            "legacy_path": str(args.legacy),
            "legacy_sha256": sha256(args.legacy),
            "new_record_count": len(records),
            "base_model_sha256": EXPECTED_BASE,
            "data_cache_sha256": EXPECTED_CACHE,
            "core_runner_sha256": EXPECTED_CORE_RUNNER,
            "reviewer_runner_sha256": EXPECTED_REVIEWER_RUNNER,
            "reviewer_runner_sha256_native_budget": (
                EXPECTED_REVIEWER_RUNNER_NATIVE_BUDGET
            ),
            "pfin_helper_sha256": EXPECTED_PFIN_HELPER,
            "retrieval_runner_sha256": EXPECTED_RAG_RUNNER,
            "corrected_retrieval_path": str(args.rag_corrected),
            "corrected_retrieval_sha256": sha256(args.rag_corrected),
            "legacy_meta_and_protocol_validated": True,
            "new_record_protocols_validated": True,
            "corrected_retrieval_protocol_validated": True,
            "grid_model_state_invariants_validated": True,
            "deterministic_algorithms_enforced": False,
            "grid_source_policy": (
                "K=5 cells are explicitly selected from validated E1 alpha-sweep rows; "
                "the alpha=1 E2 K=5 reruns remain visible in the rerun audit"
            ),
            "active_client_policy": (
                "standard FedAvg-family rows derive active clients as shard_size>=4 because the "
                "audited core skips smaller local updates; FedMME/P-FIN/local-only rows use all "
                "nonempty shards"
            ),
            "comparison_rule": (
                "same corpus, realized partitions, backbones, optimizer, batch, LR and "
                "24-local-epoch budget unless the method definition explicitly changes aggregation"
            ),
            "limitations": (
                "two seeds; CUDA algorithms were explicitly not forced deterministic; timings are "
                "from a shared H100 server; FedMME and P-FIN are declared matched adaptations"
            ),
        },
        "grid": grid,
        "federated_fusion": fusion,
        "recent_fedmme_style": fedmme,
        "recent_fedmme_native": fedmme_native,
        "pfin_missing_text": pfin,
        "initialization_random": random_init,
        "legacy_baselines": legacy["E8_baselines"],
        "legacy_anticollapse": legacy["E3_anticollapse"],
        "legacy_early_abort": {
            "provenance": {
                "kind": "validated_legacy_pooled_training_control",
                "legacy_suite": "E3b_early_abort",
                "training_scope": "centralized pooled-data training control (non-federated)",
                "federated": False,
                "epochs": 12,
                "seeds": [0, 1],
                "trainable_params": EXPECTED_MULTIMODAL_TRAINABLE_PARAMS,
                "metric_selection": "best validation epoch with the complete 12-epoch history retained",
                "base_model_sha256": EXPECTED_BASE,
                "data_cache_sha256": EXPECTED_CACHE,
                "core_runner_sha256": EXPECTED_CORE_RUNNER,
                "deterministic_algorithms_enforced": False,
                "legacy_meta_and_protocol_validated": True,
            },
            "records": legacy["E3b_early_abort"],
        },
        "legacy_initialization": legacy["E4_warmstart"],
        "legacy_branch_cost": legacy["E6_cost"],
        "retrieval": {
            "provenance": corrected_retrieval["provenance"],
            "protocol": corrected_retrieval["protocol"],
            "heldout": corrected_retrieval["result"],
        },
        "legacy_partition_source": {
            key: value
            for key, value in legacy["E1_alpha_sweep"].items()
            if key.startswith("alpha=0.1|")
        },
        "independent_rerun_audit": rerun,
        "raw_new_records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf8")
    print(f"validated legacy protocol and {len(records)} reviewer records -> {args.out}")


if __name__ == "__main__":
    main()
