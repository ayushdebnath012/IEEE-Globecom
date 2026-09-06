#!/usr/bin/env python3
"""Render the reviewer-completion result and systems figures.

The input is the JSON emitted by ``merge_reviewer_results.py``.  Every plotted
quantity is validated before rendering, and the exact plotted summaries are
echoed to stdout so the manuscript can quote values without transcribing a
figure.  Error bars and heat-map ``+/-`` values are sample standard deviations
across the two matched seeds unless an explicitly single-run or formula-derived
quantity is marked.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALPHAS = (0.1, 1.0, 5.0)
CLIENTS = (3, 5, 10, 20)
SEEDS = (0, 1)
FUSIONS = (
    "concat",
    "attention",
    "gated",
    "clip",
    "flamingo",
    "blip2",
    "coca",
    "unified_io",
)
FUSION_LABELS = (
    "Concat",
    "Cross-attn.",
    "Gated",
    "Dual-proj.",
    "Decoder A",
    "Decoder B",
    "Cross-attn.-384",
    "Token encoder",
)
PFIN_MODES = (
    "zero",
    "deterministic",
    "probabilistic",
    "probabilistic_uq",
)
PFIN_LABELS = ("Zero-fill", "Det. FIN", "P-FIN-style", "P-FIN-style + UQ")
CLASS_LABELS = ("Normal", "Pneum.", "COVID", "Effusion", "Cardio.")

TOP_LEVEL_SECTIONS = (
    "grid",
    "federated_fusion",
    "recent_fedmme_style",
    "recent_fedmme_native",
    "pfin_missing_text",
    "initialization_random",
    "legacy_baselines",
    "legacy_anticollapse",
    "legacy_early_abort",
    "legacy_initialization",
    "legacy_branch_cost",
    "legacy_partition_source",
)

# Color-blind-friendly palette (Okabe--Ito with neutral additions).
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#6F4E7C"
YELLOW = "#E69F00"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#D4D4D4"


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _record(data: Mapping[str, Any], section: str, key: str) -> Mapping[str, Any]:
    block = _mapping(data.get(section), section)
    if key not in block:
        raise ValueError(f"missing required record {section}[{key!r}]")
    return _mapping(block[key], f"{section}[{key!r}]")


def _early_abort_record(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    block = _mapping(data.get("legacy_early_abort"), "legacy_early_abort")
    records = _mapping(block.get("records"), "legacy_early_abort.records")
    if key not in records:
        raise ValueError(f"missing pooled-control record legacy_early_abort.records[{key!r}]")
    return _mapping(records[key], f"legacy_early_abort.records[{key!r}]")


def _number(record: Mapping[str, Any], field: str, context: str) -> float:
    if field not in record:
        raise ValueError(f"missing numeric field {context}.{field}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{field} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{context}.{field} is not finite: {value!r}")
    return value


def _integer(record: Mapping[str, Any], field: str, context: str) -> int:
    if field not in record:
        raise ValueError(f"missing integer field {context}.{field}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{field} must be an integer, got {value!r}")
    return value


def _numeric_sequence(record: Mapping[str, Any], field: str, context: str) -> list[float]:
    if field not in record or not isinstance(record[field], list) or not record[field]:
        raise ValueError(f"{context}.{field} must be a non-empty numeric list")
    values: list[float] = []
    for index, value in enumerate(record[field]):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}.{field}[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{context}.{field}[{index}] is not finite")
        values.append(number)
    return values


def _history(record: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    return _mapping(record.get("history"), f"{context}.history")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    clean = [float(value) for value in values]
    if not all(math.isfinite(value) for value in clean):
        raise ValueError(f"summary contains a non-finite value: {clean}")
    return {
        "values": clean,
        "mean": float(statistics.fmean(clean)),
        "sample_sd": float(statistics.stdev(clean)) if len(clean) > 1 else None,
        "n": len(clean),
    }


def _mean_and_sd(values: Sequence[float]) -> tuple[float, float]:
    summary = _summary(values)
    return summary["mean"], summary["sample_sd"] or 0.0


def _two_seed_records(
    data: Mapping[str, Any],
    section: str,
    key_for_seed: Callable[[int], str],
) -> list[Mapping[str, Any]]:
    return [_record(data, section, key_for_seed(seed)) for seed in SEEDS]


def _metric_summary(
    records: Sequence[Mapping[str, Any]], field: str, context: str
) -> dict[str, Any]:
    return _summary(
        [_number(record, field, f"{context}[seed={seed}]") for seed, record in zip(SEEDS, records)]
    )


def _grid_key(alpha: float, clients: int, seed: int) -> str:
    return f"alpha={alpha}|K={clients}|seed={seed}"


def _validate_active_clients(
    record: Mapping[str, Any], nominal_clients: int, context: str
) -> None:
    nominal = _integer(record, "nominal_clients", context)
    active = _integer(record, "active_clients", context)
    if nominal != nominal_clients:
        raise ValueError(
            f"{context}.nominal_clients={nominal} does not match key K={nominal_clients}"
        )
    if not 0 <= active <= nominal:
        raise ValueError(f"{context}.active_clients={active} is outside [0, {nominal}]")
    sizes = record.get("client_sizes")
    if not isinstance(sizes, list) or len(sizes) != nominal:
        raise ValueError(f"{context}.client_sizes must contain {nominal} shards")
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sizes):
        raise ValueError(f"{context}.client_sizes must contain nonnegative integers")
    expected_skipped = [index for index, size in enumerate(sizes) if size < 4]
    skipped = record.get("skipped_client_ids")
    if skipped != expected_skipped:
        raise ValueError(
            f"{context}.skipped_client_ids={skipped!r} does not match shard-size<4 rule "
            f"{expected_skipped!r}"
        )
    if active != nominal - len(expected_skipped):
        raise ValueError(
            f"{context}.active_clients={active} does not match derived count "
            f"{nominal - len(expected_skipped)}"
        )


def _validate_schema(data: Mapping[str, Any]) -> None:
    meta = _mapping(data.get("_meta"), "_meta")
    for field in ("legacy_meta_and_protocol_validated", "new_record_protocols_validated"):
        if meta.get(field) is not True:
            raise ValueError(f"_meta.{field} must be exactly true")
    if meta.get("deterministic_algorithms_enforced") is not False:
        raise ValueError("_meta.deterministic_algorithms_enforced must be exactly false")
    # 40 original reviewer-completion records plus the two native-budget FedMME
    # runs that remove the local-epoch departure from the source method.
    if meta.get("new_record_count") != 42:
        raise ValueError("_meta.new_record_count must equal the complete 42-record manifest")

    for section in TOP_LEVEL_SECTIONS:
        _mapping(data.get(section), section)

    early_abort_block = _mapping(data.get("legacy_early_abort"), "legacy_early_abort")
    early_abort_provenance = _mapping(
        early_abort_block.get("provenance"), "legacy_early_abort.provenance"
    )
    if early_abort_provenance.get("kind") != "validated_legacy_pooled_training_control":
        raise ValueError("legacy_early_abort provenance kind is missing or stale")
    if early_abort_provenance.get("training_scope") != (
        "centralized pooled-data training control (non-federated)"
    ):
        raise ValueError("legacy_early_abort must be labeled as a pooled non-federated control")
    if early_abort_provenance.get("federated") is not False:
        raise ValueError("legacy_early_abort.provenance.federated must be exactly false")
    if early_abort_provenance.get("epochs") != 12:
        raise ValueError("legacy_early_abort must retain the 12-epoch control protocol")
    if early_abort_provenance.get("legacy_meta_and_protocol_validated") is not True:
        raise ValueError("legacy_early_abort provenance must be validated")
    early_abort_records = _mapping(
        early_abort_block.get("records"), "legacy_early_abort.records"
    )
    expected_early_abort_keys = {
        f"early_abort={enabled}|seed={seed}"
        for enabled in (True, False)
        for seed in SEEDS
    }
    if set(early_abort_records) != expected_early_abort_keys:
        raise ValueError(
            "legacy_early_abort.records key mismatch: "
            f"missing={sorted(expected_early_abort_keys - set(early_abort_records))} "
            f"extra={sorted(set(early_abort_records) - expected_early_abort_keys)}"
        )
    for enabled in (False, True):
        for seed in SEEDS:
            key = f"early_abort={enabled}|seed={seed}"
            record = _early_abort_record(data, key)
            for field in ("f1", "accuracy", "diversity", "min_diversity", "wall_seconds"):
                _number(record, field, key)
            history = _history(record, key)
            history_values: dict[str, list[float]] = {}
            for field in ("val_f1", "val_acc", "diversity", "epoch_seconds"):
                values = _numeric_sequence(history, field, f"{key}.history")
                if len(values) != 12:
                    raise ValueError(f"{key}.history.{field} must contain exactly 12 epochs")
                history_values[field] = values
            for field in ("val_f1", "val_acc", "diversity"):
                if any(not 0.0 <= value <= 1.0 for value in history_values[field]):
                    raise ValueError(f"{key}.history.{field} contains a value outside [0, 1]")
            if any(value <= 0.0 for value in history_values["epoch_seconds"]):
                raise ValueError(f"{key}.history.epoch_seconds must be strictly positive")
            epoch = _integer(record, "epoch", key)
            if not 1 <= epoch <= 12:
                raise ValueError(f"{key}.epoch must be in [1, 12]")
            if epoch - 1 != history_values["val_f1"].index(max(history_values["val_f1"])):
                raise ValueError(f"{key}.epoch is not the first maximum-F1 pooled epoch")
            for result_field, history_field in (
                ("f1", "val_f1"),
                ("accuracy", "val_acc"),
                ("diversity", "diversity"),
            ):
                if not math.isclose(
                    _number(record, result_field, key),
                    history_values[history_field][epoch - 1],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"{key}.{result_field} is inconsistent with its selected pooled epoch"
                    )
            if not math.isclose(
                _number(record, "min_diversity", key),
                min(history_values["diversity"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{key}.min_diversity is inconsistent with its history")
            if not math.isclose(
                _number(record, "wall_seconds", key),
                sum(history_values["epoch_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(f"{key}.wall_seconds does not equal its 12 epoch timings")
            if _integer(record, "trainable_params", key) != early_abort_provenance.get(
                "trainable_params"
            ):
                raise ValueError(f"{key}.trainable_params disagrees with pooled-control provenance")

    for alpha in ALPHAS:
        for clients in CLIENTS:
            for seed in SEEDS:
                record = _record(data, "grid", _grid_key(alpha, clients, seed))
                context = f"grid[{_grid_key(alpha, clients, seed)!r}]"
                for field in ("f1", "peak_mib", "upload_bytes_per_client_per_round"):
                    _number(record, field, context)
                _validate_active_clients(record, clients, context)
                history = _history(record, context)
                round_f1 = _numeric_sequence(history, "round_f1", f"{context}.history")
                round_seconds = _numeric_sequence(
                    history, "round_seconds", f"{context}.history"
                )
                if len(round_f1) != len(round_seconds):
                    raise ValueError(f"{context} round_f1 and round_seconds lengths differ")

    for fusion in FUSIONS:
        for seed in SEEDS:
            key = f"fusion={fusion}|alpha=0.1|K=5|seed={seed}"
            _number(_record(data, "federated_fusion", key), "f1", key)

    for seed in SEEDS:
        key = f"fedmme_style|alpha=0.1|K=5|seed={seed}"
        _number(_record(data, "recent_fedmme_style", key), "f1", key)
        key = f"init=random|alpha=1.0|K=5|seed={seed}"
        _number(_record(data, "initialization_random", key), "f1", key)

        for algorithm in ("fedavg", "fedprox", "scaffold"):
            key = f"{algorithm}|alpha=0.1|seed={seed}"
            record = _record(data, "legacy_baselines", key)
            _number(record, "f1", key)
            _numeric_sequence(_history(record, key), "round_f1", f"{key}.history")

        for arm in ("full", "no_balanced", "no_diversity", "neither"):
            for alpha in (0.1, 1.0):
                key = f"{arm}|alpha={alpha}|seed={seed}"
                record = _record(data, "legacy_anticollapse", key)
                _number(record, "f1", key)
                round_diversity = _numeric_sequence(
                    _history(record, key), "round_div", f"{key}.history"
                )
                if len(round_diversity) != 8:
                    raise ValueError(f"{key}.history.round_div must contain exactly 8 rounds")
                if any(not 0.0 <= value <= 1.0 for value in round_diversity):
                    raise ValueError(f"{key}.history.round_div contains a value outside [0, 1]")

        for warm_start in (False, True):
            key = f"warm_start={warm_start}|seed={seed}"
            _number(_record(data, "legacy_initialization", key), "f1", key)

        for mode in PFIN_MODES:
            key = f"pfin={mode}|alpha=0.1|K=5|seed={seed}"
            record = _record(data, "pfin_missing_text", key)
            if record.get("matched_adaptation") is not True:
                raise ValueError(f"{key}.matched_adaptation must be exactly true")
            if record.get("mode") != mode:
                raise ValueError(f"{key}.mode does not agree with its record key")
            _number(record, "full_text_f1", key)
            # This field is deliberately mandatory: the primary P-FIN result is
            # the forced-missing-text stress condition, never record["f1"].
            forced_missing = _number(record, "forced_missing_text_f1", key)
            primary = _number(record, "primary_f1", key)
            if not math.isclose(primary, forced_missing, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{key}.primary_f1 must equal forced_missing_text_f1"
                )

    local = _record(data, "legacy_baselines", "local_only|alpha=0.1|seed=0")
    _number(local, "mean_f1", "local_only|alpha=0.1|seed=0")

    for branch in ("Fed-LLM", "Fed-ViT", "Fed-VLM"):
        record = _record(data, "legacy_branch_cost", branch)
        for field in ("f1", "upload_bytes_per_client_per_round", "wall_seconds", "peak_mib"):
            _number(record, field, f"legacy_branch_cost[{branch!r}]")

    partition = _record(data, "legacy_partition_source", "alpha=0.1|seed=0")
    histogram = partition.get("client_class_hist")
    if not isinstance(histogram, list) or len(histogram) != 5:
        raise ValueError("severe-skew client_class_hist must contain five clients")
    for client_id, row in enumerate(histogram):
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError(f"client_class_hist[{client_id}] must contain five classes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row):
            raise ValueError(f"client_class_hist[{client_id}] must contain nonnegative counts")
        if sum(row) <= 0:
            raise ValueError(f"client_class_hist[{client_id}] is empty")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.25,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_metric_axis(ax: plt.Axes, *, horizontal: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if horizontal:
        ax.xaxis.grid(True, alpha=0.28)
        ax.set_axisbelow(True)
    else:
        ax.yaxis.grid(True, alpha=0.28)
        ax.set_axisbelow(True)


def _errorbar_barh(
    ax: plt.Axes,
    labels: Sequence[str],
    summaries: Sequence[Mapping[str, Any]],
    colors: Sequence[str],
) -> None:
    y = np.arange(len(labels))
    means = np.array([item["mean"] for item in summaries], dtype=float)
    errors = np.array([item["sample_sd"] or 0.0 for item in summaries], dtype=float)
    ax.barh(
        y,
        means,
        xerr=errors,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        error_kw={"elinewidth": 0.75, "capsize": 1.8, "capthick": 0.7},
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Macro-F1 score")
    _style_metric_axis(ax, horizontal=True)


def _matched_figure(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    echo: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.36), layout="constrained")

    # (a) Common-setting baselines, alpha=0.1 and K=5.
    local = _record(data, "legacy_baselines", "local_only|alpha=0.1|seed=0")
    baseline_specs: list[tuple[str, list[Mapping[str, Any]], str, str]] = [
        (
            "Local-only (1 split)",
            [local],
            "mean_f1",
            "local-only (one severe-skew partition; mean across clients)",
        ),
        (
            "OmniMed (FedAvg)",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"fedavg|alpha=0.1|seed={seed}",
            ),
            "f1",
            "OmniMed-FL / FedAvg",
        ),
        (
            "FedProx",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"fedprox|alpha=0.1|seed={seed}",
            ),
            "f1",
            "FedProx",
        ),
        (
            "SCAFFOLD-AdamW",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"scaffold|alpha=0.1|seed={seed}",
            ),
            "f1",
            "SCAFFOLD-AdamW",
        ),
        (
            "FedMME (24 ep.)",
            _two_seed_records(
                data,
                "recent_fedmme_style",
                lambda seed: f"fedmme_style|alpha=0.1|K=5|seed={seed}",
            ),
            "f1",
            "FedMME-style matched adaptation",
        ),
        (
            "FedMME (100 ep.)",
            _two_seed_records(
                data,
                "recent_fedmme_native",
                lambda seed: f"fedmme_native|epochs=100|alpha=0.1|K=5|seed={seed}",
            ),
            "f1",
            "FedMME-style native local budget",
        ),
    ]
    baseline_labels: list[str] = []
    baseline_summaries: list[dict[str, Any]] = []
    for short_label, records, field, echo_label in baseline_specs:
        values = [
            _number(record, field, f"matched baseline {echo_label}") for record in records
        ]
        baseline_labels.append(short_label)
        baseline_summaries.append(_summary(values))
    _errorbar_barh(
        axes[0, 0],
        baseline_labels,
        baseline_summaries,
        [LIGHT_GRAY, ORANGE, SKY, PURPLE, GREEN, YELLOW],
    )
    axes[0, 0].set_title("(a) Matched baselines ($\\alpha$=0.1, K=5)")
    echo["matched_baselines"] = {
        spec[3]: summary for spec, summary in zip(baseline_specs, baseline_summaries)
    }

    # (b) Round-wise convergence for iterative methods.
    convergence_specs = (
        (
            "OmniMed (FedAvg)",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"fedavg|alpha=0.1|seed={seed}",
            ),
            ORANGE,
        ),
        (
            "FedProx",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"fedprox|alpha=0.1|seed={seed}",
            ),
            GREEN,
        ),
        (
            "SCAFFOLD-AdamW",
            _two_seed_records(
                data,
                "legacy_baselines",
                lambda seed: f"scaffold|alpha=0.1|seed={seed}",
            ),
            PURPLE,
        ),
    )
    convergence_echo: dict[str, Any] = {}
    ax = axes[0, 1]
    for label, records, color in convergence_specs:
        curves = [
            _numeric_sequence(
                _history(record, f"{label}[seed={seed}]"),
                "round_f1",
                f"{label}[seed={seed}].history",
            )
            for seed, record in zip(SEEDS, records)
        ]
        lengths = {len(curve) for curve in curves}
        if len(lengths) != 1:
            raise ValueError(f"{label} convergence curves have unequal lengths")
        array = np.asarray(curves, dtype=float)
        mean = array.mean(axis=0)
        sd = array.std(axis=0, ddof=1)
        rounds = np.arange(1, array.shape[1] + 1)
        ax.plot(rounds, mean, marker="o", markersize=2.2, color=color, label=label)
        ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.11, linewidth=0)
        convergence_echo[label] = {
            "round_mean": mean.tolist(),
            "round_sample_sd": sd.tolist(),
            "n": int(array.shape[0]),
        }
    ax.set_xlim(1, max(rounds))
    ax.set_ylim(0.0, 1.30)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Macro-F1 score")
    ax.set_title("(b) Convergence under severe skew")
    ax.legend(frameon=False, ncol=3, loc="upper center", columnspacing=0.7,
              handlelength=1.2, borderpad=0.1, handletextpad=0.4)
    _style_metric_axis(ax)
    echo["convergence"] = convergence_echo

    # (c) Anti-collapse ablation.
    anticollapse_specs = (
        ("Full", "full"),
        ("- balanced loader", "no_balanced"),
        ("- entropy term", "no_diversity"),
        ("Neither", "neither"),
    )
    anticollapse_by_alpha: dict[float, list[dict[str, Any]]] = {}
    minimum_diversity_by_alpha: dict[float, list[dict[str, Any]]] = {}
    for alpha in (0.1, 1.0):
        summaries: list[dict[str, Any]] = []
        diversity_summaries: list[dict[str, Any]] = []
        for _, arm in anticollapse_specs:
            records = _two_seed_records(
                data,
                "legacy_anticollapse",
                lambda seed, arm=arm, alpha=alpha: (
                    f"{arm}|alpha={alpha}|seed={seed}"
                ),
            )
            summaries.append(
                _metric_summary(records, "f1", f"anti-collapse {arm}, alpha={alpha}")
            )
            minimum_diversities = [
                min(
                    _numeric_sequence(
                        _history(record, f"anti-collapse {arm}, alpha={alpha}, seed={seed}"),
                        "round_div",
                        f"anti-collapse {arm}, alpha={alpha}, seed={seed}.history",
                    )
                )
                for seed, record in zip(SEEDS, records)
            ]
            diversity_summaries.append(_summary(minimum_diversities))
        anticollapse_by_alpha[alpha] = summaries
        minimum_diversity_by_alpha[alpha] = diversity_summaries
    ax = axes[0, 2]
    x = np.arange(len(anticollapse_specs))
    width = 0.36
    bar_handles = []
    diversity_handle = None
    for offset, alpha, color in ((-width / 2, 0.1, ORANGE), (width / 2, 1.0, SKY)):
        summaries = anticollapse_by_alpha[alpha]
        bars = ax.bar(
            x + offset,
            [summary["mean"] for summary in summaries],
            width,
            yerr=[summary["sample_sd"] or 0.0 for summary in summaries],
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=f"$\\alpha$={alpha:g}",
            error_kw={"elinewidth": 0.65, "capsize": 1.4, "capthick": 0.65},
        )
        bar_handles.append(bars)
        diversity_summaries = minimum_diversity_by_alpha[alpha]
        marker = ax.errorbar(
            x + offset,
            [summary["mean"] for summary in diversity_summaries],
            yerr=[summary["sample_sd"] or 0.0 for summary in diversity_summaries],
            fmt="D",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.65,
            color="black",
            elinewidth=0.65,
            capsize=1.4,
            zorder=5,
        )
        if diversity_handle is None:
            diversity_handle = marker[0]
    ax.set_xticks(x, ["Full", "-Loader", "-Div.", "Neither"], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("(c) Anti-collapse components")
    ax.legend(
        [bar_handles[0], bar_handles[1], diversity_handle],
        ["$\\alpha$=0.1 F1", "$\\alpha$=1 F1", "Min. predicted-class diversity"],
        frameon=False,
        loc="lower center",
        ncol=2,
        columnspacing=0.65,
        handletextpad=0.35,
        fontsize=5.0,
    )
    _style_metric_axis(ax)
    echo["anti_collapse"] = {
        f"alpha={alpha}": {
            label: {
                "macro_f1": f1_summary,
                "minimum_predicted_class_diversity": diversity_summary,
            }
            for (label, _), f1_summary, diversity_summary in zip(
                anticollapse_specs,
                summaries,
                minimum_diversity_by_alpha[alpha],
            )
        }
        for alpha, summaries in anticollapse_by_alpha.items()
    }

    # (d) Eight operational federated fusion rules.
    fusion_summaries: list[dict[str, Any]] = []
    for fusion in FUSIONS:
        records = _two_seed_records(
            data,
            "federated_fusion",
            lambda seed, fusion=fusion: (
                f"fusion={fusion}|alpha=0.1|K=5|seed={seed}"
            ),
        )
        fusion_summaries.append(_metric_summary(records, "f1", f"fusion {fusion}"))
    ax = axes[1, 0]
    x = np.arange(len(FUSIONS))
    fusion_means = [summary["mean"] for summary in fusion_summaries]
    fusion_sd = [summary["sample_sd"] or 0.0 for summary in fusion_summaries]
    best_index = int(np.argmax(fusion_means))
    colors = [ORANGE if index == best_index else BLUE for index in range(len(FUSIONS))]
    ax.bar(
        x,
        fusion_means,
        yerr=fusion_sd,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"elinewidth": 0.7, "capsize": 1.6, "capthick": 0.7},
    )
    ax.set_xticks(x, FUSION_LABELS, rotation=43, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro-F1 score")
    ax.set_title("(d) Operational fusion rules")
    _style_metric_axis(ax)
    echo["federated_fusion"] = {
        label: summary for label, summary in zip(FUSION_LABELS, fusion_summaries)
    }

    # (e) Three federated initialization controls plus a visually separated
    # pair from E3b.  E3b is centralized pooled-data training and is never
    # presented as a federated result.
    initialization_specs = (
        (
            "All\nrandom",
            "Same architecture, random encoders and heads",
            _two_seed_records(
                data,
                "initialization_random",
                lambda seed: f"init=random|alpha=1.0|K=5|seed={seed}",
            ),
        ),
        (
            "Public\nenc.",
            "Public encoders, random task heads",
            _two_seed_records(
                data,
                "legacy_initialization",
                lambda seed: f"warm_start=False|seed={seed}",
            ),
        ),
        (
            "Pooled\nstart",
            "Pooled-data warm start",
            _two_seed_records(
                data,
                "legacy_initialization",
                lambda seed: f"warm_start=True|seed={seed}",
            ),
        ),
    )
    initialization_summaries = [
        _metric_summary(records, "f1", f"initialization {echo_label}")
        for _, echo_label, records in initialization_specs
    ]
    initialization_round1 = [
        _summary(
            [
                _numeric_sequence(
                    _history(record, f"initialization {echo_label}[seed={seed}]"),
                    "round_f1",
                    f"initialization {echo_label}[seed={seed}].history",
                )[0]
                for seed, record in zip(SEEDS, records)
            ]
        )
        for _, echo_label, records in initialization_specs
    ]
    early_abort_specs = (
        (
            "Abort\noff",
            "Abort off",
            [
                _early_abort_record(data, f"early_abort=False|seed={seed}")
                for seed in SEEDS
            ],
        ),
        (
            "Abort\non",
            "Abort on",
            [
                _early_abort_record(data, f"early_abort=True|seed={seed}")
                for seed in SEEDS
            ],
        ),
    )
    early_abort_summaries = [
        _summary(
            [
                _numeric_sequence(
                    _history(record, f"{echo_label}[seed={seed}]"),
                    "val_f1",
                    f"{echo_label}[seed={seed}].history",
                )[-1]
                for seed, record in zip(SEEDS, records)
            ]
        )
        for _, echo_label, records in early_abort_specs
    ]
    early_abort_best_summaries = [
        _metric_summary(records, "f1", f"pooled early-abort control {echo_label}")
        for _, echo_label, records in early_abort_specs
    ]
    ax = axes[1, 1]
    initialization_x = np.asarray((0.0, 1.0, 2.0))
    early_abort_x = np.asarray((4.0, 5.0))
    ax.axvspan(3.45, 5.55, color=PURPLE, alpha=0.055, zorder=0)
    ax.axvline(3.25, color=GRAY, linewidth=0.65, linestyle="--", alpha=0.8)
    ax.bar(
        initialization_x,
        [summary["mean"] for summary in initialization_summaries],
        yerr=[summary["sample_sd"] or 0.0 for summary in initialization_summaries],
        color=[LIGHT_GRAY, BLUE, ORANGE],
        edgecolor="white",
        linewidth=0.45,
        error_kw={"elinewidth": 0.75, "capsize": 1.8, "capthick": 0.7},
    )
    ax.bar(
        early_abort_x,
        [summary["mean"] for summary in early_abort_summaries],
        yerr=[summary["sample_sd"] or 0.0 for summary in early_abort_summaries],
        color=[PURPLE, GREEN],
        edgecolor="white",
        linewidth=0.45,
        error_kw={"elinewidth": 0.75, "capsize": 1.8, "capthick": 0.7},
    )
    ax.set_xticks(
        np.concatenate((initialization_x, early_abort_x)),
        [item[0] for item in initialization_specs]
        + [item[0] for item in early_abort_specs],
        rotation=0,
        ha="center",
        fontsize=4.8,
    )
    ax.scatter(
        initialization_x,
        [summary["mean"] for summary in initialization_round1],
        marker="D",
        s=13,
        color="black",
        zorder=4,
        label="Fed. round 1",
    )
    ax.text(
        1.0,
        0.98,
        "Federated init. ($\\alpha$=1, K=5)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.2,
        color=GRAY,
    )
    ax.text(
        4.5,
        0.98,
        "Pooled control\n(non-FL, 12 epochs)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.2,
        color=PURPLE,
        linespacing=0.9,
    )
    ax.set_xlim(-0.65, 5.65)
    ax.set_ylim(0.0, 1.34)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Macro-F1 score")
    ax.set_title("(e) Initialization / abort controls")
    ax.legend(frameon=False, loc="lower left")
    _style_metric_axis(ax)
    echo["initialization"] = {
        echo_label: {"final": final, "round_1": round_one}
        for (_, echo_label, _), final, round_one in zip(
            initialization_specs, initialization_summaries, initialization_round1
        )
    }
    early_abort_block = _mapping(data["legacy_early_abort"], "legacy_early_abort")
    echo["pooled_early_abort_control"] = {
        "provenance": dict(
            _mapping(
                early_abort_block.get("provenance"),
                "legacy_early_abort.provenance",
            )
        ),
        "arms": {
            echo_label: {
                "final_epoch_macro_f1": final_summary,
                "best_validation_macro_f1": best_summary,
                "best_epoch": _summary(
                    [
                        _integer(record, "epoch", f"{echo_label}[seed={seed}]")
                        for seed, record in zip(SEEDS, records)
                    ]
                ),
                "minimum_predicted_class_diversity": _summary(
                    [
                        _number(
                            record,
                            "min_diversity",
                            f"{echo_label}[seed={seed}]",
                        )
                        for seed, record in zip(SEEDS, records)
                    ]
                ),
                "wall_seconds": _summary(
                    [
                        _number(record, "wall_seconds", f"{echo_label}[seed={seed}]")
                        for seed, record in zip(SEEDS, records)
                    ]
                ),
                "history_epochs": [
                    len(
                        _numeric_sequence(
                            _history(record, f"{echo_label}[seed={seed}]"),
                            "val_f1",
                            f"{echo_label}[seed={seed}].history",
                        )
                    )
                    for seed, record in zip(SEEDS, records)
                ],
            }
            for (_, echo_label, records), final_summary, best_summary in zip(
                early_abort_specs, early_abort_summaries, early_abort_best_summaries
            )
        },
    }

    # (f) P-FIN-style missing-modality adaptation.  The dark bars are the
    # mandatory primary metric: forced_missing_text_f1.
    full_summaries: list[dict[str, Any]] = []
    missing_summaries: list[dict[str, Any]] = []
    for mode in PFIN_MODES:
        records = _two_seed_records(
            data,
            "pfin_missing_text",
            lambda seed, mode=mode: f"pfin={mode}|alpha=0.1|K=5|seed={seed}",
        )
        full_summaries.append(_metric_summary(records, "full_text_f1", f"P-FIN {mode}"))
        missing_summaries.append(
            _metric_summary(records, "forced_missing_text_f1", f"P-FIN {mode}")
        )
    ax = axes[1, 2]
    x = np.arange(len(PFIN_MODES))
    width = 0.36
    ax.bar(
        x - width / 2,
        [summary["mean"] for summary in full_summaries],
        width,
        yerr=[summary["sample_sd"] or 0.0 for summary in full_summaries],
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        linewidth=0.45,
        label="All text observed (secondary)",
        error_kw={"elinewidth": 0.65, "capsize": 1.4, "capthick": 0.65},
    )
    ax.bar(
        x + width / 2,
        [summary["mean"] for summary in missing_summaries],
        width,
        yerr=[summary["sample_sd"] or 0.0 for summary in missing_summaries],
        color=GREEN,
        edgecolor="white",
        linewidth=0.45,
        label="Forced missing (primary)",
        error_kw={"elinewidth": 0.65, "capsize": 1.4, "capthick": 0.65},
    )
    ax.set_xticks(x, PFIN_LABELS, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro-F1 score")
    ax.set_title("(f) P-FIN-style missing-text stress")
    ax.legend(frameon=False, loc="lower left")
    _style_metric_axis(ax)
    echo["pfin_missing_text"] = {
        label: {
            "full_text": full,
            "forced_missing_text_primary": missing,
        }
        for label, full, missing in zip(PFIN_LABELS, full_summaries, missing_summaries)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350)
    plt.close(fig)
    return echo


def _annotated_heatmap(
    ax: plt.Axes,
    means: np.ndarray,
    sample_sd: np.ndarray,
    *,
    title: str,
    formatter: Callable[[float], str],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cell_notes: np.ndarray | None = None,
) -> None:
    if cell_notes is not None and cell_notes.shape != means.shape:
        raise ValueError("cell_notes must have the same shape as the heat-map values")
    image = ax.imshow(means, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(CLIENTS)), [str(value) for value in CLIENTS])
    ax.set_yticks(np.arange(len(ALPHAS)), [str(value).rstrip("0").rstrip(".") for value in ALPHAS])
    ax.set_xlabel("Nominal clients K")
    ax.set_ylabel("Dirichlet $\\alpha$")
    ax.set_title(title)
    threshold = (float(np.nanmin(means)) + float(np.nanmax(means))) / 2.0
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            mean_text = formatter(float(means[row, column]))
            sd_text = formatter(float(sample_sd[row, column]))
            note_text = "" if cell_notes is None else f"\n{cell_notes[row, column]}"
            color = "white" if means[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                f"{mean_text}\n±{sd_text}{note_text}",
                ha="center",
                va="center",
                color=color,
                fontsize=4.7 if cell_notes is not None else 5.1,
                linespacing=0.88,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Exact cell labels carry the scale; a small shared-style color key avoids
    # spending a publication panel on three full color bars.
    inset = ax.inset_axes([1.015, 0.03, 0.025, 0.94])
    plt.colorbar(image, cax=inset)
    inset.tick_params(labelsize=4.5, width=0.4, length=1.5)


def _systems_figure(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    echo: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.32), layout="constrained")

    metric_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    heatmap_echo: dict[str, Any] = {}
    active_notes = np.empty((len(ALPHAS), len(CLIENTS)), dtype=object)
    active_echo: dict[str, Any] = {}
    for row, alpha in enumerate(ALPHAS):
        for column, clients in enumerate(CLIENTS):
            records = _two_seed_records(
                data,
                "grid",
                lambda seed, alpha=alpha, clients=clients: _grid_key(
                    alpha, clients, seed
                ),
            )
            active_values = [
                _integer(record, "active_clients", _grid_key(alpha, clients, seed))
                for seed, record in zip(SEEDS, records)
            ]
            lower, upper = min(active_values), max(active_values)
            active_notes[row, column] = (
                f"A={lower}/{clients}"
                if lower == upper
                else f"A={lower}-{upper}/{clients}"
            )
            active_echo[f"alpha={alpha}|K={clients}"] = {
                "values": active_values,
                "minimum": lower,
                "maximum": upper,
                "nominal_clients": clients,
                "rule": "active iff realized shard size is at least four samples",
            }
    for metric in ("f1", "round_seconds", "peak_gib"):
        means = np.zeros((len(ALPHAS), len(CLIENTS)), dtype=float)
        sds = np.zeros_like(means)
        value_echo: dict[str, Any] = {}
        for row, alpha in enumerate(ALPHAS):
            for column, clients in enumerate(CLIENTS):
                records = _two_seed_records(
                    data,
                    "grid",
                    lambda seed, alpha=alpha, clients=clients: _grid_key(
                        alpha, clients, seed
                    ),
                )
                if metric == "f1":
                    values = [
                        _number(record, "f1", _grid_key(alpha, clients, seed))
                        for seed, record in zip(SEEDS, records)
                    ]
                elif metric == "round_seconds":
                    values = [
                        statistics.fmean(
                            _numeric_sequence(
                                _history(record, _grid_key(alpha, clients, seed)),
                                "round_seconds",
                                f"{_grid_key(alpha, clients, seed)}.history",
                            )
                        )
                        for seed, record in zip(SEEDS, records)
                    ]
                else:
                    values = [
                        _number(record, "peak_mib", _grid_key(alpha, clients, seed))
                        / 1024.0
                        for seed, record in zip(SEEDS, records)
                    ]
                summary = _summary(values)
                means[row, column] = summary["mean"]
                sds[row, column] = summary["sample_sd"] or 0.0
                value_echo[f"alpha={alpha}|K={clients}"] = summary
        metric_matrices[metric] = (means, sds)
        heatmap_echo[metric] = value_echo

    _annotated_heatmap(
        axes[0, 0],
        *metric_matrices["f1"],
        title="(a) Macro-F1 score (A=active clients)",
        formatter=lambda value: f"{value:.2f}",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cell_notes=active_notes,
    )
    _annotated_heatmap(
        axes[0, 1],
        *metric_matrices["round_seconds"],
        title="(b) Timed round section (s)",
        formatter=lambda value: f"{value:.0f}",
        cmap="YlOrBr",
    )
    _annotated_heatmap(
        axes[0, 2],
        *metric_matrices["peak_gib"],
        title="(c) Peak GPU memory (GiB)",
        formatter=lambda value: f"{value:.1f}",
        cmap="PuBu",
    )
    echo["scalability_heatmaps"] = heatmap_echo
    echo["active_clients"] = active_echo

    # (d) Formula-derived bidirectional model traffic. Recalculate from the FP32
    # payload, nominal K, and round count, and cross-check every stored total.
    # This is a protocol quantity rather than a stochastic measurement, so the
    # repeated JSON records verify it but are not treated as independent samples.
    communication_summaries: list[dict[str, Any]] = []
    for clients in CLIENTS:
        calculated: list[float] = []
        for alpha in ALPHAS:
            for seed in SEEDS:
                key = _grid_key(alpha, clients, seed)
                record = _record(data, "grid", key)
                payload = _number(record, "upload_bytes_per_client_per_round", key)
                rounds = len(
                    _numeric_sequence(
                        _history(record, key), "round_seconds", f"{key}.history"
                    )
                )
                bytes_calculated = 2.0 * clients * rounds * payload
                if "total_comm_bytes" in record:
                    reported = _number(record, "total_comm_bytes", key)
                    if not math.isclose(reported, bytes_calculated, rel_tol=1e-7, abs_tol=1.0):
                        raise ValueError(
                            f"{key} reported communication {reported} does not match "
                            f"2*K*rounds*payload={bytes_calculated}"
                        )
                calculated.append(bytes_calculated / (1024.0**3))
        reference = calculated[0]
        if any(
            not math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
            for value in calculated[1:]
        ):
            raise ValueError(
                f"K={clients} formula-derived traffic varies across alpha/seed records: "
                f"{calculated}"
            )
        communication_summaries.append(
            {
                "values": [reference],
                "mean": reference,
                "sample_sd": None,
                "n": 1,
                "verified_record_count": len(calculated),
                "definition": (
                    "2 * nominal K * 8 rounds * one-way FP32 trainable state; "
                    "formula-derived, not measured network traffic"
                ),
            }
        )
    ax = axes[1, 0]
    means = [summary["mean"] for summary in communication_summaries]
    ax.errorbar(
        CLIENTS,
        means,
        color=ORANGE,
        marker="o",
        markersize=3.2,
        capsize=2.0,
    )
    for clients, value in zip(CLIENTS, means):
        ax.annotate(
            f"{value:.1f}",
            (clients, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=5.4,
        )
    ax.set_xticks(CLIENTS)
    ax.set_xlabel("Nominal clients K")
    ax.set_ylabel("Bidirectional FP32 volume (GiB)")
    ax.set_title("(d) Formula volume (nominal K)")
    _style_metric_axis(ax)
    echo["calculated_bidirectional_communication_gib"] = {
        f"K={clients}": summary
        for clients, summary in zip(CLIENTS, communication_summaries)
    }

    # (e) Modality-branch resource audit.  Colors are normalized within each
    # row because the three measures have different physical units; annotations
    # remain absolute.
    branch_keys = ("Fed-LLM", "Fed-ViT", "Fed-VLM")
    branch_labels = ("Text only", "Image only", "Multimodal\nconcat")
    cost_values = np.zeros((4, 3), dtype=float)
    for column, branch in enumerate(branch_keys):
        record = _record(data, "legacy_branch_cost", branch)
        cost_values[0, column] = _number(record, "f1", branch)
        cost_values[1, column] = _number(
            record, "upload_bytes_per_client_per_round", branch
        ) / (1024.0**3)
        cost_values[2, column] = _number(record, "wall_seconds", branch) / 60.0
        cost_values[3, column] = _number(record, "peak_mib", branch) / 1024.0
    row_max = np.maximum(cost_values.max(axis=1, keepdims=True), np.finfo(float).eps)
    normalized = cost_values / row_max
    ax = axes[1, 1]
    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3), branch_labels)
    ax.set_yticks(
        np.arange(4),
        (
            "Macro-F1 score",
            "One-way state\nGiB",
            "Timed rounds\n(min)",
            "Peak GPU\nGiB",
        ),
    )
    ax.set_title("(e) Branch cost ($\\alpha$=1, K=5; one run)")
    formats = ("{:.3f}", "{:.2f}", "{:.1f}", "{:.1f}")
    for row in range(4):
        for column in range(3):
            ax.text(
                column,
                row,
                formats[row].format(cost_values[row, column]),
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.58 else "black",
                fontsize=5.8,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    echo["modality_branch_cost_single_run"] = {
        label.replace("\n", " "): {
            "macro_f1": float(cost_values[0, index]),
            "payload_gib_per_client_per_round": float(cost_values[1, index]),
            "shared_server_wall_minutes": float(cost_values[2, index]),
            "peak_gpu_gib": float(cost_values[3, index]),
            "n": 1,
            "legacy_source_record": branch,
        }
        for index, (branch, label) in enumerate(zip(branch_keys, branch_labels))
    }

    # (f) The actual severe-skew partition used by seed 0.  Color represents
    # within-client class share while the annotation is the raw sample count.
    partition = _record(data, "legacy_partition_source", "alpha=0.1|seed=0")
    counts = np.asarray(partition["client_class_hist"], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    ax = axes[1, 2]
    image = ax.imshow(shares, cmap="magma_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(5), CLASS_LABELS, rotation=28, ha="right")
    ax.set_yticks(np.arange(5), [f"Client {index}" for index in range(1, 6)])
    ax.set_title("(f) Severe-skew class allocation")
    for row in range(5):
        for column in range(5):
            ax.text(
                column,
                row,
                str(int(counts[row, column])),
                ha="center",
                va="center",
                color="white" if shares[row, column] < 0.18 else "black",
                fontsize=5.4,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    inset = ax.inset_axes([1.015, 0.03, 0.025, 0.94])
    colorbar = plt.colorbar(image, cax=inset)
    colorbar.set_label("Within-client share", fontsize=4.8)
    inset.tick_params(labelsize=4.5, width=0.4, length=1.5)
    echo["severe_skew_alpha_0.1_seed_0"] = {
        "client_class_counts": counts.astype(int).tolist(),
        "client_class_shares": shares.tolist(),
        "class_order": list(CLASS_LABELS),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350)
    plt.close(fig)
    return echo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "merged",
        type=Path,
        help="JSON generated by merge_reviewer_results.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <paper-root>/generated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.merged.is_file():
        raise FileNotFoundError(f"merged result file not found: {args.merged}")
    data = _mapping(json.loads(args.merged.read_text(encoding="utf8")), "root")
    _validate_schema(data)
    _configure_matplotlib()

    if args.out_dir is None:
        if len(args.merged.resolve().parents) < 2:
            raise ValueError("cannot infer paper root; pass --out-dir explicitly")
        out_dir = args.merged.resolve().parent.parent / "generated"
    else:
        out_dir = args.out_dir.resolve()

    matched_path = out_dir / "reviewer_matched_results.png"
    systems_path = out_dir / "reviewer_systems_scalability.png"
    matched_echo = _matched_figure(data, matched_path)
    systems_echo = _systems_figure(data, systems_path)

    print("PLOTTED_VALUES_BEGIN")
    print(
        json.dumps(
            {
                "uncertainty_rule": (
                    "sample SD across two matched seeds; null SD marks a single-run "
                    "or formula-derived quantity"
                ),
                "matched_results": matched_echo,
                "systems": systems_echo,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("PLOTTED_VALUES_END")
    print(f"wrote {matched_path}")
    print(f"wrote {systems_path}")


if __name__ == "__main__":
    main()
