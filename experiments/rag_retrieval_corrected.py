"""Leakage-free TF--IDF/FAISS retrieval audit for the controlled corpus.

The vectorizer is fit on the 2,400 training notes only.  The 600 validation
notes are transformed afterward and are used solely as queries.  Labels are
used only to score retrieved neighbours, never to fit the representation or
index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


LABELS = (
    "NORMAL",
    "PNEUMONIA",
    "COVID19",
    "PLEURAL_EFFUSION",
    "CARDIOMEGALY",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf8")
    temporary.replace(path)


def scalar_label(value: object) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"expected a singleton label, got {value!r}")
        value = value[0]
    label = int(value)
    if label < 0 or label >= len(LABELS):
        raise ValueError(f"label outside [0, {len(LABELS) - 1}]: {label}")
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=4096)
    args = parser.parse_args()

    if args.top_k != 5:
        raise ValueError("the audited paper protocol requires top_k=5")
    if args.max_features != 4096:
        raise ValueError("the audited paper protocol requires max_features=4096")

    with args.cache.open("rb") as handle:
        data = pickle.load(handle)

    required = {"train_texts", "val_texts", "train_tlbls", "val_tlbls"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"cache is missing required fields: {sorted(missing)}")

    corpus = [str(text) for text in data["train_texts"]]
    queries = [str(text) for text in data["val_texts"]]
    corpus_labels = np.asarray(
        [scalar_label(value) for value in data["train_tlbls"]], dtype=np.int64
    )
    query_labels = np.asarray(
        [scalar_label(value) for value in data["val_tlbls"]], dtype=np.int64
    )
    if len(corpus) != 2400 or len(queries) != 600:
        raise ValueError(
            f"expected 2,400 training and 600 validation notes, got "
            f"{len(corpus)} and {len(queries)}"
        )
    if len(corpus_labels) != len(corpus) or len(query_labels) != len(queries):
        raise ValueError("text and label counts disagree")
    if set(np.unique(corpus_labels)) != set(range(5)):
        raise ValueError("training corpus does not contain all five labels")
    if set(np.unique(query_labels)) != set(range(5)):
        raise ValueError("validation queries do not contain all five labels")

    # Crucially, validation queries are not passed to fit().
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    corpus_matrix = vectorizer.fit_transform(corpus)
    query_matrix = vectorizer.transform(queries)
    corpus_dense = corpus_matrix.toarray().astype("float32")
    query_dense = query_matrix.toarray().astype("float32")
    faiss.normalize_L2(corpus_dense)
    faiss.normalize_L2(query_dense)

    index = faiss.IndexFlatIP(corpus_dense.shape[1])
    index.add(corpus_dense)
    similarities, neighbour_indices = index.search(query_dense, args.top_k)

    per_condition: dict[str, dict] = {}
    top1_correct_all: list[float] = []
    precision_all: list[float] = []
    similarity_all: list[float] = []
    for label_id, label_name in enumerate(LABELS):
        positions = np.flatnonzero(query_labels == label_id)
        retrieved_labels = corpus_labels[neighbour_indices[positions]]
        top1_correct = (retrieved_labels[:, 0] == label_id).astype(float)
        precision = (retrieved_labels == label_id).mean(axis=1)
        top1_similarity = similarities[positions, 0].astype(float)
        top1_correct_all.extend(top1_correct.tolist())
        precision_all.extend(precision.tolist())
        similarity_all.extend(top1_similarity.tolist())
        per_condition[label_name] = {
            "n": int(len(positions)),
            "top1_accuracy": float(top1_correct.mean()),
            "precision_at_k": float(precision.mean()),
            "mean_top1_similarity": float(top1_similarity.mean()),
            "std_top1_similarity": float(top1_similarity.std(ddof=0)),
        }

    macro_top1 = float(
        np.mean([row["top1_accuracy"] for row in per_condition.values()])
    )
    macro_precision = float(
        np.mean([row["precision_at_k"] for row in per_condition.values()])
    )
    macro_similarity = float(
        np.mean([row["mean_top1_similarity"] for row in per_condition.values()])
    )
    result = {
        "n_queries": len(queries),
        "corpus_size": len(corpus),
        "top_k": args.top_k,
        "per_condition": per_condition,
        "condition_macro_top1_accuracy": macro_top1,
        "condition_macro_precision_at_k": macro_precision,
        "condition_macro_mean_top1_similarity": macro_similarity,
        # Compatibility aliases are explicitly defined as condition-macro.
        "overall_top1_accuracy": macro_top1,
        "overall_mean_top1_similarity": macro_similarity,
        "micro_top1_accuracy": float(np.mean(top1_correct_all)),
        "micro_precision_at_k": float(np.mean(precision_all)),
        "micro_mean_top1_similarity": float(np.mean(similarity_all)),
    }
    runner = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "task": "heldout_tfidf_faiss_retrieval",
        "key": "fit=train_only|index=train|query=validation|top_k=5",
        "protocol": {
            "fit_scope": "training notes only",
            "index_scope": "2,400 training notes",
            "query_scope": "600 validation notes transformed after fit",
            "labels_used_for": "evaluation only",
            "vectorizer": "TfidfVectorizer(max_features=4096)",
            "normalization": "L2",
            "search": "exact FAISS IndexFlatIP",
            "similarity_sd_ddof": 0,
        },
        "provenance": {
            "data_cache_sha256": sha256_file(args.cache),
            "runner_sha256": sha256_file(runner),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "result": result,
    }
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "condition_macro_top1_accuracy": macro_top1,
                "condition_macro_precision_at_k": macro_precision,
                "condition_macro_mean_top1_similarity": macro_similarity,
                "n_queries": len(queries),
                "vocabulary_size": len(vectorizer.vocabulary_),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
