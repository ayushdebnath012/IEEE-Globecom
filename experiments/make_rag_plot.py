"""Plot the validated 600-query retrieval evaluation used in the paper.

The input must be the merged corrected-results JSON.  This intentionally does
not reuse the five-query pre-audit heatmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf8"))["E7_rag"]["heldout"]
    keys = ["NORMAL", "PNEUMONIA", "COVID19", "PLEURAL_EFFUSION", "CARDIOMEGALY"]
    labels = ["Normal", "Pneum.", "COVID-19", "Effusion", "Cardiom."]
    rows = [data["per_condition"][key] for key in keys]

    top1 = np.array([row["top1_accuracy"] for row in rows])
    precision = np.array([row["precision_at_k"] for row in rows])
    similarity = np.array([row["mean_top1_similarity"] for row in rows])
    sim_std = np.array([row["std_top1_similarity"] for row in rows])

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.15), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, top1, width, color="#4C72B0", edgecolor="black",
           linewidth=0.35, label="Top-1 acc.")
    ax.bar(x, precision, width, color="#55A868", edgecolor="black",
           linewidth=0.35, label="P@5")
    ax.errorbar(x + width, similarity, yerr=sim_std, fmt="D", ms=3.2,
                capsize=2, color="#C44E52", linewidth=0.8, label="Top-1 sim.")
    ax.axhline(0.2, color="#777777", linestyle="--", linewidth=0.8,
               label="5-class prior")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.15, 0.86)
    ax.set_title("Validated retrieval (600 held-out queries)", fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8,
              handlelength=1.3, borderpad=0.1, labelspacing=0.25)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
