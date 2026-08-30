"""Plot the validated severe-skew client label realization used in the paper."""

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

    record = json.loads(args.results.read_text(encoding="utf8"))["E1_alpha_sweep"][
        "alpha=0.1|seed=0"
    ]
    counts = np.asarray(record["client_class_hist"], dtype=float)
    sizes = np.asarray(record["client_sizes"], dtype=float)
    shares = counts / sizes[:, None]
    classes = ["Normal", "Pneum.", "COVID-19", "Effusion", "Cardiom."]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(3.45, 1.95), constrained_layout=True)
    image = ax.imshow(shares, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_title(r"Realized severe-skew partition ($\alpha=0.1$, seed 0)")
    ax.set_xticks(range(5), classes, rotation=20, ha="right")
    ax.set_yticks(range(5), [f"C{i+1} ({int(n)})" for i, n in enumerate(sizes)])
    for row in range(5):
        for col in range(5):
            value = shares[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if value > 0.55 else "black")
    colourbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colourbar.set_label("Client label share", fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
