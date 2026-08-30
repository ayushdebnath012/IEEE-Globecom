"""Plot representative realized client label shares from validated E1 runs."""

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

    results = json.loads(args.results.read_text(encoding="utf8"))["E1_alpha_sweep"]
    keys = [("alpha=0.1|seed=0", r"(a) Severe skew: $\alpha=0.1$"),
            ("alpha=1.0|seed=0", r"(b) Moderate skew: $\alpha=1.0$")]
    class_names = ["Normal", "Pneum.", "COVID-19", "Effusion", "Cardiom."]

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.05), constrained_layout=True)
    image = None
    for ax, (key, title) in zip(axes, keys):
        record = results[key]
        counts = np.asarray(record["client_class_hist"], dtype=float)
        sizes = np.asarray(record["client_sizes"], dtype=float)
        shares = counts / sizes[:, None]
        image = ax.imshow(shares, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)), class_names, rotation=18, ha="right")
        ax.set_yticks(range(5), [f"C{i+1} (n={int(n)})" for i, n in enumerate(sizes)])
        ax.set_xlabel("Class share within client")
        for row in range(shares.shape[0]):
            for col in range(shares.shape[1]):
                value = shares[row, col]
                colour = "white" if value > 0.55 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=7, color=colour)

    assert image is not None
    colourbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colourbar.set_label("Client label share", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
