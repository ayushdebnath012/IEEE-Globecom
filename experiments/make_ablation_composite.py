"""Build a compact paper figure that preserves the original modality plot.

Panels (b) and (c) are generated only from the validated merged reviewer JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def avg_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--old-plot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.results.read_text(encoding="utf8"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.35), constrained_layout=True)

    axes[0].imshow(mpimg.imread(args.old_plot))
    axes[0].axis("off")
    axes[0].set_title("(a) Original modality ablation", fontsize=10, pad=3)

    variants = ["full", "no_balanced", "no_diversity", "neither"]
    labels = ["Full", "$-$Sampler", "$-$Div.", "$-$Both"]
    colors = ["#276FBF", "#F18F01", "#5FAD56", "#A23B72"]
    width = 0.19
    x = np.arange(2)
    for i, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        vals, errs = [], []
        for alpha in (0.1, 1.0):
            rows = [
                value["f1"]
                for key, value in data["E3_anticollapse"].items()
                if key.startswith(f"{variant}|alpha={alpha}|")
            ]
            m, s = avg_sd(rows)
            vals.append(m)
            errs.append(s)
        axes[1].bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2,
                    color=color, label=label, edgecolor="black", linewidth=0.35)
    axes[1].set_xticks(x, [r"$\alpha=0.1$", r"$\alpha=1.0$"])
    axes[1].set_ylim(0.35, 0.95)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("(b) Anti-collapse components", fontsize=10)
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.5)
    axes[1].legend(fontsize=7, ncol=2, frameon=False, loc="lower right")

    abort = data["E3b_early_abort"]
    init = data["E4_warmstart"]
    abort_on = avg_sd([v["f1"] for k, v in abort.items() if "early_abort=True" in k])
    abort_off = avg_sd([v["f1"] for k, v in abort.items() if "early_abort=False" in k])
    oracle_rows = [v for k, v in init.items() if "warm_start=True" in k]
    cold_rows = [v for k, v in init.items() if "warm_start=False" in k]
    oracle = avg_sd([v["f1"] for v in oracle_rows])
    cold = avg_sd([v["f1"] for v in cold_rows])
    values = [abort_on[0], abort_off[0], oracle[0], cold[0]]
    errors = [abort_on[1], abort_off[1], oracle[1], cold[1]]
    xpos = np.array([0, 1, 2.5, 3.5])
    axes[2].bar(xpos, values, yerr=errors, capsize=3,
                color=["#276FBF", "#9DB9D7", "#F18F01", "#F7C978"],
                edgecolor="black", linewidth=0.4)
    axes[2].scatter([2.5, 3.5],
                    [mean(v["history"]["round_f1"][0] for v in oracle_rows),
                     mean(v["history"]["round_f1"][0] for v in cold_rows)],
                    marker="D", s=24, color="black", label="Round 1")
    axes[2].set_xticks(xpos, ["Abort\non", "Abort\noff", "Oracle\ninit.", "Operational\ninit."],
                       fontsize=8)
    axes[2].set_ylim(0.6, 0.92)
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("(c) Training controls", fontsize=10)
    axes[2].grid(axis="y", alpha=0.22, linewidth=0.5)
    axes[2].legend(fontsize=7, frameon=False, loc="lower left")

    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
