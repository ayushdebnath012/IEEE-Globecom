"""Build the paper's results figure strip from the validated merged JSON only.

Unlike make_ablation_composite.py, no panel embeds a pre-audit plot image: every
number drawn here is recomputed from results_corrected_merged.json. Panels:

  (a) matched federated baselines at alpha=0.1   -> Table "baselines"  (R3.3)
  (b) per-round convergence by update rule       -> stability evidence (R3.3)
  (c) anti-collapse components at both alphas    -> Table "anticollapse" (R3.1)
  (d) client scaling at alpha=1.0                -> Table "clients"     (R3.2/R1.3)

Usage:
  python make_validated_panels.py --results results_corrected_merged.json \
      --out ../generated/fig_results_panels.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, RED, PURPLE, SAND = "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def agg(values: list[float]) -> tuple[float, float]:
    """Mean and sample std; std is 0.0 when only one seed is available."""
    if not values:
        raise ValueError("no values to aggregate")
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def seed_values(block: dict, prefix: str, field: str = "f1") -> list[float]:
    return [v[field] for k, v in block.items() if k.startswith(prefix)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.results.read_text(encoding="utf8"))
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.6), constrained_layout=True)

    # ---- (a) matched baselines at alpha = 0.1 -------------------------------
    e8 = d["E8_baselines"]
    local = e8["local_only|alpha=0.1|seed=0"]
    names = ["Local", "FedAvg", "FedProx", "SCAF-AdamW"]
    colors = [SAND, BLUE, GREEN, RED]
    means = [local["mean_f1"]]
    errs = [local["std_f1"]]
    for algo in ("fedavg", "fedprox", "scaffold"):
        m, s = agg(seed_values(e8, f"{algo}|alpha=0.1"))
        means.append(m)
        errs.append(s)
    ax[0].bar(range(4), means, yerr=errs, capsize=3, color=colors,
              edgecolor="black", linewidth=0.5, error_kw={"lw": 1.1})
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(names, fontsize=11.5, rotation=22, ha="right")
    ax[0].set_ylabel("Macro F1")
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title("(a) Matched baselines")

    # ---- (b) per-round convergence by update rule ---------------------------
    for algo, colour, label in (("fedavg", BLUE, "FedAvg"),
                                ("fedprox", GREEN, "FedProx"),
                                ("scaffold", RED, "SCAFFOLD-AdamW")):
        curves = [v["history"]["round_f1"]
                  for k, v in e8.items() if k.startswith(f"{algo}|alpha=0.1")]
        arr = np.array(curves, dtype=float)
        rounds = np.arange(1, arr.shape[1] + 1)
        mu = arr.mean(axis=0)
        ax[1].plot(rounds, mu, marker="o", ms=3, lw=1.6, color=colour, label=label)
        if arr.shape[0] > 1:
            ax[1].fill_between(rounds, arr.min(axis=0), arr.max(axis=0),
                               color=colour, alpha=0.16, linewidth=0)
    ax[1].set_xlabel("Federated round")
    ax[1].set_ylabel("Macro F1")
    ax[1].set_ylim(0, 1.0)
    ax[1].legend(frameon=False, loc="lower right", handlelength=1.2, fontsize=10.5,
                 borderpad=0.1, labelspacing=0.25)
    ax[1].set_title("(b) Convergence by round")

    # ---- (c) anti-collapse components --------------------------------------
    e3 = d["E3_anticollapse"]
    variants = ["full", "no_balanced", "no_diversity", "neither"]
    labels = ["Full", r"$-$Sampler", r"$-$Div.", r"$-$Both"]
    colors_c = [BLUE, SAND, GREEN, PURPLE]
    x = np.arange(2)
    width = 0.2
    for i, (variant, label, colour) in enumerate(zip(variants, labels, colors_c)):
        mus, sds = [], []
        for alpha in (0.1, 1.0):
            m, s = agg(seed_values(e3, f"{variant}|alpha={alpha}"))
            mus.append(m)
            sds.append(s)
        ax[2].bar(x + (i - 1.5) * width, mus, width, yerr=sds, capsize=2,
                  color=colour, edgecolor="black", linewidth=0.4,
                  label=label, error_kw={"lw": 0.9})
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([r"$\alpha$=0.1", r"$\alpha$=1.0"])
    ax[2].set_ylabel("Macro F1")
    ax[2].set_ylim(0.3, 1.18)
    ax[2].legend(frameon=False, ncol=4, loc="upper center", handlelength=0.9,
                 columnspacing=0.6, borderpad=0.1, handletextpad=0.4,
                 fontsize=10.5, borderaxespad=0.2)
    ax[2].set_title("(c) Anti-collapse parts")

    # ---- (d) client scaling -------------------------------------------------
    e2 = d["E2_client_sweep"]
    ks = [3, 5, 10, 20]
    mus, sds = [], []
    for k in ks:
        m, s = agg(seed_values(e2, f"K={k}|"))
        mus.append(m)
        sds.append(s)
    # Categorical x-axis: the sweep is not evenly spaced and a log axis leaks
    # minor tick labels into the major ones at this figure width.
    pos = np.arange(len(ks))
    ax[3].errorbar(pos, mus, yerr=sds, marker="o", ms=4, lw=1.6, capsize=3,
                   color=PURPLE)
    ax[3].set_xticks(pos)
    ax[3].set_xticklabels([str(k) for k in ks])
    ax[3].set_xlim(-0.35, len(ks) - 0.65)
    ax[3].set_xlabel("Clients $K$")
    ax[3].set_ylabel("Macro F1")
    ax[3].set_title("(d) Client scaling")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")

    # Echo the plotted aggregates so they can be checked against the tables.
    print("\nplotted values (mean +/- sample std):")
    print(f"  local-only      {means[0]:.3f} +/- {errs[0]:.3f}")
    for name, m, s in zip(["fedavg", "fedprox", "scaffold"], means[1:], errs[1:]):
        print(f"  {name:<15} {m:.3f} +/- {s:.3f}")
    for variant in variants:
        for alpha in (0.1, 1.0):
            m, s = agg(seed_values(e3, f"{variant}|alpha={alpha}"))
            print(f"  {variant:<13} a={alpha:<4} {m:.3f} +/- {s:.3f}")
    for k, m, s in zip(ks, mus, sds):
        print(f"  K={k:<13} {m:.3f} +/- {s:.3f}")


if __name__ == "__main__":
    main()
