"""Single-column retrieval figure, recomputed from the validated merged JSON.

Covers all 600 held-out queries (the corrected protocol), not the withdrawn
five-query probe.

Usage:
  python make_rag_panel.py --results reviewer_results_merged.json \
      --out ../generated/fig_rag_validated.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, SAND, RED = "#4C72B0", "#CCB974", "#C44E52"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

SHORT = {"NORMAL": "Normal", "PNEUMONIA": "Pneum.", "COVID19": "COVID",
         "PLEURAL_EFFUSION": "Pleural", "CARDIOMEGALY": "Cardio."}
CONDITIONS = tuple(SHORT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.results.read_text(encoding="utf8"))
    meta = data.get("_meta")
    if not isinstance(meta, dict) or meta.get("corrected_retrieval_protocol_validated") is not True:
        raise ValueError("merged results do not certify the corrected retrieval protocol")
    rag_group = data.get("retrieval")
    if not isinstance(rag_group, dict) or "heldout" not in rag_group:
        raise ValueError("results JSON lacks the corrected retrieval section")
    rag = rag_group["heldout"]
    pc = rag["per_condition"]
    if set(pc) != set(CONDITIONS):
        raise ValueError("retrieval section does not contain the five expected conditions")
    conds = list(CONDITIONS)
    x = np.arange(len(conds))

    top1 = [pc[c]["top1_accuracy"] for c in conds]
    pk = [pc[c]["precision_at_k"] for c in conds]
    sim = [pc[c]["mean_top1_similarity"] for c in conds]
    sim_sd = [pc[c]["std_top1_similarity"] for c in conds]
    total_queries = sum(pc[c]["n"] for c in conds)
    macro_p_at_5 = rag["condition_macro_precision_at_k"]
    if total_queries != rag["n_queries"]:
        raise ValueError("per-condition retrieval counts do not match n_queries")

    fig, ax = plt.subplots(figsize=(5.0, 2.3), constrained_layout=True)
    ax.bar(x - 0.2, top1, 0.4, color=BLUE, edgecolor="black", linewidth=0.4,
           label="Top-1 label acc.")
    ax.bar(x + 0.2, pk, 0.4, color=SAND, edgecolor="black", linewidth=0.4,
           label="Label P@5")
    ax.errorbar(x, sim, yerr=sim_sd, fmt="D", ms=5, color="black", capsize=3,
                lw=1.0, label="Top-1 sim.")
    ax.axhline(0.2, color=RED, lw=1.2, ls="--")
    ax.text(
        len(conds) - 0.45,
        0.225,
        "label prior",
        color=RED,
        fontsize=9,
        ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in conds])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=1.0,
              columnspacing=0.9, borderpad=0.1, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")
    print(f"condition-macro: top1={rag['condition_macro_top1_accuracy']:.3f} "
          f"macro_p@5={macro_p_at_5:.3f} "
          f"sim={rag['condition_macro_mean_top1_similarity']:.3f} n={rag['n_queries']}")
    for c in conds:
        print(f"  {c:<18} n={pc[c]['n']:<4} top1={pc[c]['top1_accuracy']:.3f} "
              f"p@5={pc[c]['precision_at_k']:.3f} sim={pc[c]['mean_top1_similarity']:.3f}")


if __name__ == "__main__":
    main()
