"""Second results figure. Every panel maps to a specific reviewer request.

All values are recomputed from results_corrected_merged.json; no panel reuses a
pre-audit run or embeds a bitmap of one.

  (a) Dirichlet alpha sweep         -> R3.2 "scalability ... under varying data distributions"
  (b) fusion-operator screen        -> R3.1 "quantify ... each fusion strategy"
  (c) early-abort + warm-start init -> R3.1 "warm-start initialization mechanism"
  (d) communication per branch      -> R3.2 "communication cost"; R1.3 "communication overhead"
  (e) runtime + peak memory         -> R3.2 "runtime, memory consumption"
  (f) communication vs client count -> R3.2 "scalability ... under varying numbers of clients"

Usage:
  python make_systems_panels.py --results results_corrected_merged.json \
      --out ../generated/fig_systems_panels.png
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
GIB, MIB = 1024 ** 3, 1024 ** 2

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def agg(vals: list[float]) -> tuple[float, float]:
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def by_prefix(block: dict, prefix: str, field: str = "f1") -> list[float]:
    return [v[field] for k, v in block.items() if k.startswith(prefix)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.results.read_text(encoding="utf8"))
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.2), constrained_layout=True)
    ax = axes.ravel()
    echo: list[str] = []

    # ---- (a) alpha sweep ------------------------------------- R3.2 ---------
    e1 = d["E1_alpha_sweep"]
    alphas = [0.1, 0.3, 0.5, 1.0, 5.0]
    mus, sds = zip(*(agg(by_prefix(e1, f"alpha={a}|")) for a in alphas))
    ax[0].errorbar(range(len(alphas)), mus, yerr=sds, marker="o", ms=4.5, lw=1.7,
                   capsize=3, color=RED)
    ax[0].set_xticks(range(len(alphas)))
    ax[0].set_xticklabels([str(a) for a in alphas])
    ax[0].set_xlabel(r"Dirichlet $\alpha$")
    ax[0].set_ylabel("Macro F1")
    ax[0].set_title("(a) Data distributions")
    echo += [f"  alpha={a}: {m:.3f} +/- {s:.3f}" for a, m, s in zip(alphas, mus, sds)]

    # ---- (b) fusion-operator screen -------------------------- R3.1 ---------
    e5 = d["E5_fusion_seeds"]
    pretty = {"concat": "Concat", "attention": "Cross-attn", "gated": "Gated",
              "clip": "Dual-proj", "flamingo": "Decoder A", "blip2": "Decoder B",
              "coca": "Cross-attn384", "unified_io": "Token enc."}
    ops = sorted({k.split("|")[0] for k in e5})
    stats = sorted(((pretty.get(o, o), *agg(by_prefix(e5, f"{o}|"))) for o in ops),
                   key=lambda t: t[1])
    names, vals, errs = [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats]
    ax[1].barh(range(len(names)), vals, xerr=errs, color=GREEN, capsize=2.5,
               edgecolor="black", linewidth=0.4, error_kw={"lw": 0.9})
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=9.5)
    ax[1].set_xlim(0.6, 1.0)
    ax[1].set_xlabel("Macro F1")
    ax[1].set_title("(b) Fusion strategy")
    echo += [f"  fusion {n}: {v:.3f} +/- {e:.3f}" for n, v, e in zip(names, vals, errs)]

    # ---- (c) early-abort and warm-start ---------------------- R3.1 ---------
    e3b, e4 = d["E3b_early_abort"], d["E4_warmstart"]
    groups = [("Abort\non", by_prefix(e3b, "early_abort=True"), None),
              ("Abort\noff", by_prefix(e3b, "early_abort=False"), None),
              ("Oracle\ninit", by_prefix(e4, "warm_start=True"), "warm_start=True"),
              ("Cold\ninit", by_prefix(e4, "warm_start=False"), "warm_start=False")]
    colours = [BLUE, "#A8C4E0", SAND, "#E8D9A8"]
    for i, ((label, vals_g, warm_key), colour) in enumerate(zip(groups, colours)):
        m, s = agg(vals_g)
        ax[2].bar(i, m, yerr=s, capsize=3, color=colour, edgecolor="black",
                  linewidth=0.5, error_kw={"lw": 1.0})
        if warm_key is not None:
            r1 = mean([v["history"]["round_f1"][0]
                       for k, v in e4.items() if k.startswith(warm_key)])
            ax[2].plot(i, r1, marker="D", ms=6, color="black", zorder=5)
            echo.append(f"  {label.replace(chr(10), ' ')} round-1 {r1:.3f}")
        echo.append(f"  {label.replace(chr(10), ' ')}: {m:.3f} +/- {s:.3f}")
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels([g[0] for g in groups], fontsize=10)
    ax[2].set_ylim(0.6, 0.95)
    ax[2].set_ylabel("Macro F1")
    ax[2].set_title("(c) Warm-start / abort")

    # ---- (d) communication per branch ------------------------ R3.2 / R1.3 --
    e6 = d["E6_cost"]
    branches = ["Fed-LLM", "Fed-ViT", "Fed-VLM"]
    colours_b = [BLUE, GREEN, RED]
    payload = [e6[b]["upload_bytes_per_client_per_round"] / MIB for b in branches]
    ax[3].bar(range(3), payload, color=colours_b, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(payload):
        ax[3].text(i, v + 14, f"{v:.0f}", ha="center", fontsize=10)
    ax[3].set_xticks(range(3))
    ax[3].set_xticklabels(["LLM", "ViT", "VLM"], fontsize=10.5)
    ax[3].set_ylim(0, max(payload) * 1.22)
    ax[3].set_ylabel("Payload (MiB)")
    ax[3].set_title("(d) Communication cost")
    echo += [f"  {b} payload {p:.0f} MiB, total "
             f"{e6[b]['total_comm_bytes'] / GIB:.1f} GiB"
             for b, p in zip(branches, payload)]

    # ---- (e) runtime and peak memory ------------------------- R3.2 ---------
    rounds = [len(e6[b]["history"]["round_f1"]) for b in branches]
    spr = [e6[b]["wall_seconds"] / r for b, r in zip(branches, rounds)]
    memp = [e6[b]["peak_mib"] for b in branches]
    x = np.arange(3)
    ax[4].bar(x - 0.2, spr, 0.4, color=PURPLE, edgecolor="black", linewidth=0.4,
              label="s / round")
    ax4b = ax[4].twinx()
    ax4b.bar(x + 0.2, memp, 0.4, color=SAND, edgecolor="black", linewidth=0.4,
             label="Peak MiB")
    ax4b.grid(False)
    ax[4].set_xticks(x)
    ax[4].set_xticklabels(["LLM", "ViT", "VLM"], fontsize=10.5)
    ax[4].set_ylabel("Time / round (s)")
    ax4b.set_ylabel("Peak (MiB)")
    ax[4].set_ylim(0, max(spr) * 1.35)
    ax4b.set_ylim(0, max(memp) * 1.35)
    h1, l1 = ax[4].get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax[4].legend(h1 + h2, l1 + l2, frameon=False, fontsize=9.5, ncol=2,
                 loc="upper left", handlelength=1.0, columnspacing=0.7,
                 borderpad=0.1)
    ax[4].set_title("(e) Runtime & memory")
    echo += [f"  {b}: {s:.0f} s/round, {m:.0f} MiB peak"
             for b, s, m in zip(branches, spr, memp)]

    # ---- (f) communication vs client count ------------------- R3.2 ---------
    e2 = d["E2_client_sweep"]
    ks = [3, 5, 10, 20]
    comm = [mean(by_prefix(e2, f"K={k}|", "total_comm_bytes")) / GIB for k in ks]
    ax[5].plot(ks, comm, marker="o", ms=4.5, lw=1.7, color=PURPLE)
    ax[5].set_xticks(ks)
    ax[5].set_xticklabels([str(k) for k in ks])
    ax[5].set_xlabel("Clients $K$")
    ax[5].set_ylabel("Comm. (GiB)")
    ax[5].set_title("(f) Scaling in $K$")
    echo += [f"  K={k}: {c:.1f} GiB" for k, c in zip(ks, comm)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}\n\nplotted values:")
    print("\n".join(echo))


if __name__ == "__main__":
    main()
