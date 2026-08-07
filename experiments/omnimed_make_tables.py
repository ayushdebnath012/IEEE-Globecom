"""
Turn results_v2.json (from omnimed_experiments.py) into paste-ready LaTeX
tables and figures for the OmniMed-FL paper.

    python omnimed_make_tables.py --results results_v2.json --outdir ../

Writes:
    tab_alpha_sweep.tex      E1  -> answers R3.1 (single non-IID setting)
    tab_client_sweep.tex     E2  -> answers R3.4 (scalability)
    tab_anticollapse.tex     E3  -> answers R3.2 (component ablation)
    tab_warmstart.tex        E4  -> answers R3.2 (warm start)
    tab_fusion_variance.tex  E5  -> answers R3.1 (is the ranking real?)
    tab_measured_cost.tex    E6  -> answers R1.2/R3.4 (runtime, memory, comm)
    tab_rag.tex              E7  -> answers R3.2 (retrieval at proper scale)
    fig_alpha_sweep.png      E1 plot
    fig_client_scaling.png   E2 plot
    SUMMARY.md               plain-language readout of what each result shows

Aggregation is mean +/- sample std over seeds. Where only one seed exists the
std column prints "--" rather than 0.00, so a single-seed run cannot be mistaken
for a converged estimate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_key(key: str) -> dict:
    out = {}
    for part in key.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out["variant"] = part
    return out


FUSION_DISPLAY = {
    "concat": "Concat", "attention": "Cross-Attn", "gated": "Gated",
    "clip": "CLIP", "flamingo": "Flamingo", "blip2": "BLIP-2",
    "coca": "CoCa", "unified_io": "Unified-IO",
}


def tex_escape(s: str) -> str:
    """Escape the characters that break a LaTeX table cell."""
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def agg(values):
    a = np.asarray(values, dtype=float)
    if a.size == 1:
        return a[0], None
    return a.mean(), a.std(ddof=1)


def fmt(mean, std, prec=3):
    if std is None:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}\\,$\\pm$\\,{std:.{prec}f}"


def group(exp: dict, by: str, metric: str = "f1"):
    """Collect metric values grouped by one key field."""
    buckets = defaultdict(list)
    for key, rec in exp.items():
        p = parse_key(key)
        if by in p:
            buckets[p[by]].append(rec[metric])
    return buckets


# ---------------------------------------------------------------------------


def tab_alpha(d, outdir):
    exp = d.get("E1_alpha_sweep")
    if not exp:
        return None
    rows = []
    f1 = group(exp, "alpha", "f1")
    div = group(exp, "alpha", "diversity")
    for a in sorted(f1, key=float):
        m, s = agg(f1[a])
        dm, _ = agg(div[a])
        # realized heterogeneity: mean over clients of max class share
        skews = []
        for key, rec in exp.items():
            if parse_key(key)["alpha"] != a:
                continue
            for h in rec.get("client_class_hist", []):
                tot = sum(h)
                if tot:
                    skews.append(max(h) / tot)
        skew = np.mean(skews) if skews else float("nan")
        rows.append((a, skew, fmt(m, s), f"{dm:.2f}"))

    tex = [
        r"\begin{table}[t]",
        r"\caption{Effect of partition heterogeneity on Fed-VLM. $\alpha$ is the",
        r"Dirichlet concentration; \emph{skew} is the realized mean per-client share of",
        r"the dominant class. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:alpha}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\alpha$ & Skew & Macro F1 & Diversity\\",
        r"\midrule",
    ]
    for a, skew, f, dv in rows:
        tex.append(f"{a} & {skew:.2f} & {f} & {dv} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_alpha_sweep.tex", tex)


def tab_clients(d, outdir):
    exp = d.get("E2_client_sweep")
    if not exp:
        return None
    f1 = group(exp, "K", "f1")
    tex = [
        r"\begin{table}[t]",
        r"\caption{Scalability in the number of clients $K$ at $\alpha{=}1.0$.",
        r"Communication is the measured total exchanged over the run,",
        r"$V=2KT|\theta|b$. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:clients}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$K$ & Macro F1 & Wall-clock (s) & Total comm. (GiB)\\",
        r"\midrule",
    ]
    for K in sorted(f1, key=int):
        m, s = agg(f1[K])
        secs = [r["wall_seconds"] for k, r in exp.items() if parse_key(k)["K"] == K]
        gib = [r["total_comm_bytes"] / 2 ** 30 for k, r in exp.items()
               if parse_key(k)["K"] == K]
        tex.append(f"{K} & {fmt(m, s)} & {np.mean(secs):.0f} & {np.mean(gib):.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_client_sweep.tex", tex)


def tab_anticollapse(d, outdir):
    exp = d.get("E3_anticollapse")
    if not exp:
        return None
    label = {"full": "Full stack",
             "no_balanced": r"$-$ balanced sampler",
             "no_diversity": r"$-$ diversity term",
             "neither": r"$-$ both"}
    alphas = sorted({parse_key(k)["alpha"] for k in exp}, key=float)
    tex = [
        r"\begin{table}[t]",
        r"\caption{Anti-collapse component ablation on Fed-VLM. Each row removes one",
        r"component from the full stack. Diversity is the fraction of the five classes",
        r"predicted at the final round. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:anticollapse}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{l" + "cc" * len(alphas) + "}",
        r"\toprule",
        r" & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{$\alpha={a}$}}" for a in alphas) + r"\\",
    ]
    tex.append("".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(alphas))))
    tex.append(r"Configuration & " + " & ".join(["F1 & Div."] * len(alphas)) + r"\\")
    tex.append(r"\midrule")
    for v in ["full", "no_balanced", "no_diversity", "neither"]:
        cells = []
        for a in alphas:
            f1s = [r["f1"] for k, r in exp.items()
                   if parse_key(k).get("variant") == v and parse_key(k)["alpha"] == a]
            dvs = [r["diversity"] for k, r in exp.items()
                   if parse_key(k).get("variant") == v and parse_key(k)["alpha"] == a]
            if not f1s:
                cells += ["--", "--"]
                continue
            m, s = agg(f1s)
            cells += [fmt(m, s), f"{np.mean(dvs):.2f}"]
        tex.append(f"{label[v]} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_anticollapse.tex", tex)


def tab_warmstart(d, outdir):
    exp = d.get("E4_warmstart")
    if not exp:
        return None
    f1 = group(exp, "warm_start", "f1")
    tex = [
        r"\begin{table}[t]",
        r"\caption{Warm-start ablation: federated training from centralized",
        r"pretrained weights versus from a cold initialization.}",
        r"\label{tab:warmstart}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Initialization & Macro F1 & Round-1 F1\\",
        r"\midrule",
    ]
    for w, name in [("True", "Warm start (pretrained)"), ("False", "Cold start (random)")]:
        if w not in f1:
            continue
        m, s = agg(f1[w])
        r1 = [r["history"]["round_f1"][0] for k, r in exp.items()
              if parse_key(k)["warm_start"] == w]
        tex.append(f"{name} & {fmt(m, s)} & {np.mean(r1):.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_warmstart.tex", tex)


def tab_fusion(d, outdir):
    exp = d.get("E5_fusion_seeds")
    if not exp:
        return None
    buckets = defaultdict(list)
    for k, r in exp.items():
        buckets[parse_key(k)["variant"]].append(r["f1"])
    ordered = sorted(buckets.items(), key=lambda kv: -np.mean(kv[1]))
    tex = [
        r"\begin{table}[t]",
        r"\caption{Fusion strategies with repeated seeds. The std column is what",
        r"decides whether the ordering is real; strategies whose intervals overlap",
        r"should be treated as tied.}",
        r"\label{tab:fusionvar}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Fusion & Macro F1 & Seeds\\",
        r"\midrule",
    ]
    for name, vals in ordered:
        m, s = agg(vals)
        disp = FUSION_DISPLAY.get(name, tex_escape(name))
        tex.append(f"{disp} & {fmt(m, s)} & {len(vals)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_fusion_variance.tex", tex)


def tab_cost(d, outdir):
    exp = d.get("E6_cost")
    if not exp:
        return None
    tex = [
        r"\begin{table}[t]",
        r"\caption{Measured cost per branch at $K{=}5$, $\alpha{=}1.0$. Payload is",
        r"per client per round; wall-clock and peak memory are measured on the GPU",
        r"named in the results metadata.}",
        r"\label{tab:comm}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Branch & $|\theta|$ & Payload & Total $V$ & s/round & Peak mem.\\",
        r"\midrule",
    ]
    for name in ["Fed-LLM", "Fed-ViT", "Fed-VLM"]:
        r = exp.get(name)
        if not r:
            continue
        p = r["trainable_params"] / 1e6
        pay = r["upload_bytes_per_client_per_round"] / 2 ** 20
        tot = r["total_comm_bytes"] / 2 ** 30
        spr = np.mean(r["history"]["round_seconds"])
        mem = r["peak_mib"]
        tex.append(f"{name} & {p:.1f}\\,M & {pay:.0f}\\,MiB & {tot:.1f}\\,GiB "
                   f"& {spr:.0f} & {mem:.0f}\\,MiB \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_measured_cost.tex", tex)


def tab_rag(d, outdir):
    exp = (d.get("E7_rag") or {}).get("heldout")
    if not exp or "error" in exp:
        return None
    tex = [
        r"\begin{table}[t]",
        rf"\caption{{Retrieval evaluation over {exp['n_queries']} held-out queries",
        rf"against a {exp['corpus_size']}-note index (top-$k$={exp['top_k']}).}}",
        r"\label{tab:rag}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & $n$ & Top-1 acc. & P@$k$ & Top-1 sim.\\",
        r"\midrule",
    ]
    for cond, v in exp["per_condition"].items():
        tex.append(f"{tex_escape(cond.replace('_',' ').title())} & {v['n']} & "
                   f"{v['top1_accuracy']:.3f} & {v['precision_at_k']:.3f} & "
                   f"{v['mean_top1_similarity']:.3f}\\,$\\pm$\\,{v['std_top1_similarity']:.3f} \\\\")
    tex += [
        r"\midrule",
        rf"\textbf{{Overall}} & {exp['n_queries']} & "
        rf"\textbf{{{exp['overall_top1_accuracy']:.3f}}} & -- & "
        rf"{exp['overall_mean_top1_similarity']:.3f} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return _write(outdir / "tab_rag.tex", tex)


def tab_baselines(d, outdir):
    """R3.3 -- matched-setting comparison. This is the table that replaces the
    literature-positioning Table IV, because every arm here ran on our data."""
    exp = d.get("E8_baselines")
    if not exp:
        return None
    label = {"local_only": "Local only (no federation)", "fedavg": "FedAvg",
             "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN"}
    alphas = sorted({parse_key(k)["alpha"] for k in exp}, key=float)

    tex = [
        r"\begin{table}[t]",
        r"\caption{Federated aggregation rules under \emph{matched} settings: same",
        r"Concat-VLM backbone, data, partition, local budget and hyperparameters, so",
        r"the aggregation rule is the only variable. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:baselines}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(rf"$\alpha={a}$" for a in alphas) + r"\\",
        r"\midrule",
    ]
    for m in ["local_only", "fedavg", "fedprox", "scaffold", "fedbn"]:
        cells = []
        for a in alphas:
            if m == "local_only":
                recs = [r for k, r in exp.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not recs:
                    cells.append("--")
                    continue
                r = recs[0]
                cells.append(f"{r['mean_f1']:.3f}")
            else:
                vals = [r["f1"] for k, r in exp.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not vals:
                    cells.append("--")
                    continue
                mn, sd = agg(vals)
                cells.append(fmt(mn, sd))
        tex.append(f"{label[m]} & " + " & ".join(cells) + r" \\")
    tex += [
        r"\bottomrule", r"\end{tabular}",
        r"\vspace{1pt}", "",
        r"{\scriptsize Local-only is the mean over the five clients, each trained on its",
        r"own shard alone. Every other row is the global model after $T$ rounds.}",
        r"\end{table}",
    ]
    return _write(outdir / "tab_baselines.tex", tex)


def figs(d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    e1 = d.get("E1_alpha_sweep")
    if e1:
        f1 = group(e1, "alpha", "f1")
        xs = sorted(f1, key=float)
        m = [np.mean(f1[a]) for a in xs]
        s = [np.std(f1[a], ddof=1) if len(f1[a]) > 1 else 0 for a in xs]
        fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
        ax.errorbar([float(a) for a in xs], m, yerr=s, marker="o", ms=4,
                    lw=1.6, color="#C44E52", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel(r"Dirichlet $\alpha$ (log scale)", fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.2)
        p = outdir / "fig_alpha_sweep.png"
        fig.savefig(p, bbox_inches="tight")
        made.append(p)

    e2 = d.get("E2_client_sweep")
    if e2:
        f1 = group(e2, "K", "f1")
        xs = sorted(f1, key=int)
        m = [np.mean(f1[k]) for k in xs]
        fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
        ax.plot([int(k) for k in xs], m, marker="s", ms=4, lw=1.6, color="#4C72B0")
        ax.set_xlabel("Number of clients $K$", fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.2)
        p = outdir / "fig_client_scaling.png"
        fig.savefig(p, bbox_inches="tight")
        made.append(p)
    return made


def summary(d, outdir):
    """Plain-language readout, so the numbers get interpreted before they get pasted."""
    L = ["# Experiment readout", ""]
    meta = d.get("_meta", {})
    L += [f"- tier: `{meta.get('tier')}`  GPU: `{meta.get('gpu')}`",
          f"- train/val: {meta.get('n_train')}/{meta.get('n_val')}", ""]

    e1 = d.get("E1_alpha_sweep")
    if e1:
        f1 = group(e1, "alpha", "f1")
        xs = sorted(f1, key=float)
        lo, hi = np.mean(f1[xs[0]]), np.mean(f1[xs[-1]])
        ref = np.mean(f1["1.0"]) if "1.0" in f1 else hi
        L += ["## E1 alpha sweep (R3.1)",
              f"- most heterogeneous (alpha={xs[0]}): F1 {lo:.3f}",
              f"- least heterogeneous (alpha={xs[-1]}): F1 {hi:.3f}",
              f"- cost of severe heterogeneity vs the paper's alpha=1.0: {lo-ref:+.3f}",
              "- If that cost is small, the 99.1% retention claim generalizes beyond "
              "alpha=1.0 and you can say so. If it is large, the claim must be scoped "
              "to moderate heterogeneity.", ""]

    e3 = d.get("E3_anticollapse")
    if e3:
        L += ["## E3 anti-collapse ablation (R3.2)"]
        for v in ["full", "no_balanced", "no_diversity", "neither"]:
            vals = [r["f1"] for k, r in e3.items() if parse_key(k).get("variant") == v]
            dvs = [r["diversity"] for k, r in e3.items() if parse_key(k).get("variant") == v]
            if vals:
                L.append(f"- {v}: F1 {np.mean(vals):.3f}, final diversity {np.mean(dvs):.2f}")
        L += ["- This is the table the reviewer asked for. If `neither` does not "
              "collapse, the anti-collapse stack is not doing the work the paper "
              "attributes to it -- say that rather than keep the claim.", ""]

    e5 = d.get("E5_fusion_seeds")
    if e5:
        b = defaultdict(list)
        for k, r in e5.items():
            b[parse_key(k)["variant"]].append(r["f1"])
        stds = [np.std(v, ddof=1) for v in b.values() if len(v) > 1]
        if stds:
            L += ["## E5 fusion variance (R3.1)",
                  f"- median seed std: {np.median(stds):.4f}",
                  f"- spread across strategies: "
                  f"{max(np.mean(v) for v in b.values()) - min(np.mean(v) for v in b.values()):.4f}",
                  "- If the spread is within ~2x the seed std, the fusion ranking is "
                  "noise and the paper should report them as tied.", ""]

    e6 = d.get("E6_cost")
    if e6 and "Fed-VLM" in e6 and "Fed-LLM" in e6:
        a = e6["Fed-VLM"]["upload_bytes_per_client_per_round"]
        b_ = e6["Fed-LLM"]["upload_bytes_per_client_per_round"]
        L += ["## E6 measured cost (R1.2/R3.4)",
              f"- Fed-VLM payload {a/2**20:.0f} MiB vs Fed-LLM {b_/2**20:.0f} MiB "
              f"({a/b_:.2f}x)",
              f"- Fed-VLM {np.mean(e6['Fed-VLM']['history']['round_seconds']):.0f} s/round, "
              f"peak {e6['Fed-VLM']['peak_mib']:.0f} MiB",
              "- These replace the analytic Table V. Swap them in and delete the "
              "'analytic, not measured' caveat.", ""]

    e8 = d.get("E8_baselines")
    if e8:
        L += ["## E8 matched-setting baselines (R3.3)"]
        for a in sorted({parse_key(k)["alpha"] for k in e8}, key=float):
            row = []
            for m in ["local_only", "fedavg", "fedprox", "scaffold", "fedbn"]:
                recs = [r for k, r in e8.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not recs:
                    continue
                v = (recs[0]["mean_f1"] if m == "local_only"
                     else np.mean([r["f1"] for r in recs]))
                row.append(f"{m} {v:.3f}")
            L.append(f"- alpha={a}: " + ", ".join(row))
        L += ["- This is the comparison the reviewer asked for, and the only one in "
              "the paper run under matched conditions. Two readings matter: how far "
              "every federated arm sits above local-only (that is what federation "
              "buys), and whether the drift-correcting methods separate from FedAvg "
              "at low alpha. If FedProx/SCAFFOLD beat FedAvg at alpha=0.1, say so and "
              "switch the aggregator -- do not keep FedAvg because it is what the "
              "submitted version used.", ""]

    e7 = (d.get("E7_rag") or {}).get("heldout")
    if e7 and "error" not in e7:
        L += ["## E7 retrieval (R3.2)",
              f"- {e7['n_queries']} queries, top-1 acc {e7['overall_top1_accuracy']:.3f}, "
              f"mean sim {e7['overall_mean_top1_similarity']:.3f}",
              "- Replaces the 5-query probe. Report the similarity range honestly; "
              "do not restore the 0.89 claim unless this run produces it.", ""]

    p = outdir / "SUMMARY.md"
    p.write_text("\n".join(L), encoding="utf8")
    return p


def _write(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    d = json.loads(Path(a.results).read_text())
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made = [f for f in [tab_alpha(d, outdir), tab_clients(d, outdir),
                        tab_anticollapse(d, outdir), tab_warmstart(d, outdir),
                        tab_fusion(d, outdir), tab_cost(d, outdir),
                        tab_rag(d, outdir), tab_baselines(d, outdir)] if f]
    made += figs(d, outdir)
    made.append(summary(d, outdir))

    print("Wrote:")
    for p in made:
        print("  ", p)
    print("\nRead SUMMARY.md before pasting anything into the paper.")


if __name__ == "__main__":
    main()
