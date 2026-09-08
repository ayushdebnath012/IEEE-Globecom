# OmniMed-FL

Multimodal federated learning for five-class clinical condition classification
(Normal, Pneumonia, COVID-19, Pleural Effusion, Cardiomegaly).

Submitted to IEEE GLOBECOM.

> **The earlier results in this repository are superseded.** Between submission and
> this revision the entire experimental suite was re-run under audit, and several
> headline numbers did not survive. They are not used as current evidence: the
> original comparison table, the "18 model variants" framing, the five-LLM/five-ViT
> encoder sweep, the 99.1% centralized-retention summary, and the 0.89
> retrieval-similarity claim do not support the revised conclusions. To preserve
> continuity with the accepted paper, the Results section distributes six of its pilot
> plots and the branch scores behind 99.1% near their corresponding discussions,
> clearly labeled as pre-audit, non-pooled context. The
> revised manuscript reports *different and lower* audited scores. `paper/` is the
> version; `paper/RESPONSE_TO_REVIEWERS.md` itemizes every correction and
> `experiments/METHODOLOGY_AUDIT.md` explains why the earlier runs were invalid.

```
paper/        the audited manuscript, its figures, and the compiled PDF
experiments/  reviewer-requested experiment suite (runs on Colab) and the audit
source/       training code, and the results behind the withdrawn submission
```

## paper/

| File | |
|---|---|
| `Globecom_final.tex` | manuscript source (IEEEtran, 6 pages) |
| `Globecom_final.pdf` | compiled |
| `results_corrected.tex` | the Results section, `\input` by the manuscript |
| `generated/` | audited figures, selected accepted-version pilot plots, and generated LaTeX tables |
| `RESPONSE_TO_REVIEWERS.md` | reply to the reviewers, and the withdrawn claims |

Compiles with `pdflatex` in three passes. `\graphicspath` points at `generated/`,
so the tree compiles as-is. Audited results come from
`experiments/reviewer_results_merged.json`; selected `legacy_*.png` panels preserve
the accepted-version pilot and are explicitly separated from current conclusions.

## source/

The pipeline as it stood for the original submission.

| File | |
|---|---|
| `MedFederate_Colab_Complete.py` | the full training pipeline: data, models, FedAvg |
| `MedFederate_Kaggle.py` | Kaggle variant |
| `gen_medfederate_plots.py` | regenerates the pre-audit figures from the results |
| `results/*.json` | the measured results behind the submitted version |

`results/medfederate_results.json` is the record for the **submitted** paper, not the
revision: per-variant Macro F1, per-epoch histories, federated round histories, the
retrieval probe, and the 31-system benchmark compilation. It is kept for provenance.
The audited numbers that the current manuscript reports are in `experiments/`, and
where the two disagree the audited set is the correct one.

## experiments/

The experiments the reviewers asked for that the submitted results do not contain,
plus the audit that invalidated the earlier run set.

| | Experiment | Reviewer point |
|---|---|---|
| E1 | Dirichlet α sweep | non-IID generality |
| E2 | client-count sweep K | scalability |
| E3 | anti-collapse component ablation | ablation |
| E4 | warm-start ablation | ablation |
| E5 | fusion strategies × seeds | variance |
| E6 | measured runtime / memory / communication | systems cost |
| E7 | retrieval over the full held-out split | retrieval at scale |
| E8 | FedAvg / FedProx / SCAFFOLD / FedBN / local-only | matched-setting baselines |

**Running it.** Set the Colab runtime to a T4 GPU, paste `omnimed_colab.py` into one
cell, run it, then:

```python
run("smoke")                        # ~20 min, validates the pipeline
run("standard", ["E1", "E2"])       # ~2-3 h
run("standard", ["E3", "E4"])       # ~2-3 h
run("standard", ["E8"])             # ~2-3 h
run("standard", ["E5", "E6", "E7"]) # ~1-2 h
build_tables()                      # LaTeX tables, figures, SUMMARY.md
```

Every `(experiment, config, seed)` is written to the results JSON as soon as it
finishes and skipped on re-run, so a disconnect costs at most one run. Call
`restore()` to upload a saved JSON and continue.

`omnimed_experiments.py` / `omnimed_make_tables.py` are the same code as separate
modules, with `OmniMed_Experiments.ipynb` as a cell-by-cell driver. Use whichever
you prefer.

### the audit

| File | |
|---|---|
| `METHODOLOGY_AUDIT.md` | what invalidated the earlier runs, which chunks are eligible, and the provenance chain |
| `validate_results.py` | the record-count, key-set and protocol check for the corrected artifact |
| `results_corrected_merged.json` | the validated artifact, SHA-256 recorded in the audit |
| `reviewer_results_merged.json` | the later merge the manuscript's numbers come from |
| `corrected_rag_retrieval.json` | corrected retrieval results, pinned by hash from the merge |
| `reviewer_completion.py`, `rag_retrieval_corrected.py`, `pfin_matched.py` | the runners, pinned by hash from the merge |
| `make_*.py` | regenerate each audited figure in `paper/generated/` |

Two artifacts, linked by hash. `results_corrected_merged.json` is the audited base
and hashes to the SHA-256 written down in `METHODOLOGY_AUDIT.md`:

```sh
sha256sum experiments/results_corrected_merged.json
```

`reviewer_results_merged.json` is the later merge that adds the reviewer-completion
runs, and is what the manuscript reports from. It uses a different schema, so
`validate_results.py` does not apply to it — instead its `_meta` pins its parent and
every runner by SHA-256. `METHODOLOGY_AUDIT.md` tabulates that chain and gives a
snippet that checks all of it; every hash resolves to a file committed here except
one superseded runner revision. The audited set is therefore the parent of the
reported set, and the code that produced the numbers is the code in this directory.

## Repository scope

Only `paper/`, `experiments/`, `source/`, and repository configuration are tracked
here. Administrative documents, presentations, videos, duplicate manuscript trees,
rendering scratch files, caches, and model checkpoints are excluded.

The complete project, including those materials and every commit that precedes this
repository, is archived at
[ayushdebnath012/OmniMed-FL](https://github.com/ayushdebnath012/OmniMed-FL). This
repository starts from a fresh history so a clone carries only the paper, the code,
and the results.

The experiment directory also retains `results_cold_*.json` and
`reviewer_results/*.json` as raw run records, plus `merge_results.py` and
`merge_reviewer_results.py` for validation and aggregation. The published merged
artifacts and their pinned runners are preserved byte-for-byte.
`experiments/MedFederate_Colab_Complete.py` preserves the separate experiment base
variant; `source/` retains the original submission pipeline.
The `remote_*.sh` and `remote_tail_subset.py` scripts record the original server
launch/resume workflow; their server paths and environment need adapting before
reuse. Partial diagnostic runs and superseded manuscript rewrites are excluded.

## Notes

- E8 is the only comparison in this work run under matched conditions. The
  benchmark table in the paper reports numbers other authors measured on their own
  corpora, which is a positioning exercise rather than a controlled comparison, and
  the paper says so.
- Reported Macro F1 values come from a corpus that is partly synthetic. The paper's
  Limitations section sets out what that does and does not support.
