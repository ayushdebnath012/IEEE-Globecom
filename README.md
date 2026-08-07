# OmniMed-FL

Multimodal federated learning for five-class clinical condition classification
(Normal, Pneumonia, COVID-19, Pleural Effusion, Cardiomegaly).

Submitted to IEEE GLOBECOM.

```
paper/        the manuscript, its figures, and the compiled PDF
experiments/  reviewer-requested experiment suite (runs on Colab)
source/       training code and the results that back every number in the paper
```

## paper/

| File | |
|---|---|
| `Globecom_final.tex` | manuscript source (IEEEtran, 7 pages) |
| `Globecom_final.pdf` | compiled |
| `omnimed_plots/` | every figure |

Compiles with `pdflatex` in three passes. Three duplicate figures are commented
out for length; each is one `%`-block from returning.

## source/

| File | |
|---|---|
| `MedFederate_Colab_Complete.py` | the full training pipeline: data, models, FedAvg |
| `MedFederate_Kaggle.py` | Kaggle variant |
| `gen_medfederate_plots.py` | regenerates the paper figures from the results |
| `results/*.json` | measured results — the provenance for every number reported |

`results/medfederate_results.json` is the authoritative record: per-variant Macro F1,
per-epoch histories, federated round histories, the retrieval probe, and the
31-system benchmark compilation. Every figure and table in the paper derives from it.

## experiments/

The experiments the reviewers asked for that the submitted results do not contain.

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

## Notes

- E8 is the only comparison in this work run under matched conditions. The
  benchmark table in the paper reports numbers other authors measured on their own
  corpora, which is a positioning exercise rather than a controlled comparison, and
  the paper says so.
- Reported Macro F1 values come from a corpus that is partly synthetic. The paper's
  Limitations section sets out what that does and does not support.
