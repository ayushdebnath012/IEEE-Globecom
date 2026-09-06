# OmniMed-FL - Response to Reviewers

We thank the reviewers for the constructive comments. The revision reframes the
work as a controlled proxy systems study and avoids claims of clinical
validation, deployment readiness, statistical significance, formal privacy, or
exact reproduction of prior methods.

> **Please read this section first.** Between submission and this revision we
> re-ran the entire experimental suite under audit. Several headline numbers in
> the submitted PDF did not survive, and we have withdrawn rather than defended
> them. As a result the revised manuscript reports *different and lower* scores
> than the version the reviewers saw, and the comparison table (original
> Table IV), the "18 model variants" framing, the five-LLM/five-ViT encoder
> sweep, the 99.1% centralized-retention claim, and the 0.89 retrieval-similarity
> claim are gone. This is deliberate, and it is the reason Reviewer 2's summary
> and Reviewer 1's "state-of-the-art" concern no longer map onto the text. The
> corrections are itemized immediately below; every remaining number in the
> manuscript comes from the audited run set.

## Corrections to claims made in the original submission

Three statements in the original submission are not supported by the audited
runs. We withdraw them rather than defend them, and flag them here because
Reviewer 2's summary restates them in good faith.

1. **"Electronic health records."** The original text described the second
   modality as EHR data. No patient EHR data were used at any point. The notes
   are class-conditioned synthetic clinical-style text paired to images by
   class, not by patient. Section III-C now states this explicitly ("No patient
   electronic health record (EHR) data are used"), and the abstract repeats it.
   The claim that combining images with EHRs "is crucial for accurate
   diagnosis" is therefore not a claim this paper can make.

2. **"The multimodal approach significantly outperforms isolated
   single-modality systems."** In the audited branch comparison the text-only
   branch reaches 0.893 macro-F1 against 0.782 for multimodal concatenation,
   while costing 2.3x less model state, 2.4x less wall time and 2.8x less peak
   memory (Fig. 4(e)). The revision therefore states that this proxy corpus
   does not support a blanket multimodal-superiority claim, and attributes the
   text advantage to class-conditioned template shortcuts that the text encoder
   can exploit directly. What the data do support is retained: at `$\alpha=0.1$`
   every federated arm except SCAFFOLD--AdamW exceeds local-only training
   (0.310), which is a federated-versus-isolated result, not a
   multimodal-versus-unimodal one.

3. **"Five benchmarks" and "18 model variants."** The revision evaluates one
   controlled five-class proxy corpus, not five benchmarks. The arms actually
   run are enumerated in Section IV: eight fusion rules, three initializations,
   four anti-collapse configurations, four missing-text imputation rules, five
   severe-skew baselines, a twelve-cell `$K\times\alpha$` grid, and a
   three-branch modality audit.

## Revision-wide presentation changes

- The exact title from the submitted `_6` paper is preserved: "OmniMed-FL: A
  Robust Multimodal Federated Learning Framework for Clinical Diagnosis." The
  manuscript defines "robust" as empirical stress testing, not certified
  clinical robustness.
- The paper is plot-led: four figures contain 14 visual panels (architecture,
  two new six-panel reviewer composites, and corrected retrieval). All results
  tables were removed; only the common simulation-parameter table remains.
- Relevant validated earlier evidence is retained inside the new panels, while
  withdrawn pre-audit figures and claims are not reused.

## Table and figure renumbering

Reviewer 2 refers to Table II and Table IV from the original submission. After
the visual consolidation the manuscript contains a single table, so those
numbers no longer resolve:

| Original submission | Current manuscript |
| --- | --- |
| Table II (simulation parameters) | Table I, plus the new "Why these settings" paragraph |
| Table IV (main results) | Fig. 2(a) and the quantitative sentences now closing the abstract |
| Remaining result tables | The twelve panels of Fig. 2 and Fig. 4 |

## Reviewer 1

### Synthetic-data limitations and clinical applicability

The abstract, controlled-corpus section, and Discussion and Limitations state
that all 3,000 notes and the 600-image COVID-19 class are synthetic; the other
2,400 images are public radiographs. The Conclusion reiterates the clinical-scope
caveat. Image-text pairing is by class rather than patient. The revision discusses template shortcuts, absent
patient-level discordance and missingness, site effects, lack of patient- or
institution-grouped splitting, and absence of external or prospective
validation. Neither F1 nor retrieval is interpreted as diagnostic safety,
clinical utility, generalization, or deployment readiness.

### Communication, scalability, and robustness

The revision crosses `$K={3,5,10,20}$` with
`$\alpha={0.1,1,5}$`, reports active and nominal clients, and plots macro-F1 score,
GPU-synchronized round-section time, and peak allocated GPU memory. It also
shows the realized severe-skew class allocation.

Communication is stated transparently as the formula-derived nominal FP32
model-state volume `$V_{nom}=2KT|\theta|b$`: 27.5, 45.9, 91.8, and 183.5 GiB
for the four client counts. This is not called measured network traffic. The
paper lists excluded serialization, transport, secure-aggregation, compression,
and algorithm-state overheads and distinguishes the nominal-client formula from
an alternative active-client estimate.

Robustness limitations now cover client dropout, stragglers, network latency,
heterogeneous hardware, poisoning or Byzantine updates, privacy attacks, and
secure aggregation.

On robustness we have gone beyond listing the gap and added the brief
*consideration* the reviewer asked for, grounded in our own measurements rather
than in an experiment we did not run. The Discussion now observes that at
`$\alpha=0.1$` honest client updates already disperse by `$\pm0.185$` F1, so a
trimmed-mean or coordinate-wise-median aggregator would have to discard genuine
minority-class signal in order to reject an adversarial update, while
sample-weighted FedAvg leaves a single large shard unbounded leverage. Severe
label skew and Byzantine tolerance therefore pull against each other in this
setting. We state plainly that quantifying the trade-off requires a threat model
we did not build, and we do not claim any robustness result.

### "Integration of existing methods rather than a new algorithmic framework"

We accept that we do not propose a new fusion operator or aggregation rule, and
the revision no longer implies otherwise. We do want to argue, however, that the
contribution is not merely integration. It is a *matched-condition protocol*, and
the revision now states this as the first contribution rather than leaving it
implicit: corpus, partition draw, shard-size profile, encoders, optimizer, and
local budget are held fixed across every arm, so a difference between arms is a
difference in the named factor and nothing else.

The protocol is what produces the paper's non-obvious results, and none of them
is available from the source papers read side by side:

- The FedMME-style deficit under severe skew survives quadrupling the local
  budget from our matched 24 epochs to FedMME's native 100 (`$0.537\pm0.217$` to
  `$0.472\pm0.225$`), which attributes the gap to one-shot aggregation rather
  than to a compute handicap. Only a matched-budget design can separate those.
- An eight-way fusion sweep that looks decisive (0.603 to 0.787) is noise: the
  widest per-rule standard deviation, 0.183, exceeds the entire between-rule
  spread. We therefore decline to rank fusion operators.
- Three suites that nominally share the same severe-skew FedAvg/concatenation
  configuration return `$0.652\pm0.057$`, `$0.674\pm0.185$`, and
  `$0.716\pm0.014$`. We report them separately instead of pooling or selecting,
  which sets a floor on how finely any result in this regime can be read.

Related Work now closes on this point explicitly, so the novelty claim is
testable by a reader rather than asserted.

### Stronger claims and reference accuracy

State-of-the-art, deployment-readiness, formal-privacy, and clinical-accuracy
claims were removed. Two-seed differences are described as descriptive rather
than statistically resolved. The negative SCAFFOLD--AdamW result is not
generalized to classical SGD SCAFFOLD, and the text-only branch outperforming
concatenation prevents a blanket multimodal-superiority claim.

All 21 references were checked against primary records, put in first-citation
order, and corrected where needed. FedMME and P-FIN are labeled matched
adaptations, not exact reproductions. Historical foundation-model-like code keys
are replaced in the paper by architecture-descriptive fusion names.

## Reviewer 2

### Quantitative findings in the abstract

Done as asked: the quantitative findings are now the **closing** sentences of the
abstract, after the scope caveat rather than before it. The caveat is stated
once, then the abstract ends on three quantitative results:

1. the severe-skew matched ladder - local-only 0.310, FedAvg `$0.652\pm0.057$`,
   FedProx `$0.685\pm0.064$`, matched FedMME-style one-shot `$0.537\pm0.217$`,
   SCAFFOLD--AdamW `$0.192\pm0.129$`, with the 0.033 FedProx-FedAvg gap falling
   inside either two-seed standard deviation;
2. the factor comparison - label skew costs up to 0.28 F1 (0.874-0.896 at
   `$\alpha=5$` against 0.615-0.741 at `$\alpha=0.1$`) whereas a near-sevenfold
   client increase costs at most 0.13, while bidirectional volume grows linearly
   to 183.5 GiB at `$K=20$`;
3. the branch audit - multimodal concatenation spends `$2.3\times$` the model
   state of a text-only branch scoring 0.111 F1 higher.

Because original Table IV was withdrawn (see the corrections section), we could
not report findings "in terms of Table IV" literally. Items 1-3 are the audited
replacement for what that table was meant to show, and Fig. 2(a) plots item 1.
The abstract is 251 words.

### Justification of simulation parameters

The former Table II is the paper's only remaining table and is automatically
renumbered Table I after visual consolidation. Its accompanying "Why these settings" paragraph states
that eight rounds of three local epochs provide a common 24-local-epoch budget;
batch size 16 accommodates both active encoders in FP32; the `$K\times\alpha$`
grid spans severe to near-IID partitions; and two seeds permit descriptive, not
significance, claims. Learning rate and diversity weight are fixed implemented
defaults, not per-method optima.

## Reviewer 3

### Anti-collapse, fusion, and initialization ablations

The matched six-panel figure separately covers convergence, anti-collapse
components, all eight operational fusion rules, initialization, and missing
text. At `$\alpha=0.1$`, the full anti-collapse stack obtains
`$0.716\pm0.014$`; removing the balanced loader, entropy-diversity term, or both gives
`$0.585\pm0.148$`, `$0.701\pm0.011$`, and `$0.657\pm0.057$`. Overlaid minimum
predicted-class-diversity markers distinguish transient class loss from lower
F1. The implemented loss now explicitly shows the fixed overconfidence guard;
it remains common to these arms and is not attributed a separate causal effect.

The initialization panel separates same-architecture random encoders, public
encoders plus random task modules, and a labeled pooled-data warm start. The
pooled checkpoint's extra centralized pretraining is disclosed and is used only
in that ablation; selection on the reporting validation split is disclosed, so
it is treated as diagnostic. The separate pooled abort control is explicitly
non-federated. Abort-off/on epoch-12 F1 is `$0.825\pm0.001$` and
`$0.843\pm0.032$`; the trigger never fires, so the difference is not claimed as
an early-abort benefit.

All eight fusion rules were retrained inside the operational federated loop at
`$\alpha=0.1$, $K=5$` (not screened on pooled data). Their two-seed means span
0.603 (non-residual 384-dimensional cross-attention) to 0.787 (two-token
Transformer encoder), with per-rule sample standard deviations of 0.024-0.183.
Because the widest intervals exceed the entire between-rule spread, the paper
states that the sweep supports no reliable ranking; projected concatenation, the
default used elsewhere, sits mid-pack at `$0.643\pm0.102$`.

The initialization panel now includes a third, fully random arm: same
architecture with randomly initialized encoders and task modules reaches 0.635
round-1 F1 and `$0.765\pm0.010$` at round 8, against 0.704 and
`$0.839\pm0.065$` for public encoders with random task modules, and 0.843 and
`$0.818\pm0.004$` for the labeled pooled-data warm start. Public encoders are
therefore worth roughly 0.07 F1 over full random initialization, whereas the
pooled checkpoint's round-1 head start does not survive to round 8.

### Runtime, communication, memory, and scaling

The new systems figure contains complete `$K\times\alpha$` heat maps for F1,
synchronized round-section time, and peak allocated memory, together with
formula-derived communication, one-run branch costs, and the realized
severe-skew allocation. The branch audit reports text/image/multimodal F1 of
0.893/0.740/0.782, one-way model-state sizes of 0.248/0.324/0.573 GiB, timed-round totals of
18.5/33.6/45.0 minutes, and peak memory of 1.69/3.18/4.77 GiB.

Client activity is reported per cell as `$A/K$`, where a shard is active only if
it holds at least four samples. Allocation is complete in every cell except
under severe skew, where `$\alpha=0.1$` retains 9-10 of 10 and 18-20 of 20
active clients across the two seeds; `$K=3$` and `$K=5$` stay fully active at
all three `$\alpha$` values. Over the grid the synchronized round section spans
42-114 s and peak allocated memory 4.77-5.92 GiB. The paper explicitly warns
that panels (b)-(c) must be read within a suite rather than as scaling curves:
the `$K=5$` column and the `$\alpha=1$` row are inherited from the earlier
validated suite and ran in a different session on the shared server, which
accounts for the 5.92 GiB cells and the slowest timed sections. Panel (d) states
the resulting operating point, namely that moving from `$K=3$` to `$K=20$` at
`$\alpha=5$` costs 6.7 times the nominal traffic for a 0.014 F1 change.

### Recent multimodal federated-learning comparisons

The revised study adds FedMME-style and P-FIN-style arms sharing the corpus,
partition draw, encoders, optimizer, and local budget. Two departures from the
source methods were disclosed in the first revision; we have now removed the
first outright and given the second a concrete technical justification.

**FedMME local budget: departure removed.** The first revision ran the FedMME
arm at our matched 24-local-epoch budget rather than the 100 epochs the source
paper uses ("Local client models were trained for 100 epochs with SGD optimizer
using LR 1e-3 and batch size 128"). We now report *both* points: the
matched-budget arm, which isolates the method from the budget, and a
native-budget arm at 100 local epochs, which removes the discrepancy entirely.
Reporting both separates the two confounds instead of trading one for the other.

**P-FIN fusion stack: substitution retained, with the reason stated.** P-FIN
fuses observed and imputed features with bidirectional cross-modal attention. We
keep the common concatenation classifier, and the manuscript now says why rather
than merely noting the substitution: our pipeline exposes exactly one pooled
256-dimensional feature per modality, so a cross-attention block over two
single-token sequences has a softmax over one key, i.e. unit attention weight,
and reduces exactly to a linear projection of the other modality's value vector
(verified numerically to zero error). Implementing it would therefore add the
appearance of the source architecture without any of its behavior. Faithfully
reproducing token-level bidirectional attention would require abandoning
P-FIN's own feature-level imputation premise, which is outside this revision's
scope; we state the limitation instead of disguising it.

Under the identical corpus, partition draw, encoders, and optimizer at
`$\alpha=0.1$, $K=5$`, the FedMME-style one-shot ensemble reaches
`$0.537\pm0.217$` (seeds 0.690, 0.384) at the matched 24-epoch budget and
`$0.472\pm0.225$` (seeds 0.631, 0.312) at the native 100-epoch budget, against
`$0.652\pm0.057$` for FedAvg and `$0.685\pm0.064$` for FedProx.

The native-budget run answers the obvious objection to the first revision:
quadrupling local computation does **not** close the gap to iterative
aggregation, and in these two seeds slightly widens it, though the two budgets
sit well within each other's spread. The gap therefore reflects one-shot
aggregation under severe label skew rather than a budget handicap we imposed.
Both budgets are the most seed-sensitive arms in the study, consistent with a
single aggregation having no later round in which to correct a poor local
solution.

The P-FIN-style stress test uses three multimodal and two image-only clients.
On the primary forced-missing-text metric the four imputation rules order
consistently: zero filling `$0.465\pm0.084$`, deterministic FIN
`$0.559\pm0.144$`, Gaussian `$\beta$`-NLL imputation `$0.627\pm0.042$`, and
uncertainty-weighted aggregation `$0.653\pm0.027$`. The ordering holds in both
seeds and the spread narrows as the method models its own uncertainty, matching
the qualitative behavior P-FIN reports; the secondary all-text-observed scores
(0.429-0.657, SD up to 0.274) are far noisier.

## Corrected retrieval (the retrieval component of RAG)

The earlier five-query illustration is replaced by exact TF-IDF/FAISS retrieval
over 2,400 indexed training notes for all 600 validation-note queries. TF-IDF is
fitted on the training notes only. The plot reports condition-macro top-1
same-label accuracy 0.522, same-label precision@5 0.459, and mean top-1 cosine
similarity 0.707. The manuscript explicitly identifies this as retrieval, not
generation, a fine-tuned-DistilBERT result, or clinical retrieval evidence.

## Final validation

The experiment runner, merge validator, and plotting scripts preserve protocol
and data hashes and reject incomplete or mismatched records. All 40 server
records passed validation (40 unique keys, matching base-model and data-cache
hashes, no deterministic runs, zero failure markers) and were merged into
`experiments/reviewer_results_merged.json`. Every quoted figure in the
manuscript was then re-checked programmatically against the plotted-values echo,
and the manuscript rebuilds to exactly six full letter-size pages with no
overfull boxes, no undefined references, and no unresolved citations.
