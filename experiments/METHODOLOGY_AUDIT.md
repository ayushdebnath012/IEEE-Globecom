# OmniMed-FL methodology audit

## Status

The conflict-checked publication artifact `results_corrected_merged.json` passed
`validate_results.py` with SHA-256
`59118f93c44287f144b2b1b6b0edd85373c860b9be79e7927c3683b35d47bab7`.
All expected record counts are complete: E1 10, E2 8, E3 16, E3b 4, E4 4,
E5 16, E6 3, E7 1, and E8 7.

Only `results_cold_v2.json`, `results_cold_e8.json`, and the disjoint
`results_cold_tail.json`, `results_cold_tail2.json`,
`results_cold_tail3.json`, `results_cold_cost.json`,
`results_cold_seed0.json`, and `results_cold_seed1.json` chunks from the final
corrected protocol are eligible for the revised paper. The seed chunks resume
E3/E3b/E4 from an identical six-record snapshot through the canonical runner's
`--seeds` option; the merge accepts identical overlap and rejects any conflicting
record. Files under
`diagnostic_warmstarted/` are preserved only to document the audit. Do not merge
their metrics into a manuscript table or figure.

## Why the earlier results were invalidated

1. The public-image loader omitted required Hugging Face configuration names and
   compared integer `ClassLabel` values with label strings. Its exception path
   silently supplied a full synthetic dataset, so the original image benchmark
   was not using the public radiographs described in the manuscript.
2. The attempted public-text path assigned a class by finding disease keywords
   in each passage and then fed the same passage to the classifier. That is target
   leakage, not independent ground truth.
3. Federated runs were initialized from a model trained centrally on the same
   pooled client corpus. This violates the intended FL protocol and makes high
   retention relative to centralized training circular.
4. Concurrent jobs wrote to one `warmstart_concat.pt`. The slower job overwrote
   the faster job's initialization, so a resumed result file could silently mix
   different starting weights.
5. The warm-start routine reported the best validation epoch but saved the final
   epoch's weights.
6. The balanced sampler based its epoch length on the largest local class. Under
   severe skew, balanced arms therefore received more optimizer steps than
   unbalanced arms; clients missing classes also produced undersized batches.
7. The local-only baseline used 6 local epochs versus 24 cumulative local epochs
   in the federated arms, exaggerating the apparent benefit of federation.
8. Several centralized models were constructed before their requested seed was
   set, so repeated-seed runs did not control random head initialization.
9. Runtime exceptions inside training batches were logged and swallowed, which
   could turn a partially trained run into an apparently valid result.
10. Unused unimodal heads and the inactive CNN fallback were counted as trainable
    and communicated parameters. FedBN was also not meaningful for a
    LayerNorm-based model evaluated as one server model, so it was removed from
    the matched comparison.
11. The vision forward path caught every pretrained-backbone exception and
    switched to a residual CNN. An out-of-memory error could therefore change
    architectures during a run. Pretrained-backbone load and runtime failures
    now fail closed; lightweight fallbacks are used only when explicitly
    requested at model construction.
12. Fusion registry keys borrowed names from foundation-model families even
    though the implementation contains only small fusion operators; two keys
    share the same decoder-layer parameterization. Paper tables now use
    architecture-descriptive operator labels and explicitly reject equivalence
    to the named foundation models.
13. Tokenizer loading also had a silent simple-tokenizer fallback. The corrected
    release fails closed unless a caller explicitly requests the fallback. Server
    logs for the reported suite contain no tokenizer-fallback event.

## Corrected evaluation protocol

- Images: 3,000 balanced examples (600/class). Server build logs verify 2,400
  public radiographs across Normal, Pneumonia, Pleural Effusion, and
  Cardiomegaly, plus 600 synthetic COVID-19 X-ray surrogates because the named
  COVID source was unavailable.
- Text: 3,000 class-conditioned synthetic clinical-style notes generated from
  controlled templates. They are label-level companions, not patient-matched
  EHR reports. No keyword-derived public-text labels are used.
- Split: 2,400 train / 600 validation after deterministic shuffling.
- Controlled cache SHA-256:
  `4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba`.
- Executed model source SHA-256:
  `ce473f4bca58f8920d7c22b55b3e0dd28a2de227049f4ad77141659468cbf227`;
  experiment runner SHA-256:
  `1805c5bafb5f4889ecab87fe16e3e16788d5e0d1c7d205f19c88f81555f420e4`.
  The release model source additionally fails closed on tokenizer loading and has
  SHA-256 `5fbe6392a7bed0fd3aceaae56cf0df50b55087f8411d81c0f04988c9172e49ea`;
  the reported server logs contain no tokenizer-fallback event.
- Disjoint E5/E6/E7 chunks use the canonical runner through a registry-filtering
  wrapper with SHA-256
  `644e4049c2e22d52a88136e763b9cdbb5a75f6d178076a2b7696592b985a4fce`.
  It changes only which complete keys a process attempts. The merge tool rejects
  conflicting duplicate keys.
- Model: public pretrained DistilBERT and ViT-Base/16 encoders with freshly
  initialized projection, fusion, and classification heads.
- Operational FL initialization: no model trained on pooled client data. The
  pooled-data initialization appears only as the explicitly labeled E4 oracle
  ablation and is not part of the deployable protocol.
- Optimizer and budget: AdamW, learning rate `1e-4`, weight decay `0.01`, batch
  size 16, 3 local epochs, 8 rounds, FP32 training in the reviewer suite.
- Repetition: seeds 0 and 1. A seed controls both the model/task-head
  initialization and the realized Dirichlet partition. PyTorch deterministic
  algorithms were not forced, so identical-seed CUDA reruns need not be
  bitwise-identical; variability is summarized across seeds and no
  cross-launch trajectory equality is claimed.
- E1: alpha in `{0.1, 0.3, 0.5, 1.0, 5.0}` at 5 clients.
- E2: clients in `{3, 5, 10, 20}` at alpha 1.0.
- E3: balanced sampler and diversity-loss ablations at alpha 0.1 and 1.0, with
  equal optimizer-step budgets.
- E4: pooled-data oracle initialization versus the operational initialization.
- E5: eight fusion strategies with two seeds.
- E6: measured wall-clock and peak allocated GPU memory; communication remains a
  deterministic FP32 payload calculation from trainable parameters. Runs share
  the H100 server with unrelated workloads, so timings characterize these runs
  under uncontrolled contention rather than dedicated-hardware throughput.
- E7: FAISS exact inner-product retrieval over TF-IDF vectors for every held-out
  query. It is not a fine-tuned-DistilBERT retrieval result.
- E8: FedAvg, FedProx, and a SCAFFOLD-style AdamW implementation at alpha 0.1,
  plus local-only training with the same cumulative local-epoch budget. The
  local-only arm disables the centralized early-abort heuristic so it receives
  the full matched budget and reports epoch 24 rather than selecting each
  client's best validation epoch. The
  SCAFFOLD row must retain the `-style (AdamW)` qualifier because the classical
  convergence result assumes SGD-style local steps.

## Reporting constraints

- Do not restore the old 0.956 F1, 99.1% retention, “second of 31,” deployment,
  or clinical-readiness claims unless independently supported by the corrected
  files.
- Do not describe the text as real EHR data or the modalities as patient-level
  paired data.
- Do not call computed communication volume “measured.” Runtime and peak
  allocated memory are measured; payload is calculated.
- Report the realized client class skew alongside nominal Dirichlet alpha.
- With two seeds, use mean and sample standard deviation descriptively; do not
  claim statistical significance.

## Finalization commands

```text
python merge_results.py --main results_cold_v2.json \
  --extra results_cold_tail.json --extra results_cold_tail2.json \
  --extra results_cold_tail3.json --extra results_cold_cost.json \
  --extra results_cold_seed0.json --extra results_cold_seed1.json \
  --e8 results_cold_e8.json \
  --out results_corrected_merged.json
python validate_results.py results_corrected_merged.json
python omnimed_make_tables.py --results results_corrected_merged.json \
  --outdir ../generated
```

## Provenance of the manuscript's result set

The sections above audit `results_corrected_merged.json`. The manuscript does not
read that file directly: it reports from `reviewer_results_merged.json`, the later
merge that adds the reviewer-completion runs. The two are linked by hash rather
than by shared schema, so the chain is worth stating explicitly.

`reviewer_results_merged.json` reorganizes records into different top-level groups
(`grid`, `federated_fusion`, `legacy_*`, `pfin_missing_text`, `retrieval`, …).
**`validate_results.py` does not apply to it** and will fail on its schema; that
check validates the earlier artifact only. Its `_meta` carries the provenance
instead, pinning its parent and every runner by SHA-256:

| `_meta` field | Artifact in this directory | Verified |
|---|---|---|
| `legacy_sha256` | `results_corrected_merged.json` | matches |
| `corrected_retrieval_sha256` | `corrected_rag_retrieval.json` | matches |
| `core_runner_sha256` | `omnimed_experiments.py` | matches |
| `retrieval_runner_sha256` | `rag_retrieval_corrected.py` | matches |
| `pfin_helper_sha256` | `pfin_matched.py` | matches |
| `reviewer_runner_sha256_native_budget` | `reviewer_completion.py` | matches |
| `reviewer_runner_sha256` | superseded runner revision | not retained |

`legacy_sha256` is the same digest this document records for the validated
artifact, so the audited set is the parent of the set the manuscript reports.
Every hash above except the superseded reviewer-runner revision resolves to a file
committed here, and each can be checked directly:

```sh
python - <<'PY'
import json, hashlib
m = json.load(open('reviewer_results_merged.json', encoding='utf8'))['_meta']
for field, path in [
        ('legacy_sha256', 'results_corrected_merged.json'),
        ('corrected_retrieval_sha256', 'corrected_rag_retrieval.json'),
        ('core_runner_sha256', 'omnimed_experiments.py'),
        ('retrieval_runner_sha256', 'rag_retrieval_corrected.py'),
        ('pfin_helper_sha256', 'pfin_matched.py'),
        ('reviewer_runner_sha256_native_budget', 'reviewer_completion.py')]:
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    print(f"{'ok ' if got == m[field] else 'BAD'} {path}")
PY
```

What this does and does not establish: it establishes that the manuscript's
numbers descend from the audited artifact and that the code which produced them is
the code committed here. It is not a substitute for `validate_results.py` — the
record-count, key-set, and protocol invariants that check the earlier artifact have
no equivalent for the newer schema. The `_meta` flags
`legacy_meta_and_protocol_validated`, `new_record_protocols_validated`,
`corrected_retrieval_protocol_validated`, and
`grid_model_state_invariants_validated` are assertions written by the merge step,
not independent checks.
