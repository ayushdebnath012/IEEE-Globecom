# Experiment readout

- tier: `standard`  GPU: `NVIDIA H100 NVL`
- train/val: 2400/600

## E1 alpha sweep (R3.1)
- most heterogeneous (alpha=0.1): F1 0.674
- least heterogeneous completed setting (alpha=5.0): F1 0.896
- change at severe heterogeneity relative to alpha=1.0: -0.163
- This comparison measures sensitivity to partition skew within the reviewer suite. It does not, by itself, establish retention relative to centralized training.

## E3 anti-collapse ablation (R3.2)
- full: F1 0.768, minimum diversity 0.90
- no_balanced: F1 0.697, minimum diversity 0.85
- no_diversity: F1 0.768, minimum diversity 0.85
- neither: F1 0.740, minimum diversity 0.75
- This is the table the reviewer asked for. If `neither` does not collapse, the anti-collapse stack is not doing the work the paper attributes to it -- say that rather than keep the claim.

## E2 client scaling (R3.4)
- K=3: F1 0.812+-0.043, wall-clock 545 s
- K=5: F1 0.825+-0.082, wall-clock 708 s
- K=10: F1 0.863+-0.008, wall-clock 893 s
- K=20: F1 0.846+-0.004, wall-clock 908 s

## E3b early-abort ablation (R3.2)
- early_abort=True: F1 0.884+-0.003
- early_abort=False: F1 0.845+-0.013

## E4 initialization ablation (R3.2)
- operational cold start: final F1 0.839+-0.065; round-1 0.704
- pooled-data oracle: final F1 0.818+-0.004; round-1 0.843
- The pooled-data arm is a non-deployable diagnostic oracle, not the initialization used by the operational federated experiments.

## E5 fusion variance (R3.1)
- median seed std: 0.0179
- spread across strategies: 0.1596
- If the spread is within ~2x the seed std, the fusion ranking is noise and the paper should report them as tied.

## E6 measured cost (R1.2/R3.4)
- Fed-VLM payload 587 MiB vs Fed-LLM 254 MiB (2.31x)
- Fed-VLM 337 s/round, peak 4884 MiB
- Runtime and peak allocated memory are measured. Communication remains a deterministic calculation from the FP32 payload.

## E8 matched-setting baselines (R3.3)
- alpha=0.1: local_only 0.310, fedavg 0.652, fedprox 0.685, scaffold 0.192
- This is the comparison the reviewer asked for, and the only one in the paper run under matched conditions. Two readings matter: how far every federated arm sits above local-only (that is what federation buys), and whether the drift-correcting methods separate from FedAvg at low alpha. With two seeds, report any separation descriptively; do not claim significance or switch the deployed method on this table alone. The SCAFFOLD row is an AdamW adaptation, not the classical SGD algorithm.

## E7 retrieval (R3.2)
- 600 queries, top-1 acc 0.522, mean sim 0.707
- Replaces the 5-query probe. Report the similarity range honestly; do not restore the 0.89 claim unless this run produces it.
