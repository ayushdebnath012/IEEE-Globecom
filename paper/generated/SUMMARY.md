# Experiment readout

- tier: `standard`  GPU: `NVIDIA H100 NVL`
- train/val: 2400/600

## E1 alpha sweep (R3.1)
- most heterogeneous (alpha=0.1): F1 0.643
- least heterogeneous completed setting (alpha=5.0): F1 0.906
- change at severe heterogeneity relative to alpha=1.0: -0.204
- This comparison measures sensitivity to partition skew within the reviewer suite. It does not, by itself, establish retention relative to centralized training.

## E3 anti-collapse ablation (R3.2)
- full: F1 0.765, minimum diversity 0.95
- no_balanced: F1 0.763, minimum diversity 0.85
- no_diversity: F1 0.779, minimum diversity 0.90
- neither: F1 0.791, minimum diversity 0.85
- This is the table the reviewer asked for. If `neither` does not collapse, the anti-collapse stack is not doing the work the paper attributes to it -- say that rather than keep the claim.

## E2 client scaling (R3.4)
- K=3: F1 0.862+-0.026, wall-clock 327 s
- K=5: F1 0.829+-0.017, wall-clock 334 s
- K=10: F1 0.831+-0.011, wall-clock 357 s
- K=20: F1 0.818+-0.012, wall-clock 397 s

## E3b early-abort ablation (R3.2)
- early_abort=True: F1 0.871+-0.012
- early_abort=False: F1 0.899+-0.022

## E4 initialization ablation (R3.2)
- operational cold start: final F1 0.831+-0.001; round-1 0.646
- pooled-data oracle: final F1 0.877+-0.007; round-1 0.874
- The pooled-data arm is a non-deployable diagnostic oracle, not the initialization used by the operational federated experiments.

## E5 fusion variance (R3.1)
- median seed std: 0.0053
- spread across strategies: 0.1755
- If the spread is within ~2x the seed std, the fusion ranking is noise and the paper should report them as tied.

## E6 measured cost (R1.2/R3.4)
- Fed-VLM payload 587 MiB vs Fed-LLM 254 MiB (2.31x)
- Fed-VLM 42 s/round, peak 6064 MiB
- Runtime and peak allocated memory are measured. Communication remains a deterministic calculation from the FP32 payload.

## E8 matched-setting baselines (R3.3)
- alpha=0.1: local_only 0.297, fedavg 0.662, fedprox 0.737, scaffold 0.070
- alpha=1.0: local_only 0.523, fedavg 0.820, fedprox 0.862, scaffold 0.236
- This is the comparison the reviewer asked for, and the only one in the paper run under matched conditions. Two readings matter: how far every federated arm sits above local-only (that is what federation buys), and whether the drift-correcting methods separate from FedAvg at low alpha. With two seeds, report any separation descriptively; do not claim significance or switch the deployed method on this table alone. The SCAFFOLD row is an AdamW adaptation, not the classical SGD algorithm.

## E7 retrieval (R3.2)
- 600 queries, top-1 acc 0.542, mean sim 0.706
- Replaces the 5-query probe. Report the similarity range honestly; do not restore the 0.89 claim unless this run produces it.
