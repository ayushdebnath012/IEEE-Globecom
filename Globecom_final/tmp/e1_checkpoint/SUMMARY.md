# Experiment readout

- tier: `standard`  GPU: `NVIDIA H100 NVL`
- train/val: 2400/600

## E1 alpha sweep (R3.1)
- most heterogeneous (alpha=0.1): F1 0.674
- least heterogeneous completed setting (alpha=5.0): F1 0.896
- change at severe heterogeneity relative to alpha=1.0: -0.163
- This comparison measures sensitivity to partition skew within the reviewer suite. It does not, by itself, establish retention relative to centralized training.
