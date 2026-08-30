# Experiment readout

- tier: `standard`  GPU: `NVIDIA H100 NVL`
- train/val: 2400/600

## E1 alpha sweep (R3.1)
- most heterogeneous (alpha=0.1): F1 0.674
- least heterogeneous completed setting (alpha=5.0): F1 0.896
- change at severe heterogeneity relative to alpha=1.0: -0.163
- This comparison measures sensitivity to partition skew within the reviewer suite. It does not, by itself, establish retention relative to centralized training.

## E2 client scaling (R3.4)
- K=3: F1 0.812+-0.043, wall-clock 545 s
- K=5: F1 0.825+-0.082, wall-clock 708 s
- K=10: F1 0.863+-0.008, wall-clock 893 s
- K=20: F1 0.846+-0.004, wall-clock 908 s
