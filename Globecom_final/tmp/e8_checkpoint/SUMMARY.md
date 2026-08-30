# Experiment readout

- tier: `standard`  GPU: `NVIDIA H100 NVL`
- train/val: 2400/600

## E8 matched-setting baselines (R3.3)
- alpha=0.1: local_only 0.310, fedavg 0.652, fedprox 0.685, scaffold 0.192
- This is the comparison the reviewer asked for, and the only one in the paper run under matched conditions. Two readings matter: how far every federated arm sits above local-only (that is what federation buys), and whether the drift-correcting methods separate from FedAvg at low alpha. With two seeds, report any separation descriptively; do not claim significance or switch the deployed method on this table alone. The SCAFFOLD row is an AdamW adaptation, not the classical SGD algorithm.
