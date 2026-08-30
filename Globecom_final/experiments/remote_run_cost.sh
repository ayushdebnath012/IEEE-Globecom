#!/usr/bin/env bash
set -euo pipefail
cd /home/trishita/omnimed
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/trishita/omnimed/.hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OM_ONLY=E6,E7
export OM_RESULTS_OUT=/home/trishita/omnimed/results_cold_cost.json
exec ./.venv/bin/python -u remote_tail_subset.py
