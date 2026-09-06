#!/usr/bin/env bash
set -euo pipefail

cd /home/trishita/omnimed
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/trishita/omnimed/.hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

exec /home/trishita/omnimed/.venv/bin/python -u repo/experiments/omnimed_experiments.py \
  --base repo/source/MedFederate_Colab_Complete.py \
  --tier standard --cache /home/trishita/omnimed/data_cache_standard_controlled.pkl \
  --only E8 --alphas 0.1 \
  --out /home/trishita/omnimed/results_cold_e8.json
