#!/usr/bin/env bash
set -euo pipefail
cd /home/trishita/omnimed
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/trishita/omnimed/.hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OM_ONLY=E5
export OM_FUSION_TYPES=flamingo,blip2,coca,unified_io
export OM_RESULTS_OUT=/home/trishita/omnimed/results_cold_tail2.json
exec ./.venv/bin/python -u remote_tail_subset.py
