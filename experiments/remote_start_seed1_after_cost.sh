#!/usr/bin/env bash
set -euo pipefail

cd /home/trishita/omnimed

while tmux has-session -t omnimed_cost 2>/dev/null; do
  sleep 5
done

tmux new-session -d -s omnimed_seed1 \
  "CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u repo/experiments/omnimed_experiments.py --base repo/source/MedFederate_Colab_Complete.py --tier standard --cache /home/trishita/omnimed/data_cache_standard_controlled.pkl --only E3,E4 --seeds 1 --out /home/trishita/omnimed/results_cold_seed1.json 2>&1 | tee cold_seed1.log"

echo "seed1 started after cost at $(date -Is)" >> seed1_start.log
