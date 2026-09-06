#!/usr/bin/env bash
set -euo pipefail

cd /home/trishita/omnimed

# E5--E7 are disjoint from the main file, so they can start once E8 releases a
# GPU. Prefer GPU 0 only with a generous free-memory margin; otherwise wait for
# the E1--E4 main session to release GPU 1.
while tmux has-session -t omnimed_e8 2>/dev/null; do
  sleep 30
done

E8_COUNT=$(./.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('results_cold_e8.json')
print(len(json.loads(p.read_text()).get('E8_baselines', {})) if p.exists() else -1)
PY
)

if [[ "$E8_COUNT" != "7" ]]; then
  printf 'E8 incomplete: %s/7; refusing tail launch.\n' "$E8_COUNT"
  exit 2
fi

./.venv/bin/python - <<'PY'
import json
rows = json.load(open('results_cold_e8.json')).get('E8_baselines', {})
local = [r for key, r in rows.items() if key.startswith('local_only|')]
assert len(local) == 1, f'expected one local-only record, found {len(local)}'
r = local[0]
assert r.get('epochs_per_client') == 24
assert r.get('early_abort') is False
assert 'per_client_best_epoch_f1' in r
print('Validated complete matched-budget E8 file.')
PY

while true; do
  GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 \
    | tr -d ' ')
  if [[ "$GPU0_FREE" -ge 12288 ]]; then
    TAIL_GPU=0
    break
  fi
  if ! tmux has-session -t omnimed 2>/dev/null; then
    TAIL_GPU=1
    break
  fi
  sleep 60
done

cat > run_cold_tail.sh <<'SH'
#!/usr/bin/env bash
cd /home/trishita/omnimed || exit 1
export CUDA_VISIBLE_DEVICES=__TAIL_GPU__
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/trishita/omnimed/.hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec /home/trishita/omnimed/.venv/bin/python -u repo/experiments/omnimed_experiments.py \
  --base repo/source/MedFederate_Colab_Complete.py \
  --tier standard --cache /home/trishita/omnimed/data_cache_standard_controlled.pkl \
  --only E5,E6,E7 \
  --out /home/trishita/omnimed/results_cold_tail.json
SH
sed -i "s/__TAIL_GPU__/${TAIL_GPU}/" run_cold_tail.sh
chmod +x run_cold_tail.sh
tmux new-session -d -s omnimed_tail \
  "bash /home/trishita/omnimed/run_cold_tail.sh >> /home/trishita/omnimed/cold_tail.log 2>&1"
printf 'Validated E8; E5--E7 tail launched on physical GPU %s.\n' "$TAIL_GPU"
