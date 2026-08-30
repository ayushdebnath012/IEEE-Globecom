#!/usr/bin/env bash
set -euo pipefail
cd /home/trishita/omnimed

while tmux has-session -t omnimed_tail2 2>/dev/null; do
  sleep 30
done

COUNT=$(./.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('results_cold_tail2.json')
print(len(json.loads(p.read_text()).get('E5_fusion_seeds', {})) if p.exists() else 0)
PY
)
if [[ "$COUNT" != "8" ]]; then
  printf 'Tail2 incomplete at %s/8; tail3 not launched.\n' "$COUNT"
  exit 2
fi

tmux new-session -d -s omnimed_tail3 \
  'bash /home/trishita/omnimed/run_cold_tail3.sh >> /home/trishita/omnimed/cold_tail3.log 2>&1'
printf 'Tail2 complete; attention/gated/clip chunk launched.\n'
