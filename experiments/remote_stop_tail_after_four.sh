#!/usr/bin/env bash
set -euo pipefail
cd /home/trishita/omnimed

while tmux has-session -t omnimed_tail 2>/dev/null; do
  COUNT=$(./.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('results_cold_tail.json')
print(len(json.loads(p.read_text()).get('E5_fusion_seeds', {})) if p.exists() else 0)
PY
)
  if [[ "$COUNT" -ge 2 ]]; then
    tmux kill-session -t omnimed_tail
    tmux new-session -d -s omnimed_cost \
      'bash /home/trishita/omnimed/run_cold_cost.sh >> /home/trishita/omnimed/cold_cost.log 2>&1'
    printf 'Stopped concat chunk at %s records; E6--E7 launched.\n' "$COUNT"
    exit 0
  fi
  sleep 30
done

printf 'Original tail ended before two E5 records; cost chunk not launched.\n'
exit 2
