#!/usr/bin/env bash
set -u

cd /home/trishita/omnimed || exit 1
while tmux has-session -t omnimed 2>/dev/null; do
  E4_COUNT=$(./.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("results_cold_v2.json")
if not p.exists():
    print(0)
else:
    print(len(json.loads(p.read_text()).get("E4_warmstart", {})))
PY
)
  if [ "$E4_COUNT" -eq 4 ]; then
    # E1--E4 are durable at this point. E5--E7 are assigned to the disjoint
    # GPU-0 tail store, so stop before the main process duplicates that work.
    tmux kill-session -t omnimed
    printf 'Stopped main after all four E4 records were flushed.\n'
    exit 0
  fi
  sleep 30
done

printf 'Main session ended before E4 completed; no stop action taken.\n'
exit 2
