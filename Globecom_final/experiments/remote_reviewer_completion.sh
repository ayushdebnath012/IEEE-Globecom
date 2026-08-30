#!/usr/bin/env bash
# Reviewer-completion queue for the H100 server. Valid JSON records are reused;
# invalid records are preserved with an .invalid suffix before being rerun.
set -u
set -o pipefail

ROOT=/home/trishita/omnimed
EXP="$ROOT/repo/experiments"
PY="$ROOT/.venv/bin/python"
BASE="$ROOT/repo/source/MedFederate_Colab_Complete.py"
CACHE="$ROOT/data_cache_standard_controlled.pkl"
RESULTS="$EXP/reviewer_results"
LOGS="$EXP/reviewer_logs"
EXPECTED_BASE_SHA256=ce473f4bca58f8920d7c22b55b3e0dd28a2de227049f4ad77141659468cbf227
EXPECTED_CACHE_SHA256=4286565db7ff817f6cca0894479b7c1f8836fa73aa09407fd906634dbb0969ba

mkdir -p "$RESULTS" "$LOGS"
cd "$EXP" || exit 1

# Hold the advisory lock for the lifetime of this process. A persistent lock
# file is harmless: flock state belongs to descriptor 9, not the file itself.
if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required; refusing to start an unguarded queue\n' >&2
  exit 1
fi
exec 9>"$RESULTS/.reviewer_completion_queue.lock"
if ! flock -n 9; then
  printf 'another reviewer-completion queue already holds the lock\n' >&2
  exit 1
fi

# A resumed queue must earn a new completion marker after revalidation.
rm -f "$RESULTS/QUEUE_COMPLETE"

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$ROOT/.hf"
export TOKENIZERS_PARALLELISM=false

validate_record() {
  local path="$1"
  local expected_key="$2"
  "$PY" - "$path" "$expected_key" \
      "$EXPECTED_BASE_SHA256" "$EXPECTED_CACHE_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_key, expected_base, expected_cache = sys.argv[2:5]

def require(condition, message):
    if not condition:
        raise ValueError(message)

try:
    payload = json.loads(path.read_text(encoding="utf8"))
    provenance = payload["provenance"]
    parameters = payload["parameters"]
    require(payload["key"] == expected_key, "unexpected result key")
    require(provenance["base_model_sha256"] == expected_base, "wrong base-model hash")
    require(provenance["data_cache_sha256"] == expected_cache, "wrong data-cache hash")
    require(parameters["deterministic"] is False, "deterministic parameter is not false")
    require(
        provenance["deterministic_algorithms_enforced"] is False,
        "deterministic algorithms were enforced",
    )
    require(isinstance(payload["result"], dict), "result is not an object")
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"invalid record {path.name}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_one() {
  local tag="$1"
  local expected_key="$2"
  shift 2
  local out="$RESULTS/$tag.json"
  local log="$LOGS/$tag.log"
  local invalid_copy
  if [[ -e "$out" ]]; then
    if validate_record "$out" "$expected_key"; then
      rm -f "$RESULTS/$tag.failed"
      printf 'skip valid %s\n' "$tag"
      return 0
    fi
    invalid_copy="$out.invalid.$(date -u +%Y%m%dT%H%M%SZ).$$"
    if ! mv -- "$out" "$invalid_copy"; then
      printf '%s\n' 'could not preserve invalid record' >"$RESULTS/$tag.failed"
      printf 'FAILED %s: could not preserve invalid record\n' "$tag"
      return 0
    fi
    printf 'preserved invalid %s as %s\n' "$tag" "$(basename "$invalid_copy")"
  fi
  rm -f "$RESULTS/$tag.failed"
  printf 'start %s\n' "$tag"
  if "$PY" reviewer_completion.py --base "$BASE" --cache "$CACHE" \
      --out "$out" "$@" >"$log" 2>&1; then
    if validate_record "$out" "$expected_key"; then
      printf 'done %s\n' "$tag"
    else
      printf '%s\n' 'record validation failed' >"$RESULTS/$tag.failed"
      printf 'FAILED %s: record validation failed\n' "$tag"
    fi
  else
    local rc=$?
    printf '%s\n' "$rc" >"$RESULTS/$tag.failed"
    printf 'FAILED %s rc=%s\n' "$tag" "$rc"
  fi
}

# Stage 1: the missing cells of the K x alpha systems grid. Run sequentially
# so our own jobs do not contaminate each other's runtime and memory records.
for alpha in 0.1 5.0; do
  for clients in 3 10 20; do
    for seed in 0 1; do
      tag="grid_a${alpha}_k${clients}_s${seed}"
      run_one "$tag" "alpha=${alpha}|K=${clients}|seed=${seed}" \
        --task grid --alpha "$alpha" --clients "$clients" --seed "$seed"
    done
  done
done

# Stage 2: operational federated fusion sweep. Performance is the target here,
# so three bounded workers reduce turnaround without affecting the cost grid.
fusion_worker() {
  local worker="$1"
  shift
  for fusion in "$@"; do
    for seed in 0 1; do
      tag="fusion_${fusion}_a0.1_k5_s${seed}"
      run_one "$tag" "fusion=${fusion}|alpha=0.1|K=5|seed=${seed}" \
        --task fusion --fusion "$fusion" --alpha 0.1 --clients 5 --seed "$seed"
    done
  done
  printf 'fusion worker %s complete\n' "$worker"
}

fusion_worker 0 concat gated coca &
p0=$!
fusion_worker 1 attention clip unified_io &
p1=$!
fusion_worker 2 flamingo blip2 &
p2=$!
wait "$p0" "$p1" "$p2"

# Stage 3: matched recent-method controls.
for seed in 0 1; do
  run_one "fedmme_a0.1_k5_s${seed}" \
    "fedmme_style|alpha=0.1|K=5|seed=${seed}" \
    --task fedmme --alpha 0.1 --clients 5 --seed "$seed" &
done
wait

for seed in 0 1; do
  run_one "init_random_a1.0_k5_s${seed}" \
    "init=random|alpha=1.0|K=5|seed=${seed}" \
    --task init --init-mode random --alpha 1.0 --clients 5 --seed "$seed" &
done
wait

# Stage 4: P-FIN missing-text stress test, one worker per method so paired seeds
# within a method remain sequential.
pfin_worker() {
  local mode="$1"
  for seed in 0 1; do
    tag="pfin_${mode}_a0.1_k5_s${seed}"
    run_one "$tag" "pfin=${mode}|alpha=0.1|K=5|seed=${seed}" \
      --task pfin --pfin-mode "$mode" --alpha 0.1 --clients 5 --seed "$seed"
  done
}

pfin_worker zero & q0=$!
pfin_worker deterministic & q1=$!
pfin_worker probabilistic & q2=$!
pfin_worker probabilistic_uq & q3=$!
wait "$q0" "$q1" "$q2" "$q3"

# Independently reconstruct the 40-record manifest and validate the complete
# directory. Extra/missing JSON, duplicate or wrong keys, bad provenance, a
# deterministic run, or any failure marker prevents QUEUE_COMPLETE.
if "$PY" - "$RESULTS" "$EXPECTED_BASE_SHA256" "$EXPECTED_CACHE_SHA256" <<'PY'
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
expected_base, expected_cache = sys.argv[2:4]
expected = {}

def require(condition, message):
    if not condition:
        raise ValueError(message)

for alpha in ("0.1", "5.0"):
    for clients in (3, 10, 20):
        for seed in (0, 1):
            tag = f"grid_a{alpha}_k{clients}_s{seed}"
            expected[tag] = f"alpha={alpha}|K={clients}|seed={seed}"

for fusion in ("concat", "attention", "gated", "clip", "flamingo", "blip2", "coca", "unified_io"):
    for seed in (0, 1):
        tag = f"fusion_{fusion}_a0.1_k5_s{seed}"
        expected[tag] = f"fusion={fusion}|alpha=0.1|K=5|seed={seed}"

for seed in (0, 1):
    expected[f"fedmme_a0.1_k5_s{seed}"] = f"fedmme_style|alpha=0.1|K=5|seed={seed}"
    expected[f"init_random_a1.0_k5_s{seed}"] = f"init=random|alpha=1.0|K=5|seed={seed}"

for mode in ("zero", "deterministic", "probabilistic", "probabilistic_uq"):
    for seed in (0, 1):
        tag = f"pfin_{mode}_a0.1_k5_s{seed}"
        expected[tag] = f"pfin={mode}|alpha=0.1|K=5|seed={seed}"

errors = []
if len(expected) != 40:
    errors.append(f"internal manifest has {len(expected)} entries, expected 40")

actual = {path.stem: path for path in results.glob("*.json")}
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
if missing:
    errors.append("missing JSON: " + ", ".join(missing))
if extra:
    errors.append("unexpected JSON: " + ", ".join(extra))

seen_keys = set()
for tag, expected_key in expected.items():
    path = actual.get(tag)
    if path is None:
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
        provenance = payload["provenance"]
        parameters = payload["parameters"]
        require(payload["key"] == expected_key, "unexpected result key")
        require(provenance["base_model_sha256"] == expected_base, "wrong base-model hash")
        require(provenance["data_cache_sha256"] == expected_cache, "wrong data-cache hash")
        require(parameters["deterministic"] is False, "deterministic parameter is not false")
        require(
            provenance["deterministic_algorithms_enforced"] is False,
            "deterministic algorithms were enforced",
        )
        require(isinstance(payload["result"], dict), "result is not an object")
        if payload["key"] in seen_keys:
            raise ValueError("duplicate result key")
        seen_keys.add(payload["key"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {path.name}: {exc}")

failed = sorted(path.name for path in results.glob("*.failed"))
if failed:
    errors.append("failure markers: " + ", ".join(failed))
if len(seen_keys) != 40:
    errors.append(f"validated {len(seen_keys)} unique keys, expected 40")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

print("validated json=40 failed=0 unique_keys=40")
PY
then
  date -u +%FT%TZ >"$RESULTS/QUEUE_COMPLETE"
  printf 'QUEUE_COMPLETE created after full validation\n'
else
  printf 'queue incomplete; QUEUE_COMPLETE was not created\n' >&2
  exit 1
fi
