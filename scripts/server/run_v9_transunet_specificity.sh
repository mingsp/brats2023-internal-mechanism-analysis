#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${PPTT_WORKSPACE_ROOT:-/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace}"
ASSET_ROOT="${PPTT_ASSET_ROOT:-/root/autodl-tmp/A_scheme_workspace/brats2023_data}"
PYTHON_BIN="${PPTT_PYTHON_BIN:-$WORKSPACE_ROOT/.venv/bin/python}"
CONFIG="configs/experiments/v9_transunet_specificity.yaml"
ROOT="results/v9_transunet_specificity"
FORMAL_ROOT="$ROOT/formal_test"
LOCK="$ROOT/v9_protocol_lock.json"
LOG_ROOT="logs/v9_formal"
STATUS_FILE="$ROOT/pipeline_status.json"
SEEDS=(42 123 3407)

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT"
mkdir -p "$ROOT" "$LOG_ROOT"

write_status() {
  local phase="$1"
  local status="$2"
  local message="$3"
  PIPELINE_PHASE="$phase" PIPELINE_STATUS="$status" PIPELINE_MESSAGE="$message" \
    "$PYTHON_BIN" - "$STATUS_FILE" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "phase": os.environ["PIPELINE_PHASE"],
    "status": os.environ["PIPELINE_STATUS"],
    "message": os.environ["PIPELINE_MESSAGE"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
}

fail() {
  write_status "$1" "FAILED" "$2"
  printf 'V9 failed in %s: %s\n' "$1" "$2" >&2
  exit 1
}

if [[ -n "$(git status --porcelain)" ]]; then
  fail "PREFLIGHT" "source tree is not clean"
fi
if [[ -e "$LOCK" || -d "$FORMAL_ROOT" ]]; then
  fail "PREFLIGHT" "formal lock or output already exists"
fi

write_status "PROTOCOL_LOCK" "RUNNING" \
  "locking feasible candidate and 26 common intervention-naive patients"
if ! "$PYTHON_BIN" scripts/lock_v9_transunet_protocol.py \
  --workspace-root . --asset-root "$ASSET_ROOT" \
  >"$LOG_ROOT/protocol_lock.log" 2>&1; then
  fail "PROTOCOL_LOCK" "protocol lock failed"
fi

write_status "FORMAL_INTERVENTION" "RUNNING" \
  "running three locked model seeds in parallel"
pids=()
for seed in "${SEEDS[@]}"; do
  "$PYTHON_BIN" scripts/run_v8_transunet_mechanism.py \
    --workspace-root . \
    --config "$CONFIG" \
    --protocol-lock "$LOCK" \
    --asset-root "$ASSET_ROOT" \
    --model-seed "$seed" \
    --device cuda \
    >"$LOG_ROOT/seed_${seed}.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf '%s\n' "$pid" >"$LOG_ROOT/seed_${seed}.pid"
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  fail "FORMAL_INTERVENTION" "one or more seed workers failed"
fi

write_status "FORMAL_SUMMARY" "RUNNING" \
  "auditing endpoints, dose response and intervention operators"
if ! "$PYTHON_BIN" scripts/summarize_v8_transunet_mechanism.py \
  --workspace-root . \
  --config "$CONFIG" \
  --protocol-lock "$LOCK" \
  >"$LOG_ROOT/summary.log" 2>&1; then
  fail "FORMAL_SUMMARY" "formal summary failed"
fi

scientific_status="$($PYTHON_BIN - "$FORMAL_ROOT/v9_status.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["scientific_status"])
PY
)"
write_status "COMPLETE" "PASS" "$scientific_status"
printf 'V9 pipeline complete: %s\n' "$scientific_status"
