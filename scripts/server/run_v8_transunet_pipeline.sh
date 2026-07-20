#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=""
CONFIG="configs/experiments/v8_transunet_mechanism_replication.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${PPTT_DEVICE:-cuda}"

usage() {
  printf '%s\n' \
    "Usage: $0 --workspace PATH [--config PATH]" \
    "Requires PPTT_ASSET_ROOT. Runs three validation workers and, only after" \
    "candidate registration plus protocol locking, three formal workers in parallel."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WORKSPACE" ]]; then
  printf '%s\n' "--workspace is required" >&2
  exit 2
fi
if [[ -z "${PPTT_ASSET_ROOT:-}" ]]; then
  printf '%s\n' "PPTT_ASSET_ROOT is required" >&2
  exit 2
fi

WORKSPACE="$(cd "$WORKSPACE" && pwd)"
cd "$WORKSPACE"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$WORKSPACE/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Configuration not found: %s\n' "$CONFIG" >&2
  exit 2
fi

export PYTHONPATH="$WORKSPACE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

ROOT="$WORKSPACE/results/v8_transunet_mechanism"
LOG_ROOT="$WORKSPACE/logs/v8_transunet_mechanism"
STATUS_FILE="$ROOT/pipeline_status.json"
PID_FILE="$ROOT/pipeline.pid"
mkdir -p "$ROOT" "$LOG_ROOT/validation" "$LOG_ROOT/formal"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'V8 pipeline is already running with PID %s\n' "$existing_pid" >&2
    exit 3
  fi
fi
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

write_status() {
  local phase="$1"
  local status="$2"
  local message="$3"
  PIPELINE_PHASE="$phase" \
  PIPELINE_STATUS="$status" \
  PIPELINE_MESSAGE="$message" \
  PIPELINE_STATUS_FILE="$STATUS_FILE" \
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

path = Path(os.environ["PIPELINE_STATUS_FILE"])
now = datetime.now(timezone.utc).isoformat()
previous = {}
if path.is_file():
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
payload = {
    "status": os.environ["PIPELINE_STATUS"],
    "phase": os.environ["PIPELINE_PHASE"],
    "message": os.environ["PIPELINE_MESSAGE"],
    "started_at": previous.get("started_at", now),
    "updated_at": now,
}
if payload["status"] in {"PASS", "FAILED", "TERMINATED_NO_CANDIDATE"}:
    payload["completed_at"] = now
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

fail_pipeline() {
  local phase="$1"
  local message="$2"
  write_status "$phase" "FAILED" "$message"
  printf 'FAILED [%s]: %s\n' "$phase" "$message" >&2
  exit 1
}

if [[ -n "$(git status --porcelain)" ]]; then
  fail_pipeline "PREFLIGHT" "formal V8 requires a clean git worktree"
fi

SEEDS=(42 123 3407)
MODEL="transunet_r50_vit_b16"
VALIDATION_ROOT="$ROOT/validation_workers"
FORMAL_ROOT="$ROOT/formal_test"
PROTOCOL_LOCK="$ROOT/v8_protocol_lock.json"

write_status "VALIDATION_TRACES" "RUNNING" \
  "three validation trace workers are running in parallel"
validation_pids=()
for seed in "${SEEDS[@]}"; do
  log="$LOG_ROOT/validation/seed_${seed}.log"
  "$PYTHON_BIN" scripts/run_v5_transunet.py \
    --config "$CONFIG" \
    --workspace-root "$WORKSPACE" \
    --asset-root "$PPTT_ASSET_ROOT" \
    --output-root "$VALIDATION_ROOT/seed_${seed}" \
    --jobs "$MODEL:$seed" \
    --device "$DEVICE" \
    --resume >"$log" 2>&1 &
  validation_pids+=("$!")
done

validation_failed=0
for index in "${!validation_pids[@]}"; do
  if ! wait "${validation_pids[$index]}"; then
    printf 'Validation worker seed %s failed; see its log.\n' "${SEEDS[$index]}" >&2
    validation_failed=1
  fi
done
if [[ "$validation_failed" -ne 0 ]]; then
  fail_pipeline "VALIDATION_TRACES" "one or more validation workers failed"
fi

write_status "CANDIDATE_SELECTION" "RUNNING" \
  "validation traces complete; selecting at most one registered path"
if ! "$PYTHON_BIN" scripts/select_v8_transunet_candidate.py \
  --config "$CONFIG" \
  --workspace-root "$WORKSPACE" \
  >"$LOG_ROOT/candidate_selection.log" 2>&1; then
  fail_pipeline "CANDIDATE_SELECTION" "candidate selection failed"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  fail_pipeline "PROTOCOL_LOCK" "source tree changed before protocol locking"
fi
write_status "PROTOCOL_LOCK" "RUNNING" \
  "locking candidate, source, data, patients, operators and thresholds"
if ! "$PYTHON_BIN" scripts/lock_v8_transunet_protocol.py \
  --config "$CONFIG" \
  --workspace-root "$WORKSPACE" \
  --asset-root "$PPTT_ASSET_ROOT" \
  >"$LOG_ROOT/protocol_lock.log" 2>&1; then
  fail_pipeline "PROTOCOL_LOCK" "protocol lock failed"
fi

authorized="$($PYTHON_BIN - "$PROTOCOL_LOCK" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("true" if payload.get("test_intervention_authorized") is True else "false")
PY
)"
if [[ "$authorized" != "true" ]]; then
  write_status "TERMINATED_NO_CANDIDATE" "TERMINATED_NO_CANDIDATE" \
    "validation did not register a path; formal test intervention was not run"
  printf '%s\n' "No registered TransUNet candidate; stopped before test intervention."
  exit 0
fi

write_status "FORMAL_INTERVENTION" "RUNNING" \
  "three locked test-set intervention workers are running in parallel"
formal_pids=()
for seed in "${SEEDS[@]}"; do
  log="$LOG_ROOT/formal/seed_${seed}.log"
  "$PYTHON_BIN" scripts/run_v8_transunet_mechanism.py \
    --config "$CONFIG" \
    --protocol-lock "$PROTOCOL_LOCK" \
    --workspace-root "$WORKSPACE" \
    --asset-root "$PPTT_ASSET_ROOT" \
    --output-root "$FORMAL_ROOT" \
    --model-seed "$seed" \
    --device "$DEVICE" \
    --execution-mode formal \
    --resume >"$log" 2>&1 &
  formal_pids+=("$!")
done

formal_failed=0
for index in "${!formal_pids[@]}"; do
  if ! wait "${formal_pids[$index]}"; then
    printf 'Formal worker seed %s failed; see its log.\n' "${SEEDS[$index]}" >&2
    formal_failed=1
  fi
done
if [[ "$formal_failed" -ne 0 ]]; then
  fail_pipeline "FORMAL_INTERVENTION" "one or more formal workers failed"
fi

write_status "FORMAL_SUMMARY" "RUNNING" \
  "all formal workers completed; applying the locked statistical gate"
if ! "$PYTHON_BIN" scripts/summarize_v8_transunet_mechanism.py \
  --config "$CONFIG" \
  --workspace-root "$WORKSPACE" \
  --protocol-lock "$PROTOCOL_LOCK" \
  --output-root "$FORMAL_ROOT" \
  >"$LOG_ROOT/formal_summary.log" 2>&1; then
  fail_pipeline "FORMAL_SUMMARY" "formal summary or audit failed"
fi

scientific_status="$($PYTHON_BIN - "$FORMAL_ROOT/v8_status.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("scientific_status", "UNKNOWN"))
PY
)"
write_status "COMPLETE" "PASS" "$scientific_status"
printf 'V8 pipeline complete: %s\n' "$scientific_status"
