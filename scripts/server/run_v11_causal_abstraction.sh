#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${PPTT_WORKSPACE_ROOT:-$(pwd)}"
ASSET_ROOT="${PPTT_ASSET_ROOT:-}"
PYTHON_BIN="${PPTT_PYTHON_BIN:-$WORKSPACE_ROOT/.venv/bin/python}"
CONFIG="${PPTT_V11_CONFIG:-configs/experiments/v11_causal_abstraction.yaml}"
DOSE_BATCH_SIZE="${PPTT_V11_DOSE_BATCH_SIZE:-5}"
GPU_BUDGET_MIB="${PPTT_V11_GPU_BUDGET_MIB:-22016}"
ROOT="results/v11_causal_abstraction"
LOCK="$ROOT/v11_protocol_lock.json"
FORMAL_ROOT="$ROOT/formal_jobs"
PROFILE_ROOT="$ROOT/memory_profiles"
LOG_ROOT="logs/v11_causal_abstraction"
STATUS_FILE="$ROOT/pipeline_status.json"
PID_FILE="$ROOT/pipeline.pid"

if [[ -z "$ASSET_ROOT" ]]; then
  printf '%s\n' 'PPTT_ASSET_ROOT is required' >&2
  exit 2
fi
if [[ ! "$GPU_BUDGET_MIB" =~ ^[0-9]+$ ]] || (( GPU_BUDGET_MIB <= 0 )); then
  printf '%s\n' 'PPTT_V11_GPU_BUDGET_MIB must be a positive integer' >&2
  exit 2
fi

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$ROOT" "$PROFILE_ROOT" "$LOG_ROOT/formal"

write_status() {
  local phase="$1"
  local status="$2"
  local message="$3"
  PIPELINE_PHASE="$phase" PIPELINE_STATUS="$status" PIPELINE_MESSAGE="$message" \
    "$PYTHON_BIN" - "$STATUS_FILE" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
now = datetime.now(timezone.utc).isoformat()
previous = {}
if path.is_file():
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
payload = {
    "phase": os.environ["PIPELINE_PHASE"],
    "status": os.environ["PIPELINE_STATUS"],
    "message": os.environ["PIPELINE_MESSAGE"],
    "started_at": previous.get("started_at", now),
    "updated_at": now,
}
if payload["status"] in {"PASS", "FAILED"}:
    payload["completed_at"] = now
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

fail_pipeline() {
  write_status "$1" "FAILED" "$2"
  printf 'V11 failed in %s: %s\n' "$1" "$2" >&2
  exit 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  fail_pipeline "PREFLIGHT" "Python environment is not executable: $PYTHON_BIN"
fi
if [[ ! -f "$CONFIG" || ! -f "$LOCK" ]]; then
  fail_pipeline "PREFLIGHT" "configuration or immutable protocol lock is missing"
fi
if [[ -n "$(git status --porcelain)" ]]; then
  fail_pipeline "PREFLIGHT" "formal V11 requires a clean git worktree"
fi
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    fail_pipeline "PREFLIGHT" "V11 scheduler is already running with PID $existing_pid"
  fi
fi
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

monitor_command="cd $WORKSPACE_ROOT && $PYTHON_BIN scripts/monitor_v11_causal_abstraction.py --workspace-root . --watch-seconds 10"
printf 'Monitor command:\n%s\n' "$monitor_command"

write_status "MEMORY_PROFILE" "RUNNING" \
  "profiling one locked validation patient per main architecture"
for model in unet_baseline transunet_r50_vit_b16; do
  profile="$PROFILE_ROOT/${model}_seed_42.json"
  log="$LOG_ROOT/memory_profile_${model}.log"
  if ! "$PYTHON_BIN" scripts/profile_v11_causal_abstraction_memory.py \
    --workspace-root . \
    --config "$CONFIG" \
    --protocol-lock "$LOCK" \
    --asset-root "$ASSET_ROOT" \
    --model "$model" \
    --model-seed 42 \
    --device cuda \
    --dose-batch-size "$DOSE_BATCH_SIZE" \
    --output "$profile" \
    --resume >"$log" 2>&1; then
    fail_pipeline "MEMORY_PROFILE" "memory profile failed for $model; see $log"
  fi
done

schedule="$ROOT/formal_schedule.tsv"
if ! "$PYTHON_BIN" - \
  "$CONFIG" "$PROFILE_ROOT" "$FORMAL_ROOT" "$GPU_BUDGET_MIB" "$schedule" <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import yaml

config_path, profile_root, formal_root, budget_text, output_path = sys.argv[1:]
configuration = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
profile_root = Path(profile_root)
formal_root = Path(formal_root)
budget = int(budget_text)
profiles = {}
for model in ("unet_baseline", "transunet_r50_vit_b16"):
    payload = json.loads(
        (profile_root / f"{model}_seed_42.json").read_text(encoding="utf-8")
    )
    if (
        payload.get("status") != "COMPLETE_NONFORMAL_VALIDATION_MEMORY_PROFILE"
        or payload.get("formal_claim_eligible") is not False
        or payload.get("profile_may_not_enter_formal_summary") is not True
    ):
        raise SystemExit(f"inadmissible memory profile: {model}")
    peak = max(
        float(payload["peak_allocated_mib"]),
        float(payload["peak_reserved_mib"]),
    )
    profiles[model] = int(math.ceil(peak * 1.15 + 512.0))

jobs = []
models = [*configuration["main_models"], *configuration["control_models"]]
expected = int(configuration["coverage"]["expected_formal_patients"])
for model in models:
    memory_model = "unet_baseline" if model == "unet_noskip" else model
    estimate = profiles[memory_model]
    if estimate > budget:
        raise SystemExit(
            f"one {model} job needs {estimate} MiB, above the {budget} MiB budget"
        )
    for seed in configuration["model_seeds"]:
        job_root = formal_root / model / f"seed_{int(seed)}"
        status_path = job_root / "job_status.json"
        patient_root = job_root / "patient_results"
        completed = len(list(patient_root.glob("*/patient_result.json")))
        status = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE" and completed == expected:
            continue
        jobs.append((model, int(seed), estimate))

bins = []
for job in sorted(jobs, key=lambda value: (-value[2], value[0], value[1])):
    for batch in bins:
        if sum(item[2] for item in batch) + job[2] <= budget:
            batch.append(job)
            break
    else:
        bins.append([job])

output = Path(output_path)
temporary = output.with_suffix(output.suffix + ".tmp")
with temporary.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write("batch\tmodel\tseed\testimated_peak_mib\n")
    for batch_index, batch in enumerate(bins, start=1):
        for model, seed, estimate in batch:
            stream.write(f"{batch_index}\t{model}\t{seed}\t{estimate}\n")
temporary.replace(output)
print(
    json.dumps(
        {
            "pending_jobs": len(jobs),
            "batch_count": len(bins),
            "gpu_budget_mib": budget,
            "profile_estimates_mib": profiles,
        },
        sort_keys=True,
    )
)
PY
then
  fail_pipeline "SCHEDULING" "unable to construct the memory-bounded formal schedule"
fi

batch_ids=()
if [[ $(wc -l < "$schedule") -gt 1 ]]; then
  mapfile -t batch_ids < <(tail -n +2 "$schedule" | cut -f1 | sort -n -u)
fi
write_status "FORMAL_INTERVENTION" "RUNNING" \
  "running ${#batch_ids[@]} memory-bounded batches without changing the locked matrix"

for batch in "${batch_ids[@]}"; do
  mapfile -t rows < <(awk -F '\t' -v batch="$batch" 'NR > 1 && $1 == batch {print $2 "\t" $3 "\t" $4}' "$schedule")
  pids=()
  identities=()
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r model seed estimate <<< "$row"
    job_log_root="$LOG_ROOT/formal/$model"
    mkdir -p "$job_log_root"
    log="$job_log_root/seed_${seed}.log"
    pid_file="$job_log_root/seed_${seed}.pid"
    if [[ -f "$pid_file" ]]; then
      existing_pid="$(tr -cd '0-9' < "$pid_file")"
      if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        fail_pipeline "FORMAL_INTERVENTION" \
          "refusing duplicate $model seed $seed; PID $existing_pid is alive"
      fi
    fi
    printf 'Launching batch %s: %s seed %s, estimated peak %s MiB\n' \
      "$batch" "$model" "$seed" "$estimate"
    "$PYTHON_BIN" scripts/run_v11_causal_abstraction.py \
      --workspace-root . \
      --config "$CONFIG" \
      --protocol-lock "$LOCK" \
      --asset-root "$ASSET_ROOT" \
      --model "$model" \
      --model-seed "$seed" \
      --execution-mode formal \
      --device cuda \
      --dose-batch-size "$DOSE_BATCH_SIZE" \
      --resume >"$log" 2>&1 &
    pid="$!"
    printf '%s\n' "$pid" > "$pid_file"
    pids+=("$pid")
    identities+=("$model/seed_$seed")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      printf 'Worker failed: %s\n' "${identities[$index]}" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    fail_pipeline "FORMAL_INTERVENTION" \
      "one or more workers failed in batch $batch; no summary was run"
  fi
done

write_status "FORMAL_SUMMARY" "RUNNING" \
  "all nine jobs are complete; applying the immutable patient-level gates"
if ! "$PYTHON_BIN" scripts/summarize_v11_causal_abstraction.py \
  --workspace-root . \
  --config "$CONFIG" \
  --protocol-lock "$LOCK" >"$LOG_ROOT/formal_summary.log" 2>&1; then
  fail_pipeline "FORMAL_SUMMARY" "formal summary or adversarial audit failed"
fi

scientific_status="$($PYTHON_BIN - "$ROOT/v11_status.json" <<'PY'
import json
from pathlib import Path
import sys

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
write_status "COMPLETE" "PASS" "$scientific_status"
printf 'V11 pipeline complete: %s\n' "$scientific_status"
