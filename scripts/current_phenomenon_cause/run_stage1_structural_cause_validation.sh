#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/guide_unet:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-data}"
CASE_SET_CSV="${CASE_SET_CSV:-stage1_explanation_suite_20260517/results/common_case_sets/formal_n512_cases.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-stage1_explanation_suite_20260517/results/e12_structural_cause_validation_formal_n512}"
NUM_CASES="${NUM_CASES:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BOOTSTRAP="${BOOTSTRAP:-1000}"

BASELINE_CKPT="${BASELINE_CKPT:-checkpoints/baseline_brats2023_scratch100_seed42/baseline/seed_42/baseline_seed42_best_val_loss.pth}"
NOSKIP_CKPT="${NOSKIP_CKPT:-skip_case_checkpoints/skip_case_brats2023_full100_seed42/noskip_unet/seed_42/noskip_unet_seed42_best_val_loss.pth}"

if [[ ! -f "${BASELINE_CKPT}" ]]; then echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2; exit 2; fi
if [[ ! -f "${NOSKIP_CKPT}" ]]; then echo "Missing no-skip checkpoint: ${NOSKIP_CKPT}" >&2; exit 2; fi

"${PYTHON_BIN}" -u guide_unet/process_analysis/stage1_structural_cause_validation.py \
  --model-spec "baseline=${BASELINE_CKPT}" "noskip_unet=${NOSKIP_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --case-set-csv "${CASE_SET_CSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-cases "${NUM_CASES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --bootstrap "${BOOTSTRAP}" \
  "$@"
