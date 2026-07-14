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

SAVE_DIR="${SAVE_DIR:-results/phenomenon_cause_fullnetwork_gradcam_phase0_3}"
NUM_SAMPLES="${NUM_SAMPLES:-96}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-pred_fg}"

BASELINE_CKPT="${BASELINE_CKPT:-checkpoints/baseline_brats2023_scratch100_seed42/baseline/seed_42/baseline_seed42_best_val_loss.pth}"
NOSKIP_CKPT="${NOSKIP_CKPT:-skip_case_checkpoints/skip_case_brats2023_full100_seed42/noskip_unet/seed_42/noskip_unet_seed42_best_val_loss.pth}"
STAGE1_CSV="${STAGE1_CSV:-results/skip_case_baseline_vs_noskip_mechanism_comparison/table_10_stage1_node_state_comparison.csv}"


if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${NOSKIP_CKPT}" ]]; then
  echo "Missing no-skip checkpoint: ${NOSKIP_CKPT}" >&2
  exit 2
fi

"${PYTHON_BIN}" -u guide_unet/process_analysis/analyze_phenomenon_cause_gradcam.py \
  --model_spec "baseline=${BASELINE_CKPT}" "noskip_unet=${NOSKIP_CKPT}" \
  --data_dir data \
  --split "${SPLIT}" \
  --save_dir "${SAVE_DIR}" \
  --stage1_csv "${STAGE1_CSV}" \
  --target_mode "${TARGET_MODE}" \
  --foreground_only \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --filmstrip_cases 4 \
  "$@"
