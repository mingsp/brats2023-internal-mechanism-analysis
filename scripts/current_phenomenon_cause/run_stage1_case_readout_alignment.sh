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

SAVE_DIR="${SAVE_DIR:-results/stage1_case_readout_alignment_v1}"
DATA_DIR="${DATA_DIR:-data}"
CAM_CASE_CSV="${CAM_CASE_CSV:-results/phenomenon_cause_fullnetwork_gradcam_phase0_3/table_phase1_2_gradcam_per_case.csv}"
STAGE1_REFERENCE_CSV="${STAGE1_REFERENCE_CSV:-results/skip_case_baseline_vs_noskip_mechanism_comparison/table_10_stage1_node_state_comparison.csv}"

BASELINE_CKPT="${BASELINE_CKPT:-checkpoints/baseline_brats2023_scratch100_seed42/baseline/seed_42/baseline_seed42_best_val_loss.pth}"
NOSKIP_CKPT="${NOSKIP_CKPT:-skip_case_checkpoints/skip_case_brats2023_full100_seed42/noskip_unet/seed_42/noskip_unet_seed42_best_val_loss.pth}"


if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${NOSKIP_CKPT}" ]]; then
  echo "Missing no-skip checkpoint: ${NOSKIP_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${CAM_CASE_CSV}" ]]; then
  echo "Missing CAM case CSV: ${CAM_CASE_CSV}" >&2
  exit 2
fi

"${PYTHON_BIN}" -u guide_unet/process_analysis/export_stage1_case_readout_alignment.py \
  --model_spec "baseline=${BASELINE_CKPT}" "noskip_unet=${NOSKIP_CKPT}" \
  --data_dir "${DATA_DIR}" \
  --fit_split val \
  --eval_split test \
  --save_dir "${SAVE_DIR}" \
  --cam_case_csv "${CAM_CASE_CSV}" \
  --stage1_reference_csv "${STAGE1_REFERENCE_CSV}" \
  --foreground_only \
  --match_cam_cases \
  --num_fit_samples "${NUM_FIT_SAMPLES:-256}" \
  --num_eval_samples "${NUM_EVAL_SAMPLES:-96}" \
  --fit_epochs "${FIT_EPOCHS:-4}" \
  --fit_lr "${FIT_LR:-5e-4}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --bootstrap "${BOOTSTRAP:-1000}" \
  "$@"
