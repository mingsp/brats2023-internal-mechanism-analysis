#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/guide_unet:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

PRED_DIR="${PRED_DIR:-results/phenomenon_cause_fullnetwork_gradcam_phase0_3}"
GT_DIR="${GT_DIR:-results/phenomenon_cause_fullnetwork_gradcam_phase0_3_gtfg}"
STAGE1_CSV="${STAGE1_CSV:-results/skip_case_baseline_vs_noskip_mechanism_comparison/table_10_stage1_node_state_comparison.csv}"
SAVE_DIR="${SAVE_DIR:-results/stage1_explanation_closure_v1}"
BOOTSTRAP="${BOOTSTRAP:-2000}"

"${PYTHON_BIN}" -u guide_unet/process_analysis/close_stage1_phenomenon_loop.py \
  --pred_dir "${PRED_DIR}" \
  --gt_dir "${GT_DIR}" \
  --stage1_csv "${STAGE1_CSV}" \
  --save_dir "${SAVE_DIR}" \
  --bootstrap "${BOOTSTRAP}" \
  --top_cases 3
