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

READOUT_CSV="${READOUT_CSV:-results/stage1_case_readout_alignment_n512/table_stage1_case_readout.csv}"
CLASS_REGION_CSV="${CLASS_REGION_CSV:-stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512/table_class_region_response_case_level.csv}"
PERTURBATION_CSV="${PERTURBATION_CSV:-stage1_explanation_suite_20260517/results/e4_faithfulness_perturbation_formal_n512_allnodes_main/table_perturbation_random_adjusted.csv}"
JOINED_CSV="${JOINED_CSV:-stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512/table_class_region_perturbation_joined_case_level.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-stage1_explanation_suite_20260517/results/e10_statistical_validation_formal_n512}"
BOOTSTRAP="${BOOTSTRAP:-10000}"

"${PYTHON_BIN}" -u guide_unet/process_analysis/stage1_statistical_validation.py \
  --readout-csv "${READOUT_CSV}" \
  --class-region-csv "${CLASS_REGION_CSV}" \
  --perturbation-csv "${PERTURBATION_CSV}" \
  --joined-csv "${JOINED_CSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --bootstrap "${BOOTSTRAP}" \
  "$@"
