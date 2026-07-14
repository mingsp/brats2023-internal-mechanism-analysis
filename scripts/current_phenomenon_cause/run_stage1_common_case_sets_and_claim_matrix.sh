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

OUTPUT_ROOT="${OUTPUT_ROOT:-stage1_explanation_suite_20260517/results}"

"${PYTHON_BIN}" -u guide_unet/process_analysis/stage1_build_case_sets_and_claim_matrix.py \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

