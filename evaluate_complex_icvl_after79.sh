#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/ssumamba/complex_icvl}"
START_EPOCH="${START_EPOCH:-80}"
GPU_IDS="${GPU_IDS:-0}"
ARCH="${ARCH:-ssumamba}"
PREFIX="${PREFIX:-gauss}"
LOSS="${LOSS:-char}"
CONDA_ENV="${CONDA_ENV:-ssumamba-org}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=(python)
fi

shopt -s nullglob

checkpoints=("${CHECKPOINT_DIR}"/model_epoch_*.pth)
if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "No epoch checkpoints found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

printf '%s\n' "${checkpoints[@]}" | sort -V | while IFS= read -r checkpoint; do
  filename="$(basename "${checkpoint}")"

  if [[ ! "${filename}" =~ ^model_epoch_([0-9]+)_.*\.pth$ ]]; then
    echo "Skipping unrecognized checkpoint name: ${checkpoint}"
    continue
  fi

  epoch="${BASH_REMATCH[1]}"
  if [ "${epoch}" -lt "${START_EPOCH}" ]; then
    continue
  fi

  log_file="${CHECKPOINT_DIR}/result-complex-${epoch}epoch.txt"
  echo "Evaluating epoch ${epoch}: ${checkpoint}"
  echo "Writing log: ${log_file}"

  "${PYTHON_CMD[@]}" hsi_test.py \
    -a "${ARCH}" \
    -p "${PREFIX}" \
    -r \
    -rp "${checkpoint}" \
    --gpu-ids "${GPU_IDS}" \
    --loss "${LOSS}" \
    > "${log_file}" 2>&1
done

echo "Evaluation complete for checkpoints in ${CHECKPOINT_DIR} from epoch ${START_EPOCH} onward."
