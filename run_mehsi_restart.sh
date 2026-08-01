#!/usr/bin/env bash

set -u

CMD=(
  python
  hsi_denosing_mehsi.py
  -a hssm
  -p hsdssm-mehsi
  --lr 1e-4
  --loss char
  -r
  -rp checkpoints/hssm/hsdssm-mehsi/model_latest.pth
)

RESTART_DELAY_SECONDS=5

trap 'echo "Stop requested. Exiting restart loop."; exit 130' INT TERM

while true; do
  echo "[$(date '+%F %T')] Starting training..."
  "${CMD[@]}"
  exit_code=$?

  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date '+%F %T')] Training exited cleanly."
    break
  fi

  echo "[$(date '+%F %T')] Training exited with code $exit_code. Restarting in ${RESTART_DELAY_SECONDS}s..."
  sleep "$RESTART_DELAY_SECONDS"
done
