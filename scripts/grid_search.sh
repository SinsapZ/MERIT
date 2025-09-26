#!/usr/bin/env bash

# Grid search for MERIT (ETMC-style DS fusion)
# Usage:
#   bash MERIT/MERIT/scripts/grid_search.sh /work/root /path/to/dataset [MODE]
# Args:
#   $1: workspace root (the directory that contains MERIT/)
#   $2: dataset root (the directory that contains Feature/ and Label/label.npy)
#   $3: MODE = coarse | fine   (default fine)

set -e

WS_ROOT=${1:-"$(pwd)"}
DATA_ROOT=${2:-"${WS_ROOT}/MERIT/APAVA"}
MODE=${3:-"fine"}

cd "$WS_ROOT"

# Search spaces
#
# Sweep spaces (keep history for reproducibility)
#
# COARSE (original):
#   lambda_evi: 0.15 0.20 0.30 0.40 0.50
#   lr:         1e-4 7e-5 5e-5 3e-5 2e-5
#   lambda_p:   0.3 0.5 0.8
# FINE (refined around good regions):
#   lambda_evi: 0.18 0.20 0.22 0.28 0.30
#   lr:         2e-5 3e-5 4e-5 5e-5 6e-5
#   lambda_p:   0.3 0.4 0.5

if [ "$MODE" = "coarse" ]; then
  LAMBDA_EVI_LIST=(0.15 0.20 0.30 0.40 0.50)
  LR_LIST=(1e-4 7e-5 5e-5 3e-5 2e-5)
  LAMBDA_PSEUDO_LIST=(0.3 0.5 0.8)
else
  LAMBDA_EVI_LIST=(0.18 0.20 0.22 0.28 0.30)
  LR_LIST=(2e-5 3e-5 4e-5 5e-5 6e-5)
  LAMBDA_PSEUDO_LIST=(0.3 0.4 0.5)
fi

# Fixed settings (align with your best-known config)
EVIDENCE_DROPOUT=0.2
MODEL=MERIT
DATA=APAVA

LOG_DIR=results/grid_search_logs
mkdir -p "$LOG_DIR"

# Write config snapshot to log dir (for traceability)
{
  echo "mode: ${MODE}"
  echo -n "lambda_evi_list:"; for x in "${LAMBDA_EVI_LIST[@]}"; do echo -n " ${x}"; done; echo
  echo -n "lr_list:"; for x in "${LR_LIST[@]}"; do echo -n " ${x}"; done; echo
  echo -n "lambda_pseudo_list:"; for x in "${LAMBDA_PSEUDO_LIST[@]}"; do echo -n " ${x}"; done; echo
  echo "evidence_dropout: ${EVIDENCE_DROPOUT}"
  echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
} > "${LOG_DIR}/grid_config.txt"

for LBD in "${LAMBDA_EVI_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for LP in "${LAMBDA_PSEUDO_LIST[@]}"; do
      TAG="lbd${LBD}_lr${LR}_lp${LP}"
      echo "[RUN] ${TAG}"
      CMD=(python -m MERIT.run \
        --model ${MODEL} \
        --data ${DATA} \
        --root_path ${DATA_ROOT} \
        --use_ds \
        --use_evi_loss \
        --lambda_evi ${LBD} \
        --lambda_pseudo ${LP} \
        --evidence_dropout ${EVIDENCE_DROPOUT} \
        --learning_rate ${LR})

      # Save command meta for each run
      {
        echo "tag: ${TAG}"
        echo "cmd: ${CMD[@]}"
        echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
      } > "${LOG_DIR}/run_${TAG}.meta"

      "${CMD[@]}" | tee "${LOG_DIR}/run_${TAG}.log"
    done
  done
done

echo "Grid search completed. Logs saved to ${LOG_DIR}" 


