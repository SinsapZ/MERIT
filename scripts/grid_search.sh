#!/usr/bin/env bash

# Grid search for MERIT (ETMC-style DS fusion)
# Usage:
#   bash MERIT/MERIT/scripts/grid_search.sh /home/Data1/zbl /home/Data1/zbl/dataset/APAVA
# Args:
#   $1: workspace root (the directory that contains MERIT/)
#   $2: dataset root (the directory that contains Feature/ and Label/label.npy)

set -e

WS_ROOT=${1:-"$(pwd)"}
DATA_ROOT=${2:-"${WS_ROOT}/MERIT/APAVA"}

cd "$WS_ROOT"

# Search spaces
LAMBDA_EVI_LIST=(0.05 0.10 0.20 0.30 0.50)
LR_LIST=(1e-4 7e-5 5e-5 3e-5 2e-5)
LAMBDA_PSEUDO_LIST=(0.5 0.8 1.0)

# Fixed settings (align with your best-known config)
EVIDENCE_DROPOUT=0.2
MODEL=MERIT
DATA=APAVA

LOG_DIR=results/grid_search_logs
mkdir -p "$LOG_DIR"

for LBD in "${LAMBDA_EVI_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for LP in "${LAMBDA_PSEUDO_LIST[@]}"; do
      TAG="lbd${LBD}_lr${LR}_lp${LP}"
      echo "[RUN] ${TAG}"
      CMD=(python -m MERIT.MERIT.run \
        --model ${MODEL} \
        --data ${DATA} \
        --root_path ${DATA_ROOT} \
        --use_ds \
        --use_evi_loss \
        --lambda_evi ${LBD} \
        --lambda_pseudo ${LP} \
        --evidence_dropout ${EVIDENCE_DROPOUT} \
        --learning_rate ${LR})

      "${CMD[@]}" | tee "${LOG_DIR}/run_${TAG}.log"
    done
  done
done

echo "Grid search completed. Logs saved to ${LOG_DIR}" 


