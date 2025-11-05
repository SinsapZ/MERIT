#!/bin/bash
# Helper script to fine-tune ECGFounder (Net1D) on MERIT datasets
# using MERIT/MERIT/scripts/finetune_ecgfounder.py.
#
# 默认目录假设：MERIT、ECGFounder、dataset 位于同级目录。
# 如果位置不同，可自行覆盖环境变量：
#   ECGFOUNDER_ROOT / ECGFOUNDER_CKPT / ECGFOUNDER_CONDA_ENV 等。
#
# 使用示例（默认路径下可直接运行）：
#   bash MERIT/MERIT/scripts/baseline_ecgfounder.sh PTB-XL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: bash baseline_ecgfounder.sh <DATASET> [OUTPUT_DIR]"
  exit 1
fi

ECGFOUNDER_ROOT=${ECGFOUNDER_ROOT:-"$WORKSPACE_ROOT/ECGFounder/ECGFounder"}
DEFAULT_CANDIDATES=(
  "$WORKSPACE_ROOT/dataset/$DATASET"
  "$WORKSPACE_ROOT/dataset/${DATASET}/${DATASET}"
)

if [ -n "${ECGFOUNDER_DATA_ROOT:-}" ]; then
  DEFAULT_CANDIDATES=("$ECGFOUNDER_DATA_ROOT" "${DEFAULT_CANDIDATES[@]}")
fi

ROOT_PATH="${2:-}"
if [ -n "$ROOT_PATH" ]; then
  DEFAULT_CANDIDATES=("$ROOT_PATH" "${DEFAULT_CANDIDATES[@]}")
fi

RESOLVED_PATH=""
for cand in "${DEFAULT_CANDIDATES[@]}"; do
  if [ -d "$cand/Feature" ] && [ -d "$cand/Label" ]; then
    RESOLVED_PATH="$cand"
    break
  fi
done

if [ -z "$RESOLVED_PATH" ]; then
  echo "Error: 未在以下路径找到数据集目录: ${DEFAULT_CANDIDATES[*]}"
  exit 1
fi

ROOT_PATH="$RESOLVED_PATH"
OUT_DIR=${3:-"$WORKSPACE_ROOT/MERIT/results/baselines/$DATASET/ecgfounder"}

if [ ! -d "$ROOT_PATH" ]; then
  echo "Error: 数据集路径 $ROOT_PATH 不存在，请检查。"
  exit 1
fi

GPU=${ECGFOUNDER_GPU:-0}
EPOCHS=${ECGFOUNDER_EPOCHS:-10}
LR=${ECGFOUNDER_LR:-1e-4}
BATCH=${ECGFOUNDER_BATCH:-64}

mkdir -p "$OUT_DIR"

CMD=(python -m MERIT.scripts.finetune_ecgfounder \
        --data "$DATASET" \
        --root_path "$ROOT_PATH" \
        --checkpoint "${ECGFOUNDER_CKPT:-}" \
        --output_dir "$OUT_DIR" \
        --ecgfounder_root "$ECGFOUNDER_ROOT" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH")

if [ "${ECGFOUNDER_LINEAR_PROBE:-0}" == "1" ]; then
  CMD+=(--linear_probe)
fi

if [ -z "${ECGFOUNDER_CKPT:-}" ]; then
  CMD+=(--init_random)
fi

if [ -n "${ECGFOUNDER_CONDA_ENV:-}" ]; then
  echo "Activating conda env: $ECGFOUNDER_CONDA_ENV"
  eval "$(conda shell.bash hook)"
  conda activate "$ECGFOUNDER_CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "ECGFounder baseline完成，结果已保存到 $OUT_DIR"

