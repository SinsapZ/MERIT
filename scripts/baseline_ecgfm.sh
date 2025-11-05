#!/bin/bash
# Wrapper to fine-tune ECG-FM (KED) on MERIT datasets using
# MERIT/MERIT/scripts/finetune_ecgfm.py.
#
# 默认假设工程布局如下（均在同一父目录）：
#   MERIT/  ECGFM/  ECGFounder/  FORMED/  dataset/
# 若结构不同，可通过下列环境变量覆盖：
#   ECGFM_ROOT         -> ecg-fm 仓库根目录（默认 ../ECGFM）
#   ECGFM_CHECKPOINT   -> 预训练权重路径（默认: ECGFM/ecg-fm-benchmarking/checkpoint/ecg_fm/...）
#   ECGFM_CONDA_ENV    -> 运行所需的 conda 环境
#   ECGFM_GPU          -> GPU id（默认 0）
#   ECGFM_EPOCHS       -> 训练 epoch 数（默认 10）
#   ECGFM_LR           -> 学习率（默认 5e-4）
#   ECGFM_BATCH        -> batch size（默认 64）
#   ECGFM_NUM_WORKERS  -> dataloader workers（默认 4）
#   ECGFM_EVAL_MODE    -> finetuning 模式（默认 finetuning_linear）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MERIT_ROOT="$WORKSPACE_ROOT/MERIT"

DATASET=${1:-}
OUT_DIR_ARG=${2:-}

if [ -z "$DATASET" ]; then
  echo "Usage: bash baseline_ecgfm.sh <DATASET> [OUTPUT_DIR]"
  exit 1
fi

# 推断数据集根目录（兼容 dataset/DS 以及 dataset/DS/DS 两种结构）
ROOT_CANDIDATES=(
  "$WORKSPACE_ROOT/dataset/$DATASET"
  "$WORKSPACE_ROOT/dataset/${DATASET}/${DATASET}"
)

if [ -n "${ECGFM_DATA_ROOT:-}" ]; then
  ROOT_CANDIDATES=("$ECGFM_DATA_ROOT" "${ROOT_CANDIDATES[@]}")
fi
ROOT_PATH=""
for cand in "${ROOT_CANDIDATES[@]}"; do
  if [ -d "$cand/Feature" ] && [ -d "$cand/Label" ]; then
    ROOT_PATH="$cand"
    break
  fi
done

if [ -z "$ROOT_PATH" ]; then
  echo "未找到数据集目录，请通过 ECGFM_DATA_ROOT 或脚本参数手动指定。"
  exit 1
fi

ECGFM_CHECKPOINT=${ECGFM_CHECKPOINT:-"$WORKSPACE_ROOT/ECGFM/ecg-fm-benchmarking/checkpoint/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"}
if [ ! -f "$ECGFM_CHECKPOINT" ]; then
  echo "未找到 ECG-FM 预训练权重: $ECGFM_CHECKPOINT"
  echo "请放置至上述路径或通过环境变量 ECGFM_CHECKPOINT 指定。"
  exit 1
fi

GPU=${ECGFM_GPU:-0}
EPOCHS=${ECGFM_EPOCHS:-10}
LR=${ECGFM_LR:-5e-4}
BATCH=${ECGFM_BATCH:-64}
NUM_WORKERS=${ECGFM_NUM_WORKERS:-4}
EVAL_MODE=${ECGFM_EVAL_MODE:-finetuning_linear}

OUT_DIR=${OUT_DIR_ARG:-"$MERIT_ROOT/results/baselines/$DATASET/ecgfm"}
mkdir -p "$OUT_DIR"

CMD=(python -m MERIT.scripts.finetune_ecgfm \
        --data "$DATASET" \
        --root_path "$ROOT_PATH" \
        --checkpoint "$ECGFM_CHECKPOINT" \
        --output_dir "$OUT_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH" \
        --num_workers "$NUM_WORKERS" \
        --eval_mode "$EVAL_MODE")

if [ -n "${ECGFM_CONDA_ENV:-}" ]; then
  echo "Activating conda env: $ECGFM_CONDA_ENV"
  eval "$(conda shell.bash hook)"
  conda activate "$ECGFM_CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "ECGFM baseline 完成，结果保存在 $OUT_DIR"

