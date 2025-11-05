#!/bin/bash
# Wrapper for running FORMED adapting experiments with a single GPU.
#
# Required environment variables:
#   FORMED_DIR        -> path to FORMED repo (folder containing run-adapting.py)
#   FORMED_BASE_MODEL -> TimesFM checkpoint (e.g. timesfm-2.0-500m-base.pth)
#   FORMED_CHECKPOINT -> (可选) 预训练/复用的 FORMED checkpoint 模板路径
#
# Example:
#   FORMED_DIR=../../FORMED/FORMED \
#   FORMED_BASE_MODEL=./checkpoint/timesfm-2.0-500m-base.pth \
#   bash MERIT/MERIT/scripts/baseline_formed.sh PTB-XL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATASET=${1:-}
OUT_DIR_ARG=${2:-}

if [ -z "$DATASET" ]; then
  echo "Usage: bash baseline_formed.sh <DATASET> [OUTPUT_DIR]"
  echo "当前已在 configs/adapting.toml 中添加 PTB、PTB-XL、APAVA 示例。"
  exit 1
fi

FORMED_DIR=${FORMED_DIR:-"$WORKSPACE_ROOT/FORMED/FORMED"}
FORMED_BASE_MODEL=${FORMED_BASE_MODEL:-"$FORMED_DIR/checkpoint/timesfm-2.0-500m-base.pth"}

if [ ! -d "$FORMED_DIR" ]; then
  echo "FORMED_DIR=$FORMED_DIR 不存在，请确认仓库位置。"
  exit 1
fi

RESOLVED_BASE_MODEL="$FORMED_BASE_MODEL"

if [ ! -f "$RESOLVED_BASE_MODEL" ]; then
  if [[ "$RESOLVED_BASE_MODEL" == *.safetensors ]]; then
    CONVERTED_PTH="${RESOLVED_BASE_MODEL%.safetensors}.pth"
    if [ ! -f "$CONVERTED_PTH" ]; then
      echo "检测到 TimesFM safetensors，正在转换为 .pth ..."
      python -m MERIT.scripts.convert_timesfm_checkpoint --source "$RESOLVED_BASE_MODEL" --output "$CONVERTED_PTH"
    fi
    RESOLVED_BASE_MODEL="$CONVERTED_PTH"
  elif [ -d "$RESOLVED_BASE_MODEL" ]; then
    if [ -f "$RESOLVED_BASE_MODEL/model.safetensors" ]; then
      CONVERTED_PTH="$RESOLVED_BASE_MODEL/model_timesfm.pth"
      if [ ! -f "$CONVERTED_PTH" ]; then
        echo "检测到 TimesFM safetensors 目录，正在转换为 .pth ..."
        python -m MERIT.scripts.convert_timesfm_checkpoint --source "$RESOLVED_BASE_MODEL" --output "$CONVERTED_PTH"
      fi
      RESOLVED_BASE_MODEL="$CONVERTED_PTH"
    fi
  fi
fi

if [ ! -f "$RESOLVED_BASE_MODEL" ]; then
  echo "未找到 TimesFM checkpoint: $RESOLVED_BASE_MODEL"
  echo "请提前准备 timesfm-2.0-500m-base 模型或通过 FORMED_BASE_MODEL 指定。"
  exit 1
fi

GPU=${FORMED_GPU:-0}
SEED=${FORMED_SEED:-41}
LEARNING_RATE=${FORMED_LR:-5e-4}
EPOCHS=${FORMED_EPOCHS:-50}
K_FOLD=${FORMED_K:-1}
LIMIT=${FORMED_LIMIT:-1.0}
CHECKPOINT=${FORMED_CHECKPOINT:-"$FORMED_DIR/checkpoint/FORMED-{DIM}-{SEED}.pth"}
LOG_DIR=${OUT_DIR_ARG:-${FORMED_OUT_DIR:-"$WORKSPACE_ROOT/MERIT/results/baselines/$DATASET/formed"}}
mkdir -p "$LOG_DIR"

CMD=(python "$FORMED_DIR/run-adapting.py"
      --dataset "$DATASET"
      --base_model_path "$RESOLVED_BASE_MODEL"
      --checkpoint_path "$CHECKPOINT"
      --seed "$SEED"
      --num_itr 1
      --train_epochs "$EPOCHS"
      --k_fold "$K_FOLD"
      --limit_size "$LIMIT"
      --learning_rate "$LEARNING_RATE"
      --checkpoint_seed "$SEED"
      --use_gpu --gpu "$GPU")

if [ -n "${FORMED_UV_ENV:-}" ]; then
  # Use uv to run inside project virtualenv
  CMD=(uv run --project "$FORMED_DIR" "${CMD[@]}")
fi

if [ -n "${FORMED_CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$FORMED_CONDA_ENV"
fi

CUDA_VISIBLE_DEVICES=$GPU "${CMD[@]}" | tee "$LOG_DIR/formed_${DATASET}.log"

echo "FORMED baseline 完成，日志位于 $LOG_DIR/formed_${DATASET}.log"

