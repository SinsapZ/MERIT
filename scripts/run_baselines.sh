#!/bin/bash
# 运行Baseline模型进行对比
# 使用MedGNN项目中已实现的模型

DATASET=$1  # APAVA, ADFD-Sample, PTB, PTB-XL
SEEDS="41,42,43"
GPU=0

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_baselines.sh <DATASET>"
    echo "Available datasets: APAVA, ADFD-Sample, PTB, PTB-XL"
    exit 1
fi

# 根据数据集设置路径和参数
case $DATASET in
    "APAVA")
        ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        BATCH_SIZE=128
        RESOLUTION_LIST="2"
        AUGMENTATIONS="drop0.2"
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "========================================================"
echo "运行Baseline模型 - $DATASET"
echo "Baselines: MedGNN, iTransformer, FEDformer, ECGFM, ECGFounder, FORMED"
echo "========================================================"

mkdir -p results/baselines/$DATASET
BASELINE_OUTPUT_DIR="$REPO_ROOT/results/baselines/$DATASET"

# 计算仓库根目录（本脚本位于 MERIT/MERIT/scripts 下）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# 切到 MedGNN 目录（仅为了 iTransformer）
if [ -d "$REPO_ROOT/MedGNN/MedGNN" ]; then
  pushd "$REPO_ROOT/MedGNN/MedGNN" >/dev/null
else
  echo "MedGNN/MedGNN 不存在，无法运行 iTransformer 基线。"; exit 1
fi

# ============================================================
# Baseline 1: MedGNN
# ============================================================
echo ""
echo "========================================================"
echo "Running MedGNN on $DATASET"
echo "========================================================"

for seed in ${SEEDS//,/ }; do
    echo "Seed $seed..."
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --model_id ${DATASET}-MedGNN \
        --model MedGNN \
        --data $DATASET \
        --e_layers $E_LAYERS \
        --batch_size $BATCH_SIZE \
        --d_model 256 \
        --d_ff 512 \
        --n_heads 8 \
        --augmentations $AUGMENTATIONS \
        --des "Baseline" \
        --itr 1 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --gpu $GPU \
        --seed $seed \
        2>&1 | grep -E "(Validation|Test) results"
done > "$BASELINE_OUTPUT_DIR/medgnn_results.txt"

# ============================================================
# Baseline 2: iTransformer
# ============================================================
echo ""
echo "========================================================"
echo "Running iTransformer on $DATASET"
echo "========================================================"

for seed in ${SEEDS//,/ }; do
    echo "Seed $seed..."
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --model_id ${DATASET}-iTransformer \
        --model iTransformer \
        --data $DATASET \
        --e_layers $E_LAYERS \
        --batch_size $BATCH_SIZE \
        --d_model 256 \
        --d_ff 512 \
        --n_heads 8 \
        --augmentations $AUGMENTATIONS \
        --des "Baseline" \
        --itr 1 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --gpu $GPU \
        --seed $seed \
        2>&1 | grep -E "(Validation|Test) results"
    done > "$BASELINE_OUTPUT_DIR/itransformer_results.txt"

# ============================================================
# Baseline 2: FEDformer (来自独立项目)
# ============================================================
echo ""
echo "========================================================"
echo "Running FEDformer on $DATASET"
echo "========================================================"

# 切换到 FEDformer 目录（项目根目录下的 FEDformer/FEDformer）
popd >/dev/null
if [ -d "$REPO_ROOT/FEDformer/FEDformer" ]; then
  pushd "$REPO_ROOT/FEDformer/FEDformer" >/dev/null
else
  echo "未找到 FEDformer/FEDformer，跳过FEDformer基线。"
fi

if [ -f run.py ]; then
    for seed in ${SEEDS//,/ }; do
        echo "Seed $seed..."
        python -u run.py \
            --is_training 1 \
            --task_id ${DATASET}-FEDformer \
            --model FEDformer \
            --data $DATASET \
            --root_path $ROOT_PATH \
            --seq_len 128 \
            --label_len 64 \
            --pred_len 64 \
            --d_model 256 \
            --d_ff 512 \
            --n_heads 8 \
            --e_layers $E_LAYERS \
            --batch_size $BATCH_SIZE \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --patience 3 \
            --gpu $GPU \
            --itr 1 \
            --des Baseline-FED \
            2>&1
    done > "$BASELINE_OUTPUT_DIR/fedformer_results.txt"
else
    echo "未找到 FEDformer/run.py，跳过FEDformer基线。"
fi

popd >/dev/null

# ============================================================
# Baseline 5: ECGFM (可选)
# ============================================================
DEFAULT_ECGFM_CKPT="$WORKSPACE_ROOT/ECGFM/ecg-fm-benchmarking/checkpoint/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
ALT_ECGFM_CKPT="$WORKSPACE_ROOT/ECGFM/checkpoint/last_11597276.ckpt"
SELECTED_ECGFM_CKPT=${ECGFM_CHECKPOINT:-$DEFAULT_ECGFM_CKPT}
if [ ! -f "$SELECTED_ECGFM_CKPT" ] && [ -f "$ALT_ECGFM_CKPT" ]; then
  SELECTED_ECGFM_CKPT="$ALT_ECGFM_CKPT"
fi

if [ -f "$SELECTED_ECGFM_CKPT" ]; then
  echo ""
  echo "========================================================"
  echo "Running ECGFM on $DATASET"
  echo "========================================================"
  mkdir -p "$BASELINE_OUTPUT_DIR/ecgfm"
  ECGFM_CHECKPOINT="$SELECTED_ECGFM_CKPT" bash "$SCRIPT_DIR/baseline_ecgfm.sh" "$DATASET" "$BASELINE_OUTPUT_DIR/ecgfm" \
    | tee "$BASELINE_OUTPUT_DIR/ecgfm/ecgfm_${DATASET}.log"
else
  echo "未找到 ECGFM checkpoint ($SELECTED_ECGFM_CKPT)，跳过 ECGFM baseline。"
fi

# ============================================================
# Baseline 6: ECGFounder fine-tune (可选)
# ============================================================
if [ -n "${ECGFOUNDER_CKPT:-}" ]; then
  echo ""
  echo "========================================================"
  echo "Running ECGFounder on $DATASET"
  echo "========================================================"
  mkdir -p "$BASELINE_OUTPUT_DIR/ecgfounder"
  bash "$SCRIPT_DIR/baseline_ecgfounder.sh" "$DATASET" "$ROOT_PATH" \
    "$BASELINE_OUTPUT_DIR/ecgfounder" \
    | tee "$BASELINE_OUTPUT_DIR/ecgfounder/ecgfounder_${DATASET}.log"
else
  echo "ECGFOUNDER_CKPT 未设置，跳过 ECGFounder baseline。"
fi

# ============================================================
# Baseline 7: FORMED adapting (PTB/PTB-XL/APAVA)
# ============================================================
if [[ "$DATASET" =~ ^PTB ]] || [ "$DATASET" == "APAVA" ]; then
  DEFAULT_FORMED_BASE="$WORKSPACE_ROOT/FORMED/FORMED/checkpoint/timesfm-2.0-500m-base.pth"
  SELECTED_FORMED_BASE=${FORMED_BASE_MODEL:-$DEFAULT_FORMED_BASE}
  if [ -f "$SELECTED_FORMED_BASE" ]; then
    echo ""
    echo "========================================================"
    echo "Running FORMED on $DATASET"
    echo "========================================================"
    mkdir -p "$BASELINE_OUTPUT_DIR/formed"
    FORMED_BASE_MODEL="$SELECTED_FORMED_BASE" bash "$SCRIPT_DIR/baseline_formed.sh" "$DATASET" "$BASELINE_OUTPUT_DIR/formed" \
      | tee "$BASELINE_OUTPUT_DIR/formed/formed_${DATASET}.log"
  else
    echo "未找到 TimesFM checkpoint ($SELECTED_FORMED_BASE)，跳过 FORMED baseline。"
  fi
else
  echo "FORMED baseline 暂仅支持 PTB、PTB-XL、APAVA，当前数据集 $DATASET 跳过。"
fi

# 留在脚本所在目录（可选）
cd "$SCRIPT_DIR"

echo ""
echo "========================================================"
echo "Baseline实验完成！"
echo "结果保存在: results/baselines/$DATASET/"
echo "========================================================"

