#!/bin/bash
# 一键运行不确定性实验（训练 + 保存不确定性数组 + 生成图表）
# 仅依赖现有代码与脚本参数，不需手动干预

set -e

GPU=${1:-0}

# ===================== 配置每个数据集的最佳参数 =====================
declare -A ROOTS
declare -A LRS
declare -A LAMBDA_PSEUDO_LOSS
declare -A RES_LIST

ROOTS[APAVA]="/home/Data1/zbl/dataset/APAVA"
LRS[APAVA]="1e-4"
LAMBDA_PSEUDO_LOSS[APAVA]="0.3"
RES_LIST[APAVA]="2,4,6,8"

ROOTS[PTB]="/home/Data1/zbl/dataset/PTB"
LRS[PTB]="1.5e-4"
LAMBDA_PSEUDO_LOSS[PTB]="0.2"
RES_LIST[PTB]="2,4,6,8"

ROOTS["PTB-XL"]="/home/Data1/zbl/dataset/PTB-XL"
LRS["PTB-XL"]="1.5e-4"
LAMBDA_PSEUDO_LOSS["PTB-XL"]="0.3"
RES_LIST["PTB-XL"]="2,4,6,8"

# 通用训练超参
E_LAYERS=4
DROPOUT=0.1
BATCH_SIZE=64
EPOCHS=150
PATIENCE=20
WD=1e-4
ANNEAL=50
D_MODEL=256
D_FF=512
N_HEADS=8
NODEDIM=10
SEED=41

# 输出目录
OUT_BASE="results/uncertainty"
mkdir -p "$OUT_BASE"

run_one_dataset() {
  local DS=$1
  local ROOT=${ROOTS[$DS]}
  local LR=${LRS[$DS]}
  local LPL=${LAMBDA_PSEUDO_LOSS[$DS]}
  local RES=${RES_LIST[$DS]}

  echo "\n=============================="
  echo "运行数据集: $DS"
  echo "=============================="

  # --- 1) 训练并保存不确定性数组：EviMR ---
  EVI_DIR="$OUT_BASE/$DS/evi"
  mkdir -p "$EVI_DIR"
  python -m MERIT.run \
    --model MERIT --data "$DS" --root_path "$ROOT" \
    --model_id "UNCERT-${DS}-EVI" \
    --use_ds \
    --learning_rate "$LR" \
    --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss "$LPL" \
    --annealing_epoch "$ANNEAL" \
    --resolution_list "$RES" \
    --batch_size "$BATCH_SIZE" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$EVI_DIR" \
    2>&1 | grep -E "(Validation results|Test results|Saved uncertainty|SUMMARY|SUMMARY STATISTICS)" || true

  # --- 2) 训练并保存不确定性数组：Softmax基线（不启用DS） ---
  SOFT_DIR="$OUT_BASE/$DS/softmax"
  mkdir -p "$SOFT_DIR"
  python -m MERIT.run \
    --model MERIT --data "$DS" --root_path "$ROOT" \
    --model_id "UNCERT-${DS}-SOFT" \
    --learning_rate "$LR" \
    --resolution_list "$RES" \
    --batch_size "$BATCH_SIZE" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$SOFT_DIR" \
    2>&1 | grep -E "(Validation results|Test results|Saved uncertainty|SUMMARY|SUMMARY STATISTICS)" || true

  # --- 3) 指标与图：单方法（可靠度图/选择性预测） ---
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$EVI_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_evi" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$SOFT_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_soft" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true

  # --- 4) 叠加比较：准确率-拒绝率曲线 & 不确定性分布 ---
  python -m MERIT.scripts.compare_selective --base_dir "$OUT_BASE/$DS" --dataset "$DS" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' || true
  python -m MERIT.scripts.make_uncert_density --base_dir "$OUT_BASE/$DS" --dataset "$DS" || true

  # --- 5) 噪声鲁棒性（EviMR与Softmax + 对比） ---
  python -m MERIT.scripts.make_noise_compare --dataset "$DS" --root_path "$ROOT" --resolution_list "$RES" --out_dir "$OUT_BASE/$DS" --gpu "$GPU" || true

  # --- 6) 案例图（高/低不确定度各Top-6） ---
  python -m MERIT.scripts.plot_cases \
    --dataset "$DS" \
    --root_path "$ROOT" \
    --resolution_list "$RES" \
    --uncertainty_base "$OUT_BASE/$DS" \
    --top_k_high 6 \
    --top_k_low 6 \
    --gpu "$GPU" || true
}

# ===================== 主程序 =====================
for DS in APAVA PTB PTB-XL; do
  run_one_dataset "$DS"
done

echo "\n✅ 全部完成。不确定性数据与图已保存到: $OUT_BASE/<DATASET>/...\n"


