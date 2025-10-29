#!/bin/bash
# 仅评估与出图（不训练）：
# - 从已存在的 checkpoint 加载模型
# - 保存不确定性数组（EviMR/Softmax）
# - 生成所有图（可靠度、选择性、对比、KDE/Violin、噪声鲁棒性、案例图）

set -e

GPU=${1:-0}

declare -A ROOTS
declare -A LRS
declare -A RES_LIST

ROOTS[APAVA]="/home/Data1/zbl/dataset/APAVA";   LRS[APAVA]="1e-4";  RES_LIST[APAVA]="2,4,6,8"
ROOTS[PTB]="/home/Data1/zbl/dataset/PTB";       LRS[PTB]="1.5e-4"; RES_LIST[PTB]="2,4,6,8"
ROOTS["PTB-XL"]="/home/Data1/zbl/dataset/PTB-XL"; LRS["PTB-XL"]="1.5e-4"; RES_LIST["PTB-XL"]="2,4,6,8"

E_LAYERS=4; D_MODEL=256; D_FF=512; N_HEADS=8; DROPOUT=0.1; WD=1e-4; BATCH=64; EPOCHS=150; PATIENCE=20; ANNEAL=50; NODEDIM=10; SEED=41

OUT_BASE="results/uncertainty"; mkdir -p "$OUT_BASE"

run_eval_dataset() {
  local DS=$1
  local ROOT=${ROOTS[$DS]}
  local LR=${LRS[$DS]}
  local RES=${RES_LIST[$DS]}
  echo "\n================ EVAL ONLY: $DS ================"

  local EVI_DIR="$OUT_BASE/$DS/evi";     mkdir -p "$EVI_DIR"
  local SOFT_DIR="$OUT_BASE/$DS/softmax"; mkdir -p "$SOFT_DIR"

  # 1) 仅评估：从 checkpoint 加载并保存不确定性（EviMR）
  python -m MERIT.run \
    --task_name classification \
    --is_training 0 \
    --model_id "UNCERT-${DS}-EVI" \
    --model MERIT \
    --data "$DS" \
    --root_path "$ROOT" \
    --use_ds \
    --learning_rate "$LR" \
    --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.3 \
    --annealing_epoch "$ANNEAL" \
    --resolution_list "$RES" \
    --batch_size "$BATCH" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$EVI_DIR"

  # 2) 仅评估：Softmax 基线
  python -m MERIT.run \
    --task_name classification \
    --is_training 0 \
    --model_id "UNCERT-${DS}-SOFT" \
    --model MERIT \
    --data "$DS" \
    --root_path "$ROOT" \
    --learning_rate "$LR" \
    --resolution_list "$RES" \
    --batch_size "$BATCH" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$SOFT_DIR"

  # 3) 单方法图（可靠度/选择性 + SVG）
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$EVI_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_evi" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$SOFT_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_soft" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true

  # 4) 对比曲线（EviMR vs Softmax）
  python -m MERIT.scripts.compare_selective --base_dir "$OUT_BASE/$DS" --dataset "$DS" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' || true

  # 5) KDE/Violin（近零聚焦 + 中位线 + 分离度报告）
  python -m MERIT.scripts.make_uncert_density --base_dir "$OUT_BASE/$DS" --dataset "$DS" || true

  # 6) 噪声鲁棒性（同轴对比 + 单方法曲线）
  python -m MERIT.scripts.make_noise_compare --dataset "$DS" --root_path "$ROOT" --resolution_list "$RES" --out_dir "$OUT_BASE/$DS" --gpu "$GPU" || true

  # 7) 临床案例图（最不自信Top-6）
  python -m MERIT.scripts.plot_cases \
    --dataset "$DS" \
    --root_path "$ROOT" \
    --resolution_list "$RES" \
    --uncertainty_base "$OUT_BASE/$DS" \
    --top_k_high 6 \
    --top_k_low 6 \
    --gpu "$GPU" || true
}

# 主程序：如需单独某一数据集，传入环境变量 ONLY=<DS>
if [ -n "$ONLY" ]; then
  run_eval_dataset "$ONLY"
else
  for DS in APAVA PTB PTB-XL; do
    run_eval_dataset "$DS"
  done
fi

echo "\n✅ EVAL ONLY 完成。图与报告见 $OUT_BASE/<DATASET>/ 下"


