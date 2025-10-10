#!/bin/bash
# MERIT消融实验 - 验证证据融合的有效性
# 不和MedGNN对比，专注于证明各组件的贡献

SEEDS="41,42,43,44,45"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0
BASE_EPOCHS=150
BASE_PATIENCE=20

echo "========================================================"
echo "MERIT消融实验"
echo "目标：验证证据融合机制的有效性"
echo "========================================================"

mkdir -p results/ablation

# ============================================================
# Baseline 1: 简单平均融合 (无证据理论)
# ============================================================
echo ""
echo "============================================================"
echo "实验1: Baseline - 简单平均融合"
echo "说明: Multi-Res + Transformer + Mean Pooling"
echo "============================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 1e-5 \
  --batch_size 64 \
  --train_epochs $BASE_EPOCHS \
  --patience $BASE_PATIENCE \
  --resolution_list 2,4,6,8 \
  --no_pseudo \
  --agg mean \
  --seeds "$SEEDS" \
  --log_csv results/ablation/baseline_mean.csv

# ============================================================
# Baseline 2: 学习权重融合 (无证据理论)
# ============================================================
echo ""
echo "============================================================"
echo "实验2: 学习权重融合 (无DS理论)"
echo "说明: 用简单的attention权重代替DS融合"
echo "============================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 1e-5 \
  --batch_size 64 \
  --train_epochs $BASE_EPOCHS \
  --patience $BASE_PATIENCE \
  --resolution_list 2,4,6,8 \
  --no_pseudo \
  --agg evi \
  --seeds "$SEEDS" \
  --log_csv results/ablation/learned_weights.csv

# ============================================================
# 实验3: Evidential融合 (无DS, 用简单CE loss)
# ============================================================
echo ""
echo "============================================================"
echo "实验3: Evidence-based但用交叉熵"
echo "说明: 有evidence heads但不用DS融合和EDL loss"
echo "============================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 0.0 \
  --lambda_pseudo_loss 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 1e-5 \
  --batch_size 64 \
  --train_epochs $BASE_EPOCHS \
  --patience $BASE_PATIENCE \
  --resolution_list 2,4,6,8 \
  --no_pseudo \
  --seeds "$SEEDS" \
  --log_csv results/ablation/evidence_no_ds.csv

# ============================================================
# 实验4: DS融合 (无Pseudo-view)
# ============================================================
echo ""
echo "============================================================"
echo "实验4: MERIT (DS融合, 无Pseudo-view)"
echo "说明: 完整DS融合但不用伪视图"
echo "============================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1.1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.0 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 0 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --resolution_list 2,4,6,8 \
  --no_pseudo \
  --seeds "$SEEDS" \
  --log_csv results/ablation/merit_no_pseudo.csv

# ============================================================
# 实验5: 完整MERIT (DS + Pseudo-view)
# ============================================================
echo ""
echo "============================================================"
echo "实验5: 完整MERIT模型"
echo "说明: DS融合 + Pseudo-view + EDL loss"
echo "============================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1.1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 0 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --resolution_list 2,4,6,8 \
  --seeds "$SEEDS" \
  --log_csv results/ablation/merit_full.csv

echo ""
echo "========================================================"
echo "所有消融实验完成！"
echo "结果保存在 results/ablation/ 目录"
echo "========================================================"

