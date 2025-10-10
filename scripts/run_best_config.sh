#!/bin/bash
# MERIT最佳配置 - 基于实验验证
# 这是目前实验过的最好配置：78% Test Acc

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "MERIT最佳配置实验"
echo "配置：原始参数 (lr=1.1e-4, 200 epochs, 无GNN)"
echo "预期性能: Test Acc ~78% (已验证)"
echo "========================================================"

mkdir -p results/final

# 最佳验证配置
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
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --resolution_list 2,4,6,8 \
  --seeds "$SEEDS" \
  --log_csv results/final/merit_best_config.csv

echo ""
echo "========================================================"
echo "实验完成！"
echo "结果文件: results/final/merit_best_config.csv"
echo "========================================================"
