#!/bin/bash
# MERIT模型（纯证据融合，不含GNN） - 最终实验
# 理由：GNN与evidential fusion冲突，不加GNN性能更好

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "运行MERIT模型（纯证据融合）"
echo "策略：专注于evidential DS融合，不使用GNN"
echo "预计性能: Test Acc > 80%"
echo "========================================================"

mkdir -p results/final

# 最佳配置（无GNN，优化证据融合参数）
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 1e-5 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list 2,4,6,8 \
  --seeds "$SEEDS" \
  --log_csv results/final/merit_without_gnn_final.csv

echo ""
echo "========================================================"
echo "实验完成！"
echo "结果文件: results/final/merit_without_gnn_final.csv"
echo "========================================================"

