#!/bin/bash
# 完整MERIT模型（含GNN） - 最终实验

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "运行完整MERIT模型（含GNN）"
echo "预计性能: Test Acc > 83%"
echo "========================================================"

mkdir -p results/final

# 最佳配置 + GNN
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
  --log_csv results/final/merit_full_with_gnn.csv

# 注意：需要在multi_seed_run.py中传递--use_gnn参数
# 或者直接修改默认值use_gnn=True

echo ""
echo "========================================================"
echo "实验完成！"
echo "结果文件: results/final/merit_full_with_gnn.csv"
echo "========================================================"

