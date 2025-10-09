#!/bin/bash
# 快速测试GNN效果 - 只跑5个种子
# 预计耗时: 约1小时

SEEDS="41,42,43,44,45"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "快速测试：对比有无GNN的性能差异"
echo "种子数: 5"
echo "预计总耗时: ~1小时"
echo "开始时间: $(date)"
echo "========================================================"

mkdir -p results/quick_test

# 测试1: 无GNN (之前最佳配置)
echo -e "\n[1/2] 测试 MERIT without GNN..."
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
  --log_csv results/quick_test/no_gnn.csv

# 测试2: 加上GNN
echo -e "\n[2/2] 测试 MERIT with GNN..."
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
  --use_gnn \
  --resolution_list 2,4,6,8 \
  --seeds "$SEEDS" \
  --log_csv results/quick_test/with_gnn.csv

echo ""
echo "========================================================"
echo "快速测试完成！"
echo "结束时间: $(date)"
echo "========================================================"
echo ""
echo "查看结果:"
echo "  python MERIT/scripts/quick_compare.py"

