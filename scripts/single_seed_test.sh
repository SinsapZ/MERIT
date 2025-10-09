#!/bin/bash
# 超快速单种子测试 - 5-10分钟验证GNN是否工作
# 用于快速debug和验证

ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0
SEED=41

echo "========================================================"
echo "超快速测试：单种子验证GNN"
echo "种子: $SEED"
echo "预计耗时: 5-10分钟"
echo "========================================================"

mkdir -p results/single_test

# 测试1: 无GNN
echo -e "\n测试无GNN版本..."
python -m MERIT.run \
  --model MERIT --data APAVA \
  --root_path $ROOT_PATH \
  --use_ds \
  --learning_rate 1.1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --resolution_list 2,4,6,8 \
  --gpu $GPU \
  --seed $SEED \
  2>&1 | tee results/single_test/no_gnn_seed${SEED}.log

# 测试2: 有GNN
echo -e "\n测试有GNN版本..."
python -m MERIT.run \
  --model MERIT --data APAVA \
  --root_path $ROOT_PATH \
  --use_ds \
  --learning_rate 1.1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --use_gnn \
  --resolution_list 2,4,6,8 \
  --gpu $GPU \
  --seed $SEED \
  2>&1 | tee results/single_test/with_gnn_seed${SEED}.log

echo ""
echo "========================================================"
echo "测试完成！检查日志文件查看结果:"
echo "  无GNN: results/single_test/no_gnn_seed${SEED}.log"
echo "  有GNN: results/single_test/with_gnn_seed${SEED}.log"
echo ""
echo "快速查看测试结果："
echo "  grep 'Test results' results/single_test/*.log"
echo "========================================================"

