#!/bin/bash
# 快速测试不同GNN配置 - 找最佳设置
# 预计耗时: ~1小时

SEEDS="41,42,43"  # 只用3个种子，更快
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "快速测试：不同GNN配置"
echo "种子数: 3"
echo "配置数: 4"
echo "预计总耗时: ~1小时"
echo "========================================================"

mkdir -p results/gnn_variations

BASE_ARGS="--root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --use_gnn --resolution_list 2,4,6,8 --seeds $SEEDS"

# 配置1: 原始lr + nodedim=10
echo -e "\n[1/4] GNN + lr=1.1e-4 + nodedim=10"
python -m MERIT.scripts.multi_seed_run $BASE_ARGS \
  --lr 1.1e-4 --nodedim 10 \
  --log_csv results/gnn_variations/lr1.1e4_node10.csv

# 配置2: 原始lr + nodedim=12
echo -e "\n[2/4] GNN + lr=1.1e-4 + nodedim=12"
python -m MERIT.scripts.multi_seed_run $BASE_ARGS \
  --lr 1.1e-4 --nodedim 12 \
  --log_csv results/gnn_variations/lr1.1e4_node12.csv

# 配置3: 降低lr + nodedim=10
echo -e "\n[3/4] GNN + lr=1.0e-4 + nodedim=10"
python -m MERIT.scripts.multi_seed_run $BASE_ARGS \
  --lr 1.0e-4 --nodedim 10 \
  --log_csv results/gnn_variations/lr1.0e4_node10.csv

# 配置4: 降低lr + nodedim=12 + 延长annealing
echo -e "\n[4/4] GNN + lr=1.0e-4 + nodedim=12 + anneal=60"
python -m MERIT.scripts.multi_seed_run $BASE_ARGS \
  --lr 1.0e-4 --nodedim 12 --annealing_epoch 60 \
  --log_csv results/gnn_variations/lr1.0e4_node12_anneal60.csv

echo ""
echo "========================================================"
echo "测试完成！运行对比脚本查看结果:"
echo "  python MERIT/scripts/compare_variations.py"
echo "========================================================"

