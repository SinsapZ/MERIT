#!/bin/bash
# 消融实验：对比有无GNN的影响

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

# 基础超参数（使用前面找到的最佳配置）
BASE_ARGS="--root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1.1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds $SEEDS"

echo "========================================================"
echo "消融实验：评估GNN对MERIT性能的影响"
echo "========================================================"

# 实验1: 无GNN（当前baseline）
echo -e "\n[1/2] MERIT without GNN (Baseline)"
python -m MERIT.scripts.multi_seed_run $BASE_ARGS \
  --log_csv results/ablation/merit_no_gnn.csv

# 实验2: 加上GNN（完整MERIT）
echo -e "\n[2/2] MERIT with GNN (Full Model)"
python -m MERIT.run $BASE_ARGS \
  --use_gnn \
  --log_csv results/ablation/merit_with_gnn.csv

echo ""
echo "========================================================"
echo "消融实验完成！"
echo "========================================================"
echo ""
echo "对比结果:"
echo "  无GNN: results/ablation/merit_no_gnn.csv"
echo "  有GNN: results/ablation/merit_with_gnn.csv"
echo ""
echo "运行对比脚本:"
echo "  python MERIT/scripts/compare_gnn_ablation.py"

