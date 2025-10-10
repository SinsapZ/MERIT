#!/bin/bash
# 快速调优脚本 - 用3个seeds测试不同超参数

DATASET=$1  # ADFD-Sample, PTB, PTB-XL
GPU=0
SEEDS="41,42,43"  # 只用3个seeds快速测试

if [ -z "$DATASET" ]; then
    echo "Usage: bash quick_tune.sh <DATASET>"
    echo "Available: ADFD-Sample, PTB, PTB-XL"
    exit 1
fi

# 根据数据集设置基础参数
case $DATASET in
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB/PTB"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL/PTB-XL"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        ;;
    *)
        echo "Unknown dataset"
        exit 1
        ;;
esac

mkdir -p results/tuning/$DATASET

echo "========================================================"
echo "快速调优 - $DATASET (3 seeds)"
echo "========================================================"

# ============================================================
# Config 1: 默认配置
# ============================================================
echo ""
echo "Config 1: Default (lr=1e-4, dropout=0.1)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout 0.1 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config1_default.csv

# ============================================================
# Config 2: 更高学习率
# ============================================================
echo ""
echo "Config 2: Higher LR (lr=1.5e-4)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 1.5e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout 0.1 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config2_higher_lr.csv

# ============================================================
# Config 3: 更强正则化
# ============================================================
echo ""
echo "Config 3: Stronger Regularization (dropout=0.15, wd=1e-5)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout 0.15 \
  --weight_decay 1e-5 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config3_strong_reg.csv

echo ""
echo "========================================================"
echo "调优完成！分析结果..."
echo "========================================================"

# 简单对比
echo ""
echo "Results:"
for config in config1_default config2_higher_lr config3_strong_reg; do
    if [ -f "results/tuning/$DATASET/${config}_summary.txt" ]; then
        echo ""
        echo "=== $config ==="
        grep "test_acc:" results/tuning/$DATASET/${config}_summary.txt
    fi
done

echo ""
echo "详细结果查看:"
echo "  results/tuning/$DATASET/"
echo ""

