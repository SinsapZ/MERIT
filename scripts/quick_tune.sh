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
echo "快速调优 - $DATASET (3 seeds, 基于MedGNN参数)"
echo "学习率范围: 5e-5 ~ 3e-4 (MedGNN baseline: 1e-4)"
echo "========================================================"

# ============================================================
# Config 1: MedGNN baseline (保守)
# ============================================================
echo ""
echo "Config 1: MedGNN Baseline (lr=1e-4, dropout=0.1)"
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
  --log_csv results/tuning/$DATASET/config1_baseline.csv

# ============================================================
# Config 2: 更低学习率 (更稳定)
# ============================================================
echo ""
echo "Config 2: Lower LR (lr=5e-5, 更稳定收敛)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 5e-5 \
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
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config2_lower_lr.csv

# ============================================================
# Config 3: 更高学习率 (快速训练)
# ============================================================
echo ""
echo "Config 3: Higher LR (lr=2e-4, 快速收敛)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 2e-4 \
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
  --train_epochs 100 \
  --patience 15 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config3_higher_lr.csv

# ============================================================
# Config 4: 激进学习率 (最大胆)
# ============================================================
echo ""
echo "Config 4: Aggressive LR (lr=3e-4, 激进尝试)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr 3e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 30 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout 0.15 \
  --weight_decay 1e-5 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 100 \
  --patience 15 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/tuning/$DATASET/config4_aggressive.csv

echo ""
echo "========================================================"
echo "调优完成！分析结果..."
echo "========================================================"

# 简单对比
echo ""
echo "Results Comparison:"
echo "----------------------------------------"
for config in config1_baseline config2_lower_lr config3_higher_lr config4_aggressive; do
    if [ -f "results/tuning/$DATASET/${config}_summary.txt" ]; then
        acc=$(grep "test_acc:" results/tuning/$DATASET/${config}_summary.txt | awk '{print $2}')
        echo "$config: $acc"
    fi
done

echo ""
echo "详细分析:"
echo "  - config1_baseline (1e-4):   MedGNN标准配置"
echo "  - config2_lower_lr (5e-5):   更稳定，但训练慢"
echo "  - config3_higher_lr (2e-4):  更快收敛，可能不稳定"
echo "  - config4_aggressive (3e-4): 激进尝试，高风险高回报"
echo ""
echo "建议："
echo "  1. 如果结果相近，选择训练最快的"
echo "  2. 如果方差大，选择更低的学习率"
echo "  3. 如果都不理想，尝试调整其他参数"
echo ""
echo "详细结果查看: results/tuning/$DATASET/"
echo ""

