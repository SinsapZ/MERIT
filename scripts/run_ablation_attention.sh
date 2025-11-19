#!/bin/bash
# 单独运行 Attention Fusion 消融实验
# 在PTB-XL（ECG）和ADFD（EEG）上进行

DATASET=$1  # PTB-XL 或 ADFD-Sample
GPU=${2:-0}
SEEDS="41,42,43,44,45"  # 5个seeds

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_ablation_attention.sh <DATASET> [GPU]"
    echo "Recommended: PTB-XL (ECG), PTB (ECG), or ADFD-Sample (EEG)"
    exit 1
fi

# 设置基础参数
case $DATASET in
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        DROPOUT=0.2
        LR=1e-4
        EPOCHS=150
        LAM_PSEUDO=0.3
        ANNEAL=50
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        LR=2e-4
        EPOCHS=100
        LAM_PSEUDO=0.3
        ANNEAL=50
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        LR=1.5e-4
        EPOCHS=150
        LAM_PSEUDO=0.2
        ANNEAL=40
        ;;
    *)
        echo "Unknown dataset"
        exit 1
        ;;
esac

mkdir -p results/ablation/$DATASET

echo "========================================================================"
echo "消融实验 (Attention Fusion Only) - $DATASET"
echo "========================================================================"

# ============================================================
# 变体6: Attention Fusion
# ============================================================
echo ""
echo "变体6: Attention Fusion"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --agg attention \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss $LAM_PSEUDO \
  --annealing_epoch $ANNEAL \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/attention_fusion.csv

echo ""
echo "========================================================================"
echo "结果已保存到: results/ablation/$DATASET/attention_fusion.csv"
echo "========================================================================"

