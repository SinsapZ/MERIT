#!/bin/bash
# MERIT on ADFD Dataset
# Alzheimer Detection: 88 subjects, 2 classes (HC vs AD)

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/ADFTD/ADFD"
GPU=0

echo "========================================================"
echo "MERIT - ADFD Dataset"
echo "Task: Alzheimer Detection (Binary Classification)"
echo "Data: 88 subjects, ~8800 samples"
echo "========================================================"

mkdir -p results/final_all_datasets

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data ADFD-Sample \
  --gpu $GPU \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 6 \
  --dropout 0.2 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 128 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list 2 \
  --seeds "$SEEDS" \
  --log_csv results/final_all_datasets/adfd_results.csv

echo ""
echo "========================================================"
echo "ADFD实验完成！"
echo "结果: results/final_all_datasets/adfd_results.csv"
echo "Summary: results/final_all_datasets/adfd_results_summary.txt"
echo "========================================================"

