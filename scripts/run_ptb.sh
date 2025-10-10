#!/bin/bash
# MERIT on PTB Dataset
# Myocardial Infarction Detection: 290 subjects, 2 classes

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/PTB"
GPU=0

echo "========================================================"
echo "MERIT - PTB Dataset"
echo "Task: Myocardial Infarction Detection (Binary)"
echo "Data: 290 subjects, ~14000 samples"
echo "========================================================"

mkdir -p results/final_all_datasets

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data PTB \
  --gpu $GPU \
  --lr 1e-4 \
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
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list 2,4,6,8 \
  --seeds "$SEEDS" \
  --log_csv results/final_all_datasets/ptb_results.csv

echo ""
echo "========================================================"
echo "PTB实验完成！"
echo "结果: results/final_all_datasets/ptb_results.csv"
echo "Summary: results/final_all_datasets/ptb_results_summary.txt"
echo "========================================================"

