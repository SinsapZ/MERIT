#!/bin/bash
# MERIT在所有数据集上的完整实验
# 报告完整指标：Accuracy, Precision, Recall, F1, AUROC

SEEDS="41,42,43,44,45,46,47,48,49,50"
GPU=0

echo "========================================================"
echo "MERIT - 多数据集完整实验"
echo "数据集: ADFD, APAVA, PTB, PTB-XL"
echo "指标: Accuracy, Precision, Recall, F1, AUROC"
echo "========================================================"

mkdir -p results/final_all_datasets

# ============================================================
# Dataset 1: ADFD (Sample-based)
# ============================================================
echo ""
echo "========================================================"
echo "实验 1/4: ADFD Dataset"
echo "说明: 88个受试者，2分类 (HC vs Alzheimer)"
echo "配置: 单分辨率, 6层encoder"
echo "========================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/ADFTD/ADFD \
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
echo "ADFD实验完成！"
echo "结果: results/final_all_datasets/adfd_results.csv"

# ============================================================
# Dataset 2: APAVA (Subject-based)
# ============================================================
echo ""
echo "========================================================"
echo "实验 2/4: APAVA Dataset"
echo "说明: 23个受试者，9类心律失常"
echo "配置: 多分辨率(2,4,6,8), 4层encoder"
echo "========================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu $GPU \
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
  --log_csv results/final_all_datasets/apava_results.csv

echo ""
echo "APAVA实验完成！"
echo "结果: results/final_all_datasets/apava_results.csv"

# ============================================================
# Dataset 3: PTB (Subject-based)
# ============================================================
echo ""
echo "========================================================"
echo "实验 3/4: PTB Dataset"
echo "说明: 290个受试者，2分类 (Normal vs Myocardial)"
echo "配置: 多分辨率(2,4,6,8), 4层encoder"
echo "========================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/PTB/PTB \
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
echo "PTB实验完成！"
echo "结果: results/final_all_datasets/ptb_results.csv"

# ============================================================
# Dataset 4: PTB-XL (Subject-based)
# ============================================================
echo ""
echo "========================================================"
echo "实验 4/4: PTB-XL Dataset"
echo "说明: 18885个受试者，5分类"
echo "配置: 多分辨率(2,4,6,8), 4层encoder"
echo "========================================================"

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/PTB-XL/PTB-XL \
  --data PTB-XL \
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
  --log_csv results/final_all_datasets/ptbxl_results.csv

echo ""
echo "PTB-XL实验完成！"
echo "结果: results/final_all_datasets/ptbxl_results.csv"

# ============================================================
# 汇总所有结果
# ============================================================
echo ""
echo "========================================================"
echo "所有数据集实验完成！"
echo "========================================================"
echo ""
echo "结果文件:"
echo "  1. ADFD:   results/final_all_datasets/adfd_results.csv"
echo "  2. APAVA:  results/final_all_datasets/apava_results.csv"
echo "  3. PTB:    results/final_all_datasets/ptb_results.csv"
echo "  4. PTB-XL: results/final_all_datasets/ptbxl_results.csv"
echo ""
echo "Summary文件:"
echo "  1. ADFD:   results/final_all_datasets/adfd_results_summary.txt"
echo "  2. APAVA:  results/final_all_datasets/apava_results_summary.txt"
echo "  3. PTB:    results/final_all_datasets/ptb_results_summary.txt"
echo "  4. PTB-XL: results/final_all_datasets/ptbxl_results_summary.txt"
echo ""
echo "========================================================"

