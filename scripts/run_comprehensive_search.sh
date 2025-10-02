#!/bin/bash
# 全面超参数搜索 - 适合长时间运行
# 预计总耗时: 10-15小时

SEEDS="41,42,43,44,45,46,47,48,49,50"
ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
GPU=0

echo "========================================================"
echo "开始全面超参数搜索"
echo "种子数: 10"
echo "总配置数: 15"
echo "预计总耗时: 10-15小时"
echo "开始时间: $(date)"
echo "========================================================"

# ===== 第一组: SWA变体实验 =====
echo -e "\n[1/15] SWA + 原始参数"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp01_swa_baseline.csv

echo -e "\n[2/15] SWA + 轻度weight decay"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 5e-5 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp02_swa_wd5e5.csv

# ===== 第二组: 学习率调整 =====
echo -e "\n[3/15] 学习率 9e-5 + SWA"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 9e-5 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp03_lr9e5_swa.csv

echo -e "\n[4/15] 学习率 1.1e-4 + SWA"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1.1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp04_lr1.1e4_swa.csv

echo -e "\n[5/15] 学习率 8e-5 + Cosine调度"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 8e-5 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --lr_scheduler cosine --warmup_epochs 5 \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp05_lr8e5_cosine_swa.csv

# ===== 第三组: Annealing调整 =====
echo -e "\n[6/15] Annealing 60"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 60 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp06_anneal60_swa.csv

echo -e "\n[7/15] Annealing 70"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 70 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp07_anneal70_swa.csv

echo -e "\n[8/15] Annealing 40 (更早退火)"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 40 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp08_anneal40_swa.csv

# ===== 第四组: Evidential损失权重调整 =====
echo -e "\n[9/15] Lambda_pseudo_loss 0.25"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.25 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp09_pseudo0.25_swa.csv

echo -e "\n[10/15] Lambda_pseudo_loss 0.35"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.35 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp10_pseudo0.35_swa.csv

# ===== 第五组: Evidence dropout调整 =====
echo -e "\n[11/15] Evidence dropout 0.05"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.05 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp11_evidrop0.05_swa.csv

echo -e "\n[12/15] Evidence dropout 0.08"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.08 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp12_evidrop0.08_swa.csv

# ===== 第六组: 模型容量调整 =====
echo -e "\n[13/15] NodeDim 12"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 12 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp13_node12_swa.csv

echo -e "\n[14/15] E_layers 3"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 3 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp14_elayer3_swa.csv

# ===== 第七组: 综合最优配置 =====
echo -e "\n[15/15] 综合最优配置"
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH --data APAVA --gpu $GPU \
  --lr 9.5e-5 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.28 \
  --annealing_epoch 55 --evidence_dropout 0.05 --e_layers 4 \
  --dropout 0.1 --weight_decay 5e-5 --nodedim 11 \
  --batch_size 64 --train_epochs 200 --patience 32 --swa \
  --lr_scheduler cosine --warmup_epochs 3 \
  --resolution_list 2,4,6,8 --seeds "$SEEDS" \
  --log_csv results/comprehensive/exp15_best_combined.csv

echo ""
echo "========================================================"
echo "全部实验完成！"
echo "结束时间: $(date)"
echo "========================================================"
echo ""
echo "请运行结果汇总脚本:"
echo "  python MERIT/scripts/summarize_results.py"

