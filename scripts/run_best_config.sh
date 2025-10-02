#!/bin/bash
# 最佳综合配置 - 添加正则化和优化策略

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 60 \
  --evidence_dropout 0.12 \
  --e_layers 4 \
  --dropout 0.12 \
  --weight_decay 1e-4 \
  --nodedim 12 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 32 \
  --swa \
  --lr_scheduler cosine \
  --warmup_epochs 5 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_BEST_optimized.csv

