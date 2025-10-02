#!/bin/bash
# 仅SWA策略 - 在原始配置基础上只加SWA

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
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
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --lr_scheduler none \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_only_SWA.csv

