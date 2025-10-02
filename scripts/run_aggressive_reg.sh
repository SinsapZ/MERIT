#!/bin/bash
# 激进正则化配置 - 降低方差为主要目标

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 8e-5 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.28 \
  --annealing_epoch 70 \
  --evidence_dropout 0.15 \
  --e_layers 4 \
  --dropout 0.15 \
  --weight_decay 2e-4 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 35 \
  --swa \
  --lr_scheduler cosine \
  --warmup_epochs 10 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_aggressive_regularization.csv

