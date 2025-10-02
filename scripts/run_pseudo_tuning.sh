#!/bin/bash
# 伪视图权重调优 - 降低伪视图损失权重

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.20 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_pseudo_loss_0.20.csv

