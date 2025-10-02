#!/bin/bash
# 减小模型容量 - 防止过拟合

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 1.2e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 55 \
  --evidence_dropout 0.1 \
  --e_layers 3 \
  --dropout 0.12 \
  --weight_decay 1e-4 \
  --nodedim 15 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 30 \
  --swa \
  --lr_scheduler step \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_smaller_model.csv

