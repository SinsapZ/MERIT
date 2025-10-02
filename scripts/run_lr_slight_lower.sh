#!/bin/bash
# 微调学习率 - 略微降低学习率增加稳定性

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 9e-5 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 55 \
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
  --log_csv results/config_lr9e5_anneal55_SWA.csv

