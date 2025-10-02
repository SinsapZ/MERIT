#!/bin/bash
# 退火调优 - 延长退火周期，让不确定性估计更稳定

python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --lr 1e-4 \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 80 \
  --evidence_dropout 0.0 \
  --e_layers 4 \
  --dropout 0.1 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs 200 \
  --patience 35 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43,44,45" \
  --log_csv results/config_annealing80_noWD.csv

