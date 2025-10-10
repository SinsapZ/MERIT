#!/bin/bash
# 测试综合搜索脚本 - 运行1个配置确认没有错误

DATASET=${1:-APAVA}
GPU=${2:-0}
TEST_SEEDS="41"  # 只用1个seed测试

# 根据数据集设置基础参数
case $DATASET in
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        BASE_DROPOUT=0.2
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB/PTB"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        BASE_DROPOUT=0.1
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL/PTB-XL"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        BASE_DROPOUT=0.1
        ;;
    "APAVA")
        ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        BASE_DROPOUT=0.1
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

mkdir -p results/test_search

echo "========================================================================"
echo "测试综合搜索脚本 - $DATASET"
echo "========================================================================"
echo "运行1个配置 (1 seed) 确认脚本正常工作"
echo "如果成功，可以启动完整搜索"
echo "========================================================================"

# 测试一个配置
lr=1e-4
lambda_fuse=1.0
lambda_view=1.0
lambda_pseudo=0.3

echo ""
echo "测试配置: lr=${lr}, lambda=(${lambda_fuse},${lambda_view},${lambda_pseudo})"
echo ""

# 使用case判断学习率
case $lr in
    3e-4|2e-4)
        EPOCHS=100
        PATIENCE=15
        ANNEALING=30
        ;;
    1.5e-4|1.2e-4|1.1e-4|1e-4)
        EPOCHS=150
        PATIENCE=20
        ANNEALING=50
        ;;
    *)
        EPOCHS=200
        PATIENCE=30
        ANNEALING=50
        ;;
esac

echo "自动设置: epochs=$EPOCHS, patience=$PATIENCE, annealing=$ANNEALING"
echo ""

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $lr \
  --lambda_fuse $lambda_fuse \
  --lambda_view $lambda_view \
  --lambda_pseudo_loss $lambda_pseudo \
  --annealing_epoch $ANNEALING \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $BASE_DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$TEST_SEEDS" \
  --log_csv results/test_search/test_${DATASET}.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 测试成功！脚本工作正常"
    echo "========================================================================"
    echo ""
    echo "现在可以启动完整搜索："
    echo "  bash MERIT/scripts/comprehensive_search.sh $DATASET $GPU"
    echo ""
    echo "或启动所有数据集："
    echo "  bash MERIT/scripts/run_all_comprehensive_search.sh $GPU"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ 测试失败，请检查错误信息"
    echo "========================================================================"
    exit 1
fi

