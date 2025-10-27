#!/bin/bash
# 运行Baseline模型进行对比
# 使用MedGNN项目中已实现的模型

DATASET=$1  # APAVA, ADFD-Sample, PTB, PTB-XL
SEEDS="41,42,43,44,45,46,47,48,49,50"
GPU=0

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_baselines.sh <DATASET>"
    echo "Available datasets: APAVA, ADFD-Sample, PTB, PTB-XL"
    exit 1
fi

# 根据数据集设置路径和参数
case $DATASET in
    "APAVA")
        ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        BATCH_SIZE=128
        RESOLUTION_LIST="2"
        AUGMENTATIONS="drop0.2"
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL"
        E_LAYERS=4
        BATCH_SIZE=64
        RESOLUTION_LIST="2,4,6,8"
        AUGMENTATIONS="none,drop0.35"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "========================================================"
echo "运行Baseline模型 - $DATASET"
echo "Baselines: Medformer, iTransformer, FEDformer, CrossGNN, MedGNN"
echo "========================================================"

mkdir -p results/baselines/$DATASET

# 切换到MedGNN目录
cd MedGNN/MedGNN

# ============================================================
# Baseline 1: Medformer
# ============================================================
echo ""
echo "========================================================"
echo "Running Medformer on $DATASET"
echo "========================================================"

for seed in ${SEEDS//,/ }; do
    echo "Seed $seed..."
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --model_id ${DATASET}-Medformer \
        --model Medformer \
        --data $DATASET \
        --e_layers $E_LAYERS \
        --batch_size $BATCH_SIZE \
        --d_model 256 \
        --d_ff 512 \
        --n_heads 8 \
        --augmentations $AUGMENTATIONS \
        --des "Baseline" \
        --itr 1 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --gpu $GPU \
        --seed $seed \
        2>&1 | grep -E "(Validation|Test) results"
done > ../../results/baselines/$DATASET/medformer_results.txt

# ============================================================
# Baseline 2: iTransformer
# ============================================================
echo ""
echo "========================================================"
echo "Running iTransformer on $DATASET"
echo "========================================================"

for seed in ${SEEDS//,/ }; do
    echo "Seed $seed..."
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --model_id ${DATASET}-iTransformer \
        --model iTransformer \
        --data $DATASET \
        --e_layers $E_LAYERS \
        --batch_size $BATCH_SIZE \
        --d_model 256 \
        --d_ff 512 \
        --n_heads 8 \
        --augmentations $AUGMENTATIONS \
        --des "Baseline" \
        --itr 1 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --patience 3 \
        --gpu $GPU \
        --seed $seed \
        2>&1 | grep -E "(Validation|Test) results"
done > ../../results/baselines/$DATASET/itransformer_results.txt

# ============================================================
# Baseline 3: FEDformer (来自独立项目)
# ============================================================
echo ""
echo "========================================================"
echo "Running FEDformer on $DATASET"
echo "========================================================"

# 切换到FEDformer目录（项目根目录下的 FEDformer/FEDformer）
cd ../../FEDformer/FEDformer || { echo "FEDformer 项目不存在，跳过"; cd - >/dev/null; }

if [ -f run.py ]; then
    for seed in ${SEEDS//,/ }; do
        echo "Seed $seed..."
        python -u run.py \
            --is_training 1 \
            --task_id ${DATASET}-FEDformer \
            --model FEDformer \
            --data $DATASET \
            --root_path $ROOT_PATH \
            --seq_len 128 \
            --label_len 64 \
            --pred_len 64 \
            --d_model 256 \
            --d_ff 512 \
            --n_heads 8 \
            --e_layers $E_LAYERS \
            --batch_size $BATCH_SIZE \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --patience 3 \
            --gpu $GPU \
            --itr 1 \
            --des Baseline-FED \
            2>&1
    done > ../../results/baselines/$DATASET/fedformer_results.txt
else
    echo "未找到 FEDformer/run.py，跳过FEDformer基线。"
fi

# 继续进行下一个基线

# ============================================================
# Baseline 4: CrossGNN (来自独立项目)
# ============================================================
echo ""
echo "========================================================"
echo "Running CrossGNN on $DATASET"
echo "========================================================"

# 切换到CrossGNN目录（项目根目录下的 CrossGNN/CrossGNN）
cd ../../CrossGNN/CrossGNN || { echo "CrossGNN 项目不存在，跳过"; cd - >/dev/null; }

if [ -f run_longExp.py ]; then
    for seed in ${SEEDS//,/ }; do
        echo "Seed $seed..."
        python -u run_longExp.py \
            --is_training 1 \
            --model_id ${DATASET}-CrossGNN \
            --model CrossGNN \
            --data $DATASET \
            --root_path $ROOT_PATH \
            --seq_len 128 \
            --label_len 64 \
            --pred_len 64 \
            --d_model 256 \
            --d_ff 512 \
            --n_heads 8 \
            --e_layers $E_LAYERS \
            --batch_size $BATCH_SIZE \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --patience 3 \
            --gpu $GPU \
            --itr 1 \
            --des Baseline-CrossGNN \
            2>&1
    done > ../../results/baselines/$DATASET/crossgnn_results.txt
else
    echo "未找到 CrossGNN/run_longExp.py，跳过CrossGNN基线。"
fi

# 返回到仓库根目录
cd ../..

# ============================================================
# Baseline 3: MedGNN (参考)
# ============================================================
echo ""
echo "========================================================"
echo "MedGNN Results (从论文)"
echo "说明: MedGNN的结果可以直接从论文中引用"
echo "========================================================"

# 返回MERIT目录
cd ../..

echo ""
echo "========================================================"
echo "Baseline实验完成！"
echo "结果保存在: results/baselines/$DATASET/"
echo "========================================================"

