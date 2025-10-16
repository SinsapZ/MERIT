#!/bin/bash
# 消融实验 - 验证各组件的有效性
# 在PTB-XL（ECG）和ADFD（EEG）上进行

DATASET=$1  # PTB-XL 或 ADFD-Sample
GPU=${2:-0}
SEEDS="41,42,43,44,45"  # 5个seeds

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_ablation.sh <DATASET> [GPU]"
    echo "Recommended: PTB-XL (ECG) or ADFD-Sample (EEG)"
    exit 1
fi

# 设置基础参数
case $DATASET in
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        DROPOUT=0.2
        LR=1e-4
        EPOCHS=150
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        LR=2e-4
        EPOCHS=100
        ;;
    *)
        echo "Unknown dataset"
        exit 1
        ;;
esac

mkdir -p results/ablation/$DATASET

echo "========================================================================"
echo "消融实验 - $DATASET"
echo "========================================================================"
echo "变体:"
echo "  1. Full Model (完整MERIT)"
echo "  2. w/o Evidential Fusion (无证据融合，用简单平均)"
echo "  3. w/o Pseudo-view (无伪视图)"
echo "  4. w/o Freq Branch (无频域分支)"
echo "  5. w/o Diff Branch (无差分分支)"
echo "========================================================================"

# ============================================================
# 变体1: Full Model (完整MERIT)
# ============================================================
echo ""
echo "变体1: Full Model"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.3 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/full_model.csv

# ============================================================
# 变体2: w/o Evidential Fusion (用简单平均)
# ============================================================
echo ""
echo "变体2: w/o Evidential Fusion"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --agg mean \
  --no_pseudo \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/wo_evidential.csv

# ============================================================
# 变体3: w/o Pseudo-view
# ============================================================
echo ""
echo "变体3: w/o Pseudo-view"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.0 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --no_pseudo \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/wo_pseudo.csv

# ============================================================
# 变体4: w/o Frequency Branch
# ============================================================
echo ""
echo "变体4: w/o Frequency Branch"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.3 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --no_freq \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/wo_freq.csv

# ============================================================
# 变体5: w/o Difference Branch
# ============================================================
echo ""
echo "变体5: w/o Difference Branch"

python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data $DATASET \
  --gpu $GPU \
  --lr $LR \
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.3 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers $E_LAYERS \
  --dropout $DROPOUT \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 64 \
  --train_epochs $EPOCHS \
  --patience 20 \
  --swa \
  --no_diff \
  --resolution_list $RESOLUTION_LIST \
  --seeds "$SEEDS" \
  --log_csv results/ablation/$DATASET/wo_diff.csv

echo ""
echo "========================================================================"
echo "消融实验完成！分析结果..."
echo "========================================================================"

# 分析结果
python3 - <<EOF
import pandas as pd
import os

variants = [
    ('full_model', 'Full Model'),
    ('wo_evidential', 'w/o Evidential Fusion'),
    ('wo_pseudo', 'w/o Pseudo-view'),
    ('wo_freq', 'w/o Frequency Branch'),
    ('wo_diff', 'w/o Difference Branch'),
]

print("\n" + "="*80)
print("消融实验结果 - $DATASET")
print("="*80)
print(f"{'Variant':<30} {'Acc':<15} {'F1':<15} {'AUROC':<15}")
print("-"*80)

results = []
for fname, vname in variants:
    csv_path = f"results/ablation/$DATASET/{fname}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_ok = df[df['return_code'] == 0]
        
        if len(df_ok) > 0:
            acc = df_ok['test_acc'].mean()
            acc_std = df_ok['test_acc'].std()
            f1 = df_ok['test_f1'].mean()
            auroc = df_ok['test_auroc'].mean()
            
            results.append((vname, acc, acc_std, f1, auroc))
            print(f"{vname:<30} {acc*100:.2f}±{acc_std*100:.2f}   {f1*100:.2f}      {auroc*100:.2f}")

# 计算贡献
if len(results) > 1:
    full_acc = results[0][1]
    
    print("\n" + "="*80)
    print("各组件贡献:")
    print("="*80)
    
    for vname, acc, _, _, _ in results[1:]:
        drop = (full_acc - acc) * 100
        print(f"{vname:<30} 性能下降: {drop:.2f}%")

# LaTeX表格
print("\n" + "="*80)
print("LaTeX Table:")
print("="*80)
print("\\begin{tabular}{lccc}")
print("\\hline")
print("Variant & Accuracy & F1 & AUROC \\\\\\")
print("\\hline")

for vname, acc, acc_std, f1, auroc in results:
    print(f"{vname} & {acc*100:.2f}±{acc_std*100:.2f} & {f1*100:.2f} & {auroc*100:.2f} \\\\\\")

print("\\hline")
print("\\end{tabular}")

EOF

echo ""
echo "========================================================================"
echo "结果已保存到: results/ablation/$DATASET/"
echo "========================================================================"

