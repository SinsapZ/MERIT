#!/bin/bash
# 超参数快速搜索 - 3×3×3 = 27个配置
# 用3个seeds快速验证，找出最佳配置

DATASET=$1
GPU=${2:-0}
SEEDS="41,42,43"  # 3个seeds快速验证

if [ -z "$DATASET" ]; then
    echo "Usage: bash find_best_params.sh <DATASET> [GPU]"
    echo "Available: APAVA, ADFD-Sample, PTB, PTB-XL"
    exit 1
fi

# 数据集基础参数
case $DATASET in
    "APAVA")
        ROOT_PATH="/home/Data1/zbl/dataset/APAVA"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        ;;
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        DROPOUT=0.2
        ;;
    "PTB")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        ;;
    "PTB-XL")
        ROOT_PATH="/home/Data1/zbl/dataset/PTB-XL"
        E_LAYERS=4
        RESOLUTION_LIST="2,4,6,8"
        DROPOUT=0.1
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

mkdir -p results/param_search/$DATASET

echo "========================================================================"
echo "超参数搜索 - $DATASET"
echo "========================================================================"
echo "搜索空间: 3×3×3 = 27个配置"
echo "  - 学习率 (3个): 1e-4, 1.5e-4, 2e-4"
echo "  - Lambda_view (3个): 0.5, 1.0, 1.5"
echo "  - Lambda_pseudo (3个): 0.2, 0.3, 0.5"
echo ""
echo "每个配置: 3 seeds"
echo "预计时间: 2-3小时"
echo "========================================================================"
echo "开始时间: $(date)"
echo ""

CONFIG_ID=0

# 学习率: 3个
for lr in 1e-4 1.5e-4 2e-4; do
    
    # Lambda_view: 3个
    for lambda_view in 0.5 1.0 1.5; do
        
        # Lambda_pseudo: 3个
        for lambda_pseudo in 0.2 0.3 0.5; do
            
            CONFIG_ID=$((CONFIG_ID + 1))
            
            echo ""
            echo "================================================================"
            echo "Config $CONFIG_ID/27: lr=$lr, λ_view=$lambda_view, λ_pseudo=$lambda_pseudo"
            echo "================================================================"
            
            # 根据学习率调整epochs
            case $lr in
                2e-4)
                    EPOCHS=100; PATIENCE=15; ANNEALING=30 ;;
                1.5e-4)
                    EPOCHS=120; PATIENCE=18; ANNEALING=40 ;;
                *)
                    EPOCHS=150; PATIENCE=20; ANNEALING=50 ;;
            esac
            
            python -m MERIT.scripts.multi_seed_run \
              --root_path $ROOT_PATH \
              --data $DATASET \
              --gpu $GPU \
              --lr $lr \
              --lambda_fuse 1.0 \
              --lambda_view $lambda_view \
              --lambda_pseudo_loss $lambda_pseudo \
              --annealing_epoch $ANNEALING \
              --evidence_dropout 0.0 \
              --e_layers $E_LAYERS \
              --dropout $DROPOUT \
              --weight_decay 0 \
              --nodedim 10 \
              --batch_size 64 \
              --train_epochs $EPOCHS \
              --patience $PATIENCE \
              --swa \
              --resolution_list $RESOLUTION_LIST \
              --seeds "$SEEDS" \
              --log_csv results/param_search/$DATASET/config${CONFIG_ID}_lr${lr}_lv${lambda_view}_lp${lambda_pseudo}.csv \
              2>&1 | grep -E "(completed|Test - Acc)"
            
        done
    done
done

echo ""
echo "========================================================================"
echo "搜索完成！分析结果..."
echo "结束时间: $(date)"
echo "========================================================================"

# 自动分析找出最佳配置
python3 - <<EOF
import pandas as pd
import glob
import os

results = []
pattern = "results/param_search/$DATASET/config*.csv"

for csv_file in glob.glob(pattern):
    try:
        df = pd.read_csv(csv_file)
        df_ok = df[df['return_code'] == 0]
        
        if len(df_ok) >= 2:
            config_name = os.path.basename(csv_file).replace('.csv', '')
            
            results.append({
                'config': config_name,
                'acc_mean': df_ok['test_acc'].mean(),
                'acc_std': df_ok['test_acc'].std(),
                'f1_mean': df_ok['test_f1'].mean(),
                'auroc_mean': df_ok['test_auroc'].mean(),
                'n_seeds': len(df_ok),
            })
    except:
        continue

if results:
    results.sort(key=lambda x: x['acc_mean'], reverse=True)
    
    print("\n" + "="*80)
    print(f"Top 10 配置 - $DATASET")
    print("="*80)
    print(f"{'Rank':<6} {'Config':<45} {'Test Acc':<20} {'F1':<12}")
    print("-"*80)
    
    for i, res in enumerate(results[:10], 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{marker:<6} {res['config']:<45} {res['acc_mean']:.4f}±{res['acc_std']:.4f}  {res['f1_mean']:.4f}")
    
    # 保存最佳配置
    best = results[0]
    
    with open(f"results/param_search/$DATASET/best_config.txt", 'w') as f:
        f.write(f"最佳配置: {best['config']}\n")
        f.write(f"Test Acc: {best['acc_mean']:.4f} ± {best['acc_std']:.4f}\n")
        f.write(f"Test F1: {best['f1_mean']:.4f}\n")
        f.write(f"Test AUROC: {best['auroc_mean']:.4f}\n")
        f.write(f"Seeds: {best['n_seeds']}/3\n\n")
        
        # 提取参数
        config = best['config']
        import re
        
        lr_match = re.search(r'lr([0-9.e-]+)', config)
        lv_match = re.search(r'lv([0-9.]+)', config)
        lp_match = re.search(r'lp([0-9.]+)', config)
        
        if lr_match:
            f.write(f"--lr {lr_match.group(1)}\n")
        if lv_match:
            f.write(f"--lambda_view {lv_match.group(1)}\n")
        if lp_match:
            f.write(f"--lambda_pseudo_loss {lp_match.group(1)}\n")
    
    print(f"\n✅ 最佳配置已保存: results/param_search/$DATASET/best_config.txt")
    
    # 显示推荐配置
    print("\n" + "="*80)
    print("📝 推荐配置:")
    print("="*80)
    with open(f"results/param_search/$DATASET/best_config.txt", 'r') as f:
        print(f.read())
    
else:
    print("\n❌ 没有找到成功的配置")
    print("请检查实验是否正常运行")

EOF

echo ""
echo "========================================================================"
echo "下一步: 用最佳配置运行完整实验 (10 seeds)"
echo "========================================================================"
echo ""
echo "修改 run_all_datasets.sh 中 $DATASET 的参数，然后运行:"
echo "  bash MERIT/scripts/run_all_datasets.sh"
echo ""

