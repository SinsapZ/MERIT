#!/bin/bash
# 超参数快速搜索 - 两阶段策略
# 阶段1: 27个配置×1个seed×少量epochs，快速筛选 (预计1-2小时)
# 阶段2: Top5配置×3个seeds×完整epochs，精确验证 (预计2-3小时)

DATASET=$1
GPU=${2:-0}
STAGE=${3:-all}  # all, stage1, stage2

if [ -z "$DATASET" ]; then
    echo "Usage: bash find_best_params_fast.sh <DATASET> [GPU] [STAGE]"
    echo "Available: APAVA, ADFD, ADFD-Sample, PTB, PTB-XL"
    echo "  ADFD: Subject-independent (harder, cross-subject)"
    echo "  ADFD-Sample: Sample-dependent (easier, within-subject)"
    echo "STAGE: all(默认), stage1(仅快速筛选), stage2(仅精确验证)"
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
    "ADFD")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        DROPOUT=0.2
        BATCH_SIZE=128
        # 优化（仅脚本层，不改multi_seed）：学习率调度、warmup、证据dropout、权重衰减
        LR_SCHEDULER="cosine"
        WARMUP_EPOCHS=5
        EVI_DROPOUT=0.10
        WEIGHT_DECAY=1e-4
        ;;
    "ADFD-Sample")
        ROOT_PATH="/home/Data1/zbl/dataset/ADFD"
        E_LAYERS=6
        RESOLUTION_LIST="2"
        DROPOUT=0.2
        BATCH_SIZE=128
        # 优化（仅脚本层，不改multi_seed）：学习率调度、warmup、证据dropout、权重衰减
        LR_SCHEDULER="cosine"
        WARMUP_EPOCHS=5
        EVI_DROPOUT=0.05
        WEIGHT_DECAY=5e-5
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

RESULT_DIR="results/param_search/$DATASET"
mkdir -p $RESULT_DIR

# 设置默认batch_size（如果没有在case中设置）
BATCH_SIZE=${BATCH_SIZE:-64}

# ============================================================================
# 阶段1: 快速筛选 (27个配置, 1 seed, 少量epochs)
# ============================================================================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage1" ]; then
    echo "========================================================================"
    echo "阶段1: 快速筛选 - $DATASET"
    echo "========================================================================"
    echo "搜索空间: 3×3×3 = 27个配置"
    echo "  - 学习率 (3个): 1e-4, 1.5e-4, 2e-4"
    echo "  - Lambda_view (3个): 0.5, 1.0, 1.5"
    echo "  - Lambda_pseudo (3个): 0.2, 0.3, 0.5"
    echo ""
    echo "每个配置: 1 seed, 少量epochs (30-50)"
    echo "预计时间: 1-2小时"
    echo "========================================================================"
    echo "开始时间: $(date)"
    echo ""
    
    CONFIG_ID=0
    
    for lr in 1e-4 1.5e-4 2e-4; do
        for lambda_view in 0.5 1.0 1.5; do
            for lambda_pseudo in 0.2 0.3 0.5; do
                
                CONFIG_ID=$((CONFIG_ID + 1))
                
                echo ""
                echo "Config $CONFIG_ID/27: lr=$lr, λ_view=$lambda_view, λ_pseudo=$lambda_pseudo"
                
                # 快速筛选: 少量epochs, 更激进的early stopping
                case $lr in
                    2e-4)
                        EPOCHS=30; PATIENCE=8; ANNEALING=10 ;;
                    1.5e-4)
                        EPOCHS=40; PATIENCE=10; ANNEALING=15 ;;
                    *)
                        EPOCHS=50; PATIENCE=12; ANNEALING=20 ;;
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
                  --evidence_dropout ${EVI_DROPOUT:-0.0} \
                  --e_layers $E_LAYERS \
                  --dropout $DROPOUT \
                  --weight_decay ${WEIGHT_DECAY:-0} \
                  --nodedim 10 \
                  --lr_scheduler ${LR_SCHEDULER:-none} \
                  --warmup_epochs ${WARMUP_EPOCHS:-0} \
                  --batch_size $BATCH_SIZE \
                  --train_epochs $EPOCHS \
                  --patience $PATIENCE \
                  --swa \
                  --resolution_list $RESOLUTION_LIST \
                  --seeds "41" \
                  --log_csv $RESULT_DIR/stage1_config${CONFIG_ID}.csv \
                  2>&1 | grep -E "(completed|Test - Acc)"
                
            done
        done
    done
    
    echo ""
    echo "========================================================================"
    echo "阶段1完成！分析结果..."
    echo "========================================================================"
fi

# ============================================================================
# 分析阶段1结果，选出Top5配置
# ============================================================================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage2" ]; then
    
    python3 - <<EOF
import pandas as pd
import glob
import os

results = []
pattern = "$RESULT_DIR/stage1_config*.csv"

for csv_file in glob.glob(pattern):
    try:
        df = pd.read_csv(csv_file)
        df_ok = df[df['return_code'] == 0]
        
        if len(df_ok) > 0:
            config_name = os.path.basename(csv_file).replace('.csv', '').replace('stage1_', '')
            
            results.append({
                'config_id': config_name,
                'acc': df_ok['test_acc'].iloc[0],
                'f1': df_ok['test_f1'].iloc[0],
                'auroc': df_ok['test_auroc'].iloc[0],
            })
    except:
        continue

if results:
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    print("\n" + "="*80)
    print(f"阶段1结果 - Top 10配置")
    print("="*80)
    print(f"{'Rank':<6} {'Config':<20} {'Test Acc':<12} {'F1':<12} {'AUROC':<12}")
    print("-"*80)
    
    for i, res in enumerate(results[:10], 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{marker:<6} {res['config_id']:<20} {res['acc']:.4f}      {res['f1']:.4f}      {res['auroc']:.4f}")
    
    # 保存Top5配置ID
    top5_ids = [r['config_id'].replace('config', '') for r in results[:5]]
    with open("$RESULT_DIR/top5_configs.txt", 'w') as f:
        for cid in top5_ids:
            f.write(f"{cid}\n")
    
    print(f"\n✅ Top5配置已保存到: $RESULT_DIR/top5_configs.txt")
    print(f"   配置ID: {', '.join(top5_ids)}")
    
else:
    print("\n❌ 阶段1没有成功的配置，请检查实验")
    exit 1

EOF

    # 读取Top5配置ID
    if [ ! -f "$RESULT_DIR/top5_configs.txt" ]; then
        echo "❌ 未找到Top5配置文件"
        exit 1
    fi
    
    TOP5_IDS=($(cat $RESULT_DIR/top5_configs.txt))
    
    echo ""
    echo "========================================================================"
    echo "阶段2: 精确验证 Top5 配置"
    echo "========================================================================"
    echo "配置: ${TOP5_IDS[@]}"
    echo "每个配置: 3 seeds, 完整epochs"
    echo "预计时间: 2-3小时"
    echo "========================================================================"
    echo "开始时间: $(date)"
    echo ""
    
    # 重新运行Top5配置，用完整设置
    CONFIG_ID=0
    
    for lr in 1e-4 1.5e-4 2e-4; do
        for lambda_view in 0.5 1.0 1.5; do
            for lambda_pseudo in 0.2 0.3 0.5; do
                
                CONFIG_ID=$((CONFIG_ID + 1))
                
                # 检查是否在Top5中
                if [[ ! " ${TOP5_IDS[@]} " =~ " ${CONFIG_ID} " ]]; then
                    continue
                fi
                
                echo ""
                echo "================================================================"
                echo "Config $CONFIG_ID (Top5): lr=$lr, λ_view=$lambda_view, λ_pseudo=$lambda_pseudo"
                echo "================================================================"
                
                # 完整epochs
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
                  --evidence_dropout ${EVI_DROPOUT:-0.0} \
                  --e_layers $E_LAYERS \
                  --dropout $DROPOUT \
                  --weight_decay ${WEIGHT_DECAY:-0} \
                  --nodedim 10 \
                  --lr_scheduler ${LR_SCHEDULER:-none} \
                  --warmup_epochs ${WARMUP_EPOCHS:-0} \
                  --batch_size $BATCH_SIZE \
                  --train_epochs $EPOCHS \
                  --patience $PATIENCE \
                  --swa \
                  --resolution_list $RESOLUTION_LIST \
                  --seeds "41,42,43" \
                  --log_csv $RESULT_DIR/stage2_config${CONFIG_ID}.csv \
                  2>&1 | grep -E "(completed|Test - Acc)"
                
            done
        done
    done
    
    echo ""
    echo "========================================================================"
    echo "阶段2完成！最终分析..."
    echo "========================================================================"
    
    # 最终分析
    python3 - <<EOF
import pandas as pd
import glob
import os
import re

results = []
pattern = "$RESULT_DIR/stage2_config*.csv"

for csv_file in glob.glob(pattern):
    try:
        df = pd.read_csv(csv_file)
        df_ok = df[df['return_code'] == 0]
        
        if len(df_ok) >= 2:
            config_name = os.path.basename(csv_file).replace('.csv', '').replace('stage2_', '')
            
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
    print(f"最终结果 - Top5配置完整验证")
    print("="*80)
    print(f"{'Rank':<6} {'Config':<20} {'Test Acc':<20} {'F1':<12} {'Seeds':<8}")
    print("-"*80)
    
    for i, res in enumerate(results, 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{marker:<6} {res['config']:<20} {res['acc_mean']:.4f}±{res['acc_std']:.4f}  {res['f1_mean']:.4f}      {res['n_seeds']}/3")
    
    # 保存最佳配置
    best = results[0]
    
    with open(f"$RESULT_DIR/best_config.txt", 'w') as f:
        f.write(f"最佳配置: {best['config']}\n")
        f.write(f"Test Acc: {best['acc_mean']:.4f} ± {best['acc_std']:.4f}\n")
        f.write(f"Test F1: {best['f1_mean']:.4f}\n")
        f.write(f"Test AUROC: {best['auroc_mean']:.4f}\n")
        f.write(f"Seeds: {best['n_seeds']}/3\n\n")
        
        # 提取参数
        config_id = int(best['config'].replace('config', ''))
        
        # 反推参数 (config_id从1开始)
        config_idx = config_id - 1
        lp_idx = config_idx % 3
        lv_idx = (config_idx // 3) % 3
        lr_idx = config_idx // 9
        
        lr_vals = ['1e-4', '1.5e-4', '2e-4']
        lv_vals = [0.5, 1.0, 1.5]
        lp_vals = [0.2, 0.3, 0.5]
        
        f.write(f"--lr {lr_vals[lr_idx]}\n")
        f.write(f"--lambda_view {lv_vals[lv_idx]}\n")
        f.write(f"--lambda_pseudo_loss {lp_vals[lp_idx]}\n")
    
    print(f"\n✅ 最佳配置已保存: $RESULT_DIR/best_config.txt")
    
    # 显示推荐配置
    print("\n" + "="*80)
    print("📝 推荐配置:")
    print("="*80)
    with open(f"$RESULT_DIR/best_config.txt", 'r') as f:
        print(f.read())
    
else:
    print("\n❌ 没有找到成功的配置")

EOF

fi

echo ""
echo "========================================================================"
echo "全部完成！"
echo "结束时间: $(date)"
echo "========================================================================"
echo ""
echo "下一步: 用最佳配置运行完整实验 (10 seeds)"
echo "  bash MERIT/scripts/run_all_datasets.sh"
echo ""

