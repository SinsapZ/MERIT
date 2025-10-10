#!/bin/bash
# MERIT综合超参数搜索 - 2天自动运行
# 目标：找到每个数据集的最优配置
# 策略：3 seeds快速验证 → 选出top配置 → 10 seeds完整验证

DATASET=$1
GPU=${2:-0}
QUICK_SEEDS="41,42,43"
FULL_SEEDS="41,42,43,44,45,46,47,48,49,50"

if [ -z "$DATASET" ]; then
    echo "Usage: bash comprehensive_search.sh <DATASET> [GPU]"
    echo "Available: ADFD-Sample, PTB, PTB-XL, APAVA"
    exit 1
fi

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
        echo "Unknown dataset"
        exit 1
        ;;
esac

mkdir -p results/comprehensive_search/$DATASET

echo "========================================================================"
echo "MERIT 综合超参数搜索 - $DATASET"
echo "========================================================================"
echo "阶段1: 快速筛选 (3 seeds × 多种配置)"
echo "阶段2: 完整验证 (10 seeds × top-3配置)"
echo "预计时间: ~24-36小时"
echo "========================================================================"
echo ""
echo "开始时间: $(date)"
echo ""

# ============================================================
# 阶段1: 快速筛选 (3 seeds)
# ============================================================
echo "========================================================================"
echo "阶段1: 快速筛选 - 学习率 × Lambda权重组合"
echo "========================================================================"

CONFIG_ID=0

# 学习率候选: 5e-5, 8e-5, 1e-4, 1.2e-4, 1.5e-4, 2e-4, 3e-4
LR_LIST=(5e-5 8e-5 1e-4 1.2e-4 1.5e-4 2e-4 3e-4)

# Lambda权重组合
# 组合1: 均衡 (1.0, 1.0, 0.3)
# 组合2: 强化融合 (1.0, 0.5, 0.5)
# 组合3: 弱化单视图 (1.0, 0.3, 0.5)
# 组合4: 强化所有 (1.0, 1.0, 0.5)
# 组合5: 极简 (1.0, 0.0, 0.0)

LAMBDA_CONFIGS=(
    "1.0,1.0,0.3"    # 默认均衡
    "1.0,0.5,0.5"    # 强化跨分辨率
    "1.0,0.3,0.5"    # 弱化单视图
    "1.0,1.0,0.5"    # 强化pseudo
    "1.0,1.5,0.3"    # 强化单视图
)

LAMBDA_NAMES=(
    "balanced"
    "fusion_focused"
    "weak_view"
    "strong_pseudo"
    "strong_view"
)

# 主循环：学习率 × Lambda组合
for lr in "${LR_LIST[@]}"; do
    for i in "${!LAMBDA_CONFIGS[@]}"; do
        lambda_config="${LAMBDA_CONFIGS[$i]}"
        lambda_name="${LAMBDA_NAMES[$i]}"
        
        IFS=',' read -r lambda_fuse lambda_view lambda_pseudo <<< "$lambda_config"
        
        CONFIG_ID=$((CONFIG_ID + 1))
        
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Config $CONFIG_ID: lr=${lr}, lambda=${lambda_name} (${lambda_config})"
        echo "--------------------------------------------------------------------"
        
        # 根据学习率调整训练轮数
        if (( $(echo "$lr >= 2e-4" | bc -l) )); then
            EPOCHS=100
            PATIENCE=15
            ANNEALING=30
        elif (( $(echo "$lr >= 1e-4" | bc -l) )); then
            EPOCHS=150
            PATIENCE=20
            ANNEALING=50
        else
            EPOCHS=200
            PATIENCE=30
            ANNEALING=50
        fi
        
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
          --seeds "$QUICK_SEEDS" \
          --log_csv results/comprehensive_search/$DATASET/quick_lr${lr}_${lambda_name}.csv \
          2>&1 | grep -E "(completed|Test - Acc)"
    done
done

echo ""
echo "========================================================================"
echo "阶段1完成！分析结果，选择Top-3配置..."
echo "========================================================================"

# 自动分析并选择top-3配置
python - <<EOF
import pandas as pd
import glob
import os

results = []
pattern = "results/comprehensive_search/$DATASET/quick_*.csv"

for csv_file in glob.glob(pattern):
    try:
        df = pd.read_csv(csv_file)
        df_success = df[df['return_code'] == 0]
        
        if len(df_success) >= 2:  # 至少2个seeds成功
            acc_mean = df_success['test_acc'].mean()
            acc_std = df_success['test_acc'].std()
            
            config_name = os.path.basename(csv_file).replace('quick_', '').replace('.csv', '')
            
            results.append({
                'config': config_name,
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                'n_seeds': len(df_success),
                'file': csv_file
            })
    except Exception as e:
        continue

if results:
    # 按准确率排序
    results.sort(key=lambda x: x['acc_mean'], reverse=True)
    
    print("\n" + "="*80)
    print("Top 10 配置:")
    print("="*80)
    print(f"{'Rank':<6} {'Config':<40} {'Test Acc':<20} {'Seeds':<8}")
    print("-"*80)
    
    for i, res in enumerate(results[:10], 1):
        print(f"{i:<6} {res['config']:<40} {res['acc_mean']:.4f}±{res['acc_std']:.4f}  {res['n_seeds']}/3")
    
    # 保存top-3配置
    with open("results/comprehensive_search/$DATASET/top3_configs.txt", 'w') as f:
        for i, res in enumerate(results[:3], 1):
            f.write(f"{res['config']}\n")
    
    print("\n✅ Top-3配置已保存到: results/comprehensive_search/$DATASET/top3_configs.txt")
    print("\n建议使用这些配置进行完整实验 (10 seeds)")
else:
    print("\n❌ 没有找到成功的配置")
EOF

# ============================================================
# 阶段2: 完整验证 Top-3 配置 (10 seeds)
# ============================================================
echo ""
echo "========================================================================"
echo "阶段2: 完整验证 Top-3 配置 (10 seeds)"
echo "========================================================================"

if [ ! -f "results/comprehensive_search/$DATASET/top3_configs.txt" ]; then
    echo "⚠️  Top-3配置文件不存在，跳过阶段2"
    echo "手动选择配置后可单独运行完整实验"
    exit 0
fi

# 读取top-3配置并重新运行
TOP_CONFIGS=($(cat results/comprehensive_search/$DATASET/top3_configs.txt))

for rank in 1 2 3; do
    config_name="${TOP_CONFIGS[$((rank-1))]}"
    
    if [ -z "$config_name" ]; then
        continue
    fi
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Top-$rank 配置: $config_name (10 seeds完整验证)"
    echo "--------------------------------------------------------------------"
    
    # 从配置名中提取参数
    # 格式: lr{value}_{lambda_name}
    lr=$(echo $config_name | grep -oP 'lr\K[0-9.e-]+')
    lambda_name=$(echo $config_name | sed 's/.*_//')
    
    # 设置lambda值
    case $lambda_name in
        "balanced")
            lambda_fuse=1.0; lambda_view=1.0; lambda_pseudo=0.3 ;;
        "fusion_focused")
            lambda_fuse=1.0; lambda_view=0.5; lambda_pseudo=0.5 ;;
        "weak_view")
            lambda_fuse=1.0; lambda_view=0.3; lambda_pseudo=0.5 ;;
        "strong_pseudo")
            lambda_fuse=1.0; lambda_view=1.0; lambda_pseudo=0.5 ;;
        "strong_view")
            lambda_fuse=1.0; lambda_view=1.5; lambda_pseudo=0.3 ;;
        *)
            lambda_fuse=1.0; lambda_view=1.0; lambda_pseudo=0.3 ;;
    esac
    
    # 根据学习率设置epochs
    if (( $(echo "$lr >= 2e-4" | bc -l) )); then
        EPOCHS=100; PATIENCE=15; ANNEALING=30
    elif (( $(echo "$lr >= 1e-4" | bc -l) )); then
        EPOCHS=150; PATIENCE=20; ANNEALING=50
    else
        EPOCHS=200; PATIENCE=30; ANNEALING=50
    fi
    
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
      --seeds "$FULL_SEEDS" \
      --log_csv results/comprehensive_search/$DATASET/full_top${rank}_${config_name}.csv
done

echo ""
echo "========================================================================"
echo "综合搜索完成！"
echo "结束时间: $(date)"
echo "========================================================================"
echo ""
echo "结果文件:"
echo "  快速筛选: results/comprehensive_search/$DATASET/quick_*.csv"
echo "  完整验证: results/comprehensive_search/$DATASET/full_top*.csv"
echo ""
echo "最终最佳配置:"
cat results/comprehensive_search/$DATASET/top3_configs.txt
echo ""

