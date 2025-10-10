#!/bin/bash
# 一键运行所有数据集的综合超参数搜索
# 2天自动化运行，找到每个数据集的最优配置

GPU=${1:-0}

echo "========================================================================"
echo "MERIT - 全数据集综合超参数搜索"
echo "========================================================================"
echo "数据集: ADFD-Sample, PTB, PTB-XL, APAVA"
echo "GPU: $GPU"
echo ""
echo "预计总时间: 36-48小时"
echo "  - 阶段1快速筛选: 12-18小时 (35个配置 × 4数据集)"
echo "  - 阶段2完整验证: 24-30小时 (top-3 × 4数据集)"
echo ""
echo "开始时间: $(date)"
echo "========================================================================"

# ============================================================
# APAVA (使用已知最佳配置 + 少量搜索)
# ============================================================
echo ""
echo "========================================================================"
echo "Dataset 1/4: APAVA"
echo "策略: 已知lr=1.1e-4最优，搜索lambda组合"
echo "========================================================================"

bash MERIT/scripts/comprehensive_search.sh APAVA $GPU

# ============================================================
# ADFD-Sample
# ============================================================
echo ""
echo "========================================================================"
echo "Dataset 2/4: ADFD-Sample"
echo "策略: 全面搜索 (小数据集，关键数据集)"
echo "========================================================================"

bash MERIT/scripts/comprehensive_search.sh ADFD-Sample $GPU

# ============================================================
# PTB
# ============================================================
echo ""
echo "========================================================================"
echo "Dataset 3/4: PTB"
echo "策略: 全面搜索 (大数据集，二分类)"
echo "========================================================================"

bash MERIT/scripts/comprehensive_search.sh PTB $GPU

# ============================================================
# PTB-XL
# ============================================================
echo ""
echo "========================================================================"
echo "Dataset 4/4: PTB-XL"
echo "策略: 全面搜索 (超大数据集)"
echo "========================================================================"

bash MERIT/scripts/comprehensive_search.sh PTB-XL $GPU

# ============================================================
# 汇总所有结果
# ============================================================
echo ""
echo "========================================================================"
echo "所有数据集搜索完成！汇总结果..."
echo "结束时间: $(date)"
echo "========================================================================"

python - <<'EOF'
import pandas as pd
import glob
import os

datasets = ['APAVA', 'ADFD-Sample', 'PTB', 'PTB-XL']

print("\n" + "="*80)
print("🏆 各数据集最佳配置汇总")
print("="*80)

best_configs = {}

for dataset in datasets:
    # 读取完整验证的结果
    pattern = f"results/comprehensive_search/{dataset}/full_top*.csv"
    files = glob.glob(pattern)
    
    if not files:
        # 如果没有完整验证，读取快速筛选的top配置
        top3_file = f"results/comprehensive_search/{dataset}/top3_configs.txt"
        if os.path.exists(top3_file):
            with open(top3_file, 'r') as f:
                best_config = f.readline().strip()
            print(f"\n{dataset}:")
            print(f"  最佳配置 (基于快速筛选): {best_config}")
        continue
    
    # 找到完整验证中最好的
    best_acc = 0
    best_file = None
    
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            df_success = df[df['return_code'] == 0]
            
            if len(df_success) >= 8:  # 至少8个seeds成功
                acc_mean = df_success['test_acc'].mean()
                if acc_mean > best_acc:
                    best_acc = acc_mean
                    best_file = csv_file
        except:
            continue
    
    if best_file:
        df = pd.read_csv(best_file)
        df_success = df[df['return_code'] == 0]
        
        config_name = os.path.basename(best_file).replace('full_top1_', '').replace('.csv', '')
        
        print(f"\n{dataset}:")
        print(f"  最佳配置: {config_name}")
        print(f"  Test Acc: {df_success['test_acc'].mean():.4f} ± {df_success['test_acc'].std():.4f}")
        print(f"  Test F1:  {df_success['test_f1'].mean():.4f} ± {df_success['test_f1'].std():.4f}")
        print(f"  Seeds成功: {len(df_success)}/10")
        
        best_configs[dataset] = {
            'config': config_name,
            'acc': df_success['test_acc'].mean(),
            'std': df_success['test_acc'].std(),
        }

# 保存最佳配置
if best_configs:
    with open('results/comprehensive_search/best_configs_summary.txt', 'w') as f:
        f.write("MERIT最佳配置汇总\n")
        f.write("="*80 + "\n\n")
        
        for dataset, info in best_configs.items():
            f.write(f"{dataset}:\n")
            f.write(f"  配置: {info['config']}\n")
            f.write(f"  准确率: {info['acc']:.4f} ± {info['std']:.4f}\n\n")
    
    print("\n✅ 最佳配置已保存到: results/comprehensive_search/best_configs_summary.txt")

print("\n" + "="*80 + "\n")
EOF

echo ""
echo "========================================================================"
echo "🎉 全部搜索完成！"
echo "========================================================================"
echo ""
echo "结果位置:"
echo "  - 各数据集详细结果: results/comprehensive_search/<DATASET>/"
echo "  - 最佳配置汇总: results/comprehensive_search/best_configs_summary.txt"
echo ""
echo "下一步:"
echo "  1. 查看 best_configs_summary.txt"
echo "  2. 用最佳配置更新 run_<dataset>.sh"
echo "  3. 运行baseline对比实验"
echo "  4. 生成论文表格"
echo ""

