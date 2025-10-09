#!/usr/bin/env python
"""
对比有无GNN的性能差异
"""

import pandas as pd
import numpy as np

def load_and_summarize(csv_file):
    """加载CSV并计算统计量"""
    df = pd.read_csv(csv_file)
    df_success = df[df['return_code'] == 0]
    
    if len(df_success) == 0:
        return None
    
    stats = {
        'n_seeds': len(df_success),
        'test_acc_mean': df_success['test_acc'].mean(),
        'test_acc_std': df_success['test_acc'].std(),
        'test_f1_mean': df_success['test_f1'].mean(),
        'test_f1_std': df_success['test_f1'].std(),
        'test_auroc_mean': df_success['test_auroc'].mean(),
        'test_auroc_std': df_success['test_auroc'].std(),
    }
    return stats

def main():
    print("="*80)
    print("GNN 消融实验对比")
    print("="*80)
    
    # 加载结果
    no_gnn = load_and_summarize('results/ablation/merit_no_gnn.csv')
    with_gnn = load_and_summarize('results/ablation/merit_with_gnn.csv')
    
    if no_gnn is None or with_gnn is None:
        print("错误: 无法加载实验结果")
        return
    
    # 打印对比
    print("\n📊 性能对比:")
    print("-"*80)
    print(f"{'配置':<20} {'Test Acc':<20} {'Test F1':<20} {'Test AUROC':<20}")
    print("-"*80)
    
    print(f"{'MERIT (无GNN)':<20} "
          f"{no_gnn['test_acc_mean']:.5f}±{no_gnn['test_acc_std']:.5f}    "
          f"{no_gnn['test_f1_mean']:.5f}±{no_gnn['test_f1_std']:.5f}    "
          f"{no_gnn['test_auroc_mean']:.5f}±{no_gnn['test_auroc_std']:.5f}")
    
    print(f"{'MERIT (含GNN)':<20} "
          f"{with_gnn['test_acc_mean']:.5f}±{with_gnn['test_acc_std']:.5f}    "
          f"{with_gnn['test_f1_mean']:.5f}±{with_gnn['test_f1_std']:.5f}    "
          f"{with_gnn['test_auroc_mean']:.5f}±{with_gnn['test_auroc_std']:.5f}")
    
    print("-"*80)
    
    # 计算提升
    acc_improvement = (with_gnn['test_acc_mean'] - no_gnn['test_acc_mean']) * 100
    f1_improvement = (with_gnn['test_f1_mean'] - no_gnn['test_f1_mean']) * 100
    auroc_improvement = (with_gnn['test_auroc_mean'] - no_gnn['test_auroc_mean']) * 100
    
    std_acc_change = with_gnn['test_acc_std'] - no_gnn['test_acc_std']
    
    print(f"\n📈 绝对提升:")
    print(f"  Test Acc:   +{acc_improvement:.2f}%")
    print(f"  Test F1:    +{f1_improvement:.2f}%")
    print(f"  Test AUROC: +{auroc_improvement:.2f}%")
    
    print(f"\n📉 稳定性变化 (Std):")
    print(f"  Test Acc Std: {no_gnn['test_acc_std']:.5f} → {with_gnn['test_acc_std']:.5f} "
          f"({'降低' if std_acc_change < 0 else '升高'} {abs(std_acc_change):.5f})")
    
    print("\n" + "="*80)
    
    # 与MedGNN对比
    print("\n📊 与MedGNN对比:")
    print("-"*80)
    medgnn_acc = 0.8260
    medgnn_f1 = 0.8025
    medgnn_auroc = 0.8593
    
    print(f"{'模型':<20} {'Test Acc':<15} {'Test F1':<15} {'Test AUROC':<15}")
    print("-"*80)
    print(f"{'MedGNN':<20} {medgnn_acc:.5f}      {medgnn_f1:.5f}      {medgnn_auroc:.5f}")
    print(f"{'MERIT (含GNN)':<20} {with_gnn['test_acc_mean']:.5f}      "
          f"{with_gnn['test_f1_mean']:.5f}      {with_gnn['test_auroc_mean']:.5f}")
    
    vs_medgnn_acc = (with_gnn['test_acc_mean'] - medgnn_acc) * 100
    vs_medgnn_f1 = (with_gnn['test_f1_mean'] - medgnn_f1) * 100
    
    print("-"*80)
    print(f"相对MedGNN: Acc {vs_medgnn_acc:+.2f}%, F1 {vs_medgnn_f1:+.2f}%")
    
    if with_gnn['test_acc_mean'] > medgnn_acc:
        print("\n🎉 成功！MERIT性能超过MedGNN！")
    elif with_gnn['test_acc_mean'] > medgnn_acc - 0.01:
        print("\n✅ 接近成功！MERIT与MedGNN性能相当")
    else:
        print("\n⚠️  还需要进一步调优超参数")
    
    print("="*80)

if __name__ == '__main__':
    main()

