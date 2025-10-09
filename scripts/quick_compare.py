#!/usr/bin/env python
"""
快速对比有无GNN的性能差异
"""

import pandas as pd
import numpy as np
import os

def load_results(csv_file):
    """加载并统计结果"""
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df_success = df[df['return_code'] == 0]
    
    if len(df_success) == 0:
        return None
    
    return {
        'n': len(df_success),
        'acc_mean': df_success['test_acc'].mean(),
        'acc_std': df_success['test_acc'].std(),
        'f1_mean': df_success['test_f1'].mean(),
        'f1_std': df_success['test_f1'].std(),
        'auroc_mean': df_success['test_auroc'].mean(),
        'auroc_std': df_success['test_auroc'].std(),
    }

def main():
    print("\n" + "="*70)
    print("快速测试结果对比")
    print("="*70)
    
    no_gnn = load_results('results/quick_test/no_gnn.csv')
    with_gnn = load_results('results/quick_test/with_gnn.csv')
    
    if no_gnn is None or with_gnn is None:
        print("\n错误: 无法加载结果文件")
        print("请确认实验已完成，文件位于:")
        print("  - results/quick_test/no_gnn.csv")
        print("  - results/quick_test/with_gnn.csv")
        return
    
    # 打印结果
    print(f"\n📊 性能对比 (基于 {no_gnn['n']} 个种子):")
    print("-"*70)
    print(f"{'配置':<25} {'Test Acc':<20} {'Test F1':<20}")
    print("-"*70)
    
    no_gnn_acc = f"{no_gnn['acc_mean']:.4f} ± {no_gnn['acc_std']:.4f}"
    no_gnn_f1 = f"{no_gnn['f1_mean']:.4f} ± {no_gnn['f1_std']:.4f}"
    with_gnn_acc = f"{with_gnn['acc_mean']:.4f} ± {with_gnn['acc_std']:.4f}"
    with_gnn_f1 = f"{with_gnn['f1_mean']:.4f} ± {with_gnn['f1_std']:.4f}"
    
    print(f"{'MERIT (无GNN)':<25} {no_gnn_acc:<20} {no_gnn_f1:<20}")
    print(f"{'MERIT (含GNN)':<25} {with_gnn_acc:<20} {with_gnn_f1:<20}")
    print("-"*70)
    
    # 计算提升
    acc_diff = (with_gnn['acc_mean'] - no_gnn['acc_mean']) * 100
    f1_diff = (with_gnn['f1_mean'] - no_gnn['f1_mean']) * 100
    auroc_diff = (with_gnn['auroc_mean'] - no_gnn['auroc_mean']) * 100
    
    print(f"\n📈 性能变化:")
    print(f"  Test Acc:   {acc_diff:+.2f}%")
    print(f"  Test F1:    {f1_diff:+.2f}%")
    print(f"  Test AUROC: {auroc_diff:+.2f}%")
    
    # 稳定性
    std_diff = (with_gnn['acc_std'] - no_gnn['acc_std']) * 100
    print(f"\n📉 稳定性变化 (标准差):")
    print(f"  无GNN: {no_gnn['acc_std']:.4f}")
    print(f"  含GNN: {with_gnn['acc_std']:.4f}")
    print(f"  变化:  {std_diff:+.4f} ({'更稳定' if std_diff < 0 else '略不稳定'})")
    
    # 与MedGNN对比
    print("\n" + "="*70)
    print("与MedGNN对比")
    print("="*70)
    
    medgnn_acc = 0.8260
    medgnn_f1 = 0.8025
    medgnn_std = 0.0035
    
    print(f"\n{'模型':<25} {'Test Acc':<20} {'Test F1':<20}")
    print("-"*70)
    print(f"{'MedGNN (baseline)':<25} {medgnn_acc:.4f} ± {medgnn_std:.4f}    {medgnn_f1:.4f}")
    print(f"{'MERIT (含GNN)':<25} {with_gnn['acc_mean']:.4f} ± {with_gnn['acc_std']:.4f}    {with_gnn['f1_mean']:.4f}")
    print("-"*70)
    
    vs_medgnn_acc = (with_gnn['acc_mean'] - medgnn_acc) * 100
    vs_medgnn_f1 = (with_gnn['f1_mean'] - medgnn_f1) * 100
    
    print(f"\n相对MedGNN: Acc {vs_medgnn_acc:+.2f}%, F1 {vs_medgnn_f1:+.2f}%")
    
    # 结论
    print("\n" + "="*70)
    print("🎯 结论")
    print("="*70)
    
    if with_gnn['acc_mean'] > medgnn_acc:
        print("✅ 成功！MERIT性能超过MedGNN，可以发论文！")
        print("   建议：跑完整10个种子确认结果稳定性")
    elif with_gnn['acc_mean'] > medgnn_acc - 0.005:
        print("✅ 接近成功！MERIT与MedGNN性能相当")
        print("   建议：")
        print("   1. 跑更多种子看平均值")
        print("   2. 或微调超参数（lr, annealing_epoch等）")
    elif with_gnn['acc_mean'] > no_gnn['acc_mean']:
        print("⚠️  GNN有帮助，但还未超过MedGNN")
        print("   建议：")
        print("   1. 针对GNN版本重新调整超参数")
        print("   2. 尝试不同的nodedim (10 → 12 或 15)")
        print("   3. 尝试调整lambda权重")
    else:
        print("❌ GNN似乎没有帮助，需要检查：")
        print("   1. GNN是否正确集成")
        print("   2. 是否需要调整GNN相关参数")
    
    print("="*70 + "\n")
    
    # 保存摘要
    summary = {
        'no_gnn_acc': no_gnn['acc_mean'],
        'no_gnn_std': no_gnn['acc_std'],
        'with_gnn_acc': with_gnn['acc_mean'],
        'with_gnn_std': with_gnn['acc_std'],
        'improvement': acc_diff,
        'vs_medgnn': vs_medgnn_acc,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/quick_test/summary.csv', index=False)
    print("详细摘要已保存到: results/quick_test/summary.csv")

if __name__ == '__main__':
    main()

