#!/usr/bin/env python
"""
对比不同GNN配置的性能
"""

import pandas as pd
import numpy as np
import glob
import os

def load_result(csv_file):
    """加载单个结果文件"""
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df_success = df[df['return_code'] == 0]
    
    if len(df_success) == 0:
        return None
    
    return {
        'file': os.path.basename(csv_file),
        'n': len(df_success),
        'acc': df_success['test_acc'].mean(),
        'acc_std': df_success['test_acc'].std(),
        'f1': df_success['test_f1'].mean(),
        'f1_std': df_success['test_f1'].std(),
        'auroc': df_success['test_auroc'].mean(),
    }

def main():
    print("\n" + "="*80)
    print("GNN配置对比")
    print("="*80)
    
    # 加载所有结果
    results = []
    csv_files = glob.glob('results/gnn_variations/*.csv')
    
    if not csv_files:
        print("\n错误: 未找到结果文件")
        print("请确认实验已完成，文件位于: results/gnn_variations/")
        return
    
    for csv_file in csv_files:
        res = load_result(csv_file)
        if res:
            results.append(res)
    
    if not results:
        print("\n错误: 所有实验都失败了")
        return
    
    # 按准确率排序
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    print(f"\n📊 配置排名 (基于 {results[0]['n']} 个种子):")
    print("-"*80)
    print(f"{'排名':<4} {'配置':<35} {'Test Acc':<15} {'Test F1':<15}")
    print("-"*80)
    
    for i, res in enumerate(results, 1):
        config_name = res['file'].replace('.csv', '').replace('_', ' ')
        acc_str = f"{res['acc']:.4f}±{res['acc_std']:.4f}"
        f1_str = f"{res['f1']:.4f}±{res['f1_std']:.4f}"
        
        marker = "🏆" if i == 1 else f"{i}. "
        print(f"{marker:<4} {config_name:<35} {acc_str:<15} {f1_str:<15}")
    
    print("-"*80)
    
    # 最佳配置
    best = results[0]
    print(f"\n🏆 最佳配置: {best['file'].replace('.csv', '')}")
    print(f"   Test Acc: {best['acc']:.4f} ± {best['acc_std']:.4f}")
    print(f"   Test F1:  {best['f1']:.4f} ± {best['f1_std']:.4f}")
    print(f"   Test AUROC: {best['auroc']:.4f}")
    
    # 与MedGNN对比
    medgnn_acc = 0.8260
    diff = (best['acc'] - medgnn_acc) * 100
    
    print(f"\n📊 vs MedGNN (82.60%):")
    print(f"   差距: {diff:+.2f}%")
    
    if best['acc'] >= medgnn_acc:
        print("   ✅ 已超过MedGNN！建议用此配置跑完整10种子实验")
    elif best['acc'] >= medgnn_acc - 0.01:
        print("   ⚡ 非常接近！建议跑更多种子确认")
    else:
        print("   ⚠️  还有提升空间，可以继续调优")
    
    print("\n" + "="*80)
    
    # 保存排名
    df = pd.DataFrame(results)
    df.to_csv('results/gnn_variations/ranking.csv', index=False)
    print("\n排名已保存到: results/gnn_variations/ranking.csv\n")

if __name__ == '__main__':
    main()

