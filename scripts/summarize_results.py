#!/usr/bin/env python
"""
汇总所有实验结果，生成排行榜
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_results(results_dir='results/comprehensive'):
    """加载所有实验结果"""
    csv_files = glob.glob(f'{results_dir}/exp*.csv')
    
    all_results = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # 过滤成功的运行
            df_success = df[df['return_code'] == 0]
            
            if len(df_success) == 0:
                continue
            
            # 计算统计量
            stats = {
                'experiment': Path(csv_file).stem,
                'n_seeds': len(df_success),
                'val_acc_mean': df_success['val_acc'].mean(),
                'val_acc_std': df_success['val_acc'].std(),
                'val_f1_mean': df_success['val_f1'].mean(),
                'val_f1_std': df_success['val_f1'].std(),
                'val_auroc_mean': df_success['val_auroc'].mean(),
                'val_auroc_std': df_success['val_auroc'].std(),
                'test_acc_mean': df_success['test_acc'].mean(),
                'test_acc_std': df_success['test_acc'].std(),
                'test_f1_mean': df_success['test_f1'].mean(),
                'test_f1_std': df_success['test_f1'].std(),
                'test_auroc_mean': df_success['test_auroc'].mean(),
                'test_auroc_std': df_success['test_auroc'].std(),
            }
            
            all_results.append(stats)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return pd.DataFrame(all_results)

def print_rankings(df):
    """打印排行榜"""
    
    print("\n" + "="*80)
    print("实验结果排行榜 - 按 Test Accuracy 排序")
    print("="*80)
    
    # 按test_acc排序
    df_sorted = df.sort_values('test_acc_mean', ascending=False).reset_index(drop=True)
    
    print(f"\n{'排名':<4} {'实验名称':<35} {'Test Acc':<15} {'Test F1':<15} {'Test AUROC':<15}")
    print("-"*80)
    
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        exp_name = row['experiment'].replace('exp', '').replace('_', ' ')
        test_acc = f"{row['test_acc_mean']:.5f}±{row['test_acc_std']:.5f}"
        test_f1 = f"{row['test_f1_mean']:.5f}±{row['test_f1_std']:.5f}"
        test_auroc = f"{row['test_auroc_mean']:.5f}±{row['test_auroc_std']:.5f}"
        
        print(f"{rank:<4} {exp_name:<35} {test_acc:<15} {test_f1:<15} {test_auroc:<15}")
    
    print("\n" + "="*80)
    print("实验结果排行榜 - 按 Test F1 排序")
    print("="*80)
    
    # 按test_f1排序
    df_sorted = df.sort_values('test_f1_mean', ascending=False).reset_index(drop=True)
    
    print(f"\n{'排名':<4} {'实验名称':<35} {'Test F1':<15} {'Test Acc':<15} {'Test AUROC':<15}")
    print("-"*80)
    
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        exp_name = row['experiment'].replace('exp', '').replace('_', ' ')
        test_f1 = f"{row['test_f1_mean']:.5f}±{row['test_f1_std']:.5f}"
        test_acc = f"{row['test_acc_mean']:.5f}±{row['test_acc_std']:.5f}"
        test_auroc = f"{row['test_auroc_mean']:.5f}±{row['test_auroc_std']:.5f}"
        
        print(f"{rank:<4} {exp_name:<35} {test_f1:<15} {test_acc:<15} {test_auroc:<15}")
    
    print("\n" + "="*80)
    print("最稳定的配置 - 按标准差排序（越小越好）")
    print("="*80)
    
    # 按std排序
    df_sorted = df.sort_values('test_acc_std', ascending=True).reset_index(drop=True)
    
    print(f"\n{'排名':<4} {'实验名称':<35} {'Std(Acc)':<12} {'Mean(Acc)':<15}")
    print("-"*80)
    
    for idx, row in df_sorted.head(10).iterrows():
        rank = idx + 1
        exp_name = row['experiment'].replace('exp', '').replace('_', ' ')
        std = f"{row['test_acc_std']:.5f}"
        mean = f"{row['test_acc_mean']:.5f}"
        
        print(f"{rank:<4} {exp_name:<35} {std:<12} {mean:<15}")

def main():
    print("正在加载实验结果...")
    
    # 尝试两个目录
    if os.path.exists('results/comprehensive'):
        df = load_all_results('results/comprehensive')
    elif os.path.exists('results'):
        df = load_all_results('results')
    else:
        print("错误: 找不到结果目录")
        return
    
    if len(df) == 0:
        print("没有找到有效的实验结果")
        return
    
    print(f"成功加载 {len(df)} 个实验结果\n")
    
    # 打印排行榜
    print_rankings(df)
    
    # 保存详细结果
    output_file = 'results/comprehensive_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    # 找到最佳配置
    best_idx = df['test_acc_mean'].idxmax()
    best_exp = df.loc[best_idx]
    
    print("\n" + "="*80)
    print("🏆 最佳配置")
    print("="*80)
    print(f"实验: {best_exp['experiment']}")
    print(f"Test Acc: {best_exp['test_acc_mean']:.5f} ± {best_exp['test_acc_std']:.5f}")
    print(f"Test F1:  {best_exp['test_f1_mean']:.5f} ± {best_exp['test_f1_std']:.5f}")
    print(f"Test AUROC: {best_exp['test_auroc_mean']:.5f} ± {best_exp['test_auroc_std']:.5f}")
    print("="*80)

if __name__ == '__main__':
    main()

