#!/usr/bin/env python3
"""
分析超参数搜索结果，找出Top5配置
支持两种模式：
1. 从CSV文件分析（如果有的话）
2. 从控制台输出分析（手动输入）
"""

import pandas as pd
import glob
import os
import sys
import re
from pathlib import Path


def analyze_from_csv(dataset, results_dir="results/param_search"):
    """从CSV文件分析结果"""
    pattern = f"{results_dir}/{dataset}/*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"❌ 未找到CSV文件: {pattern}")
        return None
    
    print(f"✅ 找到 {len(csv_files)} 个结果文件")
    print("")
    
    results = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_ok = df[df['return_code'] == 0]
            
            if len(df_ok) > 0:
                config_name = os.path.basename(csv_file).replace('.csv', '')
                
                # 提取配置参数
                lr_match = re.search(r'lr([0-9.e-]+)', config_name)
                lv_match = re.search(r'lv([0-9.]+)', config_name)
                lp_match = re.search(r'lp([0-9.]+)', config_name)
                
                results.append({
                    'config': config_name,
                    'lr': lr_match.group(1) if lr_match else 'unknown',
                    'lambda_view': lv_match.group(1) if lv_match else 'unknown',
                    'lambda_pseudo': lp_match.group(1) if lp_match else 'unknown',
                    'acc_mean': df_ok['test_acc'].mean(),
                    'acc_std': df_ok['test_acc'].std() if len(df_ok) > 1 else 0,
                    'f1_mean': df_ok['test_f1'].mean(),
                    'auroc_mean': df_ok['test_auroc'].mean(),
                    'recall_mean': df_ok['test_rec'].mean() if 'test_rec' in df_ok.columns else None,
                    'auprc_mean': df_ok['test_auprc'].mean() if 'test_auprc' in df_ok.columns else None,
                    'n_seeds': len(df_ok),
                })
        except Exception as e:
            print(f"⚠️  跳过 {csv_file}: {e}")
            continue
    
    return results


def analyze_from_output(output_file):
    """从控制台输出文件分析结果"""
    if not os.path.exists(output_file):
        print(f"❌ 文件不存在: {output_file}")
        return None
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配配置行和结果行
    config_pattern = r'Config (\d+)/27: lr=([\d.e-]+), λ_view=([\d.]+), λ_pseudo=([\d.]+)'
    result_pattern = r'Test - Acc: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+), F1: ([\d.]+), AUROC: ([\d.]+)'
    
    configs = re.finditer(config_pattern, content)
    results = []
    
    for config_match in configs:
        config_id = config_match.group(1)
        lr = config_match.group(2)
        lambda_view = config_match.group(3)
        lambda_pseudo = config_match.group(4)
        
        # 找到这个配置后面的所有结果（可能有多个seed）
        config_pos = config_match.end()
        next_config_match = re.search(r'Config \d+/27:', content[config_pos:])
        
        if next_config_match:
            section = content[config_pos:config_pos + next_config_match.start()]
        else:
            section = content[config_pos:]
        
        # 提取所有test结果
        test_results = re.finditer(result_pattern, section)
        accs = []
        f1s = []
        aurocs = []
        
        for test_match in test_results:
            accs.append(float(test_match.group(1)))
            f1s.append(float(test_match.group(4)))
            aurocs.append(float(test_match.group(5)))
        
        if accs:
            import numpy as np
            results.append({
                'config': f'config{config_id}',
                'config_id': int(config_id),
                'lr': lr,
                'lambda_view': lambda_view,
                'lambda_pseudo': lambda_pseudo,
                'acc_mean': np.mean(accs),
                'acc_std': np.std(accs) if len(accs) > 1 else 0,
                'f1_mean': np.mean(f1s),
                'auroc_mean': np.mean(aurocs),
                'n_seeds': len(accs),
            })
    
    print(f"✅ 从输出文件解析出 {len(results)} 个配置")
    return results


def decode_config_params(config_id):
    """根据配置ID反推参数（与find_best_params_fast.sh的循环顺序一致）"""
    # 配置ID从1开始
    config_idx = config_id - 1
    
    # 参数列表（与find_best_params_fast.sh保持一致）
    lr_vals = ['1e-4', '1.5e-4', '2e-4']
    lv_vals = ['0.5', '1.0', '1.5']
    lp_vals = ['0.2', '0.3', '0.5']
    
    # 反推索引（嵌套循环顺序：lr -> lambda_view -> lambda_pseudo）
    lp_idx = config_idx % 3
    lv_idx = (config_idx // 3) % 3
    lr_idx = config_idx // 9
    
    return {
        'lr': lr_vals[lr_idx],
        'lambda_view': lv_vals[lv_idx],
        'lambda_pseudo': lp_vals[lp_idx]
    }


def print_results(results, dataset, top_n=10, save_dir=None):
    """打印和保存结果"""
    if not results:
        print("❌ 没有结果可显示")
        return
    
    # 补充missing的参数（如果是unknown，尝试从config ID反推）
    for res in results:
        if res['lr'] == 'unknown' or res['lambda_view'] == 'unknown':
            # 尝试从config name提取ID
            config_name = res['config']
            match = re.search(r'config(\d+)', config_name)
            if match:
                config_id = int(match.group(1))
                params = decode_config_params(config_id)
                res['lr'] = params['lr']
                res['lambda_view'] = params['lambda_view']
                res['lambda_pseudo'] = params['lambda_pseudo']
                res['config_id'] = config_id
    
    # 按准确率排序
    results.sort(key=lambda x: x['acc_mean'], reverse=True)
    
    print("\n" + "="*100)
    print(f"超参数搜索结果分析 - {dataset}")
    print("="*100)
    print(f"{'Rank':<6} {'Config':<15} {'LR':<10} {'λ_view':<8} {'λ_pseudo':<10} {'Acc':<18} {'F1':<10} {'AUROC':<10} {'Recall':<10} {'AUPRC':<10} {'Seeds':<8}")
    print("-"*100)
    
    for i, res in enumerate(results[:top_n], 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        
        acc_str = f"{res['acc_mean']:.4f}"
        if res['n_seeds'] > 1:
            acc_str += f"±{res['acc_std']:.4f}"
        else:
            acc_str += "       "
        
        recall_str = f"{res['recall_mean']:.4f}" if res.get('recall_mean') is not None else "-"
        auprc_str = f"{res['auprc_mean']:.4f}" if res.get('auprc_mean') is not None else "-"
        print(f"{marker:<6} {res['config']:<15} {res['lr']:<10} {res['lambda_view']:<8} "
              f"{res['lambda_pseudo']:<10} {acc_str:<18} {res['f1_mean']:.4f}    "
              f"{res['auroc_mean']:.4f}    {recall_str:<10} {auprc_str:<10} {res['n_seeds']}")
    
    # 保存Top5
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置ID
        with open(f"{save_dir}/top5_configs.txt", 'w') as f:
            for res in results[:5]:
                # 提取config ID - 从config name中提取数字
                config_id = res.get('config_id')
                if not config_id:
                    match = re.search(r'config(\d+)', res['config'])
                    if match:
                        config_id = match.group(1)
                    else:
                        config_id = res['config']  # 如果无法提取，保存原名
                f.write(f"{config_id}\n")
        
        # 保存最佳配置
        best = results[0]
        with open(f"{save_dir}/best_config.txt", 'w') as f:
            f.write(f"最佳配置: {best['config']}\n")
            f.write(f"Test Acc: {best['acc_mean']:.4f}")
            if best['n_seeds'] > 1:
                f.write(f" ± {best['acc_std']:.4f}")
            f.write(f"\nTest F1: {best['f1_mean']:.4f}\n")
            f.write(f"Test AUROC: {best['auroc_mean']:.4f}\n")
            if best.get('recall_mean') is not None:
                f.write(f"Test Recall: {best['recall_mean']:.4f}\n")
            if best.get('auprc_mean') is not None:
                f.write(f"Test AUPRC: {best['auprc_mean']:.4f}\n")
            f.write(f"Seeds: {best['n_seeds']}\n\n")
            f.write("推荐参数:\n")
            f.write(f"--lr {best['lr']}\n")
            f.write(f"--lambda_view {best['lambda_view']}\n")
            f.write(f"--lambda_pseudo_loss {best['lambda_pseudo']}\n")
        
        print(f"\n✅ 结果已保存:")
        print(f"   - {save_dir}/top5_configs.txt")
        print(f"   - {save_dir}/best_config.txt")
        
        # 显示最佳配置
        print("\n" + "="*100)
        print("📝 最佳配置:")
        print("="*100)
        with open(f"{save_dir}/best_config.txt", 'r') as f:
            print(f.read())
    
    print("")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  1. 从CSV文件分析:")
        print("     python analyze_results.py <DATASET> [results_dir]")
        print("     示例: python analyze_results.py PTB")
        print("")
        print("  2. 从输出文件分析:")
        print("     python analyze_results.py --from-output <output_file> <DATASET>")
        print("     示例: python analyze_results.py --from-output ptb_output.txt PTB")
        sys.exit(1)
    
    if sys.argv[1] == "--from-output":
        if len(sys.argv) < 4:
            print("❌ 需要指定输出文件和数据集名称")
            sys.exit(1)
        
        output_file = sys.argv[2]
        dataset = sys.argv[3]
        results_dir = f"results/param_search/{dataset}"
        
        results = analyze_from_output(output_file)
        
    else:
        dataset = sys.argv[1]
        results_dir = sys.argv[2] if len(sys.argv) > 2 else "results/param_search"
        full_results_dir = f"{results_dir}/{dataset}"
        
        results = analyze_from_csv(dataset, results_dir)
    
    if results:
        print_results(results, dataset, top_n=10, save_dir=f"results/param_search/{dataset}")
    else:
        print("\n" + "="*100)
        print("💡 提示: 如果你有控制台输出日志，可以这样分析:")
        print("="*100)
        print(f"1. 将控制台输出保存到文件 (例如: ptb_output.txt)")
        print(f"2. 运行: python analyze_results.py --from-output ptb_output.txt {dataset}")
        print("")


if __name__ == "__main__":
    main()

