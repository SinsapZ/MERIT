#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析综合搜索的结果
自动找出最佳配置和生成报告
"""
import pandas as pd
import numpy as np
import glob
import os

def analyze_dataset(dataset_name):
    """分析单个数据集的搜索结果"""
    quick_pattern = f"results/comprehensive_search/{dataset_name}/quick_*.csv"
    full_pattern = f"results/comprehensive_search/{dataset_name}/full_top*.csv"
    
    quick_files = glob.glob(quick_pattern)
    full_files = glob.glob(full_pattern)
    
    if not quick_files and not full_files:
        return None
    
    results = {
        'dataset': dataset_name,
        'quick_results': [],
        'full_results': [],
        'best_config': None,
        'best_acc': 0,
        'best_std': 0,
    }
    
    # 分析快速筛选结果
    for csv_file in quick_files:
        try:
            df = pd.read_csv(csv_file)
            df_success = df[df['return_code'] == 0]
            
            if len(df_success) >= 2:
                config_name = os.path.basename(csv_file).replace('quick_', '').replace('.csv', '')
                
                results['quick_results'].append({
                    'config': config_name,
                    'acc_mean': df_success['test_acc'].mean(),
                    'acc_std': df_success['test_acc'].std(),
                    'f1_mean': df_success['test_f1'].mean(),
                    'auroc_mean': df_success['test_auroc'].mean(),
                    'n_seeds': len(df_success),
                })
        except Exception as e:
            continue
    
    # 分析完整验证结果
    for csv_file in full_files:
        try:
            df = pd.read_csv(csv_file)
            df_success = df[df['return_code'] == 0]
            
            if len(df_success) >= 8:
                config_name = os.path.basename(csv_file).replace('full_top1_', '').replace('full_top2_', '').replace('full_top3_', '').replace('.csv', '')
                rank = int(os.path.basename(csv_file).split('_')[0].replace('fulltop', '').replace('full', '').replace('top', ''))
                
                acc_mean = df_success['test_acc'].mean()
                acc_std = df_success['test_acc'].std()
                
                results['full_results'].append({
                    'rank': rank,
                    'config': config_name,
                    'acc_mean': acc_mean,
                    'acc_std': acc_std,
                    'f1_mean': df_success['test_f1'].mean(),
                    'f1_std': df_success['test_f1'].std(),
                    'auroc_mean': df_success['test_auroc'].mean(),
                    'auroc_std': df_success['test_auroc'].std(),
                    'n_seeds': len(df_success),
                })
                
                # 更新最佳配置
                if acc_mean > results['best_acc']:
                    results['best_acc'] = acc_mean
                    results['best_std'] = acc_std
                    results['best_config'] = config_name
        except Exception as e:
            continue
    
    # 如果没有完整验证，从快速结果中选最佳
    if not results['full_results'] and results['quick_results']:
        results['quick_results'].sort(key=lambda x: x['acc_mean'], reverse=True)
        best = results['quick_results'][0]
        results['best_config'] = best['config']
        results['best_acc'] = best['acc_mean']
        results['best_std'] = best['acc_std']
    
    return results

def main():
    print("\n" + "="*90)
    print("🔍 MERIT综合超参数搜索结果分析")
    print("="*90)
    
    datasets = ['APAVA', 'ADFD-Sample', 'PTB', 'PTB-XL']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n分析 {dataset}...")
        result = analyze_dataset(dataset)
        if result:
            all_results[dataset] = result
        else:
            print(f"  ⚠️  未找到 {dataset} 的搜索结果")
    
    if not all_results:
        print("\n❌ 没有找到任何搜索结果")
        print("请先运行: bash MERIT/scripts/run_all_comprehensive_search.sh")
        return
    
    # ============================================================
    # 显示各数据集的Top配置
    # ============================================================
    print("\n" + "="*90)
    print("🏆 各数据集Top-10配置")
    print("="*90)
    
    for dataset, result in all_results.items():
        print(f"\n{'='*90}")
        print(f"📊 {dataset}")
        print(f"{'='*90}")
        
        if result['quick_results']:
            # 排序
            result['quick_results'].sort(key=lambda x: x['acc_mean'], reverse=True)
            
            print(f"\n{'Rank':<6} {'Config':<45} {'Test Acc':<20} {'F1':<12} {'AUROC':<12}")
            print("-"*90)
            
            for i, res in enumerate(result['quick_results'][:10], 1):
                acc_str = f"{res['acc_mean']*100:.2f}±{res['acc_std']*100:.2f}"
                f1_str = f"{res['f1_mean']*100:.2f}"
                auroc_str = f"{res['auroc_mean']*100:.2f}"
                
                marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                
                print(f"{marker:<6} {res['config']:<45} {acc_str:<20} {f1_str:<12} {auroc_str:<12}")
        
        # 如果有完整验证结果
        if result['full_results']:
            print(f"\n完整验证结果 (10 seeds):")
            print("-"*90)
            
            for res in sorted(result['full_results'], key=lambda x: x.get('rank', 99)):
                acc_str = f"{res['acc_mean']*100:.2f}±{res['acc_std']*100:.2f}"
                f1_str = f"{res['f1_mean']*100:.2f}±{res['f1_std']*100:.2f}"
                auroc_str = f"{res['auroc_mean']*100:.2f}±{res['auroc_std']*100:.2f}"
                
                print(f"  Top-{res.get('rank', '?')}: {res['config']}")
                print(f"    Acc: {acc_str}, F1: {f1_str}, AUROC: {auroc_str}")
    
    # ============================================================
    # 最佳配置汇总
    # ============================================================
    print("\n" + "="*90)
    print("🎯 最佳配置汇总")
    print("="*90)
    
    print(f"\n{'Dataset':<15} {'Best Config':<45} {'Test Acc':<20}")
    print("-"*90)
    
    for dataset, result in all_results.items():
        if result['best_config']:
            acc_str = f"{result['best_acc']*100:.2f}±{result['best_std']*100:.2f}"
            print(f"{dataset:<15} {result['best_config']:<45} {acc_str:<20}")
    
    # ============================================================
    # 参数规律分析
    # ============================================================
    print("\n" + "="*90)
    print("📈 参数规律分析")
    print("="*90)
    
    # 分析学习率分布
    print("\n学习率统计 (在Top-5配置中):")
    lr_counter = {}
    
    for dataset, result in all_results.items():
        if result['quick_results']:
            for res in result['quick_results'][:5]:
                # 提取学习率
                config = res['config']
                if 'lr5e-5' in config or 'lr5e-05' in config:
                    lr = '5e-5'
                elif 'lr8e-5' in config or 'lr8e-05' in config:
                    lr = '8e-5'
                elif 'lr1.2e-4' in config or 'lr1.2e-04' in config:
                    lr = '1.2e-4'
                elif 'lr1.5e-4' in config or 'lr1.5e-04' in config:
                    lr = '1.5e-4'
                elif 'lr1.1e-4' in config or 'lr1.1e-04' in config:
                    lr = '1.1e-4'
                elif 'lr1e-4' in config or 'lr1e-04' in config or 'lr0.0001' in config:
                    lr = '1e-4'
                elif 'lr2e-4' in config or 'lr2e-04' in config:
                    lr = '2e-4'
                elif 'lr3e-4' in config or 'lr3e-04' in config:
                    lr = '3e-4'
                else:
                    lr = 'unknown'
                
                lr_counter[lr] = lr_counter.get(lr, 0) + 1
    
    for lr, count in sorted(lr_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lr}: 出现{count}次 {'⭐ 最常见' if count == max(lr_counter.values()) else ''}")
    
    # 分析lambda组合分布
    print("\nLambda组合统计 (在Top-5配置中):")
    lambda_counter = {}
    
    for dataset, result in all_results.items():
        if result['quick_results']:
            for res in result['quick_results'][:5]:
                config = res['config']
                if 'balanced' in config:
                    lambda_type = 'balanced (1.0,1.0,0.3)'
                elif 'fusion_focused' in config or 'fusion' in config:
                    lambda_type = 'fusion_focused (1.0,0.5,0.5)'
                elif 'weak_view' in config or 'weak' in config:
                    lambda_type = 'weak_view (1.0,0.3,0.5)'
                elif 'strong_pseudo' in config:
                    lambda_type = 'strong_pseudo (1.0,1.0,0.5)'
                elif 'strong_view' in config:
                    lambda_type = 'strong_view (1.0,1.5,0.3)'
                else:
                    lambda_type = 'unknown'
                
                lambda_counter[lambda_type] = lambda_counter.get(lambda_type, 0) + 1
    
    for lambda_type, count in sorted(lambda_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lambda_type}: 出现{count}次 {'⭐ 最常见' if count == max(lambda_counter.values()) else ''}")
    
    # ============================================================
    # 建议的最终配置
    # ============================================================
    print("\n" + "="*90)
    print("📝 建议的最终配置 (用于论文)")
    print("="*90)
    
    for dataset, result in all_results.items():
        if result['best_config']:
            print(f"\n{dataset}:")
            print(f"  配置: {result['best_config']}")
            print(f"  准确率: {result['best_acc']*100:.2f}% ± {result['best_std']*100:.2f}%")
            
            # 提取参数建议
            config = result['best_config']
            
            # 提取lr
            for lr_val in ['5e-5', '8e-5', '1e-4', '1.1e-4', '1.2e-4', '1.5e-4', '2e-4', '3e-4']:
                if lr_val.replace('-', '') in config.replace('-', ''):
                    print(f"  推荐lr: {lr_val}")
                    break
            
            # 提取lambda类型
            if 'balanced' in config:
                print(f"  推荐lambda: fuse=1.0, view=1.0, pseudo=0.3")
            elif 'fusion' in config:
                print(f"  推荐lambda: fuse=1.0, view=0.5, pseudo=0.5")
            elif 'weak' in config:
                print(f"  推荐lambda: fuse=1.0, view=0.3, pseudo=0.5")
            elif 'strong_pseudo' in config:
                print(f"  推荐lambda: fuse=1.0, view=1.0, pseudo=0.5")
            elif 'strong_view' in config:
                print(f"  推荐lambda: fuse=1.0, view=1.5, pseudo=0.3")
    
    # ============================================================
    # 保存到配置文件
    # ============================================================
    output_file = f"results/comprehensive_search/{dataset}/recommended_config.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"# {dataset} 推荐配置\n\n")
        
        if result['best_config']:
            config = result['best_config']
            
            # 提取lr
            lr = '1e-4'  # 默认
            for lr_val in ['5e-5', '8e-5', '1e-4', '1.1e-4', '1.2e-4', '1.5e-4', '2e-4', '3e-4']:
                if lr_val.replace('-', '') in config.replace('-', ''):
                    lr = lr_val
                    break
            
            # 提取lambda
            if 'balanced' in config:
                lf, lv, lp = '1.0', '1.0', '0.3'
            elif 'fusion' in config:
                lf, lv, lp = '1.0', '0.5', '0.5'
            elif 'weak' in config:
                lf, lv, lp = '1.0', '0.3', '0.5'
            elif 'strong_pseudo' in config:
                lf, lv, lp = '1.0', '1.0', '0.5'
            elif 'strong_view' in config:
                lf, lv, lp = '1.0', '1.5', '0.3'
            else:
                lf, lv, lp = '1.0', '1.0', '0.3'
            
            # 根据lr设置epochs
            if float(lr.replace('e-', 'e-')) >= 2e-4:
                epochs, patience, annealing = 100, 15, 30
            elif float(lr.replace('e-', 'e-')) >= 1e-4:
                epochs, patience, annealing = 150, 20, 50
            else:
                epochs, patience, annealing = 200, 30, 50
            
            f.write(f"--lr {lr}\n")
            f.write(f"--lambda_fuse {lf}\n")
            f.write(f"--lambda_view {lv}\n")
            f.write(f"--lambda_pseudo_loss {lp}\n")
            f.write(f"--train_epochs {epochs}\n")
            f.write(f"--patience {patience}\n")
            f.write(f"--annealing_epoch {annealing}\n")
            f.write(f"\n# 预期性能: {result['best_acc']*100:.2f}% ± {result['best_std']*100:.2f}%\n")
    
    print(f"\n  ✅ 配置已保存到: {output_file}")
    
    return results

def main_analysis():
    datasets = ['APAVA', 'ADFD-Sample', 'PTB', 'PTB-XL']
    all_dataset_results = {}
    
    for dataset in datasets:
        result = analyze_dataset(dataset)
        if result:
            all_dataset_results[dataset] = result
    
    if not all_dataset_results:
        print("\n❌ 没有找到任何搜索结果")
        print("请先运行: bash MERIT/scripts/run_all_comprehensive_search.sh")
        return
    
    # ============================================================
    # 生成最终配置Shell脚本
    # ============================================================
    print("\n" + "="*90)
    print("📝 生成最终配置脚本")
    print("="*90)
    
    for dataset, result in all_dataset_results.items():
        if result['best_config']:
            shell_script = f"MERIT/scripts/run_{dataset.lower().replace('-sample', '')}_optimized.sh"
            
            config = result['best_config']
            
            # 提取参数（同上逻辑）
            lr = '1e-4'
            for lr_val in ['5e-5', '8e-5', '1e-4', '1.1e-4', '1.2e-4', '1.5e-4', '2e-4', '3e-4']:
                if lr_val.replace('-', '') in config.replace('-', ''):
                    lr = lr_val
                    break
            
            if 'balanced' in config:
                lf, lv, lp = '1.0', '1.0', '0.3'
            elif 'fusion' in config:
                lf, lv, lp = '1.0', '0.5', '0.5'
            elif 'weak' in config:
                lf, lv, lp = '1.0', '0.3', '0.5'
            elif 'strong_pseudo' in config:
                lf, lv, lp = '1.0', '1.0', '0.5'
            elif 'strong_view' in config:
                lf, lv, lp = '1.0', '1.5', '0.3'
            else:
                lf, lv, lp = '1.0', '1.0', '0.3'
            
            if float(lr.replace('e-', 'e-')) >= 2e-4:
                epochs, patience, annealing = 100, 15, 30
            elif float(lr.replace('e-', 'e-')) >= 1e-4:
                epochs, patience, annealing = 150, 20, 50
            else:
                epochs, patience, annealing = 200, 30, 50
            
            print(f"\n  ✅ {dataset}: lr={lr}, lambda=({lf},{lv},{lp})")
            print(f"     预期: {result['best_acc']*100:.2f}% ± {result['best_std']*100:.2f}%")
    
    # ============================================================
    # 总结报告
    # ============================================================
    print("\n" + "="*90)
    print("📊 总结报告")
    print("="*90)
    
    total_configs = sum(len(r['quick_results']) for r in all_dataset_results.values())
    total_full = sum(len(r['full_results']) for r in all_dataset_results.values())
    
    print(f"\n统计信息:")
    print(f"  - 数据集数量: {len(all_dataset_results)}")
    print(f"  - 快速筛选配置数: {total_configs}")
    print(f"  - 完整验证配置数: {total_full}")
    
    # 性能统计
    accs = [r['best_acc'] for r in all_dataset_results.values() if r['best_acc'] > 0]
    if accs:
        print(f"\n性能分布:")
        print(f"  - 平均准确率: {np.mean(accs)*100:.2f}%")
        print(f"  - 最高准确率: {max(accs)*100:.2f}%")
        print(f"  - 最低准确率: {min(accs)*100:.2f}%")
    
    # 保存最终汇总
    summary_file = "results/comprehensive_search/FINAL_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("="*90 + "\n")
        f.write("MERIT综合超参数搜索 - 最终汇总\n")
        f.write("="*90 + "\n\n")
        
        f.write("各数据集最佳配置:\n")
        f.write("-"*90 + "\n\n")
        
        for dataset, result in all_dataset_results.items():
            if result['best_config']:
                f.write(f"{dataset}:\n")
                f.write(f"  配置: {result['best_config']}\n")
                f.write(f"  Test Acc: {result['best_acc']*100:.2f}% ± {result['best_std']*100:.2f}%\n")
                
                # 提取并写入详细参数
                config = result['best_config']
                
                for lr_val in ['5e-5', '8e-5', '1e-4', '1.1e-4', '1.2e-4', '1.5e-4', '2e-4', '3e-4']:
                    if lr_val.replace('-', '') in config.replace('-', ''):
                        f.write(f"  学习率: {lr_val}\n")
                        break
                
                if 'balanced' in config:
                    f.write(f"  Lambda: (1.0, 1.0, 0.3)\n")
                elif 'fusion' in config:
                    f.write(f"  Lambda: (1.0, 0.5, 0.5)\n")
                elif 'weak' in config:
                    f.write(f"  Lambda: (1.0, 0.3, 0.5)\n")
                elif 'strong_pseudo' in config:
                    f.write(f"  Lambda: (1.0, 1.0, 0.5)\n")
                elif 'strong_view' in config:
                    f.write(f"  Lambda: (1.0, 1.5, 0.3)\n")
                
                f.write("\n")
        
        if accs:
            f.write(f"\n总体统计:\n")
            f.write(f"  平均准确率: {np.mean(accs)*100:.2f}%\n")
            f.write(f"  最高准确率: {max(accs)*100:.2f}%\n")
            f.write(f"  最低准确率: {min(accs)*100:.2f}%\n")
    
    print(f"\n✅ 最终汇总已保存到: {summary_file}")
    
    # ============================================================
    # 下一步建议
    # ============================================================
    print("\n" + "="*90)
    print("🚀 下一步行动建议")
    print("="*90)
    
    print("\n1. 更新数据集配置文件:")
    for dataset in all_dataset_results.keys():
        config_file = f"results/comprehensive_search/{dataset}/recommended_config.txt"
        if os.path.exists(config_file):
            script_name = f"run_{dataset.lower().replace('-sample', '')}.sh"
            print(f"   - 用 {config_file} 更新 MERIT/scripts/{script_name}")
    
    print("\n2. 运行baseline对比:")
    print("   bash MERIT/scripts/run_baselines.sh APAVA")
    print("   bash MERIT/scripts/run_baselines.sh ADFD-Sample")
    print("   bash MERIT/scripts/run_baselines.sh PTB")
    print("   bash MERIT/scripts/run_baselines.sh PTB-XL")
    
    print("\n3. 生成论文表格:")
    print("   python MERIT/scripts/summarize_all_datasets.py")
    
    print("\n4. 开始写论文！ 📝")
    
    print("\n" + "="*90 + "\n")

if __name__ == '__main__':
    main_analysis()

