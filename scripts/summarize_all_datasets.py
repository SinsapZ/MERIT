#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总MERIT在所有数据集上的实验结果
生成论文表格
"""
import pandas as pd
import numpy as np
import os

def load_results(csv_path):
    """加载单个数据集的结果"""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    df_success = df[df['return_code'] == 0]
    
    if len(df_success) == 0:
        return None
    
    metrics = {}
    for metric in ['test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc']:
        values = df_success[metric].dropna()
        if len(values) > 0:
            metrics[metric + '_mean'] = values.mean()
            metrics[metric + '_std'] = values.std()
        else:
            metrics[metric + '_mean'] = 0.0
            metrics[metric + '_std'] = 0.0
    
    metrics['n_seeds'] = len(df_success)
    return metrics

def format_metric(mean, std):
    """格式化指标为mean±std"""
    return f"{mean*100:.2f}±{std*100:.2f}"

def main():
    print("\n" + "="*90)
    print("MERIT - 多数据集实验结果汇总")
    print("="*90)
    
    # 数据集配置
    datasets = [
        {
            'name': 'ADFD',
            'file': 'results/final_all_datasets/adfd_results.csv',
            'desc': 'Alzheimer Detection (88 subjects, 2 classes)',
            'samples': '~8800',
        },
        {
            'name': 'APAVA',
            'file': 'results/final_all_datasets/apava_results.csv',
            'desc': 'Arrhythmia Classification (23 subjects, 9 classes)',
            'samples': '~2300',
        },
        {
            'name': 'PTB',
            'file': 'results/final_all_datasets/ptb_results.csv',
            'desc': 'Myocardial Infarction (290 subjects, 2 classes)',
            'samples': '~14000',
        },
        {
            'name': 'PTB-XL',
            'file': 'results/final_all_datasets/ptbxl_results.csv',
            'desc': 'Multi-class ECG (18885 subjects, 5 classes)',
            'samples': '~21000',
        },
    ]
    
    # 加载所有结果
    results = []
    for ds in datasets:
        res = load_results(ds['file'])
        if res:
            res['name'] = ds['name']
            res['desc'] = ds['desc']
            res['samples'] = ds['samples']
            results.append(res)
        else:
            print(f"⚠️  未找到结果: {ds['name']} ({ds['file']})")
    
    if not results:
        print("\n❌ 没有找到任何实验结果")
        print("请先运行: bash MERIT/scripts/run_all_datasets.sh")
        return
    
    # ============================================================
    # 表格1: 完整结果表（所有指标）
    # ============================================================
    print("\n" + "="*90)
    print("📊 Table 1: Complete Results on All Datasets")
    print("="*90)
    print(f"\n{'Dataset':<12} {'Accuracy':<16} {'Precision':<16} {'Recall':<16} {'F1 Score':<16} {'AUROC':<16}")
    print("-"*90)
    
    for res in results:
        acc_str = format_metric(res['test_acc_mean'], res['test_acc_std'])
        prec_str = format_metric(res['test_prec_mean'], res['test_prec_std'])
        rec_str = format_metric(res['test_rec_mean'], res['test_rec_std'])
        f1_str = format_metric(res['test_f1_mean'], res['test_f1_std'])
        auroc_str = format_metric(res['test_auroc_mean'], res['test_auroc_std'])
        
        print(f"{res['name']:<12} {acc_str:<16} {prec_str:<16} {rec_str:<16} {f1_str:<16} {auroc_str:<16}")
    
    print("-"*90)
    print(f"Note: Results are reported as Mean±Std (%) over {results[0]['n_seeds']} random seeds.")
    
    # ============================================================
    # 表格2: LaTeX格式（论文用）
    # ============================================================
    print("\n" + "="*90)
    print("📝 Table 2: LaTeX Format (for paper)")
    print("="*90)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{MERIT Performance on Multiple Datasets}")
    print("\\label{tab:all_datasets}")
    print("\\begin{tabular}{l|ccccc}")
    print("\\hline")
    print("Dataset & Accuracy & Precision & Recall & F1 Score & AUROC \\\\")
    print("\\hline")
    
    for res in results:
        acc = format_metric(res['test_acc_mean'], res['test_acc_std'])
        prec = format_metric(res['test_prec_mean'], res['test_prec_std'])
        rec = format_metric(res['test_rec_mean'], res['test_rec_std'])
        f1 = format_metric(res['test_f1_mean'], res['test_f1_std'])
        auroc = format_metric(res['test_auroc_mean'], res['test_auroc_std'])
        
        print(f"{res['name']} & {acc} & {prec} & {rec} & {f1} & {auroc} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # ============================================================
    # 表格3: 数据集统计
    # ============================================================
    print("\n" + "="*90)
    print("📊 Table 3: Dataset Statistics")
    print("="*90)
    print(f"\n{'Dataset':<12} {'#Samples':<12} {'#Classes':<10} {'Description':<50}")
    print("-"*90)
    
    for res in results:
        # 从描述中提取类别数
        if '2 classes' in res['desc']:
            n_classes = 2
        elif '9 classes' in res['desc']:
            n_classes = 9
        elif '5 classes' in res['desc']:
            n_classes = 5
        else:
            n_classes = 'N/A'
        
        print(f"{res['name']:<12} {res['samples']:<12} {n_classes:<10} {res['desc']:<50}")
    
    print("-"*90)
    
    # ============================================================
    # 分析：哪个指标最稳定
    # ============================================================
    print("\n" + "="*90)
    print("📈 Analysis: Metric Stability")
    print("="*90)
    
    avg_stds = {
        'Accuracy': np.mean([r['test_acc_std'] for r in results]),
        'Precision': np.mean([r['test_prec_std'] for r in results]),
        'Recall': np.mean([r['test_rec_std'] for r in results]),
        'F1 Score': np.mean([r['test_f1_std'] for r in results]),
        'AUROC': np.mean([r['test_auroc_std'] for r in results]),
    }
    
    print("\nAverage standard deviation across datasets:")
    for metric, std in sorted(avg_stds.items(), key=lambda x: x[1]):
        print(f"  {metric:<12}: {std*100:.2f}%  {'✅ Most stable' if std == min(avg_stds.values()) else ''}")
    
    # ============================================================
    # 保存为CSV
    # ============================================================
    output_file = 'results/final_all_datasets/summary_all_datasets.csv'
    
    df_summary = pd.DataFrame(results)
    df_summary = df_summary[['name', 'desc', 'samples', 'n_seeds',
                               'test_acc_mean', 'test_acc_std',
                               'test_prec_mean', 'test_prec_std',
                               'test_rec_mean', 'test_rec_std',
                               'test_f1_mean', 'test_f1_std',
                               'test_auroc_mean', 'test_auroc_std']]
    
    df_summary.to_csv(output_file, index=False)
    print(f"\n✅ Summary saved to: {output_file}")
    
    # ============================================================
    # 论文写作建议
    # ============================================================
    print("\n" + "="*90)
    print("📝 Paper Writing Suggestions")
    print("="*90)
    
    print("\n1. Results Section 写法:")
    print("   'We evaluate MERIT on four publicly available ECG datasets:")
    print("   ADFD, APAVA, PTB, and PTB-XL, covering different tasks (binary")
    print("   and multi-class classification) and scales (88 to 18,885 subjects).'")
    
    print("\n2. 突出重点:")
    best_acc_ds = max(results, key=lambda x: x['test_acc_mean'])
    print(f"   - Best accuracy: {best_acc_ds['name']} ({best_acc_ds['test_acc_mean']*100:.2f}%)")
    
    best_auroc_ds = max(results, key=lambda x: x['test_auroc_mean'])
    print(f"   - Best AUROC: {best_auroc_ds['name']} ({best_auroc_ds['test_auroc_mean']*100:.2f}%)")
    
    print("\n3. 泛化性论述:")
    print("   'MERIT demonstrates consistent performance across diverse datasets,")
    print("   indicating good generalization capability of our evidential fusion")
    print("   framework.'")
    
    print("\n4. 不同数据集的特点:")
    print("   - ADFD: Alzheimer detection (binary, EEG)")
    print("   - APAVA: Multi-class arrhythmia (9 classes, challenging)")
    print("   - PTB: Myocardial infarction (binary, large scale)")
    print("   - PTB-XL: Very large scale (18K+ subjects, 5 classes)")
    
    print("\n" + "="*90 + "\n")

if __name__ == '__main__':
    main()

