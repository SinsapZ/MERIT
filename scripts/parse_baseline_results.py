#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析baseline模型的结果
从输出文本中提取指标
"""
import re
import numpy as np
import sys
import os

def parse_results_from_file(filepath):
    """从输出文件中解析结果"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 匹配模式：Validation results --- Loss: X, Accuracy: X, Precision: X, Recall: X, F1: X, AUROC: X, AUPRC: X
    pattern = r"(Validation|Test) results --- Loss: ([0-9\.]+), Accuracy: ([0-9\.]+), Precision: ([0-9\.]+), Recall: ([0-9\.]+), F1: ([0-9\.]+), AUROC: ([0-9\.]+), AUPRC: ([0-9\.]+)"
    
    results = {'val': [], 'test': []}
    for match in re.finditer(pattern, content):
        kind, loss, acc, prec, rec, f1, auroc, auprc = match.groups()
        target = 'val' if kind == 'Validation' else 'test'
        results[target].append({
            'loss': float(loss),
            'acc': float(acc),
            'prec': float(prec),
            'rec': float(rec),
            'f1': float(f1),
            'auroc': float(auroc),
            'auprc': float(auprc),
        })
    
    # 计算平均和标准差
    summary = {}
    for split in ['val', 'test']:
        if results[split]:
            for metric in ['acc', 'prec', 'rec', 'f1', 'auroc']:
                values = [r[metric] for r in results[split]]
                summary[f'{split}_{metric}_mean'] = np.mean(values)
                summary[f'{split}_{metric}_std'] = np.std(values)
        else:
            for metric in ['acc', 'prec', 'rec', 'f1', 'auroc']:
                summary[f'{split}_{metric}_mean'] = 0.0
                summary[f'{split}_{metric}_std'] = 0.0
    
    summary['n_seeds'] = len(results['test'])
    return summary

def format_result(mean, std):
    """格式化结果为百分比"""
    return f"{mean*100:.2f}±{std*100:.2f}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_baseline_results.py <DATASET>")
        print("Example: python parse_baseline_results.py APAVA")
        sys.exit(1)
    
    dataset = sys.argv[1]
    results_dir = f"results/baselines/{dataset}"
    
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        print(f"请先运行: bash MERIT/scripts/run_baselines.sh {dataset}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"Baseline Results on {dataset}")
    print("="*80)
    
    # 定义baseline模型
    baselines = [
        ('Medformer', 'medformer_results.txt'),
        ('iTransformer', 'itransformer_results.txt'),
    ]
    
    all_results = []
    
    for model_name, filename in baselines:
        filepath = os.path.join(results_dir, filename)
        res = parse_results_from_file(filepath)
        
        if res and res['n_seeds'] > 0:
            print(f"\n✅ {model_name}:")
            print(f"   Seeds: {res['n_seeds']}/10")
            print(f"   Test Acc:    {format_result(res['test_acc_mean'], res['test_acc_std'])}")
            print(f"   Test Prec:   {format_result(res['test_prec_mean'], res['test_prec_std'])}")
            print(f"   Test Rec:    {format_result(res['test_rec_mean'], res['test_rec_std'])}")
            print(f"   Test F1:     {format_result(res['test_f1_mean'], res['test_f1_std'])}")
            print(f"   Test AUROC:  {format_result(res['test_auroc_mean'], res['test_auroc_std'])}")
            
            all_results.append({
                'model': model_name,
                **res
            })
        else:
            print(f"\n⚠️  {model_name}: 未找到结果或全部失败")
    
    # 生成对比表格
    if all_results:
        print("\n" + "="*80)
        print("📊 Comparison Table")
        print("="*80)
        print(f"\n{'Model':<15} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1 Score':<18} {'AUROC':<18}")
        print("-"*115)
        
        for res in all_results:
            acc = format_result(res['test_acc_mean'], res['test_acc_std'])
            prec = format_result(res['test_prec_mean'], res['test_prec_std'])
            rec = format_result(res['test_rec_mean'], res['test_rec_std'])
            f1 = format_result(res['test_f1_mean'], res['test_f1_std'])
            auroc = format_result(res['test_auroc_mean'], res['test_auroc_std'])
            
            print(f"{res['model']:<15} {acc:<18} {prec:<18} {rec:<18} {f1:<18} {auroc:<18}")
        
        # LaTeX格式
        print("\n" + "="*80)
        print("📝 LaTeX Format")
        print("="*80)
        print()
        for res in all_results:
            acc = format_result(res['test_acc_mean'], res['test_acc_std'])
            prec = format_result(res['test_prec_mean'], res['test_prec_std'])
            rec = format_result(res['test_rec_mean'], res['test_rec_std'])
            f1 = format_result(res['test_f1_mean'], res['test_f1_std'])
            auroc = format_result(res['test_auroc_mean'], res['test_auroc_std'])
            
            print(f"{res['model']} & {acc} & {prec} & {rec} & {f1} & {auroc} \\\\")
    
    print("\n" + "="*80)
    print("💡 提示:")
    print("  - MedGNN的结果可以从论文中直接引用")
    print("  - 在APAVA上: MedGNN达到82.60%±0.35%")
    print("  - 你的MERIT结果: 运行 bash MERIT/scripts/run_apava.sh")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

