#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MERIT不确定性评估 - ESWA核心实验
评估ECE, Selective Prediction, Uncertainty-Error Correlation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
import os

def compute_ece(confidences, predictions, labels, n_bins=15):
    """计算Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accuracies = (predictions == labels).astype(float)
    
    ece = 0.0
    bin_accs, bin_confs = [], []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += np.abs(conf - acc) * in_bin.mean()
            bin_accs.append(acc)
            bin_confs.append(conf)
    
    return ece, np.array(bin_accs), np.array(bin_confs)

def selective_prediction(confidences, predictions, labels):
    """计算选择性预测"""
    sorted_idx = np.argsort(confidences)[::-1]
    coverages, accuracies = [], []
    
    for cov in np.linspace(0.1, 1.0, 20):
        n = int(len(predictions) * cov)
        if n > 0:
            acc = accuracy_score(labels[sorted_idx[:n]], predictions[sorted_idx[:n]])
            coverages.append(cov * 100)
            accuracies.append(acc * 100)
    
    return np.array(coverages), np.array(accuracies)

def parse_palette(arg_str: str):
    default = ['e1d89c','e1c59c','e1ae9c','e1909c','4a4a4a']
    if not arg_str:
        return ['#'+h for h in default]
    parts = [p.strip().lstrip('#') for p in arg_str.split(',') if p.strip()]
    if not parts:
        parts = default
    return ['#'+h for h in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='APAVA')
    parser.add_argument('--output_dir', type=str, default='results/uncertainty')
    parser.add_argument('--palette', type=str, default='e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a')
    parser.add_argument('--reject_rate', type=float, default=20.0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n评估 {args.dataset_name} 的不确定性...")
    colors = parse_palette(args.palette)
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 12
    
    # 加载数据
    try:
        uncertainties = np.load(os.path.join(args.uncertainty_dir, 'uncertainties.npy'))
        confidences = np.load(os.path.join(args.uncertainty_dir, 'confidences.npy'))
        predictions = np.load(os.path.join(args.uncertainty_dir, 'predictions.npy'))
        labels = np.load(os.path.join(args.uncertainty_dir, 'labels.npy'))
    except:
        print("❌ 未找到不确定性数据文件")
        print("需要修改exp_classification.py保存这些数据")
        return
    
    errors = (predictions != labels).astype(float)
    accuracy = 1.0 - errors.mean()
    
    # 1. ECE
    ece, bin_accs, bin_confs = compute_ece(confidences, predictions, labels)
    print(f"\nECE: {ece:.4f}")
    
    # 2. Selective Prediction
    coverages, accuracies = selective_prediction(confidences, predictions, labels)
    
    print(f"\nSelective Prediction:")
    for cov, acc in zip(coverages[::4], accuracies[::4]):
        gain = acc - accuracy * 100
        print(f"  {cov:>5.0f}% coverage: {acc:>6.2f}% ({gain:>+5.2f}%)")
    
    # 3. Correlation
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    print(f"\nUncertainty-Error Correlation: {corr:.4f}")
    
    # 4. 绘图
    # Reliability Diagram（美化配色）
    plt.figure(figsize=(8, 6))
    x = np.arange(len(bin_accs))
    plt.bar(x, bin_accs, color=colors[0], alpha=0.85, label='Accuracy')
    plt.plot(x, bin_confs, color=colors[4], marker='o', label='Confidence', linewidth=2)
    plt.xlabel('Bin')
    plt.ylabel('Value')
    plt.title('Reliability Diagram')
    plt.legend()
    out_png = os.path.join(args.output_dir, f'{args.dataset_name}_reliability.png')
    out_svg = os.path.join(args.output_dir, f'{args.dataset_name}_reliability.svg')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    
    # Selective Prediction（美化配色）
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, accuracies, color=colors[3], marker='o', linewidth=2, label='MERIT')
    plt.axhline(y=accuracy*100, color=colors[4], linestyle='--', label=f'All: {accuracy*100:.2f}%')
    plt.xlabel('Coverage (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Selective Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = os.path.join(args.output_dir, f'{args.dataset_name}_selective.png')
    out_svg = os.path.join(args.output_dir, f'{args.dataset_name}_selective.svg')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    
    # 保存LaTeX表格
    latex_file = os.path.join(args.output_dir, 'selective_prediction_table.txt')
    with open(latex_file, 'w') as f:
        f.write("\\begin{tabular}{ccc}\n\\hline\n")
        f.write("Coverage & Accuracy & Gain \\\\\n\\hline\n")
        for cov, acc in zip(coverages[::4], accuracies[::4]):
            gain = acc - accuracy * 100
            f.write(f"{cov:.0f}\\% & {acc:.2f}\\% & {gain:+.2f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    
    print(f"LaTeX表格: {latex_file}\n")

    # 5. 人机协同 triage 摘要与候选清单
    rr = float(args.reject_rate)
    rr = min(max(rr, 0.0), 100.0)
    cov_target = 100.0 - rr
    idx = (np.abs(coverages - cov_target)).argmin()
    # 阈值：按置信度从高到低排序后第k个
    k = max(1, int(len(confidences) * cov_target/100.0))
    thr = np.sort(confidences)[::-1][k-1]
    kept = confidences >= thr
    acc_after = (predictions[kept] == labels[kept]).mean() * 100.0
    summary = os.path.join(args.output_dir, 'triage_summary.txt')
    with open(summary, 'w') as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Overall accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Reject rate: {rr:.1f}% (keep {cov_target:.1f}% samples)\n")
        f.write(f"Accuracy after triage: {acc_after:.2f}% (gain {acc_after - accuracy*100:+.2f}%)\n")
        f.write(f"Confidence threshold: {thr:.6f}\n")
        f.write(f"Kept/Total: {kept.sum()}/{len(kept)}\n")
    import csv
    cand_path = os.path.join(args.output_dir, 'triage_candidates.csv')
    order = np.argsort(confidences)  # 从低到高，最不自信优先
    with open(cand_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index','label','prediction','uncertainty(approx)','confidence'])
        for i in order[: max(1, int(len(confidences)*rr/100.0)) ]:
            w.writerow([int(i), int(labels[i]), int(predictions[i]), float(1.0 - confidences[i]), float(confidences[i])])
    print(f"Triage summary: {summary}")
    print(f"Triage candidates: {cand_path}")

if __name__ == '__main__':
    main()

