#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MERIT不确定性评估 - ESWA核心实验
评估ECE, Selective Prediction, Uncertainty-Error Correlation
"""
import numpy as np
import matplotlib.pyplot as plt
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='APAVA')
    parser.add_argument('--output_dir', type=str, default='results/uncertainty')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n评估 {args.dataset_name} 的不确定性...")
    
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
    # Reliability Diagram
    plt.figure(figsize=(8, 6))
    x = np.arange(len(bin_accs))
    plt.bar(x, bin_accs, alpha=0.7, label='Accuracy')
    plt.plot(x, bin_confs, 'ro-', label='Confidence', linewidth=2)
    plt.xlabel('Bin')
    plt.ylabel('Value')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'{args.dataset_name}_reliability.png'), dpi=300)
    plt.close()
    
    # Selective Prediction
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, accuracies, 'b-o', linewidth=2, label='MERIT')
    plt.axhline(y=accuracy*100, color='r', linestyle='--', label=f'All samples: {accuracy*100:.2f}%')
    plt.xlabel('Coverage (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Selective Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, f'{args.dataset_name}_selective.png'), dpi=300)
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

if __name__ == '__main__':
    main()

