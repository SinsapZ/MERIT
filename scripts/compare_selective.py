#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较 EviMR vs Softmax 的 选择性预测曲线（Accuracy vs Rejection）
支持配色与自动美化，输出 PNG
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def selective_prediction(confidences, predictions, labels):
    idx = np.argsort(confidences)[::-1]
    coverages, accuracies = [], []
    for cov in np.linspace(0.1, 1.0, 20):
        n = int(len(predictions) * cov)
        if n > 0:
            acc = (labels[idx[:n]] == predictions[idx[:n]]).mean()
            coverages.append(cov * 100)
            accuracies.append(acc * 100)
    return np.array(coverages), np.array(accuracies)

def parse_palette(s):
    default = ['#e1d89c','#e1c59c','#e1ae9c','#e1909c','#4a4a4a']
    if not s: return default
    parts = ['#'+p.strip().lstrip('#') for p in s.split(',') if p.strip()]
    return parts if parts else default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, required=True, help='results/uncertainty/<DATASET>')
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--palette', type=str, default='e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a')
    args = ap.parse_args()

    colors = parse_palette(args.palette)
    evi_dir  = os.path.join(args.base_dir, 'evi')
    soft_dir = os.path.join(args.base_dir, 'softmax')

    def load_curve(d):
        cf, lf, pf = [os.path.join(d,n) for n in ('confidences.npy','labels.npy','predictions.npy')]
        if not (os.path.exists(cf) and os.path.exists(lf) and os.path.exists(pf)):
            return None, None
        c=np.load(cf); y=np.load(lf); p=np.load(pf)
        cov, acc = selective_prediction(c,p,y)
        return 100-cov, acc

    re_e, ac_e = load_curve(evi_dir)
    re_s, ac_s = load_curve(soft_dir)
    if re_e is None or re_s is None:
        print('Missing arrays under evi/ or softmax/.')
        return

    plt.figure(figsize=(8,5))
    plt.plot(re_e, ac_e, color=colors[3], marker='o', linewidth=2, label='EviMR-Net')
    plt.plot(re_s, ac_s, color=colors[4], marker='o', linestyle='--', linewidth=2, label='Softmax')
    plt.xlabel('Rejection rate (%)'); plt.ylabel('Accuracy (%)')
    plt.title(f'{args.dataset}: Accuracy vs Rejection')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    out_png = os.path.join(args.base_dir, 'acc_vs_reject_compare.png')
    out_svg = os.path.join(args.base_dir, 'acc_vs_reject_compare.svg')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    print('Saved', out_png, 'and', out_svg)

if __name__ == '__main__':
    main()


