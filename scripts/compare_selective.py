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
    # 置信度度量：
    # native: EviMR=pred.mean(max)=max(alpha)/sum(alpha), Softmax=max prob
    # legacy1u: EviMR=1-u, Softmax=max prob（与旧图一致）
    # prob: 两者都用预测概率最大值
    # margin: p_max - p_second
    # neg_entropy: -H(p)
    ap.add_argument('--metric', type=str, default='native', choices=['native','legacy1u','prob','margin','neg_entropy'])
    args = ap.parse_args()

    colors = parse_palette(args.palette)
    evi_dir  = os.path.join(args.base_dir, 'evi')
    soft_dir = os.path.join(args.base_dir, 'softmax')

    def load_curve(d, metric='native'):
        # 必要数组
        lf, pf = [os.path.join(d,n) for n in ('labels.npy','predictions.npy')]
        if not (os.path.exists(lf) and os.path.exists(pf)):
            return None, None
        y=np.load(lf); p=np.load(pf)

        # 尝试读取概率与不确定度
        cf = os.path.join(d, 'confidences.npy')
        uf = os.path.join(d, 'uncertainties.npy')
        probf = os.path.join(d, 'probs.npy')  # 若未来补充
        c = None

        if metric == 'legacy1u':
            if not os.path.exists(uf): return None, None
            c = 1.0 - np.load(uf)
        elif metric == 'native':
            if not os.path.exists(cf): return None, None
            c = np.load(cf)
        elif metric in ('prob','margin','neg_entropy'):
            # 需要概率分布；若无，就退化为 confidences 近似
            if os.path.exists(probf):
                probs = np.load(probf)
            else:
                # 用 confidences 近似：prob=one_hot(pred)*conf + 均匀残差（弱近似）
                if not os.path.exists(cf): return None, None
                conf = np.load(cf)
                K = int(np.max(p))+1
                probs = np.ones((len(p), K), dtype=float) * ((1.0-conf)/(K-1)).reshape(-1,1)
                probs[np.arange(len(p)), p] = conf
            if metric == 'prob':
                c = probs.max(axis=1)
            elif metric == 'margin':
                sortp = np.sort(probs, axis=1)
                c = sortp[:,-1] - sortp[:,-2]
            elif metric == 'neg_entropy':
                eps=1e-12; pe=np.clip(probs,eps,1.0); H = -(pe*np.log(pe)).sum(axis=1)
                c = -H
        if c is None: return None, None
        cov, acc = selective_prediction(c,p,y)
        return 100-cov, acc

    # 对 PTB / PTB-XL 默认 legacy1u；可用 --metric 覆盖
    metric_evi = args.metric if args.metric!='native' else ('legacy1u' if args.dataset in ('PTB','PTB-XL') else 'native')
    metric_soft = 'prob' if args.metric in ('prob','margin','neg_entropy') else 'native'
    re_e, ac_e = load_curve(evi_dir, metric=metric_evi)
    re_s, ac_s = load_curve(soft_dir, metric=metric_soft)
    if re_e is None or re_s is None:
        print('Missing arrays under evi/ or softmax/.')
        return

    # simple smoothing to减少抖动（3点移动平均，不改变端点数量）
    def smooth(y):
        if y.size < 3:
            return y
        ys = y.copy()
        for i in range(1, y.size-1):
            ys[i] = (y[i-1] + y[i] + y[i+1]) / 3.0
        return ys
    ac_e_s = smooth(ac_e)
    ac_s_s = smooth(ac_s)

    plt.figure(figsize=(8,5))
    plt.plot(re_e, ac_e_s, color=colors[3], marker='o', linewidth=2, label='EviMR-Net')
    plt.plot(re_s, ac_s_s, color=colors[4], marker='o', linestyle='--', linewidth=2, label='Softmax')
    plt.xlabel('Rejection rate (%)'); plt.ylabel('Accuracy (%)')
    plt.title(f'{args.dataset}: Accuracy vs Rejection')
    plt.grid(True, alpha=0.3, linestyle=':'); plt.legend(); plt.tight_layout()
    out_png = os.path.join(args.base_dir, 'acc_vs_reject_compare.png')
    out_svg = os.path.join(args.base_dir, 'acc_vs_reject_compare.svg')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    print('Saved', out_png, 'and', out_svg)

if __name__ == '__main__':
    main()


