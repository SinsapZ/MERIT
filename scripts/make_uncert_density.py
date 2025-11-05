#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成不确定度分布图（KDE + Violin）与分离度报告。
输入：results/uncertainty/<DATASET>/evi/{uncertainties.npy,labels.npy,predictions.npy}
输出：
  - uncert_density_evi_kde.(png|svg)
  - uncert_density_evi_violin.(png|svg)
  - uncert_separation.txt
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, required=True, help='results/uncertainty/<DATASET>')
    ap.add_argument('--dataset', type=str, required=True)
    args = ap.parse_args()

    evi = os.path.join(args.base_dir, 'evi')
    uf = os.path.join(evi, 'uncertainties.npy')
    lf = os.path.join(evi, 'labels.npy')
    pf = os.path.join(evi, 'predictions.npy')
    if not (os.path.exists(uf) and os.path.exists(lf) and os.path.exists(pf)):
        print('[Skip] missing arrays under', evi)
        return

    u = np.load(uf); y = np.load(lf); p = np.load(pf)
    err = (p != y)

    # 根据数据自适应截断上限：默认聚焦 0-0.05，如分布整体更大则放宽到99.5分位的1.05倍
    zoom_default = 0.05
    try:
        p995 = float(np.percentile(u, 99.5))
    except Exception:
        p995 = float(np.max(u))
    clip_hi = zoom_default if p995 <= zoom_default else float(min(np.max(u) * 1.05, p995 * 1.05))
    clip_hi = max(clip_hi, zoom_default)  # 至少保持默认显示范围
    print(f"[{args.dataset}] using uncertainty upper bound {clip_hi:.4f} for zoomed plots (p99.5={p995:.4f})")
    sns.set_style('whitegrid')

    # KDE 0-0.05（自适应y轴上限，避免数值过大）
    plt.figure(figsize=(7, 5))
    sns.kdeplot(u[~err], bw_method=0.2, fill=True, alpha=0.35, color='#e1d89c', label='Correct', clip=(0, clip_hi))
    sns.kdeplot(u[err],  bw_method=0.2, fill=True, alpha=0.35, color='#e1c59c', label='Misclassified', clip=(0, clip_hi))
    plt.xlim(0, clip_hi)
    # 估计直方图密度并设置上限为99分位的1.2倍，避免“尖峰”破坏观感
    try:
        uc = u[~err]; ue = u[err]
        uc = uc[(uc>=0)&(uc<=clip_hi)]; ue = ue[(ue>=0)&(ue<=clip_hi)]
        dens_c,_ = np.histogram(uc, bins=200, range=(0,clip_hi), density=True)
        dens_e,_ = np.histogram(ue, bins=200, range=(0,clip_hi), density=True)
        ymax = np.percentile(np.concatenate([dens_c, dens_e]), 99)*1.2
        if np.isfinite(ymax) and ymax>0:
            plt.ylim(0, float(ymax))
    except Exception:
        pass
    plt.axvline(np.median(u[~err]), color='#e1d89c', linestyle='--', alpha=0.8)
    plt.axvline(np.median(u[err]),  color='#e1c59c', linestyle='--', alpha=0.8)
    plt.xlabel('Uncertainty (u)'); plt.ylabel('Density'); plt.title(f'{args.dataset}: Uncertainty Distribution (EviMR)'); plt.legend(); plt.tight_layout()
    out = os.path.join(args.base_dir, 'uncert_density_evi_kde')
    plt.savefig(out + '.png', dpi=300); plt.savefig(out + '.svg'); plt.close()

    # Violin 0-0.05
    df = pd.DataFrame({'u': np.concatenate([u[~err], u[err]]),
                       'group': ['Correct']*int((~err).sum()) + ['Misclassified']*int(err.sum())})
    df = df[df['u'] <= clip_hi]
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x='group', y='u', hue='group', palette={'Correct':'#e1d89c','Misclassified':'#e1c59c'}, dodge=False, cut=0, inner=None, legend=False)
    med = df.groupby('group')['u'].median()
    for i, (_, val) in enumerate(med.items()):
        plt.plot([i-0.2, i+0.2], [val, val], color='#4a4a4a', linewidth=2)
    plt.ylabel('Uncertainty (u)'); plt.title(f'{args.dataset}: Uncertainty (zoom 0-0.05)'); plt.tight_layout()
    out = os.path.join(args.base_dir, 'uncert_density_evi_violin')
    plt.savefig(out + '.png', dpi=300); plt.savefig(out + '.svg'); plt.close()

    # 分离度
    try:
        auc = roc_auc_score(err.astype(int), u)
    except Exception:
        auc = float('nan')
    with open(os.path.join(args.base_dir, 'uncert_separation.txt'), 'w') as f:
        f.write(f"mean u (correct) : {u[~err].mean():.6f}\n")
        f.write(f"mean u (error)   : {u[err].mean():.6f}\n")
        f.write(f"median u (correct): {np.median(u[~err]):.6f}\n")
        f.write(f"median u (error)  : {np.median(u[err]):.6f}\n")
        f.write(f"AUROC(u -> error) : {auc:.6f}\n")
    print('Saved density/KDE/violin for', args.dataset)


if __name__ == '__main__':
    main()


