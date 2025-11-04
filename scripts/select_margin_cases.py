#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按“类别区分度”（top1-top2 概率差）为每个真实类别挑选 Top-K 样本：
- 读取测试集，通过模型前向得到 fused 概率
- 计算 margin = p_top1 - p_top2、uncertainty u = K/sum(alpha)、confidence = p_top1
- 每个真实类别各取 K 个（默认仅取正确样本，可切换）
- 导出 CSV: results/uncertainty/<DATASET>/cases/triage_margin.csv
可与 plot_cases.py 配合：指定 --index_csv 即可按该清单绘图。
"""
import os
import argparse as ap
import numpy as np
import csv
import torch
from MERIT.exp.exp_classification import Exp_Classification


def build_exp(ds, root, res, gpu, e_layers=4, d_model=256, d_ff=512, n_heads=8,
              batch_size=64, lr=1e-4, seed=41, use_ds=True):
    args = ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-EVI', model='MERIT',
                        data=ds, root_path=root, use_gpu=True, use_multi_gpu=False, devices='0', gpu=gpu,
                        freq='h', embed='timeF', output_attention=False, activation='gelu',
                        single_channel=False, augmentations='none,drop0.35', no_freq=False, no_diff=False,
                        use_gnn=False, use_evi_loss=False, lambda_evi=1.0, agg='evi', lambda_pseudo=1.0,
                        evidence_act='softplus', evidence_dropout=0.0,
                        d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                        e_layers=e_layers, dropout=0.1, resolution_list=res,
                        nodedim=10, batch_size=batch_size, train_epochs=150, patience=20,
                        learning_rate=lr, use_ds=use_ds, swa=True, weight_decay=1e-4,
                        lr_scheduler='none', warmup_epochs=0, seed=seed, num_workers=2)
    return Exp_Classification(args)


def main():
    p = ap.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--resolution_list', required=True)
    p.add_argument('--uncertainty_base', required=True, help='results/uncertainty/<DATASET>')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--top_k_per_class', type=int, default=3)
    p.add_argument('--correct_only', action='store_true')
    args = p.parse_args()

    exp = build_exp(args.dataset, args.root_path, args.resolution_list, args.gpu)
    test_data, test_loader = exp._get_data(flag='TEST')
    infer_model = exp.swa_model if getattr(exp, 'swa', False) else exp.model
    infer_model.to(exp.device)
    infer_model.eval()

    records = []
    with torch.no_grad():
        offset = 0
        for bx, label, pm in test_loader:
            bx = bx.float().to(exp.device)
            pm = pm.float().to(exp.device)
            fused_alpha, _ = infer_model(bx, pm, None, None)
            S = fused_alpha.sum(dim=1, keepdim=True)
            prob = fused_alpha / S
            K = prob.shape[1]
            u = (K / S).squeeze(1).cpu().numpy()
            conf = prob.max(dim=1).values.cpu().numpy()
            pred = prob.argmax(dim=1).cpu().numpy()
            prob_np = prob.cpu().numpy()
            for i in range(bx.shape[0]):
                p_sorted = np.sort(prob_np[i])[::-1]
                p1 = float(p_sorted[0]); p2 = float(p_sorted[1] if prob_np.shape[1] > 1 else 0.0)
                margin = p1 - p2
                records.append({
                    'index': offset + i,
                    'label': int(label[i].cpu().item()),
                    'prediction': int(pred[i]),
                    'margin': float(margin),
                    'p1': float(p1), 'p2': float(p2),
                    'uncertainty': float(u[i]), 'confidence': float(conf[i])
                })
            offset += bx.shape[0]

    # 分类别挑选 Top-K（默认仅正确样本）
    labels = sorted(set(r['label'] for r in records))
    selected = []
    for c in labels:
        subset = [r for r in records if r['label'] == c]
        if args.correct_only:
            subset = [r for r in subset if r['label'] == r['prediction']]
        subset = sorted(subset, key=lambda x: x['margin'], reverse=True)[:args.top_k_per_class]
        selected.extend(subset)

    out_dir = os.path.join(args.uncertainty_base, 'cases')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'triage_margin.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index','label','prediction','margin','p1','p2','uncertainty','confidence'])
        for r in selected:
            w.writerow([r['index'], r['label'], r['prediction'], r['margin'], r['p1'], r['p2'], r['uncertainty'], r['confidence']])
    print('Saved', out_csv)


if __name__ == '__main__':
    main()


