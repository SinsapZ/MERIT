#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临床友好案例图自动化：
从 triage_candidates.csv 选取最不自信的若干样本，绘制：
  1) 波形图（标注 label/pred/conf/u）
  2) 概率柱状图（高亮预测类别）

要求：不依赖重训，优先加载 checkpoint（UNCERT-<DS>-EVI）；若找不到，使用当前权重推理。
输出：PNG+SVG 到 results/uncertainty/<DATASET>/cases/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from MERIT.exp.exp_classification import Exp_Classification
import torch
import csv


PALETTE = {
    'vanilla': '#e1d89c',
    'tan': '#e1c59c',
    'melon': '#e1ae9c',
    'puce': '#e1909c',
    "gray": '#4a4a4a',
}


def load_triage(plots_evi_dir: str, top_k: int):
    triage_csv = os.path.join(plots_evi_dir, 'triage_candidates.csv')
    if not os.path.exists(triage_csv):
        raise FileNotFoundError(f'Not found: {triage_csv}')
    rows = []
    with open(triage_csv, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            # index,label,prediction,uncertainty(approx),confidence
            rows.append({
                'index': int(row['index']),
                'label': int(row['label']),
                'prediction': int(row['prediction']),
                'uncertainty': float(row.get('uncertainty(approx)', row.get('uncertainty', 0.0))),
                'confidence': float(row['confidence']),
            })
    rows = sorted(rows, key=lambda x: x['confidence'])  # 最不自信在前
    return rows[:top_k]


def build_exp(ds, root, res, e_layers, d_model, d_ff, n_heads, batch_size, lr, gpu, seed, use_ds=True):
    import argparse as ap
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
    exp = Exp_Classification(args)
    # 加载checkpoint（可选）
    setting = f"{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_res{args.resolution_list}_node{args.nodedim}_seed{args.seed}_bs{args.batch_size}_lr{args.learning_rate}"
    ckpt = os.path.join('./checkpoints', args.task_name, args.model_id, args.model, setting, 'checkpoint.pth')
    if os.path.exists(ckpt):
        try:
            if getattr(args,'swa', False):
                exp.swa_model.load_state_dict(torch.load(ckpt, map_location='cuda' if args.use_gpu else 'cpu'))
            else:
                exp.model.load_state_dict(torch.load(ckpt, map_location='cuda' if args.use_gpu else 'cpu'))
        except Exception:
            pass
    return exp


def plot_cases(ds, root, res, out_base, plots_evi_dir, top_k=6, gpu=0, e_layers=4, d_model=256, d_ff=512, n_heads=8, batch_size=64, lr=1e-4, seed=41):
    os.makedirs(out_base, exist_ok=True)
    selected = load_triage(plots_evi_dir, top_k)
    exp = build_exp(ds, root, res, e_layers, d_model, d_ff, n_heads, batch_size, lr, gpu, seed, use_ds=True)
    test_data, _ = exp._get_data(flag='TEST')
    exp.model.eval()

    for k, row in enumerate(selected):
        idx = row['index']
        label = row['label']
        # 取波形
        x = test_data.X[idx]  # (T, C)
        # 推理概率
        with torch.no_grad():
            bx = torch.from_numpy(x).float().unsqueeze(0).to(exp.device)
            pm = torch.zeros((1, x.shape[0])).float().to(exp.device)
            alpha,_ = exp.model(bx, pm, None, None)  # fused alpha
            S = alpha.sum(dim=1, keepdim=True)
            prob = (alpha / S).squeeze(0).cpu().numpy()
            pred = int(prob.argmax())
            K = prob.shape[0]
            u = float((K / S).item())
            conf = float(prob.max())

        out_dir = os.path.join(out_base, 'cases')
        os.makedirs(out_dir, exist_ok=True)

        # 1) 波形
        plt.figure(figsize=(8,2))
        plt.plot(x[:,0], color=PALETTE['gray'])
        plt.title(f'Waveform (ch=0)  label={label}  pred={pred}  conf={conf:.3f}  u={u:.3f}')
        plt.tight_layout()
        base = os.path.join(out_dir, f'clinical_case_{k}_wave')
        plt.savefig(base+'.png', dpi=300); plt.savefig(base+'.svg'); plt.close()

        # 2) 概率柱状图（高亮预测类）
        colors = [PALETTE['vanilla']]*K
        colors[pred] = PALETTE['puce']
        plt.figure(figsize=(4.2,3.4))
        plt.bar(np.arange(K), prob, color=colors)
        plt.title(f'Prob (pred={pred}, u={u:.3f})')
        plt.tight_layout()
        base = os.path.join(out_dir, f'clinical_case_{k}_prob')
        plt.savefig(base+'.png', dpi=300); plt.savefig(base+'.svg'); plt.close()

    print(f'Saved clinical cases to: {out_base}/cases')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--root_path', type=str, required=True)
    ap.add_argument('--resolution_list', type=str, required=True)
    ap.add_argument('--uncertainty_base', type=str, required=True, help='results/uncertainty/<DATASET>')
    ap.add_argument('--top_k', type=int, default=6)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--e_layers', type=int, default=4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--d_ff', type=int, default=512)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args()

    plots_evi_dir = os.path.join(args.uncertainty_base, 'plots_evi')
    plot_cases(args.dataset, args.root_path, args.resolution_list,
               out_base=args.uncertainty_base,
               plots_evi_dir=plots_evi_dir,
               top_k=args.top_k, gpu=args.gpu, e_layers=args.e_layers,
               d_model=args.d_model, d_ff=args.d_ff, n_heads=args.n_heads,
               batch_size=args.batch_size, lr=args.lr, seed=args.seed)


if __name__ == '__main__':
    main()


