#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成噪声鲁棒性对比图（同坐标轴）
从 checkpoint 加载模型，仅评估不同 sigma 下的 F1。
"""
import os
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from MERIT.exp.exp_classification import Exp_Classification


def build_exp(ds, root, res, use_ds, gpu, d_model=256, d_ff=512, n_heads=8, e_layers=4, dropout=0.1,
              batch_size=64, epochs=150, patience=20, lr=1e-4, seed=41, weight_decay=1e-4):
    args = ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-EVI' if use_ds else f'UNCERT-{ds}-SOFT', model='MERIT',
                        data=ds, root_path=root, use_gpu=True, use_multi_gpu=False, devices='0', gpu=gpu,
                        freq='h', embed='timeF', output_attention=False, activation='gelu',
                        single_channel=False, augmentations='none,drop0.35', no_freq=False, no_diff=False,
                        use_gnn=False, use_evi_loss=False, lambda_evi=1.0, agg='evi', lambda_pseudo=1.0,
                        evidence_act='softplus', evidence_dropout=0.0,
                        d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                        e_layers=e_layers, dropout=dropout, resolution_list=res,
                        nodedim=10, batch_size=batch_size, train_epochs=epochs, patience=patience,
                        learning_rate=lr, use_ds=use_ds, swa=True, weight_decay=weight_decay,
                        lr_scheduler='none', warmup_epochs=0, seed=seed, num_workers=4)
    exp = Exp_Classification(args)
    setting = f"{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_res{args.resolution_list}_node{args.nodedim}_seed{args.seed}_bs{args.batch_size}_lr{args.learning_rate}"
    ckpt = os.path.join('./checkpoints', args.task_name, args.model_id, args.model, setting, 'checkpoint.pth')
    if os.path.exists(ckpt):
        try:
            if getattr(args,'swa', False):
                exp.swa_model.load_state_dict(torch.load(ckpt, map_location='cuda'))
            else:
                exp.model.load_state_dict(torch.load(ckpt, map_location='cuda'))
        except Exception as e:
            print('Warn: failed to load checkpoint:', e)
    return exp


def compute_curve(exp, sigmas):
    _, tl = exp._get_data(flag='TEST')
    f1=[]; exp.model.eval()
    with torch.no_grad():
        for s in sigmas:
            y_all=[]; p_all=[]
            for bx, y, pm in tl:
                bx = bx.float()
                if s>0: bx = bx + s*torch.randn_like(bx)
                pm = pm.float()
                bx = bx.cuda(); pm = pm.cuda(); y = y.cuda()
                if getattr(exp.args, 'use_ds', False):
                    alpha,_ = exp.model(bx, pm, None, None); prob = alpha/alpha.sum(dim=1, keepdim=True)
                else:
                    logits,_ = exp.model(bx, pm, None, None); prob = torch.softmax(logits, dim=1)
                pred = prob.argmax(dim=1)
                y_all.append(y.cpu().numpy()); p_all.append(pred.cpu().numpy())
            y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
            f1.append(f1_score(y_all, p_all, average='macro'))
    # 3点移动平均
    f1 = np.array(f1, dtype=float)
    if f1.size >= 3:
        s=f1.copy();
        for i in range(1,f1.size-1): s[i]=(f1[i-1]+f1[i]+f1[i+1])/3.0
        f1=s
    return f1


def main():
    p = ap.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--resolution_list', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    sigmas = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    os.makedirs(args.out_dir, exist_ok=True)

    exp_e = build_exp(args.dataset, args.root_path, args.resolution_list, True,  args.gpu)
    exp_s = build_exp(args.dataset, args.root_path, args.resolution_list, False, args.gpu)

    f1_e = compute_curve(exp_e, sigmas)
    f1_s = compute_curve(exp_s, sigmas)

    # 单方法曲线（更“稳”的视觉）
    plt.figure(figsize=(6,4))
    plt.plot(sigmas, f1_e, 'o-', color='#e1909c', label='EviMR-Net')
    plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{args.dataset}: Noise Robustness (EviMR)')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    out1=os.path.join(args.out_dir, 'noise_evi')
    plt.savefig(out1+'.png', dpi=300); plt.savefig(out1+'.svg'); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(sigmas, f1_s, 'o-', color='#4a4a4a', label='Softmax')
    plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{args.dataset}: Noise Robustness (Softmax)')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    out2=os.path.join(args.out_dir, 'noise_soft')
    plt.savefig(out2+'.png', dpi=300); plt.savefig(out2+'.svg'); plt.close()

    # 对比图
    plt.figure(figsize=(7,4.5))
    plt.plot(sigmas, f1_e, 'o-', color='#e1909c', label='EviMR-Net')
    plt.plot(sigmas, f1_s, 'o--', color='#4a4a4a', label='Softmax')
    plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{args.dataset}: Noise Robustness (Comparison)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    out3=os.path.join(args.out_dir, 'noise_compare')
    plt.savefig(out3+'.png', dpi=300); plt.savefig(out3+'.svg'); plt.close()
    print('Saved', out1+'.png', out2+'.png', out3+'.png')


if __name__ == '__main__':
    main()


