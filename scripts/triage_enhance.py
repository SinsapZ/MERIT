#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版 Triage 导出：
- 读取 EviMR 的 uncertainties/confidences/predictions/labels
- 选出高/低 u 的 Top-k 样本
- 计算信号质量指标（SNR估计）与多视角冲突指标
- 导出 CSV 到 results/uncertainty/<DATASET>/cases/triage_enhanced.csv
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
    exp = Exp_Classification(args)
    # 可选加载 checkpoint
    setting = f"{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_res{args.resolution_list}_node{args.nodedim}_seed{args.seed}_bs{args.batch_size}_lr{args.learning_rate}"
    ckpt = os.path.join('./checkpoints', args.task_name, args.model_id, args.model, setting, 'checkpoint.pth')
    if os.path.exists(ckpt):
        try:
            if getattr(args,'swa', False):
                exp.swa_model.load_state_dict(torch.load(ckpt, map_location='cuda'))
            else:
                exp.model.load_state_dict(torch.load(ckpt, map_location='cuda'))
        except Exception:
            pass
    return exp


def estimate_snr(x: np.ndarray, sma_win: int = 7) -> float:
    """简易SNR估计：x减去移动平均的残差为噪声，按RMS计算SNR(dB)。
    x: [T, C] or [T] numpy array
    """
    if x.ndim == 1:
        sig = x
    else:
        sig = x.mean(axis=1)
    if sma_win < 3:
        sma_win = 3
    k = sma_win
    kernel = np.ones(k) / k
    trend = np.convolve(sig, kernel, mode='same')
    noise = sig - trend
    rms_sig = np.sqrt(np.mean(sig**2) + 1e-12)
    rms_noise = np.sqrt(np.mean(noise**2) + 1e-12)
    snr_db = 20.0 * np.log10(rms_sig / (rms_noise + 1e-12))
    return float(snr_db)


def view_conflict_metrics(alpha_list: list, fused_prob: torch.Tensor) -> (float, float):
    """基于多视角的冲突度量：
    - std_pred_prob: 对最终预测类别的各视角概率标准差
    - disagree_rate: 视角与融合argmax不一致的比例
    """
    if not alpha_list or len(alpha_list) == 0:
        return float('nan'), float('nan')
    with torch.no_grad():
        probs = [a / a.sum(dim=1, keepdim=True) for a in alpha_list]
        prob_stack = torch.stack(probs, dim=0)  # [V, B, K]
        pred = fused_prob.argmax(dim=1)  # [B]
        b = fused_prob.shape[0]
        v = prob_stack.shape[0]
        idx = pred.view(1, b, 1).expand(v, b, 1)
        p_pred = torch.gather(prob_stack, dim=2, index=idx).squeeze(-1)  # [V,B]
        std_pred_prob = p_pred.std(dim=0).mean().item()
        view_argmax = prob_stack.argmax(dim=2)  # [V,B]
        disagree = (view_argmax != pred.view(1, b)).float().mean().item()
        return float(std_pred_prob), float(disagree)


def main():
    p = ap.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--resolution_list', required=True)
    p.add_argument('--uncertainty_base', required=True, help='results/uncertainty/<DATASET>')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--top_k_high', type=int, default=20)
    p.add_argument('--top_k_low', type=int, default=20)
    args = p.parse_args()

    evi_dir = os.path.join(args.uncertainty_base, 'evi')
    u_path = os.path.join(evi_dir, 'uncertainties.npy')
    c_path = os.path.join(evi_dir, 'confidences.npy')
    l_path = os.path.join(evi_dir, 'labels.npy')
    p_path = os.path.join(evi_dir, 'predictions.npy')
    for f in (u_path, c_path, l_path, p_path):
        if not os.path.exists(f):
            print('[Skip] missing', f)
            return

    u = np.load(u_path)
    c = np.load(c_path)
    y = np.load(l_path)
    yhat = np.load(p_path)

    # 高/低 u 样本索引
    idx_all = np.arange(len(u))
    hi_idx = idx_all[np.argsort(u)[-args.top_k_high:]][::-1]
    lo_idx = idx_all[np.argsort(u)[:args.top_k_low]]

    exp = build_exp(args.dataset, args.root_path, args.resolution_list, args.gpu)
    test_data, test_loader = exp._get_data(flag='TEST')
    exp.model.eval()

    # 为了方便索引，构建 index->(snr, conflict_std, disagree)
    snr_map = {}
    conflict_map = {}
    with torch.no_grad():
        offset = 0
        for bx, label, pm in test_loader:
            bsz = bx.shape[0]
            # 逐样本算SNR（使用原始numpy）
            for i in range(bsz):
                x = test_data.X[offset + i]
                snr_map[offset+i] = estimate_snr(x)
            bx = bx.float().to(exp.device)
            pm = pm.float().to(exp.device)
            fused_alpha, alpha_list = exp.model(bx, pm, None, None)
            S = fused_alpha.sum(dim=1, keepdim=True)
            fused_prob = fused_alpha / S
            stdp, disag = view_conflict_metrics(alpha_list, fused_prob)
            for i in range(bsz):
                conflict_map[offset+i] = (stdp, disag)
            offset += bsz

    out_dir = os.path.join(args.uncertainty_base, 'cases')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'triage_enhanced.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank_type','rank','index','label','prediction','uncertainty','confidence','snr_db','conflict_std','view_disagree'])
        r = 1
        for idx in hi_idx:
            snr = snr_map.get(int(idx), float('nan'))
            cs, dg = conflict_map.get(int(idx), (float('nan'), float('nan')))
            w.writerow(['high_u', r, int(idx), int(y[idx]), int(yhat[idx]), float(u[idx]), float(c[idx]), snr, cs, dg])
            r += 1
        r = 1
        for idx in lo_idx:
            snr = snr_map.get(int(idx), float('nan'))
            cs, dg = conflict_map.get(int(idx), (float('nan'), float('nan')))
            w.writerow(['low_u', r, int(idx), int(y[idx]), int(yhat[idx]), float(u[idx]), float(c[idx]), snr, cs, dg])
            r += 1
    print('Saved', out_csv)


if __name__ == '__main__':
    main()


