#!/usr/bin/env python3
"""
Dump test-time uncertainty arrays without retraining.
Outputs (in --output_dir):
  - uncertainties.npy   # u = K / sum(alpha)
  - confidences.npy     # max soft probability per sample
  - predictions.npy     # argmax class
  - labels.npy          # ground-truth
  - (optional) x_test.npy  # if --save_inputs

It tries to locate the trained checkpoint by matching the `setting` string
used in run.py. You can also pass --checkpoint_dir to specify it explicitly.
"""
import argparse
import os
import glob
import numpy as np
import torch

from MERIT.exp.exp_classification import Exp_Classification


def find_checkpoint_dir(base='./checkpoints/classification', model_id='APAVA-Subject',
                        model='MERIT', data='APAVA', e_layers=4, resolution_list='2,4,6,8',
                        nodedim=10, batch_size=64, lr=1e-4, seed=None):
    base_dir = os.path.join(base, model_id, model)
    if not os.path.exists(base_dir):
        return None
    # Build glob pattern with key fields
    res = resolution_list
    parts = [
        f"*_{model}_{data}_dm*",
        f"*_*_nh*",
        f"_el{e_layers}_res{res}_node{nodedim}_seed",
    ]
    pattern = os.path.join(base_dir, ''.join(parts) + '*')
    candidates = glob.glob(pattern)
    # Filter by bs and lr and optional seed
    def match(p):
        name = os.path.basename(p)
        cond = (f"_bs{batch_size}_" in name) and (f"_lr{lr}" in name)
        if seed is not None:
            cond = cond and (f"_seed{seed}_" in name)
        return cond
    matches = [d for d in candidates if match(d)]
    if matches:
        # Pick the latest by mtime
        matches.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return matches[0]
    return None


def build_args(ns):
    # Mirror a minimal subset of MERIT.run arguments needed by Exp_Classification
    class A: pass
    a = A()
    a.task_name = 'classification'
    a.is_training = 0
    a.model_id = ns.model_id
    a.model = 'MERIT'
    a.checkpoints = './checkpoints/'
    a.data = ns.data
    a.root_path = ns.root_path
    a.freq = 'h'
    a.d_model = 256
    a.d_ff = 512
    a.n_heads = 8
    a.e_layers = ns.e_layers
    a.d_layers = 1
    a.dropout = ns.dropout
    a.embed = 'timeF'
    a.activation = 'gelu'
    a.output_attention = False
    a.patch_len_list = '2,2,2,4,4,4,16,16,16,16,32,32,32,32,32'
    a.single_channel = False
    a.augmentations = 'none,drop0.35'
    a.resolution_list = ns.resolution_list
    a.nodedim = ns.nodedim
    a.no_pseudo = False
    a.agg = 'evi'
    a.lambda_pseudo = 1.0
    a.lambda_pseudo_loss = ns.lambda_pseudo
    a.no_diff = False
    a.no_freq = False
    a.use_gpu = torch.cuda.is_available()
    a.gpu = ns.gpu
    a.use_multi_gpu = False
    a.device_ids = [a.gpu]
    a.learning_rate = ns.lr
    a.lambda_fuse = 1.0
    a.lambda_view = ns.lambda_view
    a.annealing_epoch = ns.annealing_epoch
    a.evidence_act = 'softplus'
    a.evidence_dropout = ns.evidence_dropout
    a.weight_decay = ns.weight_decay
    a.batch_size = ns.batch_size
    a.train_epochs = 1
    a.patience = 3
    a.swa = False
    a.lr_scheduler = 'none'
    a.warmup_epochs = 0
    a.seed = ns.seed
    a.num_workers = 4
    a.seq_len = 0
    a.pred_len = 0
    a.use_ds = True
    a.num_class = 0
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, choices=['APAVA','ADFD','ADFD-Sample','PTB','PTB-XL'])
    p.add_argument('--root_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/uncertainty')
    p.add_argument('--model_id', type=str, default='APAVA-Subject')
    # Hyperparams to locate checkpoint
    p.add_argument('--e_layers', type=int, default=4)
    p.add_argument('--resolution_list', type=str, default='2,4,6,8')
    p.add_argument('--nodedim', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=str, default='1e-4')
    p.add_argument('--lambda_view', type=float, default=1.0)
    p.add_argument('--lambda_pseudo', type=float, default=0.3)
    p.add_argument('--annealing_epoch', type=int, default=50)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--evidence_dropout', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--checkpoint_dir', type=str, default='')
    p.add_argument('--save_inputs', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoint directory
    ckpt_dir = args.checkpoint_dir or find_checkpoint_dir(
        base='./checkpoints/classification', model_id=args.model_id, model='MERIT', data=args.data,
        e_layers=args.e_layers, resolution_list=args.resolution_list, nodedim=args.nodedim,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    if not ckpt_dir:
        print('❌ 未找到匹配的checkpoint目录。请使用 --checkpoint_dir 指定。')
        return
    print(f'✅ 使用checkpoint目录: {ckpt_dir}')

    # Build args and create exp
    ns = build_args(args)
    setting = os.path.basename(ckpt_dir)
    exp = Exp_Classification(ns)

    # Load model weights
    model_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    if not os.path.exists(model_path):
        print(f'❌ 未找到权重: {model_path}')
        return
    if getattr(exp, 'swa', False):
        exp.swa_model.load_state_dict(torch.load(model_path, map_location='cuda' if ns.use_gpu else 'cpu'))
    else:
        exp.model.load_state_dict(torch.load(model_path, map_location='cuda' if ns.use_gpu else 'cpu'))

    # Build loaders
    _, test_loader = exp._get_data(flag='TEST')
    if getattr(exp, 'swa', False):
        exp.swa_model.eval()
    else:
        exp.model.eval()

    preds_cls = []
    labels = []
    uncertainties = []
    confidences = []
    x_list = []

    import torch.nn.functional as F
    with torch.no_grad():
        for batch_x, label, padding_mask in test_loader:
            batch_x = batch_x.float().to(exp.device)
            padding_mask = padding_mask.float().to(exp.device)
            label = label.to(exp.device)
            fused_alpha, _ = (exp.swa_model if getattr(exp, 'swa', False) else exp.model)(batch_x, padding_mask, None, None)
            S = fused_alpha.sum(dim=1, keepdim=True)
            probs = fused_alpha / S
            pred = torch.argmax(probs, dim=1)
            # uncertainty (vacuity): K / S
            K = probs.shape[1]
            u = (K / S.squeeze(1)).clamp(max=1e6)
            conf = probs.max(dim=1).values

            preds_cls.append(pred.cpu().numpy())
            labels.append(label.flatten().cpu().numpy())
            uncertainties.append(u.cpu().numpy())
            confidences.append(conf.cpu().numpy())
            if args.save_inputs:
                x_list.append(batch_x.cpu().numpy())

    preds_cls = np.concatenate(preds_cls, axis=0)
    labels = np.concatenate(labels, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)
    confidences = np.concatenate(confidences, axis=0)
    out_dir = os.path.join(args.output_dir, args.data)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'predictions.npy'), preds_cls)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)
    np.save(os.path.join(out_dir, 'uncertainties.npy'), uncertainties)
    np.save(os.path.join(out_dir, 'confidences.npy'), confidences)
    if args.save_inputs and x_list:
        X = np.concatenate(x_list, axis=0)
        np.save(os.path.join(out_dir, 'x_test.npy'), X)

    print(f'✅ 已保存不确定性数据到: {out_dir}')


if __name__ == '__main__':
    main()


