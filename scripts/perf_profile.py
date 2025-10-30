#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能–延迟–显存三线图：
- 不重训，从 checkpoint 加载模型（若可得）
- 在不同 batch size 下测量：
  1) 平均单 batch 延迟（ms）
  2) 吞吐（samples/s）
  3) 最大显存占用（MB, CUDA）
输出：CSV + PNG/SVG
"""
import os
import argparse as ap
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from MERIT.exp.exp_classification import Exp_Classification


def build_exp(ds, root, res, gpu, batch_size, e_layers=4, d_model=256, d_ff=512, n_heads=8, lr=1e-4, seed=41, use_ds=True):
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


def measure(exp, num_warmup=5, num_batches=30):
    _, loader = exp._get_data(flag='TEST')
    device = exp.device
    model = exp.swa_model if getattr(exp, 'swa', False) else exp.model
    model.eval()
    times=[]; seen=0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        # warmup
        for i,(bx,_,pm) in zip(range(num_warmup), loader):
            bx=bx.float().to(device); pm=pm.float().to(device)
            model(bx, pm, None, None)
        # measure
        it=0
        start=time.time()
        for bx,_,pm in loader:
            t0=time.time()
            bx=bx.float().to(device); pm=pm.float().to(device)
            model(bx, pm, None, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            dt=(time.time()-t0)
            times.append(dt)
            seen+=bx.shape[0]
            it+=1
            if it>=num_batches: break
        elapsed=time.time()-start
    avg_latency_ms = np.mean(times)*1000.0
    throughput = seen/elapsed if elapsed>0 else float('nan')
    max_mem_mb = 0.0
    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated(device)/(1024.0*1024.0)
    return avg_latency_ms, throughput, max_mem_mb


def main():
    p = ap.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--resolution_list', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--batches', type=str, default='16,32,64,128')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bss=[int(x) for x in args.batches.split(',') if x.strip()]
    rows=[]
    for bs in bss:
        exp = build_exp(args.dataset, args.root_path, args.resolution_list, args.gpu, batch_size=bs)
        lat, thr, mem = measure(exp)
        rows.append((bs, lat, thr, mem))
        print(f'bs={bs}: latency={lat:.2f}ms, throughput={thr:.1f}/s, mem={mem:.1f}MB')

    # CSV
    import csv
    csv_path=os.path.join(args.out_dir, 'perf_profile.csv')
    with open(csv_path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['batch_size','latency_ms','throughput_sps','max_mem_MB']); w.writerows(rows)

    # 三线图
    arr=np.array(rows)
    bs=arr[:,0]
    plt.figure(figsize=(8,5))
    ax1=plt.gca()
    ax1.plot(bs, arr[:,1], 'o-', color='#e1909c', label='Latency (ms)')
    ax1.set_xlabel('Batch size'); ax1.set_ylabel('Latency (ms)', color='#e1909c')
    ax1.tick_params(axis='y', labelcolor='#e1909c')
    ax2=ax1.twinx()
    ax2.plot(bs, arr[:,2], 's--', color='#4a4a4a', label='Throughput (samples/s)')
    ax2.set_ylabel('Throughput (samples/s)', color='#4a4a4a')
    # 画显存第三轴
    ax3=ax1.twinx(); ax3.spines.right.set_position(("axes", 1.12))
    ax3.plot(bs, arr[:,3], 'd-.', color='#e1c59c', label='Max GPU Mem (MB)')
    ax3.set_ylabel('Max GPU Mem (MB)', color='#e1c59c')
    plt.title(f'{args.dataset}: Performance–Latency–Memory')
    fig=plt.gcf(); fig.tight_layout()
    out=os.path.join(args.out_dir, 'perf_triple')
    plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()
    print('Saved perf profile:', out)


if __name__ == '__main__':
    main()


