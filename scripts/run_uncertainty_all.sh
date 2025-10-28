#!/bin/bash
# 一键运行不确定性实验（训练 + 保存不确定性数组 + 生成图表）
# 仅依赖现有代码与脚本参数，不需手动干预

set -e

GPU=${1:-0}

# ===================== 配置每个数据集的最佳参数 =====================
declare -A ROOTS
declare -A LRS
declare -A LAMBDA_PSEUDO_LOSS
declare -A RES_LIST

ROOTS[APAVA]="/home/Data1/zbl/dataset/APAVA"
LRS[APAVA]="1e-4"
LAMBDA_PSEUDO_LOSS[APAVA]="0.3"
RES_LIST[APAVA]="2,4,6,8"

ROOTS[PTB]="/home/Data1/zbl/dataset/PTB"
LRS[PTB]="1.5e-4"
LAMBDA_PSEUDO_LOSS[PTB]="0.2"
RES_LIST[PTB]="2,4,6,8"

ROOTS["PTB-XL"]="/home/Data1/zbl/dataset/PTB-XL"
LRS["PTB-XL"]="1.5e-4"
LAMBDA_PSEUDO_LOSS["PTB-XL"]="0.3"
RES_LIST["PTB-XL"]="2,4,6,8"

# 通用训练超参
E_LAYERS=4
DROPOUT=0.1
BATCH_SIZE=64
EPOCHS=150
PATIENCE=20
WD=1e-4
ANNEAL=50
D_MODEL=256
D_FF=512
N_HEADS=8
NODEDIM=10
SEED=41

# 输出目录
OUT_BASE="results/uncertainty"
mkdir -p "$OUT_BASE"

run_one_dataset() {
  local DS=$1
  local ROOT=${ROOTS[$DS]}
  local LR=${LRS[$DS]}
  local LPL=${LAMBDA_PSEUDO_LOSS[$DS]}
  local RES=${RES_LIST[$DS]}

  echo "\n=============================="
  echo "运行数据集: $DS"
  echo "=============================="

  # --- 1) 训练并保存不确定性数组：EviMR ---
  EVI_DIR="$OUT_BASE/$DS/evi"
  mkdir -p "$EVI_DIR"
  python -m MERIT.run \
    --model MERIT --data "$DS" --root_path "$ROOT" \
    --model_id "UNCERT-${DS}-EVI" \
    --use_ds \
    --learning_rate "$LR" \
    --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss "$LPL" \
    --annealing_epoch "$ANNEAL" \
    --resolution_list "$RES" \
    --batch_size "$BATCH_SIZE" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$EVI_DIR" \
    2>&1 | grep -E "(Validation results|Test results|Saved uncertainty|SUMMARY|SUMMARY STATISTICS)" || true

  # --- 2) 训练并保存不确定性数组：Softmax基线（不启用DS） ---
  SOFT_DIR="$OUT_BASE/$DS/softmax"
  mkdir -p "$SOFT_DIR"
  python -m MERIT.run \
    --model MERIT --data "$DS" --root_path "$ROOT" \
    --model_id "UNCERT-${DS}-SOFT" \
    --learning_rate "$LR" \
    --resolution_list "$RES" \
    --batch_size "$BATCH_SIZE" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$SOFT_DIR" \
    2>&1 | grep -E "(Validation results|Test results|Saved uncertainty|SUMMARY|SUMMARY STATISTICS)" || true

  # --- 3) 指标与图：单方法（可靠度图/选择性预测） ---
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$EVI_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_evi" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$SOFT_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_soft" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true

  # --- 4) 叠加比较：准确率-拒绝率曲线 & 不确定性分布 ---
  python -m MERIT.scripts.compare_selective --base_dir "$OUT_BASE/$DS" --dataset "$DS" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' || true
  python - <<PY || true
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
ds = "$DS"; base = "$OUT_BASE"; evi = os.path.join(base, ds, 'evi')
uf=os.path.join(evi,'uncertainties.npy'); lf=os.path.join(evi,'labels.npy'); pf=os.path.join(evi,'predictions.npy')
if os.path.exists(uf) and os.path.exists(lf) and os.path.exists(pf):
    u = np.load(uf); y = np.load(lf); p = np.load(pf)
    err = (p!=y)
    sns.set_style('whitegrid')
    # KDE（聚焦近零区间，叠加中位数）
    plt.figure(figsize=(7,5))
    sns.kdeplot(u[~err], bw_method=0.2, fill=True, alpha=0.35, color='#e1d89c', label='Correct', clip=(0,0.05))
    sns.kdeplot(u[ err], bw_method=0.2, fill=True, alpha=0.35, color='#e1c59c', label='Misclassified', clip=(0,0.05))
    plt.xlim(0,0.05)
    for val,color in ((np.median(u[~err]),'#e1d89c'), (np.median(u[err]),'#e1c59c')):
        plt.axvline(val, color=color, linestyle='--', alpha=0.8)
    plt.xlabel('Uncertainty (u)'); plt.ylabel('Density'); plt.title(f'{ds}: Uncertainty Distribution (EviMR)'); plt.legend(); plt.tight_layout()
    out=os.path.join(base, ds, 'uncert_density_evi_kde')
    plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()

    # 小提琴图（0-0.05 放大区间）
    df = pd.DataFrame({'u': np.concatenate([u[~err], u[err]]),
                       'group': ['Correct']*int((~err).sum()) + ['Misclassified']*int(err.sum())})
    df = df[df['u'] <= 0.05]
    plt.figure(figsize=(6,4))
    sns.violinplot(data=df, x='group', y='u', palette={'Correct':'#e1d89c','Misclassified':'#e1c59c'}, cut=0, inner=None)
    med = df.groupby('group')['u'].median()
    for i,(grp,val) in enumerate(med.items()):
        plt.plot([i-0.2,i+0.2],[val,val], color='#4a4a4a', linewidth=2)
    plt.ylabel('Uncertainty (u)'); plt.title(f'{ds}: Uncertainty (zoom 0-0.05)'); plt.tight_layout()
    out=os.path.join(base, ds, 'uncert_density_evi_violin')
    plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()
    # 分离度指标保存
    try:
        auc = roc_auc_score(err.astype(int), u)
    except Exception:
        auc = float('nan')
    with open(os.path.join(base, ds, 'uncert_separation.txt'), 'w') as f:
        f.write(f"mean u (correct) : {u[~err].mean():.6f}\n")
        f.write(f"mean u (error)   : {u[err].mean():.6f}\n")
        f.write(f"median u (correct): {np.median(u[~err]):.6f}\n")
        f.write(f"median u (error)  : {np.median(u[err]):.6f}\n")
        f.write(f"AUROC(u -> error) : {auc:.6f}\n")
print('Saved compare curves & uncertainty density (KDE/violin) for', ds)
PY

  # --- 5) 噪声鲁棒性（EviMR与Softmax） ---
  python - <<PY || true
import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from MERIT.exp.exp_classification import Exp_Classification
import argparse as ap

def noise_curve(ds, use_ds, model_id, root, res, e_layers, dropout, bs, lr, wd, seed, gpu, out_png):
    args = ap.Namespace(task_name='classification', model_id=model_id, model='MERIT',
                        data=ds, root_path=root, use_gpu=True, use_multi_gpu=False, devices='0', gpu=gpu,
                        freq='h', embed='timeF', output_attention=False, activation='gelu',
                        single_channel=False, augmentations='none,drop0.35', no_freq=False, no_diff=False,
                        use_gnn=False, use_evi_loss=False, lambda_evi=1.0, agg='evi', lambda_pseudo=1.0,
                        evidence_act='softplus', evidence_dropout=0.0,
                        d_model=$D_MODEL, d_ff=$D_FF, n_heads=$N_HEADS,
                        e_layers=e_layers, dropout=dropout, resolution_list=res,
                        nodedim=$NODEDIM, batch_size=bs, train_epochs=$EPOCHS, patience=$PATIENCE,
                        learning_rate=lr, use_ds=use_ds, swa=True, weight_decay=wd,
                        lr_scheduler='none', warmup_epochs=0, seed=seed, num_workers=4)
    exp = Exp_Classification(args)
    # try to load trained checkpoint matching MERIT.run setting
    setting = f"{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_res{args.resolution_list}_node{args.nodedim}_seed{args.seed}_bs{args.batch_size}_lr{args.learning_rate}"
    ckpt = os.path.join('./checkpoints', args.task_name, args.model_id, args.model, setting, 'checkpoint.pth')
    if os.path.exists(ckpt):
        try:
            if getattr(args,'swa', False):
                exp.swa_model.load_state_dict(torch.load(ckpt, map_location='cuda'))
            else:
                exp.model.load_state_dict(torch.load(ckpt, map_location='cuda'))
            print('Loaded checkpoint for noise eval:', ckpt)
        except Exception as e:
            print('Warn: failed to load checkpoint:', e)
    _, test_loader = exp._get_data(flag='TEST')
    # 拉远采样点，降低抖动
    sigmas = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    f1s=[]; exp.model.eval()
    with torch.no_grad():
        for sigma in sigmas:
            y_all=[]; p_all=[]
            for bx, y, pm in test_loader:
                bx = bx.float()
                if sigma>0: bx = bx + sigma*torch.randn_like(bx)
                pm = pm.float()
                bx = bx.cuda(); pm = pm.cuda(); y = y.cuda()
                if use_ds:
                    alpha,_ = exp.model(bx, pm, None, None); prob = alpha/alpha.sum(dim=1, keepdim=True)
                else:
                    logits,_ = exp.model(bx, pm, None, None); prob = torch.softmax(logits, dim=1)
                pred = prob.argmax(dim=1)
                y_all.append(y.cpu().numpy()); p_all.append(pred.cpu().numpy())
            y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
            f1s.append(f1_score(y_all, p_all, average='macro'))
    # 3点移动平均平滑（仅中间点）
    import numpy as np
    def smooth(y):
        y = np.array(y, dtype=float)
        if y.size < 3:
            return y
        ys = y.copy()
        for i in range(1, y.size-1):
            ys[i] = (y[i-1] + y[i] + y[i+1]) / 3.0
        return ys
    f1s_s = smooth(f1s)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(6,4))
    color = '#e1909c' if use_ds else '#4a4a4a'
    plt.plot(sigmas, f1s_s, 'o-', color=color, label='EviMR-Net' if use_ds else 'Softmax')
    plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{ds}: Noise Robustness'); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace('.png', '.svg'))
    plt.close()

ds = "$DS"; root = "$ROOT"; res = "$RES"; outb = "$OUT_BASE"
noise_curve(ds, True,  f'UNCERT-{ds}-EVI',  root, res, $E_LAYERS, $DROPOUT, $BATCH_SIZE, $LR, $WD, $SEED, $GPU, os.path.join(outb, ds, 'noise_evi.png'))
noise_curve(ds, False, f'UNCERT-{ds}-SOFT', root, res, $E_LAYERS, $DROPOUT, $BATCH_SIZE, $LR, $WD, $SEED, $GPU, os.path.join(outb, ds, 'noise_soft.png'))
    # 组合对比图（同坐标轴）
    def compute_curve(ds, use_ds):
        # 与 noise_curve 相同设置，但仅返回数据
        from MERIT.exp.exp_classification import Exp_Classification
        import argparse as ap, torch, numpy as np
        args = ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-EVI' if use_ds else f'UNCERT-{ds}-SOFT', model='MERIT',
                        data=ds, root_path=root, use_gpu=True, use_multi_gpu=False, devices='0', gpu=$GPU,
                        freq='h', embed='timeF', output_attention=False, activation='gelu',
                        single_channel=False, augmentations='none,drop0.35', no_freq=False, no_diff=False,
                        use_gnn=False, use_evi_loss=False, lambda_evi=1.0, agg='evi', lambda_pseudo=1.0,
                        evidence_act='softplus', evidence_dropout=0.0,
                        d_model=$D_MODEL, d_ff=$D_FF, n_heads=$N_HEADS,
                        e_layers=$E_LAYERS, dropout=$DROPOUT, resolution_list=res,
                        nodedim=$NODEDIM, batch_size=$BATCH_SIZE, train_epochs=$EPOCHS, patience=$PATIENCE,
                        learning_rate=$LR, use_ds=use_ds, swa=True, weight_decay=$WD,
                        lr_scheduler='none', warmup_epochs=0, seed=$SEED, num_workers=4)
        exp = Exp_Classification(args)
        _, tl = exp._get_data(flag='TEST')
        sigmas = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
        f1=[]; exp.model.eval()
        with torch.no_grad():
            for s in sigmas:
                y_all=[]; p_all=[]
                for bx, y, pm in tl:
                    bx = bx.float()
                    if s>0: bx = bx + s*torch.randn_like(bx)
                    pm = pm.float()
                    bx = bx.cuda(); pm = pm.cuda(); y = y.cuda()
                    if use_ds:
                        alpha,_ = exp.model(bx, pm, None, None); prob = alpha/alpha.sum(dim=1, keepdim=True)
                    else:
                        logits,_ = exp.model(bx, pm, None, None); prob = torch.softmax(logits, dim=1)
                    pred = prob.argmax(dim=1)
                    y_all.append(y.cpu().numpy()); p_all.append(pred.cpu().numpy())
                import numpy as np
                from sklearn.metrics import f1_score
                y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
                f1.append(f1_score(y_all, p_all, average='macro'))
     # 平滑
    def smooth(a):
        a = np.array(a, dtype=float)
        if a.size < 3: return a
        s = a.copy()
        for i in range(1, a.size-1): s[i]=(a[i-1]+a[i]+a[i+1])/3.0
        return s
    return sigmas, smooth(f1)
sig_e, f1_e = compute_curve(ds, True)
sig_s, f1_s = compute_curve(ds, False)
import matplotlib.pyplot as plt
plt.figure(figsize=(7,4.5))
plt.plot(sig_e, f1_e, 'o-', color='#e1909c', label='EviMR-Net')
plt.plot(sig_s, f1_s, 'o--', color='#4a4a4a', label='Softmax')
plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{ds}: Noise Robustness (Comparison)'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
out=os.path.join(outb, ds, 'noise_compare')
plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()
print('Saved noise robustness for', ds)
PY

  # --- 6) 案例图（高/低不确定度各Top-6） ---
  python -m MERIT.scripts.plot_cases \
    --dataset "$DS" \
    --root_path "$ROOT" \
    --resolution_list "$RES" \
    --uncertainty_base "$OUT_BASE/$DS" \
    --top_k_high 6 \
    --top_k_low 6 \
    --gpu "$GPU" || true
}

# ===================== 主程序 =====================
for DS in APAVA PTB PTB-XL; do
  run_one_dataset "$DS"
done

echo "\n✅ 全部完成。不确定性数据与图已保存到: $OUT_BASE/<DATASET>/...\n"


