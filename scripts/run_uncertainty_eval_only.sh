#!/bin/bash
# 仅评估与出图（不训练）：
# - 从已存在的 checkpoint 加载模型
# - 保存不确定性数组（EviMR/Softmax）
# - 生成所有图（可靠度、选择性、对比、KDE/Violin、噪声鲁棒性、案例图）

set -e

GPU=${1:-0}

declare -A ROOTS
declare -A LRS
declare -A RES_LIST

ROOTS[APAVA]="/home/Data1/zbl/dataset/APAVA";   LRS[APAVA]="1e-4";  RES_LIST[APAVA]="2,4,6,8"
ROOTS[PTB]="/home/Data1/zbl/dataset/PTB";       LRS[PTB]="1.5e-4"; RES_LIST[PTB]="2,4,6,8"
ROOTS["PTB-XL"]="/home/Data1/zbl/dataset/PTB-XL"; LRS["PTB-XL"]="1.5e-4"; RES_LIST["PTB-XL"]="2,4,6,8"

E_LAYERS=4; D_MODEL=256; D_FF=512; N_HEADS=8; DROPOUT=0.1; WD=1e-4; BATCH=64; EPOCHS=150; PATIENCE=20; ANNEAL=50; NODEDIM=10; SEED=41

OUT_BASE="results/uncertainty"; mkdir -p "$OUT_BASE"

run_eval_dataset() {
  local DS=$1
  local ROOT=${ROOTS[$DS]}
  local LR=${LRS[$DS]}
  local RES=${RES_LIST[$DS]}
  echo "\n================ EVAL ONLY: $DS ================"

  local EVI_DIR="$OUT_BASE/$DS/evi";     mkdir -p "$EVI_DIR"
  local SOFT_DIR="$OUT_BASE/$DS/softmax"; mkdir -p "$SOFT_DIR"

  # 1) 仅评估：从 checkpoint 加载并保存不确定性（EviMR）
  python -m MERIT.run \
    --task_name classification \
    --is_training 0 \
    --model_id "UNCERT-${DS}-EVI" \
    --model MERIT \
    --data "$DS" \
    --root_path "$ROOT" \
    --use_ds \
    --learning_rate "$LR" \
    --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.3 \
    --annealing_epoch "$ANNEAL" \
    --resolution_list "$RES" \
    --batch_size "$BATCH" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$EVI_DIR"

  # 2) 仅评估：Softmax 基线
  python -m MERIT.run \
    --task_name classification \
    --is_training 0 \
    --model_id "UNCERT-${DS}-SOFT" \
    --model MERIT \
    --data "$DS" \
    --root_path "$ROOT" \
    --learning_rate "$LR" \
    --resolution_list "$RES" \
    --batch_size "$BATCH" --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
    --d_model "$D_MODEL" --d_ff "$D_FF" --n_heads "$N_HEADS" \
    --nodedim "$NODEDIM" --gpu "$GPU" --swa \
    --seed "$SEED" \
    --save_uncertainty --uncertainty_dir "$SOFT_DIR"

  # 3) 单方法图（可靠度/选择性 + SVG）
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$EVI_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_evi" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$SOFT_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_soft" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' --reject_rate 20 || true

  # 4) 对比曲线（EviMR vs Softmax）
  python -m MERIT.scripts.compare_selective --base_dir "$OUT_BASE/$DS" --dataset "$DS" --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' || true

  # 5) KDE/Violin（近零聚焦 + 中位线 + 分离度报告）
  python - <<PY || true
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.metrics import roc_auc_score
ds="$DS"; base="$OUT_BASE"; evi=os.path.join(base, ds, 'evi')
uf=os.path.join(evi,'uncertainties.npy'); lf=os.path.join(evi,'labels.npy'); pf=os.path.join(evi,'predictions.npy')
if os.path.exists(uf) and os.path.exists(lf) and os.path.exists(pf):
    u=np.load(uf); y=np.load(lf); p=np.load(pf); err=(p!=y)
    sns.set_style('whitegrid')
    plt.figure(figsize=(7,5))
    sns.kdeplot(u[~err], bw_method=0.2, fill=True, alpha=0.35, color='#e1d89c', label='Correct', clip=(0,0.05))
    sns.kdeplot(u[ err], bw_method=0.2, fill=True, alpha=0.35, color='#e1c59c', label='Misclassified', clip=(0,0.05))
    plt.xlim(0,0.05)
    import numpy as np
    for v,c in ((np.median(u[~err]),'#e1d89c'),(np.median(u[err]),'#e1c59c')):
        plt.axvline(v, color=c, linestyle='--', alpha=0.8)
    plt.xlabel('Uncertainty (u)'); plt.ylabel('Density'); plt.title(f'{ds}: Uncertainty Distribution (EviMR)'); plt.legend(); plt.tight_layout()
    out=os.path.join(base, ds, 'uncert_density_evi_kde'); plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()
    df=pd.DataFrame({'u':np.concatenate([u[~err],u[err]]),'group':['Correct']*int((~err).sum())+['Misclassified']*int(err.sum())}); df=df[df['u']<=0.05]
    plt.figure(figsize=(6,4)); sns.violinplot(data=df, x='group', y='u', hue='group', palette={'Correct':'#e1d89c','Misclassified':'#e1c59c'}, dodge=False, cut=0, inner=None, legend=False)
    med=df.groupby('group')['u'].median();
    import matplotlib.pyplot as plt
    for i,(g,v) in enumerate(med.items()): plt.plot([i-0.2,i+0.2],[v,v], color='#4a4a4a', linewidth=2)
    plt.ylabel('Uncertainty (u)'); plt.title(f'{ds}: Uncertainty (zoom 0-0.05)'); plt.tight_layout(); out=os.path.join(base, ds, 'uncert_density_evi_violin'); plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()
    try: auc=roc_auc_score(err.astype(int), u)
    except Exception: auc=float('nan')
    with open(os.path.join(base, ds, 'uncert_separation.txt'),'w') as f:
        f.write(f"mean u (correct): {u[~err].mean():.6f}\nmean u (error): {u[err].mean():.6f}\nmedian u (correct): {np.median(u[~err]):.6f}\nmedian u (error): {np.median(u[err]):.6f}\nAUROC(u->error): {auc:.6f}\n")
print('Saved density/KDE/violin for', ds)
PY

  # 6) 噪声鲁棒性（同轴对比 + 各自单图已在 compare/noise 中生成）
  python - <<PY || true
import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from MERIT.exp.exp_classification import Exp_Classification
import argparse as ap
ds="$DS"; root="$ROOT"; res="$RES"; outb="$OUT_BASE"; gpu=$GPU
def curve(use_ds):
    args=ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-EVI' if use_ds else f'UNCERT-{ds}-SOFT', model='MERIT', data=ds, root_path=root, use_gpu=True, use_multi_gpu=False, devices='0', gpu=gpu, freq='h', embed='timeF', output_attention=False, activation='gelu', single_channel=False, augmentations='none,drop0.35', no_freq=False, no_diff=False, use_gnn=False, use_evi_loss=False, lambda_evi=1.0, agg='evi', lambda_pseudo=1.0, evidence_act='softplus', evidence_dropout=0.0, d_model=$D_MODEL, d_ff=$D_FF, n_heads=$N_HEADS, e_layers=$E_LAYERS, dropout=$DROPOUT, resolution_list=res, nodedim=$NODEDIM, batch_size=$BATCH, train_epochs=$EPOCHS, patience=$PATIENCE, learning_rate=$LR, use_ds=use_ds, swa=True, weight_decay=$WD, lr_scheduler='none', warmup_epochs=0, seed=$SEED, num_workers=4)
    exp=Exp_Classification(args); _,tl=exp._get_data(flag='TEST'); sig=[0.0,0.02,0.05,0.10,0.20,0.30]; f1=[]; exp.model.eval()
    with torch.no_grad():
        for s in sig:
            ya=[]; pa=[]
            for bx,y,pm in tl:
                bx=bx.float();
                if s>0: bx=bx+s*torch.randn_like(bx)
                pm=pm.float(); bx=bx.cuda(); pm=pm.cuda(); y=y.cuda()
                if use_ds:
                    alpha,_=exp.model(bx,pm,None,None); prob=alpha/alpha.sum(dim=1,keepdim=True)
                else:
                    logit,_=exp.model(bx,pm,None,None); prob=torch.softmax(logit,dim=1)
                pred=prob.argmax(dim=1); ya.append(y.cpu().numpy()); pa.append(pred.cpu().numpy())
            ya=np.concatenate(ya); pa=np.concatenate(pa); f1.append(f1_score(ya,pa,average='macro'))
    def smooth(a):
        a=np.array(a,dtype=float)
        if a.size<3: return a
        s=a.copy()
        for i in range(1,a.size-1): s[i]=(a[i-1]+a[i]+a[i+1])/3.0
        return s
    return sig, smooth(f1)
sig_e,f1_e=curve(True); sig_s,f1_s=curve(False)
plt.figure(figsize=(7,4.5))
plt.plot(sig_e,f1_e,'o-',color='#e1909c',label='EviMR-Net')
plt.plot(sig_s,f1_s,'o--',color='#4a4a4a',label='Softmax')
plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{ds}: Noise Robustness (Comparison)'); plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
out=os.path.join(outb, ds, 'noise_compare'); plt.savefig(out+'.png',dpi=300); plt.savefig(out+'.svg'); plt.close()
print('Saved noise compare for', ds)
PY

  # 7) 临床案例图（最不自信Top-6）
  python -m MERIT.scripts.plot_cases \
    --dataset "$DS" \
    --root_path "$ROOT" \
    --resolution_list "$RES" \
    --uncertainty_base "$OUT_BASE/$DS" \
    --top_k 6 \
    --gpu "$GPU" || true
}

# 主程序：如需单独某一数据集，传入环境变量 ONLY=<DS>
if [ -n "$ONLY" ]; then
  run_eval_dataset "$ONLY"
else
  for DS in APAVA PTB PTB-XL; do
    run_eval_dataset "$DS"
  done
fi

echo "\n✅ EVAL ONLY 完成。图与报告见 $OUT_BASE/<DATASET>/ 下"


