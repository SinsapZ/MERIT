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
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$EVI_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_evi" || true
  python -m MERIT.scripts.evaluate_uncertainty --uncertainty_dir "$SOFT_DIR" --dataset_name "$DS" --output_dir "$OUT_BASE/$DS/plots_soft" || true

  # --- 4) 叠加比较：准确率-拒绝率曲线 & 不确定性分布 ---
  python - <<PY || true
import os, numpy as np, matplotlib.pyplot as plt
from MERIT.scripts.evaluate_uncertainty import selective_prediction
ds = "$DS"; base = "$OUT_BASE"
evi = os.path.join(base, ds, 'evi')
soft= os.path.join(base, ds, 'softmax')
os.makedirs(os.path.join(base, ds), exist_ok=True)

# 覆盖-准确率曲线对比（转为拒绝率）
def load_curve(d):
    c=np.load(os.path.join(d,'confidences.npy')); y=np.load(os.path.join(d,'labels.npy')); p=np.load(os.path.join(d,'predictions.npy'))
    cov, acc = selective_prediction(c,p,y)
    rej = 100 - cov
    return rej, acc
re_evi, ac_evi = load_curve(evi)
re_sft, ac_sft = load_curve(soft)
plt.figure(figsize=(8,5))
plt.plot(re_evi, ac_evi, 'b-o', label='EviMR-Net')
plt.plot(re_sft, ac_sft, 'r--o', label='Softmax')
plt.xlabel('Rejection rate (%)'); plt.ylabel('Accuracy (%)'); plt.title(f'{ds}: Accuracy vs Rejection'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(base, ds, 'acc_vs_reject_compare.png'), dpi=300)
plt.close()

# 不确定性分布（近似KDE：使用高分辨率直方图作为密度替代）
u = np.load(os.path.join(evi,'uncertainties.npy'))
y = np.load(os.path.join(evi,'labels.npy'))
p = np.load(os.path.join(evi,'predictions.npy'))
err = (p!=y)
plt.figure(figsize=(7,5))
plt.hist(u[~err], bins=60, density=True, alpha=0.4, label='Correct')
plt.hist(u[err],  bins=60, density=True, alpha=0.4, label='Misclassified')
plt.xlabel('Uncertainty (u)'); plt.ylabel('Density'); plt.title(f'{ds}: Uncertainty Distribution (EviMR)'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(base, ds, 'uncert_density_evi.png'), dpi=300)
plt.close()
print('Saved compare curves & uncertainty density for', ds)
PY

  # --- 5) 噪声鲁棒性（EviMR与Softmax） ---
  python - <<PY || true
import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from MERIT.exp.exp_classification import Exp_Classification
import argparse as ap

def noise_curve(ds, use_ds, root, res, e_layers, dropout, bs, lr, wd, seed, gpu, out_png):
    args = ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-NOISE-{"EVI" if use_ds else "SOFT"}', model='MERIT',
                        data=ds, root_path=root, use_gpu=True, gpu=gpu,
                        e_layers=e_layers, dropout=dropout, resolution_list=res,
                        nodedim=$NODEDIM, batch_size=bs, train_epochs=$EPOCHS, patience=$PATIENCE,
                        learning_rate=lr, use_ds=use_ds, swa=True, weight_decay=wd,
                        lr_scheduler='none', warmup_epochs=0, seed=seed, num_workers=4)
    exp = Exp_Classification(args)
    _, test_loader = exp._get_data(flag='TEST')
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1]
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
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(sigmas, f1s, 'o-', label='EviMR-Net' if use_ds else 'Softmax')
    plt.xlabel('Gaussian noise sigma'); plt.ylabel('F1'); plt.title(f'{ds}: Noise Robustness'); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

ds = "$DS"; root = "$ROOT"; res = "$RES"; outb = "$OUT_BASE"
noise_curve(ds, True,  root, res, $E_LAYERS, $DROPOUT, $BATCH_SIZE, $LR, $WD, $SEED, $GPU, os.path.join(outb, ds, 'noise_evi.png'))
noise_curve(ds, False, root, res, $E_LAYERS, $DROPOUT, $BATCH_SIZE, $LR, $WD, $SEED, $GPU, os.path.join(outb, ds, 'noise_soft.png'))
print('Saved noise robustness for', ds)
PY

  # --- 6) 案例图（2-3个样本，EviMR） ---
  python - <<PY || true
import os, torch, numpy as np, matplotlib.pyplot as plt
from MERIT.exp.exp_classification import Exp_Classification
import argparse as ap
ds = "$DS"; root = "$ROOT"; res = "$RES"; outb = "$OUT_BASE"; out_dir = os.path.join(outb, ds, 'cases')
os.makedirs(out_dir, exist_ok=True)
args = ap.Namespace(task_name='classification', model_id=f'UNCERT-{ds}-CASE', model='MERIT',
                    data=ds, root_path=root, use_gpu=True, gpu=$GPU,
                    e_layers=$E_LAYERS, dropout=$DROPOUT, resolution_list=res,
                    nodedim=$NODEDIM, batch_size=16, train_epochs=$EPOCHS, patience=$PATIENCE,
                    learning_rate=$LR, use_ds=True, swa=True, weight_decay=$WD,
                    lr_scheduler='none', warmup_epochs=0, seed=$SEED, num_workers=2)
exp = Exp_Classification(args)
_, test_loader = exp._get_data(flag='TEST')
exp.model.eval(); cnt=0
with torch.no_grad():
  for bx, y, pm in test_loader:
    for i in range(min(3, bx.shape[0])):
      x = bx[i].numpy()
      bx1 = bx[i:i+1].float().cuda(); pm1 = pm[i:i+1].float().cuda()
      alpha,_ = exp.model(bx1, pm1, None, None); prob = (alpha/alpha.sum(dim=1, keepdim=True)).squeeze(0).cpu().numpy()
      K = prob.shape[0]; S = (prob*K).sum(); u = K/S
      plt.figure(figsize=(8,2)); plt.plot(x[:,0]); plt.title(f'Waveform (ch=0), label={int(y[i])}'); plt.tight_layout()
      plt.savefig(os.path.join(out_dir, f'sample{cnt}_wave.png'), dpi=300); plt.close()
      plt.figure(figsize=(4,3)); plt.bar(np.arange(K), prob); plt.title(f'Prob, u={u:.3f}'); plt.tight_layout()
      plt.savefig(os.path.join(out_dir, f'sample{cnt}_prob.png'), dpi=300); plt.close(); cnt+=1
    break
print('Saved case figures for', ds)
PY
}

# ===================== 主程序 =====================
for DS in APAVA PTB PTB-XL; do
  run_one_dataset "$DS"
done

echo "\n✅ 全部完成。不确定性数据与图已保存到: $OUT_BASE/<DATASET>/...\n"


