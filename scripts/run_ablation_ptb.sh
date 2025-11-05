#!/bin/bash
# PTB 消融实验（5个配置 × 多seed）
# 1) Full (EviMR-Net)
# 2) w/o Evidential Fusion (Concat + Softmax)
# 3) w/o Difference Branch
# 4) w/o Frequency Branch
# 5) w/o Pseudo-view

set -e

GPU=${1:-0}
SEEDS=${2:-"41,42,43"}

DATA=PTB
ROOT=/home/Data1/zbl/dataset/PTB

# 来自最优配置（PTB）
LR=1.5e-4
ANNEAL=40
E_LAYERS=4
RES="2,4,6,8"
BATCH=64
EPOCHS=150
PATIENCE=20
DROPOUT=0.1
WD=0
NODEDIM=10
LAM_FUSE=1.0
LAM_VIEW=1.0
LAM_PSEUDO_LOSS=0.2

OUT_DIR=results/ablation/PTB
mkdir -p "$OUT_DIR"

echo "==================================="
echo "PTB Ablation running on GPU=$GPU"
echo "Seeds: $SEEDS"
echo "Output: $OUT_DIR"
echo "==================================="

declare -A NAMES
declare -A EXTRA

NAMES[full]="Full (EviMR-Net)"
EXTRA[full]="--use_ds"

NAMES[no_evi]="w/o Evidential Fusion (Softmax)"
EXTRA[no_evi]=""

NAMES[no_diff]="w/o Difference Branch"
EXTRA[no_diff]="--use_ds --no_diff"

NAMES[no_freq]="w/o Frequency Branch"
EXTRA[no_freq]="--use_ds --no_freq"

NAMES[no_pseudo]="w/o Pseudo-view"
EXTRA[no_pseudo]="--use_ds --no_pseudo"

CONFIGS=(full no_evi no_diff no_freq no_pseudo)

parse_and_append() {
  local txt="$1"; local csv="$2"; local seed="$3"
  # 提取 Test results 指标
  python - <<PY "$txt" "$csv" "$seed"
import re,sys,csv,os
txt=open(sys.argv[1],'r',encoding='utf-8',errors='ignore').read()
csv_path=sys.argv[2]
seed=sys.argv[3]
pat=r"Test results --- Loss: ([0-9\.]+), Accuracy: ([0-9\.]+), Precision: ([0-9\.]+), Recall: ([0-9\.]+), F1: ([0-9\.]+), AUROC: ([0-9\.]+), AUPRC: ([0-9\.]+)"
m=re.search(pat,txt)
row=[seed,1,'', '', '', '', '', '', '', '', '', '', '', '']
if m:
    loss,acc,prec,rec,f1,auroc,auprc=m.groups()
    row=[seed,0,'', '', '', '', '', '', loss,acc,prec,rec,f1,auroc,auprc]
hdr=['seed','return_code','duration_sec','val_loss','val_acc','val_prec','val_rec','val_f1','test_loss','test_acc','test_prec','test_rec','test_f1','test_auroc','test_auprc']
write_header=not os.path.exists(csv_path)
with open(csv_path,'a',newline='') as f:
    w=csv.writer(f)
    if write_header: w.writerow(hdr)
    w.writerow(row)
PY
}

for key in "${CONFIGS[@]}"; do
  NAME=${NAMES[$key]}
  EXTRA_ARGS=${EXTRA[$key]}
  CSV="$OUT_DIR/${key}.csv"
  rm -f "$CSV"
  echo "\n================ ${NAME} ================"
  IFS=',' read -ra ARR <<< "$SEEDS"
  for s in "${ARR[@]}"; do
    echo "Seed $s ..."
    LOG="$OUT_DIR/run_${key}_seed${s}.log"
    set +e
    python -m MERIT.run \
      --model MERIT --data "$DATA" --root_path "$ROOT" \
      --learning_rate "$LR" --annealing_epoch "$ANNEAL" \
      --resolution_list "$RES" --batch_size "$BATCH" \
      --train_epochs "$EPOCHS" --patience "$PATIENCE" \
      --e_layers "$E_LAYERS" --dropout "$DROPOUT" --weight_decay "$WD" \
      --nodedim "$NODEDIM" --gpu "$GPU" --swa \
      --lambda_fuse "$LAM_FUSE" --lambda_view "$LAM_VIEW" --lambda_pseudo_loss "$LAM_PSEUDO_LOSS" \
      --seed "$s" $EXTRA_ARGS \
      > "$LOG" 2>&1
    RC=$?
    set -e
    parse_and_append "$LOG" "$CSV" "$s"
  done

  # 汇总
  python - <<PY "$CSV" "$NAME"
import pandas as pd,sys
df=pd.read_csv(sys.argv[1])
ok=df[df['return_code']==0]
if ok.empty:
    print('No successful runs for', sys.argv[2]); sys.exit(0)
print('\n---- Summary:', sys.argv[2], '----')
for k in ['test_acc','test_f1','test_auroc']:
    print(f"{k}: {ok[k].mean():.4f} ± {ok[k].std():.4f}")
PY
done

echo "\n✅ Done. CSV files at: $OUT_DIR"


