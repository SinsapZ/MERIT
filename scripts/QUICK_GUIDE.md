# MERIT 实验快速指南

## 核心脚本

```
scripts/
├── multi_seed_run.py            # 多随机种子运行器
├── find_best_params.sh          # 超参数搜索 (3×3×3=27)
├── run_all_datasets.sh          # 主实验 (3数据集：APAVA, PTB, PTB-XL)
├── run_baselines.sh             # （可选）Baseline对比（MedGNN / iTransformer / FEDformer / ECGFM / ECGFounder / FORMED）
├── run_ablation.sh / run_ablation_ptb.sh   # 消融实验 (5变体)
├── run_ablation_all.py          # 消融实验快速版（多数据集一键）
├── summarize_all_datasets.py    # 结果汇总（含 LaTeX）
├── evaluate_uncertainty.py      # 不确定性评估
├── analyze_uncertainty.py       # 不确定性分析
└── README.md / QUICK_GUIDE.md   # 文档
```

---

## 实验流程

### 0. 超参数搜索

```bash
bash MERIT/scripts/find_best_params.sh APAVA 0
bash MERIT/scripts/find_best_params.sh ADFD-Sample 0
bash MERIT/scripts/find_best_params.sh PTB 0
bash MERIT/scripts/find_best_params.sh PTB-XL 0
```

### 1. 主实验

```bash
bash MERIT/scripts/run_all_datasets.sh
```

### 2. Baseline 对比

使用各仓库官方多数据集脚本：

- iTransformer（支持 `--with_swa`）：
  ```bash
  python -m iTransformer.scripts.run_ecg_subject_all \
    --datasets APAVA,PTB,PTB-XL \
    --apava_root dataset/APAVA \
    --ptb_root dataset/PTB \
    --ptbxl_root dataset/PTB-XL \
    --gpu 0 --seeds 3 \
    --out_dir results/iTransformer_final_all_datasets \
    --with_swa
  ```
- MedGNN：
  ```bash
  python -m MedGNN.scripts.run_subject_all \
    --datasets APAVA,PTB,PTB-XL \
    --apava_root dataset/APAVA \
    --ptb_root dataset/PTB \
    --ptbxl_root dataset/PTB-XL \
    --gpu 0 --num_seeds 3 \
    --out_dir results/MedGNN_final_all_datasets
  ```

> 依赖说明：需要安装 MedGNN、FEDformer、FORMED 等外部仓库；TimesFM 若仅有 `.safetensors`，脚本会自动转换为 `.pth`。

### 3. 消融实验

```bash
bash MERIT/scripts/run_ablation.sh PTB-XL 0
bash MERIT/scripts/run_ablation.sh ADFD-Sample 0
bash MERIT/scripts/run_ablation_ptb.sh 0
```

快速版（多数据集一键，迭代较少，便于快速验证）：

```bash
# 说明：
# - 默认跑 APAVA,PTB,PTB-XL（如需子集，用 --datasets 指定）
# - 用 --root_paths 指定各数据集根目录
# - --max_epochs / --patience 控制更快的轮次与早停
# - 结果写入 results/ablation_all_quick/<DATASET>/<variant>.csv

python -m MERIT.scripts.run_ablation_all \
  --datasets APAVA,PTB,PTB-XL \
  --root_paths APAVA=dataset/APAVA,PTB=dataset/PTB/PTB,PTB-XL=dataset/PTB-XL/PTB-XL \
  --gpu 0 \
  --seeds 41,42 \
  --max_epochs 80 \
  --patience 10
```

变体说明：
- Full Model：证据融合（DS）+ pseudo-view + 频域分支 + 差分分支
- w/o Evidential Fusion：`--agg mean --no_pseudo`（关闭 DS，用简单平均，且不使用伪视图）
- w/o Pseudo-view：`--no_pseudo --lambda_pseudo_loss 0.0`
- w/o Frequency Branch：`--no_freq`
- w/o Difference Branch：`--no_diff`

### 4. 不确定性分析

一键运行三数据集（APAVA, PTB, PTB-XL）：
```bash
bash MERIT/scripts/run_uncertainty_all.sh 0
```

输出（每个数据集 `results/uncertainty/<DATASET>/`）：
- evi/ 与 softmax/: uncertainties.npy, confidences.npy, predictions.npy, labels.npy
- 单方法图（PNG+SVG）：
  - plots_evi/<DATASET>_reliability.png|svg（可靠度图）
  - plots_evi/<DATASET>_selective.png|svg（选择性预测）
  - 同理 plots_soft/ 下为 Softmax 基线
- 对比图（PNG+SVG）：
  - acc_vs_reject_compare.png|svg（EviMR vs Softmax）
- 不确定度分布（自适应y上限）：
  - uncert_density_evi.png
- 噪声鲁棒性：
  - noise_evi.png, noise_soft.png
- 人机协同（默认拒绝率20%）：
  - plots_evi/triage_summary.txt
  - plots_evi/triage_candidates.csv
- 案例图：
  - cases/sample*_wave.png, sample*_prob.png

只跑单个数据集（以 APAVA 为例）：
```bash
# 训练并保存不确定性数组
python -m MERIT.run --model MERIT --data APAVA --root_path dataset/APAVA \
  --use_ds --learning_rate 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.3 \
  --annealing_epoch 50 --resolution_list 2,4,6,8 --batch_size 64 --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --weight_decay 1e-4 --nodedim 10 --gpu 0 --swa \
  --save_uncertainty --uncertainty_dir results/uncertainty/APAVA/evi

python -m MERIT.run --model MERIT --data APAVA --root_path dataset/APAVA \
  --learning_rate 1e-4 --resolution_list 2,4,6,8 --batch_size 64 --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --weight_decay 1e-4 --nodedim 10 --gpu 0 --swa \
  --save_uncertainty --uncertainty_dir results/uncertainty/APAVA/softmax

# iTransformer（支持 SWA 开关）
python -m iTransformer.scripts.run_ecg_subject_all \
  --datasets APAVA \
  --apava_root dataset/APAVA \
  --gpu 0 --seeds 3 \
  --out_dir results/iTransformer_final_all_datasets \
  --with_swa

# 单方法评估（支持自定义调色板与拒绝率，导出PNG+SVG）
python -m MERIT.scripts.evaluate_uncertainty \
  --uncertainty_dir results/uncertainty/APAVA/evi \
  --dataset_name APAVA \
  --output_dir results/uncertainty/APAVA/plots_evi \
  --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a' \
  --reject_rate 20

# EviMR vs Softmax 对比曲线（导出PNG+SVG）
python -m MERIT.scripts.compare_selective \
  --base_dir results/uncertainty/APAVA \
  --dataset APAVA \
  --palette 'e1d89c,e1c59c,e1ae9c,e1909c,4a4a4a'
```

仅评估并导出不确定性（复用已训练 checkpoint）：
```bash
# 注意：需要已有 checkpoint，参数需与训练一致（用于定位 setting）
python -m MERIT.scripts.multi_seed_run \
  --root_path dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --nodedim 10 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43" \
  --eval_only
# 导出的数组保存在对应 checkpoint/setting/uncertainty 目录
```

配色规范（Tailwind）：
- Vanilla #e1d89c, Tan #e1c59c, Melon #e1ae9c, Puce #e1909c, Davy’s gray #4a4a4a
- 约定：EviMR 主线用 Puce，Softmax 主线用 Davy’s gray；辅助填充用 Vanilla/Tan/Melon。

### 5. 结果汇总

```bash
python MERIT/scripts/summarize_all_datasets.py
```

---

更多参数与说明见 `README.md`。

