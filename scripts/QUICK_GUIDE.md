# MERIT 实验快速指南

## ✅ 9个核心文件

```
scripts/
├── 1. multi_seed_run.py           # 核心运行器
├── 2. find_best_params.sh         # 超参数搜索 (3×3×3=27)
├── 3. run_all_datasets.sh         # 主实验 (4数据集)
├── 4. run_baselines.sh            # Baseline对比
├── 5. run_ablation.sh             # 消融实验 (5变体)
├── 6. summarize_all_datasets.py   # 结果汇总+LaTeX
├── 7. evaluate_uncertainty.py     # 不确定性评估
├── 8. analyze_uncertainty.py      # 不确定性全面分析
└── 9. README.md / QUICK_GUIDE.md  # 文档
```

---

## 🚀 完整实验流程（ESWA要求）

### 0️⃣ 超参数搜索（10小时）

```bash
bash MERIT/scripts/find_best_params.sh APAVA 0
bash MERIT/scripts/find_best_params.sh ADFD-Sample 0
bash MERIT/scripts/find_best_params.sh PTB 0
bash MERIT/scripts/find_best_params.sh PTB-XL 0
```

### 1️⃣ 主实验（更新配置后，8小时）

```bash
bash MERIT/scripts/run_all_datasets.sh
```

### 2️⃣ Baseline对比（4小时）

```bash
bash MERIT/scripts/run_baselines.sh <DATASET>
```

### 3️⃣ 消融实验（4小时，在PTB-XL和ADFD上）

```bash
bash MERIT/scripts/run_ablation.sh PTB-XL 0
bash MERIT/scripts/run_ablation.sh ADFD-Sample 0
```

### 4️⃣ 不确定性分析（一键执行 + 人机协同产物 + SVG 导出）

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
  - plots_evi/triage_summary.txt（放行后准确率提升等）
  - plots_evi/triage_candidates.csv（最不自信样本清单，供医生复核）
- 案例图：
  - cases/sample*_wave.png, sample*_prob.png

只跑单个数据集（以 APAVA 为例）：
```bash
# 训练并保存不确定性数组
python -m MERIT.run --model MERIT --data APAVA --root_path /home/Data1/zbl/dataset/APAVA \
  --use_ds --learning_rate 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.3 \
  --annealing_epoch 50 --resolution_list 2,4,6,8 --batch_size 64 --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --weight_decay 1e-4 --nodedim 10 --gpu 0 --swa \
  --save_uncertainty --uncertainty_dir results/uncertainty/APAVA/evi

python -m MERIT.run --model MERIT --data APAVA --root_path /home/Data1/zbl/dataset/APAVA \
  --learning_rate 1e-4 --resolution_list 2,4,6,8 --batch_size 64 --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --weight_decay 1e-4 --nodedim 10 --gpu 0 --swa \
  --save_uncertainty --uncertainty_dir results/uncertainty/APAVA/softmax

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

配色规范（Tailwind）：
- Vanilla #e1d89c, Tan #e1c59c, Melon #e1ae9c, Puce #e1909c, Davy’s gray #4a4a4a
- 约定：EviMR 主线用 Puce，Softmax 主线用 Davy’s gray；辅助填充用 Vanilla/Tan/Melon。

### 5️⃣ 生成论文表格

```bash
python MERIT/scripts/summarize_all_datasets.py
```

---

## 🎯 ESWA投稿策略

### 核心创新
- 多视角证据融合（DS理论）
- 不确定性量化
- Pseudo-view机制

### 超越MedGNN的点
**Selective Prediction**: 70% coverage时达到~84%，超过MedGNN的82.6%

### 论文角度
不确定性感知的医疗AI系统，支持临床人机协作

---

## ⏱️ 时间规划

| 任务 | 时间 |
|------|------|
| 超参数搜索 | 10小时 |
| 主实验 | 8小时 |
| Baseline对比 | 4小时 |
| 不确定性实验 | 1天 |
| 写论文 | 2周 |

---

**完整文档**: 见 `README.md`

