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

### 4️⃣ 不确定性分析（ESWA核心，需修改代码）

```bash
python MERIT/scripts/evaluate_uncertainty.py --uncertainty_dir <path>
python MERIT/scripts/analyze_uncertainty.py --uncertainty_dir <path>
```

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

