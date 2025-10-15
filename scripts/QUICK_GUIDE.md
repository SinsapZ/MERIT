# MERIT 实验快速指南

## ✅ 7个核心文件

```
scripts/
├── 1. multi_seed_run.py           # 核心运行器
├── 2. find_best_params.sh         # 超参数搜索 (3×3×3=27配置)
├── 3. run_all_datasets.sh         # 主实验 (4数据集×10seeds)
├── 4. run_baselines.sh            # Baseline对比
├── 5. summarize_all_datasets.py   # 结果汇总+LaTeX表格
├── 6. evaluate_uncertainty.py     # 不确定性评估(ESWA核心)
└── 7. README.md                   # 使用文档
```

---

## 🚀 完整流程

### 0️⃣ 超参数搜索（10小时）

```bash
cd /home/Data1/zbl

# 每个数据集搜索27个配置
bash MERIT/scripts/find_best_params.sh APAVA 0
bash MERIT/scripts/find_best_params.sh ADFD-Sample 0
bash MERIT/scripts/find_best_params.sh PTB 0
bash MERIT/scripts/find_best_params.sh PTB-XL 0

# 查看最佳配置
cat results/param_search/*/best_config.txt
```

---

### 1️⃣ 更新配置 → 2️⃣ 运行主实验 → 3️⃣ Baseline对比 → 4️⃣ 生成表格

详见 `README.md`

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

