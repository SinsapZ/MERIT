# MERIT 实验脚本使用指南

## 📁 核心脚本（9个）

### 1️⃣ `multi_seed_run.py` ⭐核心
多随机种子实验运行器，自动统计Mean±Std。

### 2️⃣ `find_best_params.sh` ⭐调参
快速搜索最佳超参数（3×3×3=27个配置，2-3小时）

### 3️⃣ `run_all_datasets.sh` ⭐主实验
一键运行4个数据集的完整实验（用最佳配置）

### 4️⃣ `run_baselines.sh` ⭐对比
运行Medformer和iTransformer baseline

### 5️⃣ `run_ablation.sh` ⭐消融
5个变体消融实验（证明各组件有效性）

### 6️⃣ `summarize_all_datasets.py` ⭐汇总
生成论文表格（LaTeX格式）

### 7️⃣ `evaluate_uncertainty.py` ⭐ESWA核心
不确定性评估：ECE, Selective Prediction等

### 8️⃣ `analyze_uncertainty.py` ⭐ESWA分析
全面不确定性分析：噪声鲁棒性、分布、拒绝实验、案例

### 9️⃣ `README.md` + `QUICK_GUIDE.md`
使用文档

---

## 🚀 完整实验流程（4步）

### Step 0: 超参数搜索（先做这个！）⭐

```bash
cd /home/Data1/zbl

# 为每个数据集找最佳参数（2-3小时/数据集）
bash MERIT/scripts/find_best_params.sh APAVA 0
bash MERIT/scripts/find_best_params.sh ADFD-Sample 0
bash MERIT/scripts/find_best_params.sh PTB 0
bash MERIT/scripts/find_best_params.sh PTB-XL 0

# 查看最佳配置
cat results/param_search/APAVA/best_config.txt
cat results/param_search/ADFD-Sample/best_config.txt
cat results/param_search/PTB/best_config.txt
cat results/param_search/PTB-XL/best_config.txt
```

**搜索空间**:
- 学习率: 1e-4, 1.5e-4, 2e-4
- Lambda_view: 0.5, 1.0, 1.5
- Lambda_pseudo: 0.2, 0.3, 0.5

**时间**: 约10小时（4个数据集）

---

## 🚀 然后三步完成主实验

### Step 1: 用最佳配置更新 run_all_datasets.sh

根据`best_config.txt`，修改`run_all_datasets.sh`中各数据集的参数。

---

### Step 2: 运行完整实验 (6-8小时)

```bash
bash MERIT/scripts/run_all_datasets.sh
```

**输出**: 4个数据集 × 10 seeds × 5指标

---

### Step 3: Baseline对比 (4小时)

```bash
bash MERIT/scripts/run_baselines.sh APAVA
bash MERIT/scripts/run_baselines.sh ADFD-Sample
bash MERIT/scripts/run_baselines.sh PTB
bash MERIT/scripts/run_baselines.sh PTB-XL
```

---

### Step 4: 生成论文表格 (1分钟)

```bash
python MERIT/scripts/summarize_all_datasets.py
```

**输出**: LaTeX表格，直接用于论文

---

## 📊 各数据集配置（已优化）

| Dataset | lr | epochs | 其他关键参数 |
|---------|-----|--------|--------------|
| APAVA | 1.1e-4 | 200 | lambda=(1.0,1.0,0.3) |
| ADFD | 1e-4 | 150 | e_layers=6, dropout=0.2 |
| PTB | 1e-4 | 150 | - |
| PTB-XL | 2e-4 | 100 | - |

**注**: 配置已在脚本中设置好

---

## 🎯 ESWA投稿要点

### 核心创新
1. 多视角证据融合 (DS理论)
2. 不确定性量化 (MedGNN缺失)
3. Pseudo-view机制

### 关键卖点: Selective Prediction

| Coverage | MERIT | MedGNN |
|----------|-------|--------|
| 100% | 77% | 82.6% |
| 70% | ~84% | 82.6% ← **超越** |

**论文角度**: 不确定性感知的医疗AI系统，支持人机协作

---

## 📋 ESWA完整实验清单

### 必做实验（8个）

1. ✅ **4个数据集性能** - `run_all_datasets.sh`
2. ✅ **Baseline对比** - `run_baselines.sh` (Medformer, iTransformer)
3. ✅ **消融实验** - `run_ablation.sh` (5个变体)
4. ✅ **ECE校准** - `evaluate_uncertainty.py`
5. ✅ **Selective Prediction** - `evaluate_uncertainty.py`
6. ✅ **不确定性分布** - `analyze_uncertainty.py`
7. ✅ **拒绝实验** - `analyze_uncertainty.py`
8. ✅ **案例可视化** - `analyze_uncertainty.py`

### 可选实验（增强）

9. ⭐ 噪声鲁棒性实验
10. ⭐ OOD检测实验

---

## ⏱️ 完整时间规划

| 任务 | 脚本 | 时间 |
|------|------|------|
| 超参数搜索 | find_best_params.sh | 10小时 |
| 主实验(4数据集) | run_all_datasets.sh | 8小时 |
| Baseline对比 | run_baselines.sh | 4小时 |
| 消融实验(2数据集) | run_ablation.sh | 4小时 |
| 不确定性评估 | evaluate/analyze_uncertainty.py | 2小时 |
| **总计** | - | **~28小时** |

**写论文**: 2周

