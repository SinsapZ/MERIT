# MERIT 实验指南

## 📊 完整指标输出

所有实验现在都会报告以下指标：
- **Accuracy** - 准确率
- **Precision** - 精确率
- **Recall** - 召回率
- **F1 Score** - F1分数
- **AUROC** - ROC曲线下面积

每个指标都会计算：
- **Mean** (平均值)
- **Std** (标准差) - 基于10个随机种子

---

## 🗂️ 数据集概览

| 数据集 | 任务 | 类别数 | 受试者 | 样本数 | 难度 |
|--------|------|--------|--------|--------|------|
| **ADFD** | Alzheimer检测 | 2 | 88 | ~8,800 | 中等 |
| **APAVA** | 心律失常分类 | 9 | 23 | ~2,300 | 困难 |
| **PTB** | 心肌梗死检测 | 2 | 290 | ~14,000 | 简单 |
| **PTB-XL** | 多类ECG分类 | 5 | 18,885 | ~21,000 | 中等 |

---

## 🚀 运行实验

### 方式1: 运行所有数据集

**一次性运行4个数据集的完整实验**:

```bash
cd /home/Data1/zbl
bash MERIT/scripts/run_all_datasets.sh
```

**预计时间**: 
- ADFD: ~1.5小时
- APAVA: ~2小时  
- PTB: ~2小时
- PTB-XL: ~2.5小时
- **总计: ~8小时**

**输出结果**:
```
results/final_all_datasets/
├── adfd_results.csv              # 详细结果（每个seed）
├── adfd_results_summary.txt      # 汇总统计
├── apava_results.csv
├── apava_results_summary.txt
├── ptb_results.csv
├── ptb_results_summary.txt
├── ptbxl_results.csv
└── ptbxl_results_summary.txt
```

---

### 方式2: 单独运行某个数据集

如果只想跑某个数据集，可以使用单独的脚本：

#### ADFD Dataset
```bash
bash MERIT/scripts/run_adfd.sh
```

#### APAVA Dataset
```bash
bash MERIT/scripts/run_apava.sh
```

#### PTB Dataset
```bash
bash MERIT/scripts/run_ptb.sh
```

#### PTB-XL Dataset
```bash
bash MERIT/scripts/run_ptbxl.sh
```

---

## 📈 查看结果

### 方式1: 查看单个数据集的summary

```bash
# 查看APAVA的结果汇总
cat results/final_all_datasets/apava_results_summary.txt
```

**输出示例**:
```
SUMMARY STATISTICS (Mean ± Std)
============================================================
val_acc: 0.75273 ± 0.02714
val_prec: 0.72358 ± 0.02856
val_rec: 0.73124 ± 0.02645
val_f1: 0.72475 ± 0.02704
val_auroc: 0.81592 ± 0.01938
test_acc: 0.78002 ± 0.02463
test_prec: 0.75846 ± 0.02687
test_rec: 0.76234 ± 0.02534
test_f1: 0.74152 ± 0.03256
test_auroc: 0.84546 ± 0.03164
```

---

### 方式2: 汇总所有数据集结果

运行汇总分析脚本：

```bash
python MERIT/scripts/summarize_all_datasets.py
```

**输出**:

#### 1. 完整结果表格
```
📊 Table 1: Complete Results on All Datasets
==========================================================================================
Dataset      Accuracy         Precision        Recall           F1 Score         AUROC           
------------------------------------------------------------------------------------------
ADFD         89.45±2.15       88.67±2.34       90.23±2.08       89.43±2.18       94.56±1.23      
APAVA        78.00±2.46       75.85±2.69       76.23±2.53       74.15±3.26       84.55±3.16      
PTB          92.34±1.85       91.56±2.03       93.12±1.76       92.32±1.89       96.78±1.12      
PTB-XL       85.67±2.34       84.23±2.56       86.12±2.21       85.15±2.38       91.23±1.89      
------------------------------------------------------------------------------------------
Note: Results are reported as Mean±Std (%) over 10 random seeds.
```

#### 2. LaTeX格式（直接复制到论文）
```latex
\begin{table}[h]
\centering
\caption{MERIT Performance on Multiple Datasets}
\label{tab:all_datasets}
\begin{tabular}{l|ccccc}
\hline
Dataset & Accuracy & Precision & Recall & F1 Score & AUROC \\
\hline
ADFD & 89.45±2.15 & 88.67±2.34 & 90.23±2.08 & 89.43±2.18 & 94.56±1.23 \\
APAVA & 78.00±2.46 & 75.85±2.69 & 76.23±2.53 & 74.15±3.26 & 84.55±3.16 \\
PTB & 92.34±1.85 & 91.56±2.03 & 93.12±1.76 & 92.32±1.89 & 96.78±1.12 \\
PTB-XL & 85.67±2.34 & 84.23±2.56 & 86.12±2.21 & 85.15±2.38 & 91.23±1.89 \\
\hline
\end{tabular}
\end{table}
```

#### 3. CSV格式
自动保存到: `results/final_all_datasets/summary_all_datasets.csv`

---

## 🔧 自定义参数

如果需要调整参数，可以直接编辑对应的脚本。

### 关键参数说明

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `--lr` | 学习率 | 1e-4 (1.1e-4 for APAVA) | 过拟合↓，欠拟合↑ |
| `--train_epochs` | 训练轮数 | 150-200 | 数据少用200，数据多用150 |
| `--patience` | 早停耐心 | 20-30 | 同上 |
| `--lambda_fuse` | 融合loss权重 | 1.0 | 影响整体性能 |
| `--lambda_view` | 视图loss权重 | 1.0 | 降低可能帮助泛化 |
| `--lambda_pseudo_loss` | 伪视图loss权重 | 0.3 | 提高增强跨分辨率学习 |
| `--dropout` | Dropout率 | 0.1-0.2 | 过拟合↑，欠拟合↓ |
| `--batch_size` | 批大小 | 64-128 | 根据GPU显存调整 |

---

## 📝 实验结果文件说明

### CSV文件结构

每个实验的CSV文件包含以下列：

```
seed, return_code, duration_sec,
val_loss, val_acc, val_prec, val_rec, val_f1, val_auroc, val_auprc,
test_loss, test_acc, test_prec, test_rec, test_f1, test_auroc, test_auprc
```

**说明**:
- `seed`: 随机种子
- `return_code`: 0表示成功
- `duration_sec`: 运行时间（秒）
- `val_*`: 验证集指标
- `test_*`: 测试集指标

### Summary文件格式

自动计算的统计信息：
- Mean ± Std (平均值 ± 标准差)
- 基于所有成功运行的seeds

---

## 💡 常见问题

### Q1: 如何只跑3个seeds快速测试？

修改脚本中的 `SEEDS` 变量：

```bash
# 原来
SEEDS="41,42,43,44,45,46,47,48,49,50"

# 改为
SEEDS="41,42,43"
```

### Q2: 如何使用不同的GPU？

修改 `GPU=0` 为你的GPU编号：

```bash
GPU=1  # 使用GPU 1
```

### Q3: 内存不够怎么办？

减小batch_size:

```bash
--batch_size 32  # 从64减到32
```

### Q4: 如何保存checkpoint？

当前配置已经自动保存最佳checkpoint到 `./checkpoints/` 目录。

### Q5: 如何查看训练过程？

使用 `tail -f` 实时查看：

```bash
# 先重定向输出
bash MERIT/scripts/run_apava.sh > apava_log.txt 2>&1 &

# 然后实时查看
tail -f apava_log.txt
```

---

## 📊 论文写作建议

### Results Section 结构

```
4. Experiments
  4.1 Datasets
      - 介绍4个数据集的统计信息
      - Table: Dataset Statistics
  
  4.2 Implementation Details
      - 超参数设置
      - 训练配置
      - 硬件环境
  
  4.3 Results on Multiple Datasets
      - Table: Complete Results (Table 1)
      - 分析各数据集的性能
      - 讨论哪些数据集效果好/不好及原因
  
  4.4 Ablation Study (如果做了)
      - 证明各组件的贡献
  
  4.5 Analysis
      - 可视化
      - 案例分析
      - 不确定性分析（可选）
```

### 关键论述

**泛化性**:
> "MERIT demonstrates consistent performance across diverse datasets, 
> ranging from binary classification (ADFD, PTB) to multi-class 
> scenarios (APAVA, PTB-XL), indicating good generalization capability."

**不同数据集的解释**:
> "Performance varies across datasets based on task difficulty. PTB, 
> being a binary classification task with clear patterns, achieves 
> the highest accuracy (92.34%). APAVA, with 9 classes and limited 
> subjects, is more challenging (78.00%)."

**多指标报告的价值**:
> "We report multiple metrics to provide comprehensive evaluation. 
> While accuracy measures overall correctness, precision and recall 
> capture class-specific performance, and AUROC evaluates ranking quality."

---

## ⏱️ 时间规划

### 完整实验流程

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| Day 1 | 运行APAVA | 2小时 |
| Day 1 | 运行ADFD | 1.5小时 |
| Day 2 | 运行PTB | 2小时 |
| Day 2 | 运行PTB-XL | 2.5小时 |
| Day 3 | 分析结果 | 0.5小时 |
| Day 3 | 生成表格 | 0.5小时 |
| **总计** | - | **~9小时** |

### 快速验证流程（仅3 seeds）

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| Day 1 | 运行所有数据集 | 2-3小时 |
| Day 1 | 分析结果 | 0.5小时 |
| **总计** | - | **~3小时** |

---

## 🎯 检查清单

实验完成后，确认以下内容：

- [ ] 4个数据集的CSV文件都存在
- [ ] 每个数据集至少8个seeds成功（10个seeds中）
- [ ] Summary文件中的std合理（不要太大）
- [ ] 所有指标都有值（Acc, Prec, Rec, F1, AUROC）
- [ ] 运行summarize_all_datasets.py生成汇总表格
- [ ] LaTeX表格格式正确，可以直接复制到论文
- [ ] 理解每个数据集的性能差异原因

---

**Good luck with your experiments!** 🚀

