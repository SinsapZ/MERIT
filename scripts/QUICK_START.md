# MERIT 快速开始指南

## 🎯 一键运行

### 运行所有数据集（推荐用于最终论文结果）

```bash
cd /home/Data1/zbl
bash MERIT/scripts/run_all_datasets.sh
```

**时间**: ~8小时  
**输出**: 4个数据集的完整结果 + 5个指标

---

### 运行单个数据集（快速测试）

```bash
# APAVA (心律失常，9类)
bash MERIT/scripts/run_apava.sh

# ADFD (Alzheimer，2类)  
bash MERIT/scripts/run_adfd.sh

# PTB (心肌梗死，2类)
bash MERIT/scripts/run_ptb.sh

# PTB-XL (大规模，5类)
bash MERIT/scripts/run_ptbxl.sh
```

---

## 📊 查看结果

### 1. 单个数据集汇总

```bash
cat results/final_all_datasets/apava_results_summary.txt
```

### 2. 所有数据集汇总表格

```bash
python MERIT/scripts/summarize_all_datasets.py
```

**输出**:
- ✅ 完整结果表格（5个指标）
- ✅ LaTeX格式（直接用于论文）
- ✅ CSV文件（Excel可打开）

---

## 📁 文件结构

```
MERIT/scripts/
├── run_all_datasets.sh          # 一键运行所有数据集
├── run_apava.sh                 # 单独运行APAVA
├── run_adfd.sh                  # 单独运行ADFD
├── run_ptb.sh                   # 单独运行PTB
├── run_ptbxl.sh                 # 单独运行PTB-XL
├── summarize_all_datasets.py    # 汇总分析脚本
└── EXPERIMENTS_GUIDE.md         # 详细实验指南

results/final_all_datasets/
├── apava_results.csv            # APAVA详细结果
├── apava_results_summary.txt    # APAVA汇总
├── adfd_results.csv             # ADFD详细结果
├── adfd_results_summary.txt     # ADFD汇总
├── ptb_results.csv              # PTB详细结果
├── ptb_results_summary.txt      # PTB汇总
├── ptbxl_results.csv            # PTB-XL详细结果
├── ptbxl_results_summary.txt    # PTB-XL汇总
└── summary_all_datasets.csv     # 所有数据集汇总
```

---

## 📈 输出的5个指标

每个实验都会报告：

1. **Accuracy** (准确率)
2. **Precision** (精确率)  
3. **Recall** (召回率)
4. **F1 Score** (F1分数)
5. **AUROC** (ROC曲线下面积)

每个指标包括：
- Mean (平均值)
- Std (标准差) - 基于10个随机种子

---

## ⚡ 快速测试（3个seeds）

如果只想快速验证，修改脚本中的seeds：

```bash
# 编辑任意脚本，将第一行改为
SEEDS="41,42,43"  # 从10个改为3个

# 时间缩短到原来的30%
```

---

## 🎓 论文写作

运行完成后：

```bash
# 生成汇总表格（包含LaTeX格式）
python MERIT/scripts/summarize_all_datasets.py > results_summary.txt

# 直接复制LaTeX表格到论文
```

**表格标题建议**:
> Table 1: MERIT Performance on Multiple Medical Datasets. Results are reported as Mean±Std (%) over 10 random seeds.

---

## 🐛 常见问题

**Q: 运行很慢？**  
A: 正常。10个seeds × 150-200 epochs 需要1-2小时/数据集

**Q: 显存不够？**  
A: 减小batch_size（64→32）

**Q: 某个seed失败？**  
A: 没关系，只要≥8个seeds成功就可以

---

## 📞 需要帮助？

查看详细文档：`EXPERIMENTS_GUIDE.md`

