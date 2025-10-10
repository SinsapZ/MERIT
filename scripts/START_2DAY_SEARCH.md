# 🚀 2天自动搜索 - 启动指南

## ⚡ 快速开始

### 离开前执行这一条命令：

```bash
cd /home/Data1/zbl
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &
echo $! > search.pid
```

**就这样！** 2天后回来查看结果。

---

## 📊 会做什么

### 搜索参数空间

- **学习率**: 5e-5, 8e-5, 1e-4, 1.2e-4, 1.5e-4, 2e-4, 3e-4 (7个)
- **Lambda组合**: 5种不同策略
- **数据集**: APAVA, ADFD, PTB, PTB-XL (4个)

**总配置数**: 7 × 5 × 4 = **140个配置**

### 两阶段策略

```
阶段1: 快速筛选 (3 seeds)
  140个配置 × 3 seeds = 420次运行
  ↓
  自动排序，选择每个数据集的Top-3
  ↓
阶段2: 完整验证 (10 seeds)
  12个Top配置 × 10 seeds = 120次运行
  ↓
  输出最佳配置
```

---

## ⏱️ 时间预估

| 阶段 | 配置数 | 预计时间 |
|------|--------|----------|
| 快速筛选 | 140 | 12-18小时 |
| 完整验证 | 12 | 18-24小时 |
| **总计** | 152 | **30-42小时** |

**安全边界**: < 48小时（2天内完成）✅

---

## 📁 输出结果

### 2天后你会得到

```
results/comprehensive_search/
├── APAVA/
│   ├── quick_*.csv (35个快速筛选结果)
│   ├── full_top*.csv (3个完整验证结果)
│   ├── top3_configs.txt
│   └── recommended_config.txt  ← 最佳配置
├── ADFD-Sample/
│   └── (同上)
├── PTB/
│   └── (同上)
├── PTB-XL/
│   └── (同上)
├── best_configs_summary.txt    ← 所有数据集汇总
└── FINAL_SUMMARY.txt           ← 最终报告
```

---

## 🔍 回来后怎么做

### Step 1: 查看是否完成

```bash
# 检查进程
ps aux | grep run_all_comprehensive_search

# 查看日志最后几行
tail -50 search_all.log

# 查看最终汇总
cat results/comprehensive_search/FINAL_SUMMARY.txt
```

### Step 2: 分析结果

```bash
python MERIT/scripts/analyze_comprehensive_results.py
```

**输出**:
- 🏆 各数据集Top-10配置
- 📈 参数规律分析  
- 📝 推荐的最终配置
- 🎯 下一步建议

### Step 3: 更新配置文件

根据推荐配置，更新 `run_*.sh` 文件：

```bash
# 查看ADFD的推荐配置
cat results/comprehensive_search/ADFD-Sample/recommended_config.txt

# 手动复制参数到 run_adfd.sh
```

---

## 🎯 关键参数说明

### Lambda权重组合

```bash
# balanced (默认)
--lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.3
→ 均衡策略，适合大多数情况

# fusion_focused
--lambda_fuse 1.0 --lambda_view 0.5 --lambda_pseudo_loss 0.5
→ 让DS融合主导，弱化单视图监督

# weak_view
--lambda_fuse 1.0 --lambda_view 0.3 --lambda_pseudo_loss 0.5
→ 极大弱化单视图，强化跨分辨率

# strong_pseudo
--lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.5
→ 强化伪视图的作用

# strong_view
--lambda_fuse 1.0 --lambda_view 1.5 --lambda_pseudo_loss 0.3
→ 强化单视图监督
```

### 学习率策略

```
5e-5:   保守，适合小数据集或不稳定的情况
8e-5:   略保守
1e-4:   MedGNN基线 ⭐
1.2e-4: 略激进
1.5e-4: 中等激进
2e-4:   激进，适合大数据集
3e-4:   很激进，高风险高回报
```

---

## 🐛 可能的问题

### Q: 搜索中途停止怎么办？

**A**: 使用 `screen` 或 `tmux` 更保险：

```bash
screen -S merit_search
bash MERIT/scripts/run_all_comprehensive_search.sh 0
# Ctrl+A, D 断开
# 离开

# 回来后重连
screen -r merit_search
```

### Q: GPU显存不足？

**A**: 脚本会自动调整，或者修改batch_size：
```bash
# 在comprehensive_search.sh中
--batch_size 32  # 从64改为32
```

### Q: 某个数据集一直失败？

**A**: 没关系，脚本会跳过失败的配置。只要有足够成功的配置就行。

---

## 🎉 成功标准

### 回来后应该看到

- [x] `FINAL_SUMMARY.txt` 文件存在
- [x] 每个数据集至少20个配置成功
- [x] Top-3配置有完整验证结果
- [x] 最佳配置准确率合理（不是0%或100%）

### 如果一切顺利

**你会得到**:
1. ✅ 4个数据集的最优超参数
2. ✅ 每个数据集的Top-10配置排名
3. ✅ 完整的性能数字
4. ✅ 可以立即开始论文写作

**你需要做**:
1. 更新run_*.sh配置文件 (10分钟)
2. 运行baseline对比 (4小时)
3. 生成论文表格 (5分钟)
4. 开始写论文！📝

---

## 📋 离开前检查清单

- [ ] 在MERIT conda环境中
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 数据集路径正确
- [ ] 磁盘空间充足 (> 10GB)
- [ ] 使用nohup或screen
- [ ] 命令已执行
- [ ] 日志文件开始写入
- [ ] 等待5-10分钟确认正常运行
- [ ] 保存了进程ID

---

## 🎯 一键启动命令

```bash
cd /home/Data1/zbl

# 使用screen (推荐)
screen -S merit_search
bash MERIT/scripts/run_all_comprehensive_search.sh 0

# 按 Ctrl+A, 然后按 D 断开
# 可以安全离开了！

# 或使用nohup
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &
echo $! > search.pid
tail -f search_all.log  # 确认开始运行后可以Ctrl+C
```

---

## 🎊 2天后回来

```bash
# 重连screen
screen -r merit_search

# 或查看nohup日志
tail -100 search_all.log

# 分析结果
python MERIT/scripts/analyze_comprehensive_results.py

# 查看最终汇总
cat results/comprehensive_search/FINAL_SUMMARY.txt
```

---

**一切都准备好了！祝你离开愉快，2天后回来收获满满的实验结果！** 🎉🚀

记得用screen，不要用nohup（screen更稳定）！

