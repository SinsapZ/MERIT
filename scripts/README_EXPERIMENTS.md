# 🚀 MERIT 全面超参数搜索指南

## 📋 实验概览

### 总配置数: 15个
### 每个配置种子数: 10个
### 预计总耗时: 10-15小时

---

## 🎯 运行方法

### 方案1: 后台运行（推荐）

```bash
# 创建results/comprehensive目录
mkdir -p results/comprehensive

# 后台运行，输出重定向到日志文件
nohup bash MERIT/scripts/run_comprehensive_search.sh > results/experiment_log.txt 2>&1 &

# 查看进程
ps aux | grep run_comprehensive_search

# 实时查看日志
tail -f results/experiment_log.txt
```

### 方案2: Screen/Tmux运行

```bash
# 使用screen
screen -S merit_exp
bash MERIT/scripts/run_comprehensive_search.sh
# 按 Ctrl+A 然后按 D 来detach

# 重新连接
screen -r merit_exp

# 或使用tmux
tmux new -s merit_exp
bash MERIT/scripts/run_comprehensive_search.sh
# 按 Ctrl+B 然后按 D 来detach
tmux attach -t merit_exp
```

---

## 📊 监控进度

### 实时监控脚本

```bash
# 在另一个终端运行
python MERIT/scripts/monitor_progress.py
```

### 手动查看进度

```bash
# 查看已完成的实验数量
ls results/comprehensive/exp*_summary.txt | wc -l

# 查看最新的实验日志
tail -50 results/experiment_log.txt
```

---

## 📈 实验配置详情

| 编号 | 配置名称 | 主要变化 |
|------|---------|---------|
| 1 | swa_baseline | SWA + 原始参数 |
| 2 | swa_wd5e5 | SWA + 轻度weight decay |
| 3 | lr9e5_swa | 学习率 9e-5 + SWA |
| 4 | lr1.1e4_swa | 学习率 1.1e-4 + SWA |
| 5 | lr8e5_cosine_swa | 学习率 8e-5 + Cosine调度 |
| 6 | anneal60_swa | Annealing 60 + SWA |
| 7 | anneal70_swa | Annealing 70 + SWA |
| 8 | anneal40_swa | Annealing 40 + SWA |
| 9 | pseudo0.25_swa | Pseudo loss 0.25 + SWA |
| 10 | pseudo0.35_swa | Pseudo loss 0.35 + SWA |
| 11 | evidrop0.05_swa | Evidence dropout 0.05 + SWA |
| 12 | evidrop0.08_swa | Evidence dropout 0.08 + SWA |
| 13 | node12_swa | NodeDim 12 + SWA |
| 14 | elayer3_swa | E_layers 3 + SWA |
| 15 | best_combined | 综合最优配置 |

---

## 📊 结果分析

### 实验完成后，运行汇总脚本

```bash
python MERIT/scripts/summarize_results.py
```

### 输出内容包括:

1. **按Test Accuracy排序的排行榜**
2. **按Test F1排序的排行榜**
3. **最稳定配置排行榜**（标准差最小）
4. **最佳配置详细信息**
5. **详细CSV报告**: `results/comprehensive_summary.csv`

---

## 🎯 预期效果

**目标**: 
- Test Acc > 0.80 (平均)
- Test Acc Std < 0.03 (标准差)

**如果达到**: 
- Test Acc > 0.81 ± 0.025 → 🎉 非常成功！
- Test Acc > 0.80 ± 0.030 → ✅ 达标
- Test Acc < 0.80 或 Std > 0.04 → 需要进一步调整

---

## 📁 输出文件结构

```
results/
├── comprehensive/
│   ├── exp01_swa_baseline.csv
│   ├── exp01_swa_baseline_summary.txt
│   ├── exp02_swa_wd5e5.csv
│   ├── exp02_swa_wd5e5_summary.txt
│   ├── ...
│   └── exp15_best_combined.csv
├── comprehensive_summary.csv
└── experiment_log.txt
```

---

## 🛠️ 故障排除

### 如果实验中断

```bash
# 查看哪些实验已完成
ls results/comprehensive/exp*.csv

# 手动运行未完成的实验
python -m MERIT.scripts.multi_seed_run \
  --root_path /home/Data1/zbl/dataset/APAVA \
  --data APAVA --gpu 0 \
  --lr 1e-4 --lambda_fuse 1.0 --lambda_view 1.0 --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 --evidence_dropout 0.0 --e_layers 4 \
  --dropout 0.1 --weight_decay 0 --nodedim 10 \
  --batch_size 64 --train_epochs 200 --patience 30 --swa \
  --resolution_list 2,4,6,8 --seeds "41,42,43,44,45,46,47,48,49,50" \
  --log_csv results/comprehensive/expXX_name.csv
```

### GPU内存不足

```bash
# 减少batch size
--batch_size 32

# 或减少种子数
--seeds "41,42,43,44,45"
```

---

## ✅ 使用检查清单

- [ ] 确认GPU可用: `nvidia-smi`
- [ ] 创建输出目录: `mkdir -p results/comprehensive`
- [ ] 启动实验: `nohup bash ... &`
- [ ] 启动监控: `python monitor_progress.py`
- [ ] 定期检查日志: `tail -f results/experiment_log.txt`
- [ ] 完成后运行汇总: `python summarize_results.py`

---

## 🎁 祝您旅途愉快！

实验会在实验室安静地运行，等你回来就能看到完整的结果排行榜！

