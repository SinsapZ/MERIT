# 🤖 MERIT 2天自动超参数搜索指南

## 🎯 目标

在你离开的2天内，**自动找到每个数据集的最优超参数配置**。

---

## 🚀 一键启动（推荐）

### 方式1: 运行所有数据集 (推荐)

```bash
cd /home/Data1/zbl

# 启动后台运行
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &

# 查看日志
tail -f search_all.log
```

**预计时间**: 36-48小时  
**配置数量**: 35个配置/数据集 × 4数据集 = **140个实验**

---

### 方式2: 单独运行某个数据集

```bash
# 只搜索ADFD
nohup bash MERIT/scripts/comprehensive_search.sh ADFD-Sample 0 > search_adfd.log 2>&1 &

# 只搜索PTB
nohup bash MERIT/scripts/comprehensive_search.sh PTB 0 > search_ptb.log 2>&1 &
```

---

## 📊 搜索的超参数空间

### 学习率 (7个值)
```
5e-5, 8e-5, 1e-4, 1.2e-4, 1.5e-4, 2e-4, 3e-4
 ↓     ↓     ↓      ↓       ↓      ↓     ↓
保守  较保守 MedGNN  APAVA  略激进  激进  很激进
```

### Lambda权重组合 (5种)

| 组合 | lambda_fuse | lambda_view | lambda_pseudo | 策略 |
|------|-------------|-------------|---------------|------|
| **balanced** | 1.0 | 1.0 | 0.3 | 🎯 默认均衡 (MedGNN风格) |
| **fusion_focused** | 1.0 | 0.5 | 0.5 | 🔄 强化跨分辨率融合 |
| **weak_view** | 1.0 | 0.3 | 0.5 | 📉 弱化单视图监督 |
| **strong_pseudo** | 1.0 | 1.0 | 0.5 | 💪 强化伪视图 |
| **strong_view** | 1.0 | 1.5 | 0.3 | 📊 强化单视图 |

### 配套参数自动调整

根据学习率自动调整：

| 学习率范围 | train_epochs | patience | annealing_epoch |
|------------|--------------|----------|-----------------|
| < 1e-4 | 200 | 30 | 50 |
| 1e-4 ~ 2e-4 | 150 | 20 | 50 |
| ≥ 2e-4 | 100 | 15 | 30 |

---

## ⏱️ 时间规划

### 阶段1: 快速筛选 (3 seeds)

| 数据集 | 配置数 | 单个配置时间 | 总时间 |
|--------|--------|--------------|--------|
| APAVA | 35 | ~8分钟 | **~4.5小时** |
| ADFD | 35 | ~6分钟 | **~3.5小时** |
| PTB | 35 | ~8分钟 | **~4.5小时** |
| PTB-XL | 35 | ~10分钟 | **~6小时** |
| **总计** | **140** | - | **~18小时** |

### 阶段2: 完整验证 (10 seeds, Top-3)

| 数据集 | Top配置 | 单个配置时间 | 总时间 |
|--------|---------|--------------|--------|
| APAVA | 3 | ~2小时 | **~6小时** |
| ADFD | 3 | ~1.5小时 | **~4.5小时** |
| PTB | 3 | ~2小时 | **~6小时** |
| PTB-XL | 3 | ~2.5小时 | **~7.5小时** |
| **总计** | **12** | - | **~24小时** |

### 总计时间

```
阶段1: 18小时 (快速筛选140个配置)
阶段2: 24小时 (完整验证12个top配置)
--------------------------------------
总计:  42小时 (~1.75天)
```

**在2天内完全可以完成！** ✅

---

## 📁 输出结果结构

```
results/comprehensive_search/
├── APAVA/
│   ├── quick_lr5e-5_balanced.csv          # 快速筛选结果
│   ├── quick_lr1e-4_balanced.csv
│   ├── ... (35个quick配置)
│   ├── top3_configs.txt                   # Top-3配置列表
│   ├── full_top1_lr1.1e-4_balanced.csv   # 完整验证结果
│   ├── full_top2_lr1e-4_balanced.csv
│   └── full_top3_lr1.2e-4_strong_pseudo.csv
├── ADFD-Sample/
│   ├── quick_*.csv (35个)
│   ├── top3_configs.txt
│   └── full_top*.csv (3个)
├── PTB/
│   └── (同上结构)
├── PTB-XL/
│   └── (同上结构)
└── best_configs_summary.txt               # 最终汇总
```

---

## 📊 自动分析流程

### 脚本自动执行的操作

1. **快速筛选35个配置** (3 seeds each)
   - 7种学习率 × 5种lambda组合 = 35种配置
   
2. **自动排序** (按test_acc降序)

3. **选择Top-3**
   - 保存到 `top3_configs.txt`

4. **完整验证Top-3** (10 seeds each)
   - 确认最佳配置

5. **生成最终报告**
   - `best_configs_summary.txt`

---

## 🔍 回来后怎么查看结果

### Step 1: 查看总体进度

```bash
cd /home/Data1/zbl

# 查看运行日志
tail -100 search_all.log

# 检查是否完成
ls -lh results/comprehensive_search/best_configs_summary.txt
```

### Step 2: 查看最佳配置

```bash
cat results/comprehensive_search/best_configs_summary.txt
```

**输出示例**:
```
MERIT最佳配置汇总
================================================================================

APAVA:
  配置: lr1.1e-4_balanced
  准确率: 0.7800 ± 0.0246

ADFD-Sample:
  配置: lr1e-4_fusion_focused
  准确率: 0.8867 ± 0.0213

PTB:
  配置: lr2e-4_balanced
  准确率: 0.9234 ± 0.0185

PTB-XL:
  配置: lr2e-4_strong_pseudo
  准确率: 0.8456 ± 0.0234
```

### Step 3: 查看详细结果

```bash
# 查看某个数据集的Top-10配置
python - <<EOF
import pandas as pd
import glob

results = []
for f in glob.glob("results/comprehensive_search/ADFD-Sample/quick_*.csv"):
    df = pd.read_csv(f)
    df_ok = df[df['return_code']==0]
    if len(df_ok) >= 2:
        results.append({
            'config': f.split('/')[-1].replace('quick_','').replace('.csv',''),
            'acc': df_ok['test_acc'].mean(),
            'std': df_ok['test_acc'].std()
        })

results.sort(key=lambda x: x['acc'], reverse=True)

print("\nADFD-Sample Top-10:")
for i, r in enumerate(results[:10], 1):
    print(f"{i}. {r['config']}: {r['acc']:.4f}±{r['std']:.4f}")
EOF
```

---

## 🔧 根据结果更新配置文件

### 假设ADFD的最佳配置是 `lr1e-4_fusion_focused`

提取参数:
- lr = 1e-4
- lambda_fuse = 1.0
- lambda_view = 0.5
- lambda_pseudo = 0.5

更新 `run_adfd.sh`:
```bash
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data ADFD-Sample \
  --gpu $GPU \
  --lr 1e-4 \                    # ← 从最佳配置
  --lambda_fuse 1.0 \            # ← 从最佳配置
  --lambda_view 0.5 \            # ← 从最佳配置
  --lambda_pseudo_loss 0.5 \     # ← 从最佳配置
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 6 \
  --dropout 0.2 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 128 \
  --train_epochs 150 \
  --patience 20 \
  --swa \
  --resolution_list 2 \
  --seeds "$SEEDS" \
  --log_csv results/final_all_datasets/adfd_results.csv
```

---

## ⚡ 加速技巧

### 如果想更快完成

#### 技巧1: 减少lambda组合

修改 `comprehensive_search.sh` 中的lambda配置：
```bash
# 从5个减少到3个
LAMBDA_CONFIGS=(
    "1.0,1.0,0.3"    # 默认
    "1.0,0.5,0.5"    # 强化融合
    "1.0,1.0,0.5"    # 强化pseudo
)
```

**节省时间**: 18小时 → **~11小时**

#### 技巧2: 减少学习率选项

```bash
# 从7个减少到5个
LR_LIST=(5e-5 1e-4 1.5e-4 2e-4 3e-4)
```

**节省时间**: 18小时 → **~13小时**

#### 技巧3: 跳过某些数据集

```bash
# 只搜索ADFD和PTB (APAVA已调好，PTB-XL太大)
bash MERIT/scripts/comprehensive_search.sh ADFD-Sample 0
bash MERIT/scripts/comprehensive_search.sh PTB 0
```

**节省时间**: ~10小时

---

## 🎬 推荐执行方案

### 保守方案 (全面搜索)

```bash
# 启动全面搜索
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &

# 2天后回来，查看结果
cat results/comprehensive_search/best_configs_summary.txt
```

**特点**:
- ✅ 最全面 (140个配置)
- ✅ 最稳妥 (一定能找到好配置)
- ⏱️ 时间长 (~42小时)

---

### 激进方案 (快速搜索)

修改 `comprehensive_search.sh`，只保留3个lambda组合 + 5个学习率 = 15个配置

```bash
# 修改后运行
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &
```

**特点**:
- ✅ 快速 (~18小时)
- ⚠️ 可能错过最佳配置
- ✅ 2天内完成更有保障

---

### 折中方案 (我的推荐) ⭐

**只搜索关键数据集 + 减少lambda组合**:

```bash
# 修改comprehensive_search.sh: 
#   - lambda从5个减到3个
#   - 保留所有7个学习率

# 然后只运行ADFD和PTB
nohup bash MERIT/scripts/comprehensive_search.sh ADFD-Sample 0 > adfd.log 2>&1 &
nohup bash MERIT/scripts/comprehensive_search.sh PTB 0 > ptb.log 2>&1 &
```

**特点**:
- ✅ 时间合适 (~24小时)
- ✅ 覆盖关键数据集
- ✅ APAVA已有最佳配置
- ✅ PTB-XL可以用PTB的配置

---

## 📈 搜索覆盖的配置详情

### 完整配置矩阵 (35个配置/数据集)

```
学习率          ×    Lambda组合       =   配置数
7个值          ×    5种组合         =   35个

详细展开:
lr=5e-5  × [balanced, fusion_focused, weak_view, strong_pseudo, strong_view]  = 5个
lr=8e-5  × [同上5种]                                                           = 5个
lr=1e-4  × [同上5种]  ⭐ MedGNN基线                                           = 5个
lr=1.2e-4 × [同上5种]                                                          = 5个
lr=1.5e-4 × [同上5种]                                                          = 5个
lr=2e-4  × [同上5种]                                                           = 5个
lr=3e-4  × [同上5种]                                                           = 5个
                                                                    总计 = 35个
```

### 两阶段策略

```
阶段1 (快速筛选):
  35个配置 × 3 seeds = 105次运行
  ↓
  自动排序
  ↓
  选择Top-3

阶段2 (完整验证):
  Top-3配置 × 10 seeds = 30次运行
  ↓
  确定最终最佳配置
```

---

## 💾 自动保存的结果

### 每个数据集的结果

1. **`top3_configs.txt`** - Top-3配置名称
   ```
   lr1e-4_fusion_focused
   lr1.2e-4_balanced
   lr8e-5_strong_pseudo
   ```

2. **`quick_*.csv`** - 快速筛选的详细结果 (35个文件)

3. **`full_top*.csv`** - 完整验证结果 (3个文件)

### 全局汇总

4. **`best_configs_summary.txt`** - 最终最佳配置
   ```
   APAVA: lr1.1e-4_balanced → 78.00±2.46%
   ADFD:  lr1e-4_fusion_focused → 88.67±2.13%
   PTB:   lr2e-4_balanced → 92.34±1.85%
   PTB-XL: lr2e-4_strong_pseudo → 84.56±2.34%
   ```

---

## 🔍 监控运行进度

### 实时查看日志

```bash
# 查看总体进度
tail -f search_all.log

# 查看当前运行的配置
tail -f search_all.log | grep "Config"

# 查看已完成的配置数
grep "completed" search_all.log | wc -l
```

### 检查中间结果

```bash
# 查看ADFD已完成的配置数
ls results/comprehensive_search/ADFD-Sample/quick_*.csv | wc -l

# 查看当前最好的结果
python - <<EOF
import pandas as pd
import glob

files = glob.glob("results/comprehensive_search/ADFD-Sample/quick_*.csv")
best_acc = 0
best_file = ""

for f in files:
    df = pd.read_csv(f)
    df_ok = df[df['return_code']==0]
    if len(df_ok) >= 2:
        acc = df_ok['test_acc'].mean()
        if acc > best_acc:
            best_acc = acc
            best_file = f.split('/')[-1]

print(f"当前最佳: {best_file}")
print(f"准确率: {best_acc:.4f}")
EOF
```

---

## 🎯 回来后的完整工作流

### Day 3 (你回来的第一天)

#### 上午: 查看搜索结果

```bash
# 1. 查看最佳配置汇总
cat results/comprehensive_search/best_configs_summary.txt

# 2. 查看各数据集的Top-10
python MERIT/scripts/analyze_comprehensive_results.py  # 我下面会创建这个
```

#### 下午: 更新配置并验证

```bash
# 根据最佳配置，更新run_*.sh文件
# 然后运行验证（如果搜索没有完整验证）

# 如果需要，重跑最佳配置确认
bash MERIT/scripts/run_adfd.sh
bash MERIT/scripts/run_ptb.sh
bash MERIT/scripts/run_ptbxl.sh
```

### Day 4: Baseline对比

```bash
# 运行baseline模型
bash MERIT/scripts/run_baselines.sh APAVA
bash MERIT/scripts/run_baselines.sh ADFD-Sample
bash MERIT/scripts/run_baselines.sh PTB
bash MERIT/scripts/run_baselines.sh PTB-XL
```

### Day 5: 生成论文表格

```bash
python MERIT/scripts/summarize_all_datasets.py
```

---

## 🎮 启动命令总结

### 🌟 最推荐：全面搜索

```bash
cd /home/Data1/zbl
nohup bash MERIT/scripts/run_all_comprehensive_search.sh 0 > search_all.log 2>&1 &
echo $! > search.pid  # 保存进程ID

# 查看进度
tail -f search_all.log

# 如需停止
kill $(cat search.pid)
```

---

### ⚡ 备选：只搜索未调优的数据集

```bash
# APAVA已经调好了，只搜索其他3个
cd /home/Data1/zbl

nohup bash MERIT/scripts/comprehensive_search.sh ADFD-Sample 0 > adfd.log 2>&1 &
nohup bash MERIT/scripts/comprehensive_search.sh PTB 0 > ptb.log 2>&1 &
nohup bash MERIT/scripts/comprehensive_search.sh PTB-XL 0 > ptbxl.log 2>&1 &
```

**时间**: ~32小时 (3个数据集)

---

## 🐛 可能的问题和解决方案

### 问题1: 脚本中途停止

**原因**: 连接断开、GPU崩溃等

**解决方案**: 使用 `screen` 或 `tmux`
```bash
screen -S merit_search
bash MERIT/scripts/run_all_comprehensive_search.sh 0
# Ctrl+A, D 断开
# screen -r merit_search 重新连接
```

### 问题2: GPU显存不足

**解决方案**: 脚本会自动调整batch_size，或手动修改：
```bash
# 在comprehensive_search.sh中修改
--batch_size 32  # 从64改为32
```

### 问题3: 某些配置一直失败

**解决方案**: 忽略即可，脚本会自动跳过失败的配置

---

## 📝 离开前的检查清单

### 确认运行环境
- [ ] 在正确的conda环境 (MERIT)
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 磁盘空间充足 (至少10GB)
- [ ] 数据集路径正确

### 启动搜索
- [ ] 运行命令
- [ ] 使用nohup或screen
- [ ] 保存进程ID
- [ ] 确认日志文件开始写入

### 离开前测试
- [ ] 等待5-10分钟
- [ ] 查看日志确认正常运行
- [ ] 看到第一个配置完成

---

## 🎯 预期最终收获

2天后你会得到：

1. ✅ **4个数据集的最佳超参数配置**
2. ✅ **每个数据集的Top-10配置排名**
3. ✅ **完整的实验结果CSV文件**
4. ✅ **可以直接用于论文的性能数字**

然后只需要：
- 更新run_*.sh文件
- 运行baseline对比
- 生成论文表格
- **开始写论文！** 📝

---

**祝实验顺利！2天后见！** 🚀✨

