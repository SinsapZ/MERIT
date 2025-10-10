# 🔧 MERIT超参数调优指南

## 🎯 基于MedGNN的参数基线

### MedGNN的成功配置

| 参数 | MedGNN值 | 说明 |
|------|----------|------|
| `learning_rate` | **1e-4** | 基准学习率 |
| `train_epochs` | 10 | 快速收敛 |
| `patience` | 3 | 早停 |
| `batch_size` | 64-128 | 根据数据集 |
| `dropout` | 0.1 | 标准正则化 |
| `e_layers` | 4-6 | 编码器层数 |

**MERIT的调整空间**：由于MERIT有额外的证据融合模块，需要更多训练。

---

## 📊 快速调优：4种学习率配置

### 使用方法

```bash
# 测试4种学习率配置
bash MERIT/scripts/quick_tune.sh ADFD-Sample
bash MERIT/scripts/quick_tune.sh PTB
bash MERIT/scripts/quick_tune.sh PTB-XL
```

### 4种配置详解

| 配置 | 学习率 | Epochs | Dropout | 适用场景 |
|------|--------|--------|---------|----------|
| **Config 1** | **1e-4** | 150 | 0.1 | 🎯 MedGNN基线，优先尝试 |
| **Config 2** | **5e-5** | 200 | 0.1 | 🐌 数据少/不稳定时用 |
| **Config 3** | **2e-4** | 100 | 0.1 | 🚀 数据多/想快速训练 |
| **Config 4** | **3e-4** | 100 | 0.15 | 💥 激进尝试，可能很好或很差 |

---

## 🎓 学习率选择指南

### 学习率范围：5e-5 ~ 3e-4

```
更低 ←────────────────────────→ 更高
5e-5      1e-4      2e-4      3e-4

稳定        MedGNN     快速       激进
慢收敛      基准       可能不稳    高风险
```

### 如何选择？

#### 1️⃣ 优先级1: MedGNN基线 (lr=1e-4)

**什么时候用**:
- ✅ 第一次尝试新数据集
- ✅ 不确定用什么参数
- ✅ MedGNN在该数据集上work

**预期**:
- 稳定收敛
- 性能中等偏上
- 是其他配置的参考基准

---

#### 2️⃣ 更低学习率 (lr=5e-5)

**什么时候用**:
- ⚠️ 数据集很小 (< 3000 samples)
- ⚠️ 方差很大 (std > 5%)
- ⚠️ 训练不稳定，loss震荡
- ⚠️ 过拟合严重

**优点**:
- ✅ 更稳定的收敛
- ✅ 更小的方差
- ✅ 泛化性更好

**缺点**:
- ❌ 训练慢 (需要200 epochs)
- ❌ 可能陷入局部最优

**配合调整**:
```bash
--train_epochs 200    # 给更多时间
--patience 30         # 更宽容
```

---

#### 3️⃣ 更高学习率 (lr=2e-4)

**什么时候用**:
- ✅ 数据集大 (> 10000 samples)
- ✅ 想快速训练
- ✅ baseline收敛太慢

**优点**:
- ✅ 训练快 (100 epochs足够)
- ✅ 可能找到更好的解
- ✅ 省GPU时间

**缺点**:
- ⚠️ 可能不稳定
- ⚠️ 需要观察是否震荡

**配合调整**:
```bash
--train_epochs 100    # 更少epochs
--patience 15         # 早停
```

---

#### 4️⃣ 激进学习率 (lr=3e-4)

**什么时候用**:
- 🎲 其他配置都不理想
- 🎲 想赌一把
- 🎲 有时间和GPU资源

**特点**:
- 💥 高风险高回报
- 💥 可能非常好，也可能爆炸
- 💥 需要配合强正则化

**配合调整**:
```bash
--dropout 0.15        # 增大dropout
--weight_decay 1e-5   # 加weight decay
--annealing_epoch 30  # 快速退火
```

---

## 📈 如何根据结果选择最佳配置

### 运行完quick_tune.sh后

查看输出：
```
Results Comparison:
----------------------------------------
config1_baseline: 0.78234 ± 0.02456
config2_lower_lr: 0.79123 ± 0.01834  ← 最好且最稳定
config3_higher_lr: 0.78567 ± 0.03245
config4_aggressive: 0.76234 ± 0.05123
```

### 决策树

```
┌─ 看平均值 (mean)
│   ├─ config2最高？ → 选config2
│   ├─ config3最高？ → 检查方差
│   │   ├─ 方差<3%？ → 选config3 ✅
│   │   └─ 方差>3%？ → 降回config1
│   └─ config4最高？ → 
│       ├─ 方差<2%？ → 幸运！选config4 🎉
│       └─ 方差>3%？ → 太不稳定，选config1或config2
│
└─ 如果都差不多
    ├─ 选训练最快的 (config3或config4)
    └─ 如果要稳定性，选config2
```

---

## 🎯 按数据集的推荐配置

### APAVA (已验证)
```bash
--lr 1.1e-4              # 略高于MedGNN
--train_epochs 200       # 充分训练
--patience 30
--dropout 0.1
```
**原因**: 数据少(2300)，需要充分训练

---

### ADFD (需调优)

**建议尝试顺序**:
1. 先试 `lr=1e-4` (MedGNN基线)
2. 如果不稳定 → `lr=5e-5`
3. 如果太慢 → `lr=2e-4`

**预期最佳**:
```bash
--lr 5e-5 或 1e-4        # 数据少，保守为主
--train_epochs 200       # 充分训练
--patience 30
--dropout 0.2            # 比APAVA更强的正则化
--e_layers 6             # 与MedGNN一致
```

---

### PTB (需调优)

**建议尝试顺序**:
1. 先试 `lr=1e-4`
2. 如果好 → 试 `lr=2e-4` (可能更快)
3. 数据大，可以用较高学习率

**预期最佳**:
```bash
--lr 1e-4 或 2e-4        # 数据多，可以大胆点
--train_epochs 100       # 数据多，不需要太多轮
--patience 15
--dropout 0.1
```

---

### PTB-XL (需调优)

**建议尝试顺序**:
1. 先试 `lr=1e-4`
2. 数据最多(18K+)，可以试 `lr=2e-4` 甚至 `lr=3e-4`

**预期最佳**:
```bash
--lr 2e-4                # 大数据集，大胆用高lr
--train_epochs 100       # 数据多，快速收敛
--patience 15
--dropout 0.1
```

---

## 🔬 高级调优策略

### 如果quick_tune.sh结果都不理想

#### 调整1: Lambda权重

```bash
# 降低per-view监督，让DS融合主导
--lambda_fuse 1.0
--lambda_view 0.5        # 从1.0降到0.5
--lambda_pseudo_loss 0.5 # 从0.3增到0.5
```

#### 调整2: Annealing策略

```bash
# 更快退火（适合小数据集）
--annealing_epoch 30     # 从50降到30

# 更慢退火（适合大数据集）
--annealing_epoch 70     # 从50增到70
```

#### 调整3: Evidence dropout

```bash
# 增加evidence层的dropout
--evidence_dropout 0.1   # 从0增到0.1
```

#### 调整4: 学习率调度

```bash
# 使用cosine annealing
--lr_scheduler cosine
--warmup_epochs 10
```

---

## 📊 调优结果分析模板

### 运行完后查看

```bash
# 查看所有配置的准确率
for config in config1_baseline config2_lower_lr config3_higher_lr config4_aggressive; do
    echo "=== $config ==="
    cat results/tuning/ADFD-Sample/${config}_summary.txt | grep "test_acc:"
done
```

### 记录模板

```
数据集: ADFD-Sample
==================
Config 1 (lr=1e-4):   Acc = __.__ ± __.__
Config 2 (lr=5e-5):   Acc = __.__ ± __.__  ← 最稳定
Config 3 (lr=2e-4):   Acc = __.__ ± __.__  ← 最快
Config 4 (lr=3e-4):   Acc = __.__ ± __.__

最佳选择: Config _
原因: _____________________
```

---

## ⏱️ 时间成本

### quick_tune.sh (3 seeds × 4 configs)

| 数据集 | 单次配置 | 总时间 |
|--------|----------|--------|
| ADFD | ~15分钟 | **~60分钟** |
| PTB | ~15分钟 | **~60分钟** |
| PTB-XL | ~20分钟 | **~80分钟** |

**总计**: ~3小时可以完成3个数据集的调优

---

## 🎯 实战建议

### Day 1: 快速调优

```bash
# 上午：运行调优
bash MERIT/scripts/quick_tune.sh ADFD-Sample  # 1小时
bash MERIT/scripts/quick_tune.sh PTB          # 1小时

# 下午：分析结果并选择最佳配置
# 修改run_adfd.sh和run_ptb.sh中的参数

bash MERIT/scripts/quick_tune.sh PTB-XL       # 1.5小时
```

### Day 2: 完整实验

```bash
# 用Day 1选定的最佳配置，跑10个seeds
bash MERIT/scripts/run_adfd.sh    # 1.5小时
bash MERIT/scripts/run_ptb.sh     # 2小时
bash MERIT/scripts/run_ptbxl.sh   # 2.5小时
```

### Day 3: Baseline对比

```bash
# 运行Medformer和iTransformer
bash MERIT/scripts/run_baselines.sh APAVA
bash MERIT/scripts/run_baselines.sh ADFD-Sample
bash MERIT/scripts/run_baselines.sh PTB
bash MERIT/scripts/run_baselines.sh PTB-XL
```

---

## 💡 经验法则

### 数据集大小 vs 学习率

```
小数据集 (<5000)   → lr = 5e-5 ~ 1e-4   (保守)
中数据集 (5K-15K)  → lr = 1e-4 ~ 2e-4   (标准)
大数据集 (>15K)    → lr = 2e-4 ~ 3e-4   (激进)
```

### 类别数 vs 训练轮数

```
二分类              → epochs = 100-150   (容易收敛)
多分类 (5-9类)      → epochs = 150-200   (需要更多训练)
```

### Dropout vs 数据量

```
数据少 + 简单任务   → dropout = 0.05      (弱正则化)
数据少 + 复杂任务   → dropout = 0.15-0.2  (强正则化)
数据多              → dropout = 0.1       (标准)
```

---

## 🔍 诊断指南

### 症状1: 训练loss下降但val/test不提升

**原因**: 过拟合

**解决方案**:
```bash
--dropout 0.15           # 增大
--weight_decay 1e-4      # 增大
--train_epochs 100       # 减少
--lambda_view 0.5        # 减少过度监督
```

---

### 症状2: 训练loss和val loss都很高

**原因**: 欠拟合或学习率太低

**解决方案**:
```bash
--lr 2e-4                # 提高学习率
--e_layers 6             # 增加模型容量
--train_epochs 200       # 更多训练
```

---

### 症状3: Loss震荡，不收敛

**原因**: 学习率太高

**解决方案**:
```bash
--lr 5e-5                # 降低学习率
--patience 30            # 增加耐心
--lr_scheduler cosine    # 使用调度器
--warmup_epochs 10       # 加warmup
```

---

### 症状4: 方差巨大 (std > 5%)

**原因**: 模型不稳定或数据集太小

**解决方案**:
```bash
--lr 5e-5                # 降低学习率
--dropout 0.15           # 增大dropout
--weight_decay 1e-5      # 加正则化
--train_epochs 200       # 更充分训练
--seeds "41,42,...,55"   # 增加seeds数量
```

---

## 📋 调优检查清单

### 运行前
- [ ] 确认数据路径正确
- [ ] 确认数据集名称正确 (ADFD-Sample, PTB, PTB-XL)
- [ ] 确认GPU可用
- [ ] 预留足够时间 (~1小时/数据集)

### 运行中
- [ ] 观察第一个config是否正常运行
- [ ] 检查loss是否下降
- [ ] 观察是否触发早停

### 运行后
- [ ] 检查是否有3个seeds成功
- [ ] 对比4个配置的准确率
- [ ] 选择最佳配置
- [ ] 记录选择的原因

---

## 🎯 最终配置模板

### 调优完成后，修改run_*.sh

以ADFD为例，假设config2 (lr=5e-5) 最好：

```bash
# 修改 MERIT/scripts/run_adfd.sh
python -m MERIT.scripts.multi_seed_run \
  --root_path $ROOT_PATH \
  --data ADFD-Sample \
  --gpu $GPU \
  --lr 5e-5 \              # ← 从quick_tune选出的最佳lr
  --lambda_fuse 1.0 \
  --lambda_view 1.0 \
  --lambda_pseudo_loss 0.30 \
  --annealing_epoch 50 \
  --evidence_dropout 0.0 \
  --e_layers 6 \
  --dropout 0.2 \
  --weight_decay 0 \
  --nodedim 10 \
  --batch_size 128 \
  --train_epochs 200 \     # ← 与lr配套
  --patience 30 \          # ← 与lr配套
  --swa \
  --resolution_list 2 \
  --seeds "$SEEDS" \
  --log_csv results/final_all_datasets/adfd_results.csv
```

---

## 🚀 快速参考

### 保守策略（稳定优先）
```bash
--lr 5e-5 或 1e-4
--train_epochs 200
--patience 30
--dropout 0.15
```

### 激进策略（速度优先）
```bash
--lr 2e-4 或 3e-4
--train_epochs 100
--patience 15
--dropout 0.1
```

### 平衡策略（推荐）
```bash
--lr 1e-4                # MedGNN基线
--train_epochs 150
--patience 20
--dropout 0.1
```

---

## 💡 额外Tips

### Tip 1: APAVA已经调好了
```bash
# APAVA的最佳配置（已验证）
--lr 1.1e-4
--train_epochs 200
--patience 30
```
不需要再调，直接用就行！

### Tip 2: 如果GPU显存不够
```bash
--batch_size 32          # 从64降到32
```

### Tip 3: 如果想更快速的调优
```bash
# 改用2个seeds
SEEDS="41,42"
```

### Tip 4: 观察训练过程
```bash
# 运行时加上输出重定向
bash MERIT/scripts/quick_tune.sh ADFD-Sample 2>&1 | tee adfd_tune.log

# 可以实时查看
tail -f adfd_tune.log
```

---

**记住: 调优的目标不是找到完美参数，而是找到"足够好"的参数！** 🎯

不要花太多时间在0.5%的提升上，专注于论文写作和分析！

