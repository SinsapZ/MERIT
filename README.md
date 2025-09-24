MERIT

MERIT 是一个多分辨率时序表示学习的分类模型：
- 以多分辨率下的频域嵌入与差分注意作为底座；


- 使用EviMR模块对各分辨率图特征进行基于证据不确定性的加权融合；
- 加入伪视图（Pseudo-view）：将所有分辨率的图特征在通道维拼接，经 1x1 卷积得到聚合表征，独立产生证据与权重，作为额外视图参与融合。
- 完整复用数据与训练流程（分类任务）。

运行：
```bash
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA
# 关闭伪视图
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --no_pseudo

# 常用消融开关
# 1) 聚合：evi vs mean
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --agg mean
# 2) 伪视图强度
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --lambda_pseudo 0.7
# 3) 证据头
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --evidence_act relu --evidence_dropout 0.1
# 4) 分支关闭
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --no_freq
python -m MERIT.MERIT.run --model MERIT --data APAVA --root_path ./dataset/APAVA --no_diff
```

