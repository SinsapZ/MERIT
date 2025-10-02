#!/bin/bash
# 自动运行所有超参数配置实验
# 按优先级顺序执行

echo "=================================="
echo "开始运行所有超参数实验"
echo "=================================="

# 实验1: 仅SWA策略
echo ""
echo "实验1/5: 仅SWA策略"
echo "----------------------------------"
bash MERIT/scripts/run_only_swa.sh
if [ $? -eq 0 ]; then
    echo "✓ 实验1完成"
else
    echo "✗ 实验1失败"
fi

# 实验2: 微调学习率+SWA
echo ""
echo "实验2/5: 微调学习率+SWA"
echo "----------------------------------"
bash MERIT/scripts/run_lr_slight_lower.sh
if [ $? -eq 0 ]; then
    echo "✓ 实验2完成"
else
    echo "✗ 实验2失败"
fi

# 实验3: 退火策略调整
echo ""
echo "实验3/5: 退火策略调整"
echo "----------------------------------"
bash MERIT/scripts/run_annealing_tuning.sh
if [ $? -eq 0 ]; then
    echo "✓ 实验3完成"
else
    echo "✗ 实验3失败"
fi

# 实验4: 降低伪视图权重
echo ""
echo "实验4/5: 降低伪视图权重"
echo "----------------------------------"
bash MERIT/scripts/run_pseudo_tuning.sh
if [ $? -eq 0 ]; then
    echo "✓ 实验4完成"
else
    echo "✗ 实验4失败"
fi

# 实验5: 轻度正则+SWA
echo ""
echo "实验5/5: 轻度正则+SWA"
echo "----------------------------------"
bash MERIT/scripts/run_balanced_v1.sh
if [ $? -eq 0 ]; then
    echo "✓ 实验5完成"
else
    echo "✗ 实验5失败"
fi

echo ""
echo "=================================="
echo "所有实验完成！"
echo "=================================="
echo ""
echo "结果文件位于 results/ 目录:"
echo "  - config_only_SWA.csv"
echo "  - config_lr9e5_anneal55_SWA.csv"
echo "  - config_annealing80_noWD.csv"
echo "  - config_pseudo_loss_0.20.csv"
echo "  - config_balanced_v1_lightReg_SWA.csv"
echo ""
echo "请比较各配置的 test_acc 和标准差，选择最佳配置"

