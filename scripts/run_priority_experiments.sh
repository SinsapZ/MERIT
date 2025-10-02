#!/bin/bash
# 只运行最高优先级的两个实验（节省时间）

echo "=================================="
echo "运行高优先级实验（2个）"
echo "=================================="

# 实验1: 仅SWA策略（最推荐）
echo ""
echo "实验1/2: 仅SWA策略"
echo "----------------------------------"
bash MERIT/scripts/run_only_swa.sh

# 实验2: 微调学习率+SWA
echo ""
echo "实验2/2: 微调学习率+SWA"
echo "----------------------------------"
bash MERIT/scripts/run_lr_slight_lower.sh

echo ""
echo "=================================="
echo "高优先级实验完成！"
echo "=================================="
echo "查看结果:"
echo "  results/config_only_SWA.csv"
echo "  results/config_lr9e5_anneal55_SWA.csv"

