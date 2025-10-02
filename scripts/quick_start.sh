#!/bin/bash
# 快速启动脚本 - 一键启动全面搜索

echo "=================================="
echo "MERIT 全面超参数搜索"
echo "=================================="
echo ""
echo "准备启动 15 个配置，每个 10 个种子"
echo "预计总耗时: 10-15 小时"
echo ""

# 检查GPU
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
else
    echo "警告: 未找到 nvidia-smi"
fi

echo ""
read -p "确认开始实验? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 创建目录
mkdir -p results/comprehensive

# 后台运行
echo ""
echo "正在后台启动实验..."
nohup bash MERIT/scripts/run_comprehensive_search.sh > results/experiment_log.txt 2>&1 &

PID=$!
echo "✓ 实验已启动！"
echo "  进程ID: $PID"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f results/experiment_log.txt"
echo "  查看进度: python MERIT/scripts/monitor_progress.py"
echo "  查看进程: ps -p $PID"
echo "  停止实验: kill $PID"
echo ""
echo "实验完成后运行:"
echo "  python MERIT/scripts/summarize_results.py"
echo ""
echo "祝您旅途愉快！ 🚀"

