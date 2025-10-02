#!/usr/bin/env python
"""
自动运行所有超参数配置实验（Python版本）
如果bash脚本有问题可以用这个
"""

import subprocess
import sys
import time
from datetime import datetime

experiments = [
    {
        "name": "仅SWA策略",
        "script": "MERIT/scripts/run_only_swa.sh",
        "priority": 1
    },
    {
        "name": "微调学习率+SWA",
        "script": "MERIT/scripts/run_lr_slight_lower.sh",
        "priority": 1
    },
    {
        "name": "退火策略调整",
        "script": "MERIT/scripts/run_annealing_tuning.sh",
        "priority": 2
    },
    {
        "name": "降低伪视图权重",
        "script": "MERIT/scripts/run_pseudo_tuning.sh",
        "priority": 2
    },
    {
        "name": "轻度正则+SWA",
        "script": "MERIT/scripts/run_balanced_v1.sh",
        "priority": 3
    },
]

def run_experiment(exp):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"实验: {exp['name']}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['bash', exp['script']],
            check=False,
            capture_output=False
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {exp['name']} 完成 (耗时: {duration/60:.1f}分钟)")
            return True
        else:
            print(f"✗ {exp['name']} 失败 (返回码: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ {exp['name']} 出错: {e}")
        return False

def main():
    print("="*50)
    print("开始运行所有超参数实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 如果命令行参数包含 --priority-only，只运行优先级1的实验
    priority_only = '--priority-only' in sys.argv
    
    results = []
    total_start = time.time()
    
    for i, exp in enumerate(experiments, 1):
        if priority_only and exp['priority'] > 1:
            continue
            
        print(f"\n进度: {i}/{len(experiments)}")
        success = run_experiment(exp)
        results.append((exp['name'], success))
    
    total_duration = time.time() - total_start
    
    # 输出总结
    print("\n" + "="*50)
    print("所有实验完成！")
    print(f"总耗时: {total_duration/60:.1f}分钟")
    print("="*50)
    
    print("\n实验结果汇总:")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print("\n结果文件位于 results/ 目录")
    print("请比较各配置的 test_acc 和标准差")

if __name__ == '__main__':
    main()

