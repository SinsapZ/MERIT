#!/usr/bin/env python
"""
实时监控实验进度
用法: python MERIT/scripts/monitor_progress.py
"""

import os
import glob
import time
from datetime import datetime
from pathlib import Path

def count_completed_experiments(results_dir='results/comprehensive'):
    """统计已完成的实验数量"""
    if not os.path.exists(results_dir):
        return 0, []
    
    csv_files = glob.glob(f'{results_dir}/exp*.csv')
    completed = []
    
    for csv_file in csv_files:
        exp_name = Path(csv_file).stem
        # 检查是否有summary文件，表示该实验已完成
        summary_file = csv_file.replace('.csv', '_summary.txt')
        if os.path.exists(summary_file):
            completed.append(exp_name)
    
    return len(completed), completed

def monitor_loop():
    """循环监控进度"""
    total_experiments = 15
    
    print("="*60)
    print("实验进度监控器")
    print("="*60)
    print(f"总实验数: {total_experiments}")
    print("按 Ctrl+C 退出监控\n")
    
    try:
        while True:
            completed_count, completed_list = count_completed_experiments()
            
            progress = (completed_count / total_experiments) * 100
            
            # 清屏（可选）
            # os.system('clear')
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"进度: {completed_count}/{total_experiments} "
                  f"({progress:.1f}%) ", end='', flush=True)
            
            if completed_count >= total_experiments:
                print("\n\n✓ 所有实验已完成！")
                print("\n已完成的实验:")
                for exp in sorted(completed_list):
                    print(f"  ✓ {exp}")
                print("\n运行汇总脚本查看结果:")
                print("  python MERIT/scripts/summarize_results.py")
                break
            
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print(f"当前进度: {completed_count}/{total_experiments}")

if __name__ == '__main__':
    monitor_loop()

