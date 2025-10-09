#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析MERIT实验结果并提供改进建议
"""
import pandas as pd
import numpy as np
import sys

def main():
    print("\n" + "="*80)
    print("MERIT vs MedGNN 性能对比分析")
    print("="*80)
    
    # MedGNN基准性能
    medgnn = {
        'acc': 0.8260,
        'f1': 0.8025,
        'auroc': 0.8593,
        'acc_std': 0.0035
    }
    
    # 上次运行的MERIT结果（不含GNN）
    merit_no_gnn = {
        'acc': 0.78002,
        'f1': 0.74152,
        'auroc': 0.84546,
        'acc_std': 0.02463,
        'f1_std': 0.03256
    }
    
    print("\n📊 性能对比:")
    print("-"*80)
    print(f"{'模型':<20} {'Test Acc':<20} {'Test F1':<20} {'Test AUROC':<15}")
    print("-"*80)
    print(f"{'MedGNN (基准)':<20} {medgnn['acc']:.4f}±{medgnn['acc_std']:.4f}     {medgnn['f1']:.4f}           {medgnn['auroc']:.4f}")
    print(f"{'MERIT (无GNN)':<20} {merit_no_gnn['acc']:.4f}±{merit_no_gnn['acc_std']:.4f}     {merit_no_gnn['f1']:.4f}±{merit_no_gnn['f1_std']:.4f}   {merit_no_gnn['auroc']:.4f}")
    print("-"*80)
    
    # 计算差距
    acc_gap = (merit_no_gnn['acc'] - medgnn['acc']) * 100
    f1_gap = (merit_no_gnn['f1'] - medgnn['f1']) * 100
    
    print(f"\n❌ 性能差距:")
    print(f"   Accuracy: {acc_gap:+.2f}% (差距: {abs(acc_gap):.2f}个百分点)")
    print(f"   F1 Score: {f1_gap:+.2f}% (差距: {abs(f1_gap):.2f}个百分点)")
    
    print("\n" + "="*80)
    print("🔍 问题诊断")
    print("="*80)
    
    problems = [
        ("❌ 关键问题", "GNN未启用！脚本中缺少 --use_gnn 参数"),
        ("⚠️  训练配置", "训练轮数过多 (200 vs 10 epochs)，可能导致过拟合"),
        ("⚠️  早停配置", "patience过大 (30 vs 3)，模型训练时间过长"),
        ("⚠️  学习率", "学习率略高 (1.1e-4 vs 1e-4)"),
    ]
    
    for i, (severity, issue) in enumerate(problems, 1):
        print(f"{i}. {severity}")
        print(f"   {issue}")
    
    print("\n" + "="*80)
    print("✅ 已实施的修复方案")
    print("="*80)
    
    fixes = [
        "✅ 添加 --use_gnn 参数以启用多分辨率GNN",
        "✅ train_epochs: 200 → 10 (与MedGNN一致)",
        "✅ patience: 30 → 3 (早停更及时)",
        "✅ learning_rate: 1.1e-4 → 1e-4 (与MedGNN一致)",
        "✅ 保留数据增强: none,drop0.35 (已正确配置)",
        "✅ 保留SWA优化 (有助于提升性能)",
    ]
    
    for fix in fixes:
        print(f"  {fix}")
    
    print("\n" + "="*80)
    print("🎯 预期改进")
    print("="*80)
    
    improvements = [
        ("启用GNN", "+3~5%", "多分辨率图神经网络可以捕获通道间依赖关系"),
        ("减少过拟合", "+1~2%", "更短的训练轮数和早停可以提高泛化性"),
        ("学习率优化", "+0.5~1%", "更稳定的学习率有助于收敛"),
    ]
    
    print(f"{'改进项':<15} {'预期提升':<12} {'原因':<50}")
    print("-"*80)
    for item, gain, reason in improvements:
        print(f"{item:<15} {gain:<12} {reason:<50}")
    
    total_expected = "+4.5~8%"
    print("-"*80)
    print(f"{'总计':<15} {total_expected:<12}")
    
    expected_acc = merit_no_gnn['acc'] + 0.06  # 中间值 6%
    print(f"\n预期性能: Test Acc ≈ {expected_acc:.4f} ({expected_acc*100:.2f}%)")
    
    if expected_acc >= medgnn['acc']:
        print(f"🎉 有希望超过MedGNN ({medgnn['acc']*100:.2f}%)!")
    else:
        remaining_gap = (medgnn['acc'] - expected_acc) * 100
        print(f"⚠️  预计仍有 {remaining_gap:.2f}% 的差距")
    
    print("\n" + "="*80)
    print("📋 后续优化建议")
    print("="*80)
    
    suggestions = [
        ("1. 如果性能仍不理想", [
            "尝试调整 nodedim: 10 → 12 或 15",
            "调整 lambda_pseudo_loss: 0.3 → 0.2 或 0.4",
            "尝试不同的 annealing_epoch: 50 → 30 或 70",
        ]),
        ("2. 稳定性优化", [
            "增加运行种子数: 10 → 20 seeds",
            "检查方差: 如果std过大，考虑增加正则化",
        ]),
        ("3. 模型分析", [
            "可视化注意力权重，检查GNN是否学到有效模式",
            "对比各个分辨率的贡献度",
            "分析失败样本，找出模型弱点",
        ]),
    ]
    
    for title, items in suggestions:
        print(f"\n{title}:")
        for item in items:
            print(f"   • {item}")
    
    print("\n" + "="*80)
    print("🚀 下一步操作")
    print("="*80)
    print("\n在Linux服务器上运行修复后的脚本：")
    print("\n   bash MERIT/scripts/run_final_with_gnn.sh")
    print("\n预计运行时间: ~40-50分钟 (10 seeds × 10 epochs)")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

