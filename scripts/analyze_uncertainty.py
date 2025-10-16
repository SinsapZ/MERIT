#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MERIT不确定性全面分析
包括：噪声鲁棒性、不确定性分布、拒绝实验、案例可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import os

def plot_noise_robustness(args):
    """
    噪声鲁棒性实验
    对比Full Model vs w/o Evidential Fusion在不同噪声下的性能
    """
    print("\n" + "="*80)
    print("1. 噪声鲁棒性实验")
    print("="*80)
    
    # 需要加载在不同噪声水平下的实验结果
    # 这需要事先运行噪声实验
    
    print("说明: 需要先运行噪声实验")
    print("运行方式: 在测试时对数据添加不同强度的高斯噪声")
    print("噪声水平: 0, 0.05, 0.1, 0.15, 0.2, 0.25")
    
    # 示例代码框架
    code_example = """
# 在exp_classification.py的test函数中添加:
for noise_level in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    # 添加噪声
    batch_x_noisy = batch_x + torch.randn_like(batch_x) * noise_level
    
    # 预测
    outputs = model(batch_x_noisy, ...)
    
    # 记录结果
    save_results(noise_level, accuracy, ...)
"""
    print(code_example)

def plot_uncertainty_distribution(uncertainties, errors, save_path):
    """绘制不确定性分布 (正确 vs 错误样本)"""
    print("\n" + "="*80)
    print("2. 不确定性分布分析")
    print("="*80)
    
    correct_unc = uncertainties[errors == 0]
    wrong_unc = uncertainties[errors == 1]
    
    print(f"正确样本不确定性: {correct_unc.mean():.4f} ± {correct_unc.std():.4f}")
    print(f"错误样本不确定性: {wrong_unc.mean():.4f} ± {wrong_unc.std():.4f}")
    print(f"差异: {(wrong_unc.mean() - correct_unc.mean()):.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # KDE分布
    from scipy.stats import gaussian_kde
    
    if len(correct_unc) > 10:
        kde_correct = gaussian_kde(correct_unc)
        x_range = np.linspace(uncertainties.min(), uncertainties.max(), 200)
        plt.plot(x_range, kde_correct(x_range), 'g-', linewidth=2, label='Correct Predictions')
    
    if len(wrong_unc) > 10:
        kde_wrong = gaussian_kde(wrong_unc)
        plt.plot(x_range, kde_wrong(x_range), 'r-', linewidth=2, label='Wrong Predictions')
    
    plt.xlabel('Uncertainty', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Uncertainty Distribution: Correct vs Wrong', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 分布图已保存: {save_path}")

def plot_rejection_curve(uncertainties, predictions, labels, save_path):
    """绘制拒绝率 vs 准确率曲线"""
    print("\n" + "="*80)
    print("3. 拒绝实验 (Rejection Experiment)")
    print("="*80)
    
    # 按不确定性排序
    sorted_idx = np.argsort(uncertainties)  # 低到高
    
    rejection_rates = []
    accuracies = []
    
    baseline_acc = accuracy_score(labels, predictions)
    
    for reject_ratio in np.linspace(0, 0.5, 21):  # 0%-50%拒绝率
        n_reject = int(len(predictions) * reject_ratio)
        
        if n_reject < len(predictions):
            # 拒绝最不确定的n_reject个样本
            keep_idx = sorted_idx[:-n_reject] if n_reject > 0 else sorted_idx
            
            if len(keep_idx) > 0:
                acc = accuracy_score(labels[keep_idx], predictions[keep_idx])
                rejection_rates.append(reject_ratio * 100)
                accuracies.append(acc * 100)
    
    rejection_rates = np.array(rejection_rates)
    accuracies = np.array(accuracies)
    
    # 打印关键点
    print(f"Baseline (0% rejection): {baseline_acc*100:.2f}%")
    for i, (rej, acc) in enumerate(zip(rejection_rates, accuracies)):
        if i % 4 == 0:  # 每隔4个打印
            gain = acc - baseline_acc * 100
            marker = "✅" if gain > 0 else ""
            print(f"  {rej:>5.0f}% rejection: {acc:>6.2f}% ({gain:>+5.2f}%) {marker}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(rejection_rates, accuracies, 'b-o', linewidth=2, markersize=6, label='MERIT')
    plt.axhline(y=baseline_acc*100, color='r', linestyle='--', linewidth=2, 
                label=f'No Rejection: {baseline_acc*100:.2f}%')
    
    # 标注超过baseline的点
    improved = accuracies > baseline_acc * 100
    if improved.any():
        plt.fill_between(rejection_rates[improved], baseline_acc*100, accuracies[improved], 
                        alpha=0.3, color='green', label='Improvement Area')
    
    plt.xlabel('Rejection Rate (%)', fontsize=14)
    plt.ylabel('Accuracy on Remaining Samples (%)', fontsize=14)
    plt.title('Rejection-Accuracy Trade-off', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 拒绝曲线已保存: {save_path}")

def visualize_cases(args):
    """可视化案例分析"""
    print("\n" + "="*80)
    print("4. 案例可视化")
    print("="*80)
    
    print("说明: 需要加载原始时序数据和预测结果")
    print("步骤:")
    print("  1. 选择高置信度正确样本")
    print("  2. 选择高不确定性样本")
    print("  3. 可视化波形 + 预测 + 不确定性")
    
    code_example = """
# 选择样本
high_conf_correct = (confidence > 0.9) & (predictions == labels)
high_unc = (uncertainties > 0.7)

# 各选2-3个样本
selected_indices = list(np.where(high_conf_correct)[0][:3]) + \\
                   list(np.where(high_unc)[0][:3])

# 加载原始信号
for idx in selected_indices:
    signal = X_test[idx]  # (seq_len, channels)
    
    # 绘制多通道波形
    plt.figure(figsize=(15, 8))
    for ch in range(min(6, signal.shape[1])):
        plt.subplot(3, 2, ch+1)
        plt.plot(signal[:, ch])
        plt.title(f'Channel {ch}')
    
    plt.suptitle(f'Sample {idx}: Pred={predictions[idx]}, True={labels[idx]}, '
                 f'Uncertainty={uncertainties[idx]:.3f}')
    plt.tight_layout()
    plt.savefig(f'case_{idx}.png')
"""
    print(code_example)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_dir', type=str, required=True, 
                        help='包含uncertainties.npy等文件的目录')
    parser.add_argument('--dataset_name', type=str, default='APAVA')
    parser.add_argument('--output_dir', type=str, default='results/uncertainty_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print(f"MERIT不确定性全面分析 - {args.dataset_name}")
    print("="*80)
    
    # 加载数据
    try:
        uncertainties = np.load(os.path.join(args.uncertainty_dir, 'uncertainties.npy'))
        confidences = np.load(os.path.join(args.uncertainty_dir, 'confidences.npy'))
        predictions = np.load(os.path.join(args.uncertainty_dir, 'predictions.npy'))
        labels = np.load(os.path.join(args.uncertainty_dir, 'labels.npy'))
        
        print(f"\n✅ 数据加载成功: {len(predictions)} 个样本")
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        print("\n需要先修改exp_classification.py保存不确定性数据")
        print("参考: evaluate_uncertainty.py中的说明")
        return
    
    errors = (predictions != labels).astype(float)
    
    # 1. 噪声鲁棒性
    plot_noise_robustness(args)
    
    # 2. 不确定性分布
    dist_path = os.path.join(args.output_dir, f'{args.dataset_name}_uncertainty_distribution.png')
    plot_uncertainty_distribution(uncertainties, errors, dist_path)
    
    # 3. 拒绝实验
    reject_path = os.path.join(args.output_dir, f'{args.dataset_name}_rejection_curve.png')
    plot_rejection_curve(uncertainties, predictions, labels, reject_path)
    
    # 4. 案例可视化
    visualize_cases(args)
    
    print("\n" + "="*80)
    print("✅ 不确定性分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - {dist_path}")
    print(f"  - {reject_path}")
    print("\n这些图可以直接用于ESWA论文的不确定性分析部分！")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

