#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decision Curve / Utility vs Rejection:
- 基于置信度阈值（或拒绝率），计算总体期望收益：
  kept: 机器自动判读；rejected: 人工复核（给定准确率与成本）
- 成本参数：
  --cost_fp, --cost_fn  错误代价；--cost_review 人工复核代价；--human_acc 人工准确率
输出：PNG/SVG 与 txt 摘要
"""
import os
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt


def load_arrays(base_dir):
    arr = {}
    for n in ('confidences.npy','predictions.npy','labels.npy'):
        p=os.path.join(base_dir,n)
        if not os.path.exists(p):
            return None
        arr[n]=np.load(p)
    return arr


def expected_utility(conf, pred, label, rejection_rate, cost_fp, cost_fn, cost_review, human_acc):
    # 使用置信度排序（高到低），低置信度部分被拒绝
    n = len(pred)
    k = max(0, int(n * (rejection_rate/100.0)))
    order = np.argsort(conf)  # 从低到高
    rejected = order[:k]
    kept = order[k:]

    # kept 由机器判读
    yk = label[kept]; pk = pred[kept]
    correct_k = (yk == pk).sum()
    fp_k = ((yk != pk) & (pk == 1)).sum()  # 若非二分类，此处仅作示例：将预测为1的误报计为FP
    fn_k = ((yk != pk) & (pk != 1)).sum()  # 非1类的错误计为FN

    # rejected 由人工复核
    yr = label[rejected]
    # 人工以概率 human_acc 判对（简单期望）
    exp_correct_r = human_acc * len(rejected)
    exp_wrong_r = (1.0 - human_acc) * len(rejected)
    # 审核成本
    cost_review_total = cost_review * len(rejected)

    # 效用：正确 +1，错误-(cost_fp/cost_fn)，加上复核成本
    utility = 1.0*correct_k - cost_fp*fp_k - cost_fn*fn_k
    utility += 1.0*exp_correct_r - (cost_fp+cost_fn)/2.0*exp_wrong_r
    utility -= cost_review_total

    # 归一化到每100例
    return utility / n * 100.0


def main():
    p = ap.ArgumentParser()
    p.add_argument('--base_dir', required=True, help='results/uncertainty/<DATASET>')
    p.add_argument('--dataset', required=True)
    p.add_argument('--cost_fp', type=float, default=1.0)
    p.add_argument('--cost_fn', type=float, default=2.0)
    p.add_argument('--cost_review', type=float, default=0.2)
    p.add_argument('--human_acc', type=float, default=0.98)
    p.add_argument('--out_dir', type=str, default='')
    args = p.parse_args()

    out_dir = args.out_dir or args.base_dir
    os.makedirs(out_dir, exist_ok=True)

    evi = load_arrays(os.path.join(args.base_dir, 'evi'))
    soft = load_arrays(os.path.join(args.base_dir, 'softmax'))
    if evi is None or soft is None:
        print('Missing arrays under evi/ or softmax/')
        return

    xs = np.arange(0, 95, 5)  # 拒绝率0-90%
    ue=[]; us=[]
    for r in xs:
        ue.append(expected_utility(evi['confidences.npy'], evi['predictions.npy'], evi['labels.npy'], r,
                                   args.cost_fp, args.cost_fn, args.cost_review, args.human_acc))
        us.append(expected_utility(soft['confidences.npy'], soft['predictions.npy'], soft['labels.npy'], r,
                                   args.cost_fp, args.cost_fn, args.cost_review, args.human_acc))
    ue=np.array(ue); us=np.array(us)

    plt.figure(figsize=(7.5,5))
    plt.plot(xs, ue, 'o-', color='#e1909c', label='EviMR-Net')
    plt.plot(xs, us, 'o--', color='#4a4a4a', label='Softmax')
    plt.xlabel('Rejection rate (%)'); plt.ylabel('Expected utility per 100 cases')
    plt.title(f'{args.dataset}: Decision Curve (utility vs rejection)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    out=os.path.join(out_dir, 'decision_curve')
    plt.savefig(out+'.png', dpi=300); plt.savefig(out+'.svg'); plt.close()

    with open(os.path.join(out_dir,'decision_curve_summary.txt'),'w') as f:
        best_e = xs[np.argmax(ue)]; best_s = xs[np.argmax(us)]
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Best utility (EviMR) at reject={best_e}%: {ue.max():.2f}\n')
        f.write(f'Best utility (Softmax) at reject={best_s}%: {us.max():.2f}\n')
        f.write(f'Params: cost_fp={args.cost_fp}, cost_fn={args.cost_fn}, cost_review={args.cost_review}, human_acc={args.human_acc}\n')
    print('Saved decision curve and summary to', out_dir)


if __name__ == '__main__':
    main()


