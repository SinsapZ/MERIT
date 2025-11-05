"""Fine-tune ECGFounder Net1D backbone on MERIT datasets.

Usage:
  python -m MERIT.scripts.finetune_ecgfounder \
      --data PTB-XL \
      --root_path /home/Data1/zbl/dataset/PTB-XL \
      --checkpoint ./checkpoints/ecgfounder/12_lead_ECGFounder.pth \
      --output_dir results/baselines/PTB-XL/ecgfounder

Environment:
  Make sure the ECGFounder repository is cloned at ../../ECGFounder/ECGFounder
  relative to this script, or use --ecgfounder_root to specify the path.

Note:
  This script assumes 12-lead ECG input; for single-lead variants extend the
  loader accordingly.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def import_ecgfounder(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from finetune_model import ft_12lead_ECGFounder  # type: ignore
        from net1d import Net1D  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            f"无法导入 ECGFounder 模型，请确认路径 {root} 是否正确，且包含 finetune_model.py"
        ) from exc
    return ft_12lead_ECGFounder, Net1D


def build_loaders(args):
    from MERIT.data_provider.data_factory import data_provider

    base_args = SimpleNamespace(
        task_name="classification",
        data=args.data,
        root_path=args.root_path,
        freq="h",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )

    train_data, train_loader = data_provider(base_args, flag="TRAIN")
    seq_len = getattr(train_data, "max_seq_len", args.seq_len)
    base_args.seq_len = seq_len
    # rebuild loaders with updated seq_len so VAL/TEST share the same length
    train_data, train_loader = data_provider(base_args, flag="TRAIN")
    val_data, val_loader = data_provider(base_args, flag="VAL")
    test_data, test_loader = data_provider(base_args, flag="TEST")

    n_classes = len(np.unique(train_data.y))
    return train_loader, val_loader, test_loader, n_classes


def evaluate(model, loader, device, num_classes):
    model.eval()
    preds = []
    scores = []
    trues = []
    with torch.no_grad():
        for batch_x, labels, _ in loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)  # [B, C, L]
            y = labels.to(device).long().squeeze(-1)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(probs, dim=1).cpu())
            trues.append(y.cpu())
            scores.append(probs.cpu())
    if not preds:
        return {"acc": 0.0, "f1": 0.0, "auroc": 0.0}
    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    y_score = torch.cat(scores).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auroc = roc_auc_score(
            torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=num_classes).numpy(),
            y_score,
            multi_class="ovr",
        )
    except Exception:
        auroc = 0.0
    return {"acc": acc, "f1": f1, "auroc": auroc}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ecgfounder_root = (
        Path(args.ecgfounder_root)
        if args.ecgfounder_root
        else PROJECT_ROOT.parent / "ECGFounder" / "ECGFounder"
    )
    ft_12lead, Net1D = import_ecgfounder(ecgfounder_root)

    train_loader, val_loader, test_loader, num_classes = build_loaders(args)

    if args.checkpoint and Path(args.checkpoint).exists():
        model = ft_12lead(
            device=device,
            pth=args.checkpoint,
            n_classes=num_classes,
            linear_prob=args.linear_probe,
        )
    elif args.init_random or not args.checkpoint:
        model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=num_classes,
        ).to(device)
    else:
        raise FileNotFoundError(
            f"未找到 checkpoint: {args.checkpoint}，若需随机初始化请增加 --init_random 或提供权重路径"
        )

    if args.linear_probe:
        for name, param in model.named_parameters():
            if "dense" not in name:
                param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_val = -float("inf")
    best_metrics = None

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []
        for batch_x, labels, _ in train_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            y = labels.to(device).long().squeeze(-1)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, num_classes)
        test_metrics = evaluate(model, test_loader, device, num_classes)

        msg = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_loss)) if epoch_loss else 0.0,
            "val": val_metrics,
            "test": test_metrics,
        }
        print(json.dumps(msg))
        with open(Path(args.output_dir) / "progress.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")

        if val_metrics[args.monitor_metric] > best_val:
            best_val = val_metrics[args.monitor_metric]
            best_metrics = {"val": val_metrics, "test": test_metrics, "epoch": epoch}
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"Best metrics saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="PTB-XL", help="MERIT dataset name")
    parser.add_argument("--root_path", type=str, required=True, help="数据集根目录")
    parser.add_argument("--checkpoint", type=str, default="", help="ECGFounder 预训练权重路径，可为空表示随机初始化")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
    parser.add_argument("--ecgfounder_root", type=str, default="", help="ECGFounder 仓库路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--linear_probe", action="store_true", help="仅优化最后一层")
    parser.add_argument("--init_random", action="store_true", help="不加载预训练权重，随机初始化")
    parser.add_argument("--monitor_metric", type=str, default="f1", choices=["acc", "f1", "auroc"])
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

