"""Fine-tune ECG-FM (KED) backbone on MERIT datasets.

This script reuses MERIT's dataloaders so it works for APAVA/ADFD/PTB/PTB-XL
without having to convert data into the ecg-fm memmap format.
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

from clinical_ts.models.fm_ecg import EcgFmKEDWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

ECGFM_CODE_ROOT = Path(os.environ.get("ECGFM_CODE_ROOT", WORKSPACE_ROOT / "ECGFM" / "ecg-fm-benchmarking" / "code"))
if str(ECGFM_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGFM_CODE_ROOT))


def build_loaders(args):
    """Reuse MERIT data_provider to obtain train/val/test loaders."""
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
    # Rebuild loaders with the updated sequence length to keep padding masks aligned.
    train_data, train_loader = data_provider(base_args, flag="TRAIN")
    val_data, val_loader = data_provider(base_args, flag="VAL")
    test_data, test_loader = data_provider(base_args, flag="TEST")

    n_classes = len(np.unique(train_data.y))
    return train_loader, val_loader, test_loader, n_classes, seq_len, train_data.X.shape[2]


def adapt_channels(batch_x, target_channels=12):
    """Convert MERIT tensors (B, L, C) to (B, target_channels, L)."""
    if batch_x.dim() != 3:
        raise ValueError("Expected batch_x shape (B, L, C)")
    b, length, channels = batch_x.shape
    if channels > target_channels:
        batch_x = batch_x[:, :, :target_channels]
    elif channels < target_channels:
        pad = torch.zeros(b, length, target_channels - channels, device=batch_x.device)
        batch_x = torch.cat([batch_x, pad], dim=2)
    return batch_x.permute(0, 2, 1).contiguous()


def evaluate(model, loader, device, num_classes):
    model.eval()
    preds = []
    probs = []
    trues = []
    with torch.no_grad():
        for batch_x, labels, _ in loader:
            batch_x = adapt_channels(batch_x.to(device).float())
            y = labels.to(device).long().view(-1)
            logits = model(batch_x)
            probs.append(torch.softmax(logits, dim=1).cpu())
            preds.append(torch.argmax(logits, dim=1).cpu())
            trues.append(y.cpu())
    if not preds:
        return {"acc": 0.0, "f1": 0.0, "auroc": 0.0}
    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        y_true_onehot = torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=num_classes).numpy()
        auroc = roc_auc_score(y_true_onehot, y_prob, multi_class="ovr")
    except Exception:
        auroc = 0.0
    return {"acc": acc, "f1": f1, "auroc": auroc}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader, num_classes, seq_len, _ = build_loaders(args)

    if not args.checkpoint:
        raise ValueError("ECG-FM 需要预训练 checkpoint，请通过 --checkpoint 指定路径")

    model = EcgFmKEDWrapper(
        num_classes=num_classes,
        num_output_tokens=seq_len,
        pretrained_path=args.checkpoint,
        eval_mode=args.eval_mode,
        lr=args.lr,
        discriminative_lr_factor=args.discriminative_lr_factor,
    ).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val = -float("inf")
    best_metrics = None

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []
        for batch_x, labels, _ in train_loader:
            batch_x = adapt_channels(batch_x.to(device).float())
            y = labels.to(device).long().view(-1)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, num_classes)
        test_metrics = evaluate(model, test_loader, device, num_classes)

        log_entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_loss)) if epoch_loss else 0.0,
            "val": val_metrics,
            "test": test_metrics,
        }
        print(json.dumps(log_entry))
        with open(Path(args.output_dir) / "progress.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        monitor = val_metrics[args.monitor_metric]
        if monitor > best_val:
            best_val = monitor
            best_metrics = {"val": val_metrics, "test": test_metrics, "epoch": epoch}
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")

    if best_metrics is None:
        best_metrics = {"val": val_metrics, "test": test_metrics, "epoch": args.epochs}

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"Best metrics saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="PTB-XL", help="MERIT dataset name")
    parser.add_argument("--root_path", type=str, required=True, help="数据集根目录")
    parser.add_argument("--checkpoint", type=str, default="", help="ECG-FM 预训练权重")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--discriminative_lr_factor", type=float, default=0.1)
    parser.add_argument("--eval_mode", type=str, default="finetuning_linear", choices=["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"])
    parser.add_argument("--monitor_metric", type=str, default="f1", choices=["acc", "f1", "auroc"])
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover
    main()

