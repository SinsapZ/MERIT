#!/usr/bin/env python3
import os
import re
import sys
import argparse
from glob import glob


VAL_RE = re.compile(
    r"Validation results --- Loss: ([0-9eE+\-\.]+), Accuracy: ([0-9eE+\-\.]+), Precision: ([0-9eE+\-\.]+), Recall: ([0-9eE+\-\.]+), F1: ([0-9eE+\-\.]+), AUROC: ([0-9eE+\-\.]+), AUPRC: ([0-9eE+\-\.]+)"
)
TEST_RE = re.compile(
    r"Test results --- Loss: ([0-9eE+\-\.]+), Accuracy: ([0-9eE+\-\.]+), Precision: ([0-9eE+\-\.]+), Recall: ([0-9eE+\-\.]+), F1: ([0-9eE+\-\.]+), AUROC: ([0-9eE+\-\.]+), AUPRC: ([0-9eE+\-\.]+)"
)


def parse_log(path):
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m_val = VAL_RE.search(lines[i])
        if m_val and i + 1 < len(lines):
            m_test = TEST_RE.search(lines[i + 1])
            if m_test:
                val = list(map(float, m_val.groups()))
                test = list(map(float, m_test.groups()))
                entries.append({
                    "val_loss": val[0], "val_acc": val[1], "val_prec": val[2], "val_rec": val[3], "val_f1": val[4], "val_auroc": val[5], "val_auprc": val[6],
                    "test_loss": test[0], "test_acc": test[1], "test_prec": test[2], "test_rec": test[3], "test_f1": test[4], "test_auroc": test[5], "test_auprc": test[6],
                })
                i += 2
                continue
        i += 1
    return entries


def main():
    parser = argparse.ArgumentParser(description="Select best run from logs")
    parser.add_argument("log_dir", nargs="?", default=os.path.join("results", "grid_search_logs"), help="directory containing run_*.log")
    parser.add_argument("--by", choices=["val", "test"], default="val", help="rank by validation or test metrics")
    parser.add_argument("--score", choices=["single", "composite"], default="single", help="rank by single metric (AUROC) or composite score")
    parser.add_argument("--w_auroc", type=float, default=0.4, help="weight for AUROC in composite score")
    parser.add_argument("--w_auprc", type=float, default=0.3, help="weight for AUPRC in composite score")
    parser.add_argument("--w_f1", type=float, default=0.2, help="weight for F1 in composite score")
    parser.add_argument("--w_acc", type=float, default=0.1, help="weight for Accuracy in composite score")
    parser.add_argument("--min_acc", type=float, default=0.0, help="minimum Accuracy constraint (0~1), filter out runs below")
    parser.add_argument("--min_f1", type=float, default=0.0, help="minimum F1 constraint (0~1), filter out runs below")
    args = parser.parse_args()

    log_dir = args.log_dir
    paths = sorted(glob(os.path.join(log_dir, "run_*.log")))
    if not paths:
        print(f"No logs found under {log_dir}")
        return 1

    rows = []
    for p in paths:
        tag = os.path.basename(p).replace("run_", "").replace(".log", "")
        entries = parse_log(p)
        if not entries:
            continue
        # pick best per-log
        def score_fn(e):
            if args.by == "val":
                acc, f1, auroc, auprc = e["val_acc"], e["val_f1"], e["val_auroc"], e["val_auprc"]
            else:
                acc, f1, auroc, auprc = e["test_acc"], e["test_f1"], e["test_auroc"], e["test_auprc"]
            if args.score == "single":
                return (auroc, f1)
            # composite
            comp = (
                args.w_auroc * auroc +
                args.w_auprc * auprc +
                args.w_f1 * f1 +
                args.w_acc * acc
            )
            return (comp, auroc)

        # optional constraints filter
        cand = [e for e in entries if (e["val_acc"] >= args.min_acc and e["val_f1"] >= args.min_f1)] if args.by == "val" else \
               [e for e in entries if (e["test_acc"] >= args.min_acc and e["test_f1"] >= args.min_f1)]
        if not cand:
            cand = entries
        best = max(cand, key=score_fn)
        best["tag"] = tag
        rows.append(best)

    if not rows:
        print("No valid entries parsed from logs.")
        return 1

    # global sort
    def row_score_key(r):
        if args.by == "val":
            acc, f1, auroc, auprc = r["val_acc"], r["val_f1"], r["val_auroc"], r["val_auprc"]
        else:
            acc, f1, auroc, auprc = r["test_acc"], r["test_f1"], r["test_auroc"], r["test_auprc"]
        if args.score == "single":
            return (auroc, f1)
        comp = (
            args.w_auroc * auroc +
            args.w_auprc * auprc +
            args.w_f1 * f1 +
            args.w_acc * acc
        )
        return (comp, auroc)

    rows.sort(key=row_score_key, reverse=True)

    # write summary csv
    out_csv = os.path.join(log_dir, "summary.csv")
    headers = [
        "tag",
        "val_loss","val_acc","val_prec","val_rec","val_f1","val_auroc","val_auprc",
        "test_loss","test_acc","test_prec","test_rec","test_f1","test_auroc","test_auprc",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    # write best txt
    best = rows[0]
    out_txt = os.path.join(log_dir, "summary_best.txt")
    # Also decode hyperparameters from tag if present (lbd*_lr*_lp*)
    hp = {}
    m_lbd = re.search(r"lbd([0-9eE\-\.]+)", best['tag'])
    m_lr = re.search(r"_lr([0-9eE\-\.]+)", best['tag'])
    m_lp = re.search(r"_lp([0-9eE\-\.]+)", best['tag'])
    if m_lbd: hp['lambda_evi'] = m_lbd.group(1)
    if m_lr: hp['learning_rate'] = m_lr.group(1)
    if m_lp: hp['lambda_pseudo'] = m_lp.group(1)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Best tag: {best['tag']} (by {args.by})\n")
        if hp:
            f.write("Hyperparams: " + ", ".join([f"{k}={v}" for k,v in hp.items()]) + "\n")
        f.write(f"Val - AUROC: {best['val_auroc']:.5f}, F1: {best['val_f1']:.5f}, Acc: {best['val_acc']:.5f}\n")
        f.write(f"Test - AUROC: {best['test_auroc']:.5f}, F1: {best['test_f1']:.5f}, Acc: {best['test_acc']:.5f}\n")

    if args.by == "val":
        print(f"Top-1 (by val/{args.score}): {best['tag']} | Val AUROC={best['val_auroc']:.5f}, F1={best['val_f1']:.5f} | Test AUROC={best['test_auroc']:.5f}, F1={best['test_f1']:.5f}")
    else:
        print(f"Top-1 (by test/{args.score}): {best['tag']} | Test AUROC={best['test_auroc']:.5f}, F1={best['test_f1']:.5f} | Val AUROC={best['val_auroc']:.5f}, F1={best['val_f1']:.5f}")
    print(f"Summary saved: {out_csv}\nBest saved: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


