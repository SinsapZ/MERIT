#!/usr/bin/env python3
import os
import re
import sys
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
    log_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "grid_search_logs")
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
        # pick by best validation AUROC, then F1 as tie-breaker
        best = max(entries, key=lambda x: (x["val_auroc"], x["val_f1"]))
        best["tag"] = tag
        rows.append(best)

    if not rows:
        print("No valid entries parsed from logs.")
        return 1

    # sort by val_auroc desc then val_f1 desc
    rows.sort(key=lambda x: (x["val_auroc"], x["val_f1"]), reverse=True)

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
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Best tag: {best['tag']}\n")
        f.write(f"Val - AUROC: {best['val_auroc']:.5f}, F1: {best['val_f1']:.5f}, Acc: {best['val_acc']:.5f}\n")
        f.write(f"Test - AUROC: {best['test_auroc']:.5f}, F1: {best['test_f1']:.5f}, Acc: {best['test_acc']:.5f}\n")

    print(f"Top-1: {best['tag']} | Val AUROC={best['val_auroc']:.5f}, F1={best['val_f1']:.5f} | Test AUROC={best['test_auroc']:.5f}, F1={best['test_f1']:.5f}")
    print(f"Summary saved: {out_csv}\nBest saved: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


