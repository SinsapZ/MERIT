#!/usr/bin/env python3
"""
Summarize ablation CSV across multiple seeds.

Input CSV columns (at least):
seed,return_code,duration_sec,val_loss,val_acc,val_prec,val_rec,val_f1,
test_loss,test_acc,test_prec,test_rec,test_f1,test_auroc,test_auprc

This script computes mean/std/var over seeds for five test metrics:
- test_acc, test_prec, test_rec, test_f1, test_auroc

Usage:
  python summarize_ablation_csv.py --csv your_file.csv --out summary.csv
If --csv is omitted, it looks for 'results.csv' in the same folder.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results.csv",
                        help="Input ablation csv (default: results.csv in this folder)")
    parser.add_argument("--out", type=str, default="summary.csv",
                        help="Output summary csv (default: summary.csv)")
    args = parser.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        raise FileNotFoundError(f"CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # Coerce relevant columns to float
    metrics = ["test_acc", "test_prec", "test_rec", "test_f1", "test_auroc"]
    for m in metrics:
        if m in df.columns:
            df[m] = df[m].apply(safe_float)
        else:
            df[m] = np.nan

    # Drop rows with all-NaN metrics
    df_valid = df.dropna(subset=metrics, how="all").copy()
    if len(df_valid) == 0:
        raise ValueError(f"No valid metric rows found in {in_path}")

    # Compute mean/std/var ignoring NaNs
    summary = {}
    for m in metrics:
        vals = df_valid[m].astype(float).to_numpy()
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=0))
        var = float(np.nanvar(vals, ddof=0))
        summary[m] = {"mean": mean, "std": std, "var": var}

    # Build summary DataFrame
    out_rows = []
    for m in metrics:
        out_rows.append({
            "metric": m,
            "mean": summary[m]["mean"],
            "std": summary[m]["std"],
            "var": summary[m]["var"],
            "n": int(np.sum(~np.isnan(df_valid[m].to_numpy())))
        })
    out_df = pd.DataFrame(out_rows, columns=["metric", "mean", "std", "var", "n"])

    # Save
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)

    # Pretty print
    print(f"\nSummary ({in_path} -> {out_path}):")
    for _, row in out_df.iterrows():
        print(f"{row['metric']}: mean={row['mean']:.6f}  std={row['std']:.6f}  var={row['var']:.6f}  (n={int(row['n'])})")


if __name__ == "__main__":
    main()


