"""Quick launcher for running MERIT ablation variants across datasets.

This script wraps `multi_seed_run.py` to sequentially execute the default
ablation variants (w/o evidential fusion, w/o pseudo-view,
w/o frequency branch, w/o difference branch) on each dataset you specify.

Key features
------------
- Supports multiple datasets in one command (default: PTB, PTB-XL).
- Caps training epochs via `--max_epochs` (default 80) and patience via
  `--patience` (default 10) to speed up quick sweeps.
- Allows custom dataset root paths using `--root_paths`, e.g.
  `APAVA=/data/APAVA,PTB=/data/PTB`.
- Keeps seeds configurable (default: `41,42`) and writes outputs under
  `results/ablation_all_quick/<DATASET>/<variant>.csv`.

Example
-------
```bash
python -m MERIT.scripts.run_ablation_all \
  --root_paths APAVA=/data/APAVA,PTB=/data/PTB,PTB-XL=/data/PTBXL,ADFD-Sample=/data/ADFD \
  --gpu 0 --max_epochs 60 --seeds 41,42
```
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


DATASET_CONFIGS = {
    "PTB": {
        "default_root": "/home/Data1/zbl/dataset/PTB",
        "lr": 1.0e-4,
        "annealing_epoch": 50,
        "e_layers": 4,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "nodedim": 10,
        "batch_size": 64,
        "resolution_list": "2,4,6,8",
        "lambda_pseudo_loss": 0.30,
        "lambda_fuse": 1.0,
        "lambda_view": 1.0,
        "lambda_pseudo": 1.0,
        "swa": True,
        "default_epochs": 150,
        "patience": 20,
    },
    "PTB-XL": {
        "default_root": "/home/Data1/zbl/dataset/PTB-XL",
        "lr": 2.0e-4,
        "annealing_epoch": 50,
        "e_layers": 4,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "nodedim": 10,
        "batch_size": 64,
        "resolution_list": "2,4,6,8",
        "lambda_pseudo_loss": 0.30,
        "lambda_fuse": 1.0,
        "lambda_view": 1.0,
        "lambda_pseudo": 1.0,
        "swa": True,
        "default_epochs": 150,
        "patience": 20,
    }
}


VARIANTS = {
    "wo_evi": {
        "label": "w/o Evidential Fusion",
        "extra_args": ["--agg", "mean", "--no_pseudo"],
        "disable_ds": True,
    },
    "wo_pseudo": {
        "label": "w/o Pseudo-view",
        "extra_args": ["--no_pseudo", "--lambda_pseudo_loss", "0.0"],
        "disable_ds": False,
    },
    "wo_freq": {
        "label": "w/o Frequency Branch",
        "extra_args": ["--no_freq"],
        "disable_ds": False,
    },
    "wo_diff": {
        "label": "w/o Difference Branch",
        "extra_args": ["--no_diff"],
        "disable_ds": False,
    },
}


def parse_root_paths(raw: str):
    mapping = {}
    if not raw:
        return mapping
    pairs = [p.strip() for p in raw.split(',') if p.strip()]
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid root_paths entry: '{pair}'. Expected format DATASET=/path/to/root")
        key, value = pair.split('=', 1)
        mapping[key.strip()] = value.strip()
    return mapping


def build_base_cmd(dataset: str, config: dict, args, root_path: str, log_csv: str):
    epochs = min(config["default_epochs"], args.max_epochs)
    patience = min(config["patience"], args.patience)

    cmd = [
        sys.executable,
        '-m',
        'MERIT.scripts.multi_seed_run',
        '--root_path',
        root_path,
        '--data',
        dataset,
        '--gpu',
        str(args.gpu),
        '--lr',
        str(config["lr"]),
        '--lambda_fuse',
        str(config["lambda_fuse"]),
        '--lambda_view',
        str(config["lambda_view"]),
        '--lambda_pseudo_loss',
        str(config["lambda_pseudo_loss"]),
        '--lambda_pseudo',
        str(config["lambda_pseudo"]),
        '--annealing_epoch',
        str(config["annealing_epoch"]),
        '--evidence_dropout',
        '0.0',
        '--e_layers',
        str(config["e_layers"]),
        '--dropout',
        str(config["dropout"]),
        '--weight_decay',
        str(config["weight_decay"]),
        '--nodedim',
        str(config["nodedim"]),
        '--batch_size',
        str(config["batch_size"]),
        '--train_epochs',
        str(epochs),
        '--patience',
        str(patience),
        '--resolution_list',
        config["resolution_list"],
        '--seeds',
        args.seeds,
        '--log_csv',
        log_csv,
    ]

    if config.get("swa", False):
        cmd.append('--swa')

    if args.eval_only:
        cmd.append('--eval_only')

    if args.extra_multi_seed_args:
        cmd.extend(args.extra_multi_seed_args)

    return cmd


def run_variant(dataset: str, variant_key: str, base_cmd: list, variant: dict, args):
    cmd = list(base_cmd)
    if variant.get('disable_ds'):
        cmd.append('--disable_ds')
    cmd.extend(variant.get('extra_args', []))

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    friendly_name = variant['label']
    print(f"\n[{timestamp}] >>> {dataset} :: {friendly_name}")
    print('Command:', ' '.join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, env=os.environ.copy())
    if proc.returncode != 0:
        print(f"[warn] Variant '{variant_key}' on {dataset} exited with code {proc.returncode}")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run MERIT ablation study across multiple datasets with quick settings",
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default=','.join(DATASET_CONFIGS.keys()),
        help="Comma-separated datasets to run (default: all)"
    )
    parser.add_argument(
        '--variants',
        type=str,
        default='wo_evi,wo_pseudo,wo_freq,wo_diff',
        help="Comma-separated ablation variants to execute"
    )
    parser.add_argument(
        '--root_paths',
        type=str,
        default='',
        help="Override dataset root paths, e.g. 'APAVA=/data/APAVA,PTB=/data/PTB'"
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='41,42')
    parser.add_argument('--max_epochs', type=int, default=150, help='Upper bound of epochs for each dataset')
    parser.add_argument('--patience', type=int, default=10, help='Upper bound of early stopping patience')
    parser.add_argument('--output_dir', type=str, default='results/ablation_all_quick')
    parser.add_argument('--dry_run', action='store_true', help='Only print commands without executing')
    parser.add_argument('--eval_only', action='store_true', help='Pass --eval_only to multi_seed_run for evaluation-only mode')
    parser.add_argument(
        '--extra_multi_seed_args',
        nargs=argparse.REMAINDER,
        help='Additional arguments passed verbatim to multi_seed_run after the standard ones'
    )

    args = parser.parse_args()

    selected_datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    selected_variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    root_overrides = parse_root_paths(args.root_paths)

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in selected_datasets:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {', '.join(DATASET_CONFIGS.keys())}")
        config = DATASET_CONFIGS[dataset]
        root_path = root_overrides.get(dataset, config.get('default_root', ''))
        if not root_path:
            raise ValueError(f"Root path for dataset '{dataset}' is not specified. Use --root_paths to provide it.")

        dataset_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"Dataset: {dataset}")
        print(f"Root path: {root_path}")
        print(f"Output dir: {dataset_dir}")
        print(f"Variants: {', '.join(selected_variants)}")
        print("=" * 72)

        for variant_key in selected_variants:
            if variant_key not in VARIANTS:
                raise ValueError(f"Unknown variant '{variant_key}'. Available: {', '.join(VARIANTS.keys())}")

            variant = VARIANTS[variant_key]
            log_csv = os.path.join(dataset_dir, f"{variant_key}.csv")

            base_cmd = build_base_cmd(dataset, config, args, root_path, log_csv)
            run_variant(dataset, variant_key, base_cmd, variant, args)


if __name__ == '__main__':
    main()


