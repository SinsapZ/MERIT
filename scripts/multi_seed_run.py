import argparse
import os
import subprocess
import sys
import time
import csv
import re
import numpy as np


def run_once(base_cmd, env=None):
    start = time.time()
    try:
        proc = subprocess.run(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
        dur = time.time() - start
        return proc.returncode, proc.stdout, dur
    except Exception as e:
        return 1, str(e), 0.0


def parse_metrics(txt):
    # Parse Validation and Test metrics from exp output
    pattern = r"(Validation|Test) results --- Loss: ([0-9\.]+), Accuracy: ([0-9\.]+), Precision: ([0-9\.]+), Recall: ([0-9\.]+), F1: ([0-9\.]+), AUROC: ([0-9\.]+), AUPRC: ([0-9\.]+)"
    results = {"val": {}, "test": {}}
    for kind, loss, acc, prec, rec, f1, auroc, auprc in re.findall(pattern, txt):
        target = "val" if kind == "Validation" else "test"
        results[target] = {
            "loss": float(loss),
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed run for MERIT (ETMC evidential mode)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="APAVA")
    parser.add_argument("--model", type=str, default="MERIT")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_csv", type=str, default="multi_seed_results.csv")
    
    # Fixed hyperparameters (best config)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_fuse", type=float, default=1.0)
    parser.add_argument("--lambda_view", type=float, default=1.0)
    parser.add_argument("--lambda_pseudo_loss", type=float, default=0.30)
    parser.add_argument("--annealing_epoch", type=int, default=50)
    parser.add_argument("--evidence_dropout", type=float, default=0.0)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--resolution_list", type=str, default="2,4,6,8")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--disable_ds", action="store_true", default=False,
                        help="skip adding --use_ds flag when invoking MERIT.run")
    
    # Additional optimization parameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--nodedim", type=int, default=10)
    parser.add_argument("--swa", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--use_gnn", action="store_true", default=False, help="enable multi-resolution GNN")
    parser.add_argument("--use_evi_loss", action="store_true", default=False)
    parser.add_argument("--lambda_evi", type=float, default=1.0)
    parser.add_argument("--lambda_pseudo", type=float, default=1.0)
    parser.add_argument("--agg", type=str, default="evi")
    
    # Multi-seed configuration
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45", help="Comma-separated list of seeds")
    
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    
    os.makedirs(os.path.dirname(args.log_csv) or '.', exist_ok=True)
    
    all_results = []
    
    with open(args.log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'return_code', 'duration_sec',
            'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auroc', 'val_auprc',
            'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc', 'test_auprc',
        ])
        
        for seed in seeds:
            cmd = [
                sys.executable, '-m', 'MERIT.run',
                '--model', args.model,
                '--data', args.data,
                '--root_path', args.root_path,
                '--learning_rate', str(args.lr),
                '--lambda_fuse', str(args.lambda_fuse),
                '--lambda_view', str(args.lambda_view),
                '--lambda_pseudo_loss', str(args.lambda_pseudo_loss),
                '--annealing_epoch', str(args.annealing_epoch),
                '--evidence_dropout', str(args.evidence_dropout),
                '--resolution_list', args.resolution_list,
                '--batch_size', str(args.batch_size),
                '--train_epochs', str(args.train_epochs),
                '--patience', str(args.patience),
                '--e_layers', str(args.e_layers),
                '--dropout', str(args.dropout),
                '--weight_decay', str(args.weight_decay),
                '--nodedim', str(args.nodedim),
                '--lr_scheduler', args.lr_scheduler,
                '--warmup_epochs', str(args.warmup_epochs),
                '--agg', args.agg,
                '--lambda_evi', str(args.lambda_evi),
                '--lambda_pseudo', str(args.lambda_pseudo),
                '--gpu', str(args.gpu),
                '--seed', str(seed),
            ]

            if not args.disable_ds:
                cmd.append('--use_ds')

            if args.eval_only:
                cmd.extend(['--task_name', 'classification', '--is_training', '0', '--save_uncertainty'])

            if args.swa:
                cmd.append('--swa')
            
            if args.use_gnn:
                cmd.append('--use_gnn')
            
            if args.use_evi_loss:
                cmd.append('--use_evi_loss')

            print(f'\n{"="*60}')
            print(f'Running seed {seed}...')
            print(f'Command: {" ".join(cmd)}')
            print(f'{"="*60}')
            
            code, out, dur = run_once(cmd, env=os.environ.copy())
            
            # Print subprocess output for debugging
            if code != 0:
                print(f'\n[ERROR] Seed {seed} failed with return_code={code}')
                print('Subprocess output:')
                print(out)
            
            metrics = parse_metrics(out)
            val = metrics.get('val', {})
            test = metrics.get('test', {})
            
            row = [
                seed, code, f"{dur:.2f}",
                val.get('loss', ''), val.get('acc', ''), val.get('prec', ''), val.get('rec', ''), 
                val.get('f1', ''), val.get('auroc', ''), val.get('auprc', ''),
                test.get('loss', ''), test.get('acc', ''), test.get('prec', ''), test.get('rec', ''), 
                test.get('f1', ''), test.get('auroc', ''), test.get('auprc', ''),
            ]
            writer.writerow(row)
            f.flush()
            
            print(f'\nSeed {seed} completed: return_code={code}, duration={dur:.2f}s')
            print(f'Val - Acc: {val.get("acc", "N/A")}, Prec: {val.get("prec", "N/A")}, Rec: {val.get("rec", "N/A")}, F1: {val.get("f1", "N/A")}, AUROC: {val.get("auroc", "N/A")}')
            print(f'Test - Acc: {test.get("acc", "N/A")}, Prec: {test.get("prec", "N/A")}, Rec: {test.get("rec", "N/A")}, F1: {test.get("f1", "N/A")}, AUROC: {test.get("auroc", "N/A")}')
            
            if code == 0:
                all_results.append({
                    'seed': seed,
                    'val_acc': val.get('acc', 0.0),
                    'val_prec': val.get('prec', 0.0),
                    'val_rec': val.get('rec', 0.0),
                    'val_f1': val.get('f1', 0.0),
                    'val_auroc': val.get('auroc', 0.0),
                    'test_acc': test.get('acc', 0.0),
                    'test_prec': test.get('prec', 0.0),
                    'test_rec': test.get('rec', 0.0),
                    'test_f1': test.get('f1', 0.0),
                    'test_auroc': test.get('auroc', 0.0),
                })
    
    # Summary statistics
    if all_results:
        print(f'\n{"="*60}')
        print('SUMMARY STATISTICS (Mean ± Std)')
        print(f'{"="*60}')
        
        metrics_to_report = ['val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auroc', 
                             'test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc']
        
        for metric in metrics_to_report:
            values = [r[metric] for r in all_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            print(f'{metric}: {mean:.5f} ± {std:.5f}')
        
        # Save summary
        summary_path = args.log_csv.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('SUMMARY STATISTICS (Mean ± Std)\n')
            f.write('='*60 + '\n')
            for metric in metrics_to_report:
                values = [r[metric] for r in all_results]
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                f.write(f'{metric}: {mean:.5f} ± {std:.5f}\n')
        print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()

