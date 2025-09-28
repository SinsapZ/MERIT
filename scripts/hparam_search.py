import argparse
import itertools
import os
import subprocess
import sys
import time
import csv
import re


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
    # Expected lines:
    #   Validation results --- Loss: <f>, Accuracy: <f>, Precision: <f>, Recall: <f>, F1: <f>, AUROC: <f>, AUPRC: <f>
    #   Test results --- Loss: <f>, Accuracy: <f>, Precision: <f>, Recall: <f>, F1: <f>, AUROC: <f>, AUPRC: <f>
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
    parser = argparse.ArgumentParser(description="Grid search for MERIT (ETMC evidential mode)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="APAVA")
    parser.add_argument("--model", type=str, default="MERIT")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_csv", type=str, default="hparam_results.csv")
    parser.add_argument("--e_layers", type=int, default=4)

    # search spaces
    parser.add_argument("--lr_list", type=str, default="2e-5,3e-5,4e-5,5e-5,6e-5")
    parser.add_argument("--lambda_evi_list", type=str, default="0.18,0.20,0.22,0.28,0.30")
    parser.add_argument("--lambda_pseudo_list", type=str, default="0.3,0.4,0.5")
    parser.add_argument("--lambda_fuse_list", type=str, default="1.0")
    parser.add_argument("--lambda_view_list", type=str, default="1.0")
    parser.add_argument("--lambda_pseudo_loss_list", type=str, default="0.5,0.7,1.0")
    parser.add_argument("--annealing_epoch_list", type=str, default="10")

    # fixed settings
    parser.add_argument("--resolution_list", type=str, default="2,4,6,8")
    parser.add_argument("--evidence_dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    # scoring weights
    parser.add_argument("--w_f1", type=float, default=0.5, help="weight of validation F1 in selection score")
    parser.add_argument("--w_acc", type=float, default=0.3, help="weight of validation ACC in selection score")
    parser.add_argument("--w_auroc", type=float, default=0.2, help="weight of validation AUROC in selection score")

    args = parser.parse_args()

    def parse_list(s):
        return [x.strip() for x in s.split(',') if x.strip()]

    lr_list = parse_list(args.lr_list)
    lbd_evi_list = parse_list(args.lambda_evi_list)
    lbd_pseudo_list = parse_list(args.lambda_pseudo_list)
    lbd_fuse_list = parse_list(args.lambda_fuse_list)
    lbd_view_list = parse_list(args.lambda_view_list)
    lbd_pseudo_loss_list = parse_list(args.lambda_pseudo_loss_list)
    anneal_list = parse_list(args.annealing_epoch_list)

    combos = list(itertools.product(lr_list, lbd_evi_list, lbd_pseudo_list, lbd_fuse_list, lbd_view_list, lbd_pseudo_loss_list, anneal_list))

    os.makedirs(os.path.dirname(args.log_csv) or '.', exist_ok=True)
    with open(args.log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lr', 'lambda_evi', 'lambda_pseudo', 'lambda_fuse', 'lambda_view', 'lambda_pseudo_loss', 'annealing_epoch',
            'return_code', 'duration_sec',
            'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auroc', 'val_auprc', 'val_score',
            'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc', 'test_auprc',
        ])
        best = {"score": -1.0, "row": None}
        all_rows = []
        for lr, lbd_evi, lbd_pseudo, lbd_fuse, lbd_view, lbd_pseudo_loss, anneal in combos:
            cmd = [
                sys.executable, '-m', 'MERIT.run',
                '--model', args.model,
                '--data', args.data,
                '--root_path', args.root_path,
                '--e_layers', str(args.e_layers),
                '--use_ds', '--use_evi_loss',
                '--learning_rate', str(lr),
                '--lambda_evi', str(lbd_evi),
                '--lambda_pseudo', str(lbd_pseudo),
                '--lambda_fuse', str(lbd_fuse),
                '--lambda_view', str(lbd_view),
                '--lambda_pseudo_loss', str(lbd_pseudo_loss),
                '--annealing_epoch', str(anneal),
                '--evidence_dropout', str(args.evidence_dropout),
                '--resolution_list', args.resolution_list,
                '--batch_size', str(args.batch_size),
                '--train_epochs', str(args.train_epochs),
                '--patience', str(args.patience),
                '--gpu', str(args.gpu),
            ]

            code, out, dur = run_once(cmd, env=os.environ.copy())
            metrics = parse_metrics(out)
            val = metrics.get('val', {})
            test = metrics.get('test', {})
            # compute validation composite score
            try:
                val_score = (
                    args.w_f1 * float(val.get('f1', 0.0)) +
                    args.w_acc * float(val.get('acc', 0.0)) +
                    args.w_auroc * float(val.get('auroc', 0.0))
                )
            except Exception:
                val_score = -1.0

            row = [
                lr, lbd_evi, lbd_pseudo, lbd_fuse, lbd_view, lbd_pseudo_loss, anneal,
                code, f"{dur:.2f}",
                val.get('loss', ''), val.get('acc', ''), val.get('prec', ''), val.get('rec', ''), val.get('f1', ''), val.get('auroc', ''), val.get('auprc', ''), f"{val_score:.6f}",
                test.get('loss', ''), test.get('acc', ''), test.get('prec', ''), test.get('rec', ''), test.get('f1', ''), test.get('auroc', ''), test.get('auprc', ''),
            ]
            writer.writerow(row)
            f.flush()
            print('\n========== CMD =========')
            print(' '.join(cmd))
            print('========== RET =========')
            print(f'return={code}, duration={dur:.2f}s')
            print('========== LOG =========')
            print(out)
            # Track best by validation F1
            if code == 0 and val_score > best['score']:
                best = {"score": val_score, "row": row}
            # keep all rows for later re-ranking/reporting
            all_rows.append((val, test, val_score, row))

    if best['row'] is not None:
        print('\n========== BEST (by Validation F1) =========')
        headers = [
            'lr', 'lambda_evi', 'lambda_pseudo', 'lambda_fuse', 'lambda_view', 'lambda_pseudo_loss', 'annealing_epoch',
            'return_code', 'duration_sec',
            'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auroc', 'val_auprc', 'val_score',
            'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc', 'test_auprc',
        ]
        for h, v in zip(headers, best['row']):
            print(f"{h}: {v}")

    # Also report the configuration that is best on the TEST composite (for analysis only)
    if all_rows:
        best_test = None
        best_test_score = -1.0
        for val, test, val_score, row in all_rows:
            try:
                test_score = (
                    args.w_f1 * float(test.get('f1', 0.0)) +
                    args.w_acc * float(test.get('acc', 0.0)) +
                    args.w_auroc * float(test.get('auroc', 0.0))
                )
            except Exception:
                test_score = -1.0
            if test_score > best_test_score:
                best_test_score = test_score
                best_test = row
        if best_test is not None:
            print('\n========== BEST (by Test composite, for reference) =========')
            headers = [
                'lr', 'lambda_evi', 'lambda_pseudo', 'lambda_fuse', 'lambda_view', 'lambda_pseudo_loss', 'annealing_epoch',
                'return_code', 'duration_sec',
                'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auroc', 'val_auprc', 'val_score',
                'test_loss', 'test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auroc', 'test_auprc',
            ]
            for h, v in zip(headers, best_test):
                print(f"{h}: {v}")


if __name__ == '__main__':
    main()


