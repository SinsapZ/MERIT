import argparse
import itertools
import os
import subprocess
import sys
import time
import csv


def run_once(base_cmd, env=None):
    start = time.time()
    try:
        proc = subprocess.run(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
        dur = time.time() - start
        return proc.returncode, proc.stdout, dur
    except Exception as e:
        return 1, str(e), 0.0


def main():
    parser = argparse.ArgumentParser(description="Grid search for MERIT (ETMC evidential mode)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="APAVA")
    parser.add_argument("--model", type=str, default="MERIT")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_csv", type=str, default="hparam_results.csv")

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
            'return_code', 'duration_sec'
        ])

        for lr, lbd_evi, lbd_pseudo, lbd_fuse, lbd_view, lbd_pseudo_loss, anneal in combos:
            cmd = [
                sys.executable, '-m', 'MERIT.run',
                '--model', args.model,
                '--data', args.data,
                '--root_path', args.root_path,
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
            writer.writerow([lr, lbd_evi, lbd_pseudo, lbd_fuse, lbd_view, lbd_pseudo_loss, anneal, code, f"{dur:.2f}"])
            f.flush()
            print('\n========== CMD =========')
            print(' '.join(cmd))
            print('========== RET =========')
            print(f'return={code}, duration={dur:.2f}s')
            print('========== LOG =========')
            print(out)


if __name__ == '__main__':
    main()


