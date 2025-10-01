import argparse
import os
import torch
from .exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERIT")
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="APAVA-Subject", help="model id")
    parser.add_argument("--model", type=str, default="MERIT", help="[MERIT]")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument("--data", type=str, default="APAVA", help="dataset type")
    parser.add_argument("--root_path", type=str, default="../dataset/APAVA", help="root path of the data file")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--patch_len_list", type=str, default="2,2,2,4,4,4,16,16,16,16,32,32,32,32,32")
    parser.add_argument("--single_channel", action="store_true", default=False)
    parser.add_argument("--augmentations", type=str, default="none,drop0.35")
    parser.add_argument('--resolution_list', type=str, default="2,4,6,8")
    parser.add_argument('--nodedim', type=int, default=10)
    parser.add_argument("--no_pseudo", action="store_true", default=False, help="disable pseudo-view in evidential fusion")
    parser.add_argument("--agg", type=str, default="evi", choices=["evi", "mean"], help="fusion: evidential weighting or simple mean")
    parser.add_argument("--lambda_pseudo", type=float, default=1.0, help="scaling factor for pseudo-view weight")
    parser.add_argument("--evidence_act", type=str, default="softplus", choices=["softplus", "relu"], help="activation for evidence head")
    parser.add_argument("--evidence_dropout", type=float, default=0.0, help="dropout before evidence head")
    parser.add_argument("--no_freq", action="store_true", default=False, help="disable frequency embedding branch")
    parser.add_argument("--no_diff", action="store_true", default=False, help="disable difference attention branch")
    parser.add_argument("--use_evi_loss", action="store_true", default=False, help="add evidential KL regularization to prior")
    parser.add_argument("--lambda_evi", type=float, default=1.0, help="weight for evidential KL to uniform prior")
    parser.add_argument("--use_ds", action="store_true", default=False, help="strict ETMC-style DS fusion over Dirichlet alphas")
    # evidential loss weights
    parser.add_argument("--lambda_fuse", type=float, default=1.0, help="weight for fused alpha loss")
    parser.add_argument("--lambda_view", type=float, default=1.0, help="weight for per-view alpha loss")
    parser.add_argument("--lambda_pseudo_loss", type=float, default=1.0, help="weight for pseudo-view alpha loss")
    parser.add_argument("--annealing_epoch", type=int, default=10, help="annealing steps for evidential KL")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--swa", action="store_true", default=False)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", default=False)
    parser.add_argument("--devices", type=str, default="0, 1, 2, 3")
    parser.add_argument("--seed", type=int, default=None, help="manual seed (overrides itr-based seed)")
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print("Args in experiment:")
    print(args)
    if args.task_name == "classification":
        Exp = Exp_Classification
    if args.is_training:
        for ii in range(args.itr):
            seed = args.seed if args.seed is not None else (41 + ii)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            args.seed = seed
            setting = "{}_{}_{}_dm{}_df{}_nh{}_el{}_res{}_node{}_seed{}_bs{}_lr{}".format(
                args.model_id, args.model, args.data, args.d_model, args.d_ff, args.n_heads, args.e_layers, args.resolution_list, args.nodedim, args.seed, args.batch_size, args.learning_rate,
            )
            exp = Exp(args)
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = args.seed if args.seed is not None else (41 + ii)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            args.seed = seed
            setting = "{}_{}_{}_dm{}_df{}_nh{}_el{}_res{}_node{}_seed{}_bs{}_lr{}".format(
                args.model_id, args.model, args.data, args.d_model, args.d_ff, args.n_heads, args.e_layers, args.resolution_list, args.nodedim, args.seed, args.batch_size, args.learning_rate,
            )
            exp = Exp(args)
            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()


