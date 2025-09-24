from copy import deepcopy
from ..data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from ..utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len
        self.args.pred_len = 0
        self.args.enc_in = test_data.X.shape[2]
        self.args.num_class = len(np.unique(test_data.y))
        model = self.model_dict["MERIT"](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                if self.swa:
                    outputs, _ = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs, _ = self.model(batch_x, padding_mask, None, None)
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)
                preds.append(outputs.detach())
                trues.append(label)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(-1,).to(torch.long),
                num_classes=self.args.num_class,
            ).float().cpu().numpy()
        )
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }
        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-5)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()
        print(f"Total Trainable Params: {total_params}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs, alphas = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long())
                if getattr(self.args, 'use_evi_loss', False):
                    kl = self._evi_kl_loss(alphas, self.args)
                    loss = loss + self.args.lambda_evi * kl
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            if hasattr(self, 'swa_model'):
                self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics['Accuracy']:.5f}, Precision: {val_metrics['Precision']:.5f}, Recall: {val_metrics['Recall']:.5f}, F1: {val_metrics['F1']:.5f}, AUROC: {val_metrics['AUROC']:.5f}, AUPRC: {val_metrics['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics['Accuracy']:.5f}, Precision: {test_metrics['Precision']:.5f}, Recall: {test_metrics['Recall']:.5f} F1: {test_metrics['F1']:.5f}, AUROC: {test_metrics['AUROC']:.5f}, AUPRC: {test_metrics['AUPRC']:.5f}"
            )

            early_stopping(-val_metrics["F1"], self.swa_model if getattr(self, 'swa', False) else self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        if getattr(self, 'swa', False):
            self.swa_model.load_state_dict(torch.load(best_model_path, map_location='cuda' if self.args.use_gpu else 'cpu'))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location='cuda' if self.args.use_gpu else 'cpu'))
        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/" + self.args.task_name + "/" + self.args.model_id + "/" + self.args.model + "/" + setting + "/"
            )
            model_path = os.path.join(path, "checkpoint.pth")
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if getattr(self, 'swa', False):
                self.swa_model.load_state_dict(torch.load(model_path, map_location='cuda' if self.args.use_gpu else 'cpu'))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location='cuda' if self.args.use_gpu else 'cpu'))

        criterion = self._select_criterion()
        vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics['Accuracy']:.5f}, Precision: {val_metrics['Precision']:.5f}, Recall: {val_metrics['Recall']:.5f}, F1: {val_metrics['F1']:.5f}, AUROC: {val_metrics['AUROC']:.5f}, AUPRC: {val_metrics['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics['Accuracy']:.5f}, Precision: {test_metrics['Precision']:.5f}, Recall: {test_metrics['Recall']:.5f}, F1: {test_metrics['F1']:.5f}, AUROC: {test_metrics['AUROC']:.5f}, AUPRC: {test_metrics['AUPRC']:.5f}"
        )
        return

    def _evi_kl_loss(self, alphas, args):
        # KL( Dir(alpha) || Dir(1) ), sum over views, mean over batch
        # Dirichlet KL to uniform prior: use closed-form
        import torch
        import torch.nn.functional as F
        kl_total = 0.0
        for alpha in alphas:
            S = torch.sum(alpha, dim=1, keepdim=True)
            K = alpha.shape[1]
            # log Beta functions
            logB_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(S)
            logB_one = - torch.lgamma(torch.tensor([K], device=alpha.device, dtype=alpha.dtype))
            # sum (alpha_i - 1)*(psi(alpha_i) - psi(S))
            digamma_sum = torch.digamma(alpha) - torch.digamma(S)
            kl = (logB_alpha - logB_one) + ((alpha - 1.0) * digamma_sum).sum(dim=1, keepdim=True)
            kl_total = kl_total + kl
        kl_total = kl_total.mean()
        return kl_total


