from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
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
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
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


