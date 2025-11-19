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
        weight_decay = getattr(self.args, 'weight_decay', 1e-4)
        if weight_decay > 0:
            model_optim = optim.AdamW(self.model.parameters(), 
                                      lr=self.args.learning_rate,
                                      weight_decay=weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_scheduler(self, optimizer):
        scheduler_type = getattr(self.args, 'lr_scheduler', 'none')
        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, 
                                         T_max=self.args.train_epochs - warmup_epochs, 
                                         eta_min=self.args.learning_rate * 0.01)
            return scheduler, warmup_epochs
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
            return scheduler, warmup_epochs
        else:
            return None, 0

    def _select_criterion(self):
        # DS 模式下，模型前向返回 fused alpha，不使用 NLLLoss；
        # 非 DS 模式返回未归一化 logits，使用 CrossEntropyLoss。
        if getattr(self.args, 'use_ds', False):
            return None
        else:
            return nn.CrossEntropyLoss()

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
                if getattr(self.args, 'use_ds', False):
                    outputs, alphas = (self.swa_model if self.swa else self.model)(batch_x, padding_mask, None, None)
                    fused_alpha = outputs
                    # 验证集损失：使用最大退火系数
                    loss = self.model.evidential_loss(
                        label.long(), fused_alpha, alphas, global_step=10**9,
                        annealing_epoch=getattr(self.args, 'annealing_epoch', 10), num_classes=self.args.num_class,
                        lambda_fuse=getattr(self.args, 'lambda_fuse', 1.0),
                        lambda_view=getattr(self.args, 'lambda_view', 1.0),
                        lambda_pseudo_loss=getattr(self.args, 'lambda_pseudo_loss', 1.0),
                    )
                    total_loss.append(loss.item())
                    preds.append(fused_alpha.detach())
                    trues.append(label)
                else:
                    outputs, _ = (self.swa_model if self.swa else self.model)(batch_x, padding_mask, None, None)
                    pred = outputs.detach().cpu()
                    loss = criterion(pred, label.long().cpu())
                    total_loss.append(loss)
                    preds.append(outputs.detach())
                    trues.append(label)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        if getattr(self.args, 'use_ds', False):
            # alpha -> p = alpha / sum(alpha)
            probs = preds / torch.sum(preds, dim=1, keepdim=True)
        else:
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
        scheduler, warmup_epochs = self._select_scheduler(model_optim)
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
                if getattr(self.args, 'use_ds', False):
                    # ETMC 风格 evidential 损失：监督 fused 与各分辨率 alpha
                    loss = self.model.evidential_loss(
                        label.long(), outputs, alphas,
                        global_step=epoch, annealing_epoch=getattr(self.args, 'annealing_epoch', 10),
                        num_classes=self.args.num_class,
                        lambda_fuse=getattr(self.args, 'lambda_fuse', 1.0),
                        lambda_view=getattr(self.args, 'lambda_view', 1.0),
                        lambda_pseudo_loss=getattr(self.args, 'lambda_pseudo_loss', 1.0),
                    )
                else:
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

            # Learning rate scheduling
            if scheduler is not None and epoch >= warmup_epochs:
                scheduler.step()
                current_lr = model_optim.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")

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

        # Latency Measurement
        if getattr(self.args, 'measure_latency', False):
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(x, **kwargs): return x
            
            print("="*50)
            print(f"Measuring Inference Latency...")
            mc_dropout = getattr(self.args, 'mc_dropout', 0)
            if mc_dropout > 0:
                print(f"Mode: MC-Dropout (iterations={mc_dropout})")
                self.model.train() # Enable dropout
            else:
                print(f"Mode: Standard Inference (EviMR-Net/Single Pass)")
                self.model.eval()

            # Warmup
            print("Warming up GPU...")
            with torch.no_grad():
                for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                    if i >= 10: break
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    if mc_dropout > 0:
                        for _ in range(mc_dropout):
                            _ = self.model(batch_x, padding_mask, None, None)
                    else:
                        _ = self.model(batch_x, padding_mask, None, None)
            
            torch.cuda.synchronize()
            
            # Measurement
            print("Measuring...")
            latencies = []
            with torch.no_grad():
                for batch_x, label, padding_mask in tqdm(test_loader, desc='Measuring Latency'):
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    
                    start_time = time.time()
                    
                    if mc_dropout > 0:
                        for _ in range(mc_dropout):
                            _ = self.model(batch_x, padding_mask, None, None)
                    else:
                        _ = self.model(batch_x, padding_mask, None, None)
                        
                    torch.cuda.synchronize()
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000) # ms

            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            print("-" * 30)
            print(f"Batch Size: {self.args.batch_size}")
            print(f"Average Latency: {avg_latency:.2f} ms/batch")
            print(f"Std Dev: {std_latency:.2f} ms")
            print("="*50)
            return # Stop here if measuring latency

        criterion = self._select_criterion()
        vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics['Accuracy']:.5f}, Precision: {val_metrics['Precision']:.5f}, Recall: {val_metrics['Recall']:.5f}, F1: {val_metrics['F1']:.5f}, AUROC: {val_metrics['AUROC']:.5f}, AUPRC: {val_metrics['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics['Accuracy']:.5f}, Precision: {test_metrics['Precision']:.5f}, Recall: {test_metrics['Recall']:.5f}, F1: {test_metrics['F1']:.5f}, AUROC: {test_metrics['AUROC']:.5f}, AUPRC: {test_metrics['AUPRC']:.5f}"
        )
        # optional: save per-sample uncertainty for downstream plotting
        try:
            if getattr(self.args, 'save_uncertainty', False):
                import numpy as np
                try:
                    from tqdm import tqdm  # optional
                except Exception:
                    def tqdm(x, **kwargs):
                        return x
                self.model.eval()
                all_alpha = []
                all_pred = []
                all_label = []
                all_conf  = []
                
                # MC-Dropout Logic
                mc_dropout = getattr(self.args, 'mc_dropout', 0)
                if mc_dropout > 0:
                    print(f"Running MC-Dropout with {mc_dropout} iterations...")
                    # Enable dropout
                    self.model.train()
                
                with torch.no_grad():
                    for batch_x, label, padding_mask in tqdm(test_loader, desc='Saving uncertainty'):
                        batch_x = batch_x.float().to(self.device)
                        padding_mask = padding_mask.float().to(self.device)
                        label = label.to(self.device)
                        
                        if mc_dropout > 0:
                            # MC-Dropout: Run T forward passes
                            mc_probs = []
                            for _ in range(mc_dropout):
                                if getattr(self.args, 'use_ds', False):
                                    fused_alpha, _ = self.model(batch_x, padding_mask, None, None)
                                    S = torch.sum(fused_alpha, dim=1, keepdim=True)
                                    prob = fused_alpha / S
                                else:
                                    logits, _ = self.model(batch_x, padding_mask, None, None)
                                    prob = torch.softmax(logits, dim=1)
                                mc_probs.append(prob.unsqueeze(0)) # (1, B, C)
                            
                            # Stack: (T, B, C)
                            mc_probs = torch.cat(mc_probs, dim=0)
                            # Mean probability: (B, C)
                            mean_prob = mc_probs.mean(dim=0)
                            
                            # Uncertainty: Predictive Entropy = -sum(p * log(p))
                            # Add epsilon to avoid log(0)
                            entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=1)
                            
                            # For compatibility with existing code structure
                            # We treat 'alpha' as mean_prob * K (pseudo-alpha) or just store entropy directly
                            # Here we store entropy as 'uncertainties' later
                            
                            pred = torch.argmax(mean_prob, dim=1)
                            conf = torch.max(mean_prob, dim=1).values
                            
                            # Store entropy in a temporary way or handle it below
                            # Let's store mean_prob as 'alpha' (normalized) for now, 
                            # and we will override the uncertainty calculation below.
                            alpha = mean_prob # (B, C)
                            
                            # Hack: Store entropy in a separate list or use a flag
                            # To minimize code changes, we'll calculate uncertainty here and store it
                            batch_uncertainty = entropy
                            
                        else:
                            if getattr(self.args, 'use_ds', False):
                                fused_alpha, _ = self.model(batch_x, padding_mask, None, None)
                                alpha = fused_alpha
                                S = torch.sum(alpha, dim=1, keepdim=True)
                                prob = alpha / S
                                pred = torch.argmax(prob, dim=1)
                                conf = torch.max(alpha, dim=1).values / S.squeeze(1)  # predictive mean of predicted class
                                batch_uncertainty = None # Calculated later
                            else:
                                logits, _ = self.model(batch_x, padding_mask, None, None)
                                prob = torch.softmax(logits, dim=1)
                                # as pseudo-alpha for uncertainty only
                                K = prob.shape[1]
                                alpha = prob * K  # keep for u computation
                                pred = torch.argmax(prob, dim=1)
                                conf = torch.max(prob, dim=1).values  # max softmax prob as confidence
                                batch_uncertainty = None # Calculated later

                        all_alpha.append(alpha.cpu())
                        all_pred.append(pred.cpu())
                        all_label.append(label.cpu())
                        all_conf.append(conf.detach().cpu())
                        
                        # If MC-Dropout, we already have uncertainty (entropy)
                        if batch_uncertainty is not None:
                            if 'all_uncertainty' not in locals():
                                all_uncertainty = []
                            all_uncertainty.append(batch_uncertainty.cpu())

                all_alpha = torch.cat(all_alpha, dim=0).numpy()
                all_pred = torch.cat(all_pred, dim=0).numpy()
                all_label = torch.cat(all_label, dim=0).numpy()
                all_conf = torch.cat(all_conf, dim=0).numpy()
                
                if mc_dropout > 0:
                    uncertainties = torch.cat(all_uncertainty, dim=0).numpy()
                    print("Using MC-Dropout Predictive Entropy as uncertainty.")
                else:
                    # uncertainty u = K / sum(alpha)
                    S = all_alpha.sum(axis=1, keepdims=True)
                    K = all_alpha.shape[1]
                    uncertainties = (K / S).squeeze(1)
                
                # confidences: DS -> predictive max mean; Softmax -> max prob
                confidences = all_conf
                out_dir = self.args.uncertainty_dir if getattr(self.args, 'uncertainty_dir', '') else os.path.join('./checkpoints', self.args.task_name, self.args.model_id, self.args.model, setting, 'uncertainty')
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, 'uncertainties.npy'), uncertainties)
                np.save(os.path.join(out_dir, 'confidences.npy'), confidences)
                np.save(os.path.join(out_dir, 'predictions.npy'), all_pred)
                np.save(os.path.join(out_dir, 'labels.npy'), all_label)
                print(f"Saved uncertainty arrays to: {out_dir}")
        except Exception as e:
            print(f"[warn] saving uncertainty failed: {e}")
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
            kl = (logB_one - logB_alpha) + ((alpha - 1.0) * digamma_sum).sum(dim=1, keepdim=True)
            kl_total = kl_total + kl
        kl_total = kl_total.mean()
        return kl_total


