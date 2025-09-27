import torch
import torch.nn as nn
import torch.nn.functional as F


class EviMR(nn.Module):
    def __init__(self, enc_in, d_model, dropout, nodedim, res_len, num_classes, resolution_list, use_pseudo=True, agg='evi', lambda_pseudo=1.0, evidence_act='softplus', evidence_dropout=0.0, use_ds=True):
        super(EviMR, self).__init__()
        self.resolution_list = list(map(int, resolution_list)) if isinstance(resolution_list, (list, tuple)) else list(map(int, resolution_list.split(',')))
        self.res_num = len(self.resolution_list)
        self.res_len = res_len
        self.num_classes = num_classes
        self.use_pseudo = use_pseudo
        self.agg = agg
        self.lambda_pseudo = lambda_pseudo
        self.evidence_act = evidence_act
        self.drop = nn.Dropout(evidence_dropout)
        self.use_ds = use_ds
        # per-resolution evidence head (no graph)
        self.evidence_heads = nn.ModuleList([nn.Linear(d_model, self.num_classes) for _ in range(self.res_num)])
        # pseudo view modules
        self.pseudo_reduce = nn.Conv1d(in_channels=d_model * self.res_num, out_channels=d_model, kernel_size=1, padding=0, stride=1, bias=True)
        self.pseudo_head = nn.Linear(d_model, self.num_classes)

    def forward(self, x):
        feat_list = []
        weights = []
        alphas = []
        for l in range(self.res_num):
            # x[l]: (B, enc_in, d_model) -> feature for fusion
            feat = x[l].permute(0, 2, 1)  # (B, d_model, enc_in)
            feat_list.append(feat)
            pooled = feat.mean(dim=2)
            logits = self.evidence_heads[l](self.drop(pooled))
            evidence = F.softplus(logits) if self.evidence_act == 'softplus' else F.relu(logits)
            alpha = evidence + 1.0
            alphas.append(alpha)
            S = torch.sum(alpha, dim=1, keepdim=True)
            u = self.num_classes / S
            w = 1.0 - u
            weights.append(w)
        if self.use_pseudo:
            # build pseudo view from concatenated features along channel axis
            pseudo_feat = torch.cat(feat_list, dim=1)  # (B, d_model*res_num, enc_in)
            pseudo_feat = self.pseudo_reduce(pseudo_feat)  # (B, d_model, enc_in)
            pooled_pseudo = pseudo_feat.mean(dim=2)
            pseudo_logits = self.pseudo_head(self.drop(pooled_pseudo))
            pseudo_evidence = F.softplus(pseudo_logits) if self.evidence_act == 'softplus' else F.relu(pseudo_logits)
            pseudo_alpha = pseudo_evidence + 1.0
            alphas.append(pseudo_alpha)
            pseudo_S = torch.sum(pseudo_alpha, dim=1, keepdim=True)
            pseudo_u = self.num_classes / pseudo_S
            pseudo_w = self.lambda_pseudo * (1.0 - pseudo_u)
            # combine weights of real views and pseudo view
            weights = torch.cat(weights + [pseudo_w], dim=1)
        else:
            weights = torch.cat(weights, dim=1)
        if self.use_ds:
            # ETMC-style evidential DS fusion over alphas (including optional pseudo alpha)
            alpha_a = self._ds_combin(alphas)
            return alpha_a, alphas
        else:
            weights = torch.clamp(weights, min=1e-6)
            if self.agg == 'evi':
                weights = weights / torch.sum(weights, dim=1, keepdim=True)
            else:
                weights = torch.full_like(weights, 1.0 / weights.shape[1])
            feats = torch.stack(feat_list + ([pseudo_feat] if self.use_pseudo else []), dim=-1)
            weights = weights.unsqueeze(1).unsqueeze(1)
            out = torch.sum(feats * weights, dim=-1)
            return out, alphas

    def _ds_combin(self, alpha_list):
        def ds_two(alpha1, alpha2):
            S0 = torch.sum(alpha1, dim=1, keepdim=True)
            S1 = torch.sum(alpha2, dim=1, keepdim=True)
            E0 = alpha1 - 1.0
            E1 = alpha2 - 1.0
            b0 = E0 / S0.expand_as(E0)
            b1 = E1 / S1.expand_as(E1)
            u0 = self.num_classes / S0
            u1 = self.num_classes / S1
            bb = torch.bmm(b0.view(-1, self.num_classes, 1), b1.view(-1, 1, self.num_classes))
            bu = b0 * u1.expand_as(b0)
            ub = b1 * u0.expand_as(b0)
            bb_sum = torch.sum(bb, dim=(1, 2))
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            K = bb_sum - bb_diag
            b_a = (b0 * b1 + bu + ub) / (1.0 - K).view(-1, 1).expand_as(b0)
            u_a = (u0 * u1) / (1.0 - K).view(-1, 1).expand_as(u0)
            S_a = self.num_classes / u_a
            e_a = b_a * S_a.expand_as(b_a)
            alpha_a = e_a + 1.0
            return alpha_a
        alpha_a = alpha_list[0]
        for v in range(1, len(alpha_list)):
            alpha_a = ds_two(alpha_a, alpha_list[v])
        return alpha_a


