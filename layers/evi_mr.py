import torch
import torch.nn as nn
import torch.nn.functional as F


class EviMR(nn.Module):
    def __init__(self, enc_in, d_model, dropout, nodedim, res_len, num_classes, resolution_list, use_pseudo=True, agg='evi', lambda_pseudo=1.0, evidence_act='softplus', evidence_dropout=0.0):
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
        weights = torch.clamp(weights, min=1e-6)
        if self.agg == 'evi':
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
        else:
            # mean aggregation
            weights = torch.full_like(weights, 1.0 / weights.shape[1])

        # aggregate features including pseudo view
        feats = torch.stack(feat_list + ([pseudo_feat] if self.use_pseudo else []), dim=-1)
        weights = weights.unsqueeze(1).unsqueeze(1)
        out = torch.sum(feats * weights, dim=-1)
        return out, alphas


