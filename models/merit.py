import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.embed import MultiResolutionData, FrequencyEmbedding
from ..layers.encdec import Encoder, EncoderLayer
from ..layers.self_attention import FormerLayer, DifferenceFormerLayer
from ..layers.difference import DifferenceDataEmb, DataRestoration
from ..layers.multi_resolution_gnn import MRGNN
from ..layers.evi_mr import EviMR


# ===== ETMC-style evidential loss (KL with annealing) =====
def _KL(alpha, c):
    device = alpha.device
    beta = torch.ones((1, c), device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss_edl(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1.0
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1.0, float(global_step) / float(max(1, annealing_step)))
    alp = E * (1.0 - label) + 1.0
    B = annealing_coef * _KL(alp, c)
    return torch.mean(A + B)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.activation = configs.activation
        self.resolution_list = list(map(int, configs.resolution_list.split(',')))

        self.res_num = len(self.resolution_list)
        self.stride_list = self.resolution_list
        self.res_len = [int(self.seq_len // res) + 1 for res in self.resolution_list]
        self.augmentations = configs.augmentations.split(',')

        # step1: multi_resolution_data
        self.multi_res_data = MultiResolutionData(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: frequency convolution network
        self.enable_freq = not getattr(configs, 'no_freq', False)
        if self.enable_freq:
            self.freq_embedding = FrequencyEmbedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: difference attention network
        self.enable_diff = not getattr(configs, 'no_diff', False)
        if self.enable_diff:
            self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
            self.difference_attention = Encoder(
                [
                    EncoderLayer(
                        DifferenceFormerLayer(
                            self.enc_in,
                            self.res_num,
                            self.d_model,
                            self.n_heads,
                            self.dropout,
                            self.output_attention,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        # step 3: transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # step 3.5: multi-resolution GNN (optional, controlled by use_gnn flag)
        self.use_gnn = getattr(configs, 'use_gnn', True)
        if self.use_gnn:
            self.mrgnn = MRGNN(configs, self.res_len)

        # step 4: evidential multi-resolution aggregation
        self.evimr = EviMR(
            self.enc_in,
            self.d_model,
            self.dropout,
            configs.nodedim,
            self.res_len,
            configs.num_class,
            configs.resolution_list,
            use_pseudo=(not getattr(configs, 'no_pseudo', False)),
            agg=getattr(configs, 'agg', 'evi'),
            lambda_pseudo=getattr(configs, 'lambda_pseudo', 1.0),
            evidence_act=getattr(configs, 'evidence_act', 'softplus'),
            evidence_dropout=getattr(configs, 'evidence_dropout', 0.0),
            use_ds=getattr(configs, 'use_ds', True),
        )

        # step 5: projection
        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x_enc.shape
        multi_res_data = self.multi_res_data(x_enc)
        if self.enable_freq:
            enc_out_1 = self.freq_embedding(multi_res_data)
        else:
            enc_out_1 = [multi_res_data[l].new_zeros((multi_res_data[l].shape[0], self.enc_in, self.d_model)) for l in range(self.res_num)]
        if self.enable_diff:
            x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
            x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
            enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
            enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]
        else:
            enc_out_2 = [torch.zeros((multi_res_data[l].shape[0], self.enc_in, self.d_model), device=multi_res_data[l].device, dtype=multi_res_data[l].dtype) for l in range(self.res_num)]
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        enc_out, attns = self.encoder(data_enc, attn_mask=None)
        
        # Optional GNN refinement
        if self.use_gnn:
            enc_out, adjacency_matrices = self.mrgnn(enc_out)
        
        output, alphas = self.evimr(enc_out)
        if getattr(self.evimr, 'use_ds', False):
            # Return fused alpha and per-resolution alphas for evidential losses
            return output, alphas
        else:
            output = output.reshape(B, -1)
            output = self.projection(output)
            return output, alphas

    def classify(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Forward pass to obtain alpha from DS fusion
        fused_alpha, alphas = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not getattr(self.evimr, 'use_ds', False):
            raise ValueError('classify() requires use_ds=True to output alpha')
        # Predict by argmax over alpha
        pred = torch.argmax(fused_alpha, dim=1)
        return pred, fused_alpha, alphas

    def evidential_loss(self, target, fused_alpha, alphas, global_step, annealing_epoch, num_classes,
                        lambda_fuse=1.0, lambda_view=1.0, lambda_pseudo_loss=1.0):
        """ETMC-style evidential loss with adjustable weights for fused, per-view and pseudo-view alphas.
        - fused_alpha: (B, K)
        - alphas: list of per-view alphas; if pseudo-view启用，最后一个为伪视图 alpha
        """
        loss = lambda_fuse * ce_loss_edl(target, fused_alpha, num_classes, global_step, annealing_epoch)
        if len(alphas) > 0:
            # 如果存在伪视图，视作 alphas 的最后一个
            if len(alphas) >= 2 and hasattr(self, 'evimr') and getattr(self.evimr, 'use_pseudo', False):
                *view_alphas, pseudo_alpha = alphas
            else:
                view_alphas, pseudo_alpha = alphas, None
            for alpha in view_alphas:
                loss = loss + lambda_view * ce_loss_edl(target, alpha, num_classes, global_step, annealing_epoch)
            if pseudo_alpha is not None:
                loss = loss + lambda_pseudo_loss * ce_loss_edl(target, pseudo_alpha, num_classes, global_step, annealing_epoch)
        return loss


