import torch
import torch.nn as nn
from ..layers.embed import MultiResolutionData, FrequencyEmbedding
from ..layers.encdec import Encoder, EncoderLayer
from ..layers.self_attention import FormerLayer, DifferenceFormerLayer
from ..layers.difference import DifferenceDataEmb, DataRestoration
from ..layers.evi_mr import EviMR


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
        )

        # step 5: projection
        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x_enc.shape
        multi_res_data = self.multi_res_data(x_enc)
        enc_out_1 = self.freq_embedding(multi_res_data) if self.enable_freq else [torch.zeros_like(multi_res_data[l]) for l in range(self.res_num)]
        if self.enable_diff:
            x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
            x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
            enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
            enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]
        else:
            enc_out_2 = [torch.zeros((multi_res_data[l].shape[0], self.enc_in, self.d_model), device=multi_res_data[l].device, dtype=multi_res_data[l].dtype) for l in range(self.res_num)]
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        enc_out, attns = self.encoder(data_enc, attn_mask=None)
        output, alphas = self.evimr(enc_out)
        output = output.reshape(B, -1)
        output = self.projection(output)
        return output, alphas


