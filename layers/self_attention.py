import math
import torch
import torch.nn as nn
import numpy as np


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class FormerLayer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()
        self.intra_attentions = nn.ModuleList([
            AttentionLayer(FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)
            attn_out.append(_attn)
        return x_out, attn_out


class DifferenceFormerLayer(nn.Module):
    def __init__(self, enc_in, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super(DifferenceFormerLayer, self).__init__()
        self.intra_attentions = nn.ModuleList([
            AttentionLayer(FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)
            attn_out.append(_attn)
        return x_out, attn_out


