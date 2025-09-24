import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :, : x.size(2), :]


class CrossChannelTokenEmbedding(nn.Module):
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class ListPatchEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, patch_len_list, stride_list, dropout, augmentation=["none"], single_channel=False):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
        self.single_channel = single_channel

        linear_layers = [
            CrossChannelTokenEmbedding(c_in=enc_in if not single_channel else 1, l_patch=patch_len, d_model=d_model)
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

        self.learnable_embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.single_channel:
            B, C, L = x.shape
            x = torch.reshape(x, (B * C, 1, L))

        x_list = []
        for padding, value_embedding in zip(self.paddings, self.value_embeddings):
            x_new = padding(x).unsqueeze(1)
            x_new = value_embedding(x_new)
            x_new = x_new.squeeze().transpose(1, 2)
            x_list.append(x_new)

        x = [x + cxt + self.position_embedding(x) for x, cxt in zip(x_list, self.learnable_embeddings)]
        return x


class MultiResolutionData(nn.Module):
    def __init__(self, enc_in, resolution_list, stride_list):
        super(MultiResolutionData, self).__init__()
        self.paddings = nn.ModuleList([nn.ReplicationPad1d((0, stride)) for stride in stride_list])
        self.multi_res = nn.ModuleList([
            nn.Conv1d(in_channels=enc_in, out_channels=enc_in, kernel_size=res, stride=res, padding=0, padding_mode='circular')
            for res in resolution_list
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_list = []
        for l in range(len(self.multi_res)):
            out = self.paddings[l](x)
            out = self.multi_res[l](out)
            x_list.append(out)
        return x_list


class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model, res_len, augmentation=["none"]):
        super(FrequencyEmbedding, self).__init__()
        self.d_model = d_model
        self.embeddings = nn.ModuleList([
            nn.Linear(int(res/2)+1, int(self.d_model/2)+1).to(torch.cfloat)
            for res in res_len
        ])

    def forward(self, x_list):
        x_out = []
        for l in range(len(x_list)):
            x = torch.fft.rfft(x_list[l], dim=-1)
            out = self.embeddings[l](x)
            out = torch.fft.irfft(out, dim=-1, n=self.d_model)
            x_out.append(out)
        return x_out


