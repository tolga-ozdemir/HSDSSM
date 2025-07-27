import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .sepconv import S3Conv
Conv3d = S3Conv.of(nn.Conv3d)
from .ss2d import SS2D


class GSSA(nn.Module):

    def __init__(self, channel, num_bands, flex=False, mod=True):
        super().__init__()
        self.channel = channel
        self.num_bands = num_bands
        self.flex = flex
        self.mod = mod

        # learnable query
        self.attn_proj = nn.Linear(channel, channel)
        self.modulator = nn.Embedding(num_bands, channel)
        self.value_proj = nn.Linear(channel, channel, bias=False)
        self.fc = nn.Linear(channel, channel, bias=False)
        self.scale = channel ** -0.5

    def forward(self, x):
        B, C, D, H, W = x.shape

        residual = x

        tmp = x.reshape(B, C, D, H * W).mean(-1).permute(0, 2, 1)

        if self.mod:
            mod = self.attn_proj(self.modulator.weight.repeat(B, 1, 1))
            attn = (tmp + mod) @ tmp.transpose(1, 2)
        else:
            attn = tmp @ tmp.transpose(1, 2)
        attn = attn * self.scale
        attn = attn.reshape(B, self.num_bands, self.num_bands)
        attn = F.softmax(attn, dim=-1)  # B, band, band
        attn = attn.unsqueeze(1).unsqueeze(1)

        v = self.value_proj(rearrange(x, 'b c d h w -> b h w d c'))

        q = torch.matmul(attn, v)

        q = self.fc(q)
        q = rearrange(q, 'b h w d c -> b c d h w')

        q += residual

        return q, attn




class SMFFN(nn.Module):
    
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, input):
        x = self.w_1(input)
        x, w = torch.chunk(x, 2, dim=-1)
        x2 = x * w
  
        return x2


""" Transformer Block
"""

class TransformerBlockSSM(nn.Module):
    def __init__(self, channels, num_bands=31, bias=False, flex=False, mod=True):
        super().__init__()
        self.channels = channels
        self.attn = GSSA(channels, num_bands, flex=flex, mod=mod)
        self.ssm = SS2D(d_model=31, d_channel=channels, bias=True)
        self.ffn = SMFFN(channels, channels * 2, bias=bias)
        self.dwconv = nn.Conv3d(channels, channels, 3, 1, 1, bias=True, groups=channels)

    def forward(self, inputs):
        r1, _ = self.attn(inputs)
        r2 = self.ssm(r1)
        r3 = self.dwconv(r2)
        r = rearrange(r3, 'b c d h w -> b d h w c')
        r = self.ffn(r)
        r = rearrange(r, 'b d h w c -> b c d h w')
        return r + r3



