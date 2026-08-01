from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import numbers

from .attention import TransformerBlockSSM
from .sepconv import SepConv_DP, SepConv_DP_CA, S3Conv
from .ss2d import SS2D, SS2DC
# from.ss2d_local import SS2D
# from.ss2d_vmambair import SS2D

BatchNorm3d = nn.BatchNorm3d
Conv3d = S3Conv.of(nn.Conv3d)
#Conv3d = nn.Conv3d
TransformerBlock = TransformerBlockSSM
IsConvImpl = False
UseBN = True


def PlainConvEncoder(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
        ('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()), 
        #('ln', LayerNorm(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True, train_mode="attention"))
        #('ssm', SS2D(d_model=31, d_channel=out_ch, bias=True))
        #('ssm', SS2DC(d_model=out_ch, bias=True))
    ]))

def DownConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv3d(in_ch, out_ch, 3, (1, 2, 2), 1, bias=False)),
        #('bn', BatchNorm3d(out_ch)if UseBN else nn.Identity()),
        ('ln', LayerNorm(out_ch)if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True, train_mode="attention"))
        #('ssm', SS2D(d_model=31, d_channel=out_ch, bias=True))
        #('ssm', SS2DC(d_model=out_ch, bias=True))
    ]))


def PlainConvDecoder(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
        ('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()), 
        #('ln', LayerNorm(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True, train_mode="ssm"))
        #('ssm', SS2D(d_model=31, d_channel=out_ch, bias=True))
        #('ssm', SS2DC(d_model=out_ch, bias=True))
    ]))


def UpConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('up', nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)),
        ('conv', nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
        #('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()),
        ('ln', LayerNorm(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True, train_mode="ssm"))
        #('ssm', SS2D(d_model=31, d_channel=out_ch, bias=True))
        #('ssm', SS2DC(d_model=out_ch, bias=True))
    ]))

def UpConvPS(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv3d(in_ch, out_ch*4, 3, 1, 1, bias=False)),
        ('re1', Rearrange('b c d h w -> b d c h w')),
        ('ps', nn.PixelShuffle(2)),
        ('re2', Rearrange('b d c h w -> b c d h w')),
        #('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()),
        ('ln', LayerNorm(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True, train_mode="ssm"))
        #('ssm', SS2D(d_model=31, d_channel=out_ch, bias=True))
        #('ssm', SS2DC(d_model=out_ch, bias=True))
    ]))


class Encoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = PlainConvEncoder(channels, channels)
            else:
                encoder_layer = DownConv(channels, 2 * channels)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x)
            xs.append(x)
        x = self.layers[-1](x)
        return x


class Decoder(nn.Module):
    count = 1
    def __init__(self, channels, num_half_layer, sample_idx, Fusion=None, ps=False):
        super(Decoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(num_half_layer)):
                fusion_layer = Fusion(ch)
                if i in sample_idx:
                    ch //= 2
                self.fusions.append(fusion_layer)

        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = PlainConvDecoder(channels, channels)
            else:
                if ps:
                    decoder_layer = UpConv(channels, channels // 2)
                else:
                    decoder_layer = UpConvPS(channels, channels // 2)
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        x = self.layers[0](x)
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x)
        return x


class TSSM(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None, ps=False):
        super(TSSM, self).__init__()
        self.head = PlainConvEncoder(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, sample_idx)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx, Fusion=Fusion, ps=ps)
        self.tail = nn.Conv3d(channels, 1, 3, 1, 1, bias=True)

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()[:, 0:1, :, :, :]
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        if IsConvImpl:
            new_state_dict = {}
            for k, v in state_dict.items():
                if ('attn.attn' in k) and 'weight' in k and 'attn_proj' not in k:
                    new_state_dict[k] = v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        return super().load_state_dict(state_dict, strict)

def pad_mod(x, mod):
    h, w = x.shape[-2:]
    h_out = (h // mod + 1) * mod
    w_out = (w // mod + 1) * mod
    out = torch.zeros(*x.shape[:-2], h_out, w_out).type_as(x)
    out[..., :h, :w] = x
    return out.to(x.device), h, w


def to_3d(x):
    return rearrange(x, 'b c d h w -> b (h w d) c')


def to_5d(x, h, w, d):
    return rearrange(x, 'b (h w d) c -> b c d h w', h=h, w=w, d=d)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        x = to_5d(self.body(to_3d(x)), h, w, d)
        return x
