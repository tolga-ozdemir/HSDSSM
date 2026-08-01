import torch
from torch import nn
from mamba_ssm import Mamba
from einops import rearrange


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        channel_token=False,
        SSAS=None,
        bias=False,
    ):
        super().__init__()
        # print(f"SS2D: d_model: {d_model}")
        self.SSAS = "b t c h w"
        # [
        #                 'b c t h w',
        #                 'b c w t h',
        #                 'b c t w h',
        #                 'b c h w t',
        #                 'b c h t w',
        #                 'b c w h t',
        #             ]
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model * 2)
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba= True,
            bias=bias,
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x, r=False):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> " + self.SSAS, b=B, c=C, t=T, h=H, w=W)
        print(x.shape)
        B, d_model = x.shape[:2]
        assert d_model == self.d_model
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
        x_norm = self.norm(x_flat)
        torch.cuda.empty_cache()
        x_mamba = self.mamba(x_norm)
        # x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
        # x_mamba = x_mamba + x_mamba_b
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims).contiguous()

        out = rearrange(out, self.SSAS + " -> b c t h w", b=B, c=C, t=T, h=H, w=W)
        return out

    def forward_channel_token(self, x, r=False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.d_model, (
            f"d_model: {d_model}, self.d_model: {self.d_model}"
        )
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, (
            f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        )
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    # @autocast(enabled=False)
    def forward(self, x, r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out
