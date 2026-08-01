# the original version, with FCU before and after
from functools import partial
from .MUNet_detail import MUNet


def munet():
    net = MUNet(
        img_size=64,
        patch_size=1,
        in_chans=31,
        out_chans=31,
        embed_dim=48,
        depths=[4, 4, 4, 4],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    net.use_2dconv = False
    net.bandwise = False
    return net
