from .LAMU import LAMU
import torch.nn as nn


def lamamba():
    net = LAMU(
        input_channels=32,
        n_stages=6,
        input_size=[31, 64, 64],
        #input_size=[34, 128, 128],
        strides=[[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2]],
        num_classes=32,
        features_per_stage=[32, 64, 64, 128, 128, 256],
        n_conv_per_stage=[2, 2, 2, 1, 1, 1],
        n_conv_per_stage_decoder=[2, 2, 2, 1, 1],
        kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        depths=[1, 1, 1, 2, 2, 2],
        num_heads=[1, 2, 2, 4, 4, 8],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=False,
        use_checkpoint=False,
    )

    net.use_2dconv = False
    net.bandwise = False
    return net
