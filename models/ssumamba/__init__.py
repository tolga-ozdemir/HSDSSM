from .multilayer import MultilayerModel
from .brcn import brcn
from .SSUMamba import UMambaEnc as SSUMamba
from .SSUMamba_SSCS import UMambaEnc as SSUMamba_SSCS


def ssumamba():
    # net = SSUMamba(input_channels=34,n_stages=6, input_size=[34, 128, 128],
    # strides=[[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[1,2,2]],
    # num_classes=34,features_per_stage=[32, 64, 64, 128, 128, 256],
    # n_conv_per_stage=[2,2,2,2,2,2],n_conv_per_stage_decoder=[2,2,2,2,2],

    net = SSUMamba(
        # input_channels=32,
        input_channels=32,
        n_stages=6,
        input_size=[31, 64, 64],
        strides=[[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2]],
        num_classes=32,
        # num_classes=32,
        features_per_stage=[32, 64, 64, 128, 128, 256],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        SSAS=[
            "b c t h w",
            "b c w t h",
            "b c t w h",
            "b c h w t",
            "b c h t w",
            "b c w h t",
        ],
    )
    net.use_2dconv = False
    net.bandwise = False
    return net


def ssumamba_sscs():
    net = SSUMamba_SSCS(
        input_channels=32,
        n_stages=6,
        input_size=[31, 64, 64],
        strides=[[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2]],
        num_classes=32,
        features_per_stage=[32, 64, 64, 128, 128, 256],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        SSAS=[
            "b c t h w",
            "b c w t h",
            "b c t w h",
            "b c h w t",
            "b c w h t",
            "b c h t w",
        ],
    )
    net.use_2dconv = False
    net.bandwise = False
    return net
