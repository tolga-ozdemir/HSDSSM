from .arch import HSSM


def hssm():
    net = HSSM(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hssm_24():
    net = HSSM(1, 24, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


