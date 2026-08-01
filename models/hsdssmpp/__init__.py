from .arch import TSSM

def hssm_8():
    net = TSSM(1, 8, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def hssm():
    net = TSSM(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hssm_24():
    net = TSSM(1, 24, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hssm_32():
    net = TSSM(1, 32, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net
