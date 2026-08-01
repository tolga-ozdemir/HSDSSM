
from .HCANet import HCANet

def hcanet():
    net = HCANet()
    net.use_2dconv = False
    net.bandwise = False
    return net

