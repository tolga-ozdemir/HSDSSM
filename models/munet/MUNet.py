import torch.nn as nn
from Net.MUNet_detail import MUNet


class MUNet_model(nn.Module):
    def __init__(self, config):
        super(MUNet_model, self).__init__()
        self.config = config
        self.swin_unet = MUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=31,
                               out_chans=31,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],

                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],

                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
    
if __name__ == '__main__':
    from utils_swin.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils_swin.model_utils import network_parameters

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 128
    width = 128
    x = torch.randn((2, 31, height, width)).to('cuda')  # .cuda()
    model = MUNet_model(opt).to('cuda')  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)
