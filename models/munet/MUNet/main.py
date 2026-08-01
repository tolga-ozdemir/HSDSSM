# coding=utf-8
# Version:python 3.7

from train import *
from dataset import makedataloader
from setup import *
import scipy.io as io
import h5py
import yaml


def main():
    arg = option()
    device = torch.device(f'cuda:{arg.cuda}' if torch.cuda.is_available() else 'cpu')


    with open('./training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    model = getmodel(opt)




    model = model.to(device)
    # Loading pre-trained models
    model, optimizer, scheduler = load_model(model, arg.pretrain_modle, mode='train')
    
    # icvl
    trainloader, testloader = makedataloader(arg.mat, arg.dataset, batch_size=arg.batch_size, blind=arg.blind,
                                             noise=arg.noise, patch=(31, 128, 128), strides=(31, 128, 128),
                                             mode='notsingle', channel_first=True, loader=h5py.File, key='rad')

    save_name_best = '{}_{}_{}_best'.format(arg.model, arg.dataset_name, 'guass_{}'.format(arg.noise) if arg.blind == 'no' else arg.blind)
    save_name_now = '{}_{}_{}_now'.format(arg.model, arg.dataset_name, 'guass_{}'.format(arg.noise) if arg.blind == 'no' else arg.blind)

    train(arg.epochs, model, optimizer, scheduler, trainloader, testloader, device, save_name_best, save_name_now)


if __name__ == '__main__':
    main()
