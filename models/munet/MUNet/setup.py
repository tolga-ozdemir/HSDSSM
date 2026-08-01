# coding=utf-8
# Version:python 3.7

import argparse
import os
import torch.optim as optim
import torch

from Net.MUNet import MUNet_model



def option():
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--path')
    parse.add_argument('-i', '--input', default=31, type=int)
    parse.add_argument('-c', '--cuda', default=0)
    parse.add_argument('-e', '--epochs', default=200, type=int)
    parse.add_argument('-l', '--lr', default=0.0002)
    #parse.add_argument('-mp', '--mat', default='/home/zqcz4080/下载/Pycharmprojects/dataset/ICVL/train/')
    parse.add_argument('-mp', '--mat', default='/home/zqcz4080/下载/Pycharmprojects/dataset/CAVE/train_only/')

    #parse.add_argument('-d', '--dataset', default='./data/dataset_p_icvl/')
    parse.add_argument('-d', '--dataset', default='./data/dataset_pcave_only/')

    #parse.add_argument('-dn', '--dataset_name', default='icvl')
    parse.add_argument('-dn', '--dataset_name', default='cave')

    parse.add_argument('-bs', '--batch_size', default=(2, 1))

    parse.add_argument('-m', '--model', default='MUNet_model', type=str)

    parse.add_argument('-b', '--blind', default='gauss', type=str)#gauss no case5
    parse.add_argument('-n', '--noise', default=50, type=int)#噪声，30.50.70
    parse.add_argument('-pm', '--pretrain_modle', default='./model/20240724_182508/MUNet_model_icvl_gauss_best.pkl', type=str)
    #parse.add_argument('-pm', '--pretrain_modle', default='', type=str)

    arg = parse.parse_args()
    return arg



def getmodel(config,model_name = "MUNet_model"):
    model = MUNet_model(config)


    return model


def load_model(model, path, mode='train'):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    # [10, 50, 100, 150] 是一个列表，指定了调整学习率的epoch数。当训练到达这些epoch数时，学习率将被调整。
    # gamma=0.5 指定了学习率调整的乘数。每次调整时，当前学习率将乘以这个乘数。在这个例子中，学习率将在指定的epoch数处减半。
    # last_epoch=-1 是一个可选参数，用于设置初始epoch数。在大多数情况下，如果这是训练的开始，你可以将其设置为-1，表示之前的epoch数为-1（即，这是第一个epoch）。但是，如果你正在从之前的训练中断点继续训练，你应该设置last_epoch为之前的最后一个epoch数。
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 50, 100, 150, 200], gamma=0.5, last_epoch=-1)

    if os.path.exists(path):
        dic = torch.load(path)
        netp, optp, opts = dic['net'], dic['optimizer'], dic['scheduler']
        # print(netp.keys())
        model.load_state_dict(netp)
        print('model load successfully!')
        if mode == 'train':
            optimizer.load_state_dict(optp)
            scheduler.load_state_dict(opts)
            return model, optimizer, scheduler
    else:
        print('model does not exist at {}'.format(path))
        return model, optimizer, scheduler
    return model
