# coding=utf-8
# Version:python 3.7

import torch
from utils import cal_loss, cal_sam, channel_minmax_normalize
from utils import mssim, mpsnr
from tqdm import tqdm
import pandas as pd
import cv2
import os
import logging
from datetime import datetime

def train(epochs, model, optimizer, scheduler, trainDataloader, testDataloader, device, name_best, name_now):

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    data_df = {'epoch':[], 'denoise_loss':[], 'psnr':[], 'ss':[], 'val_loss':[], 'best_psnr':[], 'sam':[]}
    pre_loss = 1
    best_psnr = 0

    # 获取当前时间并格式化
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{current_time}.log'

    # 配置日志文件路径
    log_dir = './logs'  # 你可以根据需要更改日志文件存储的目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_filename)

    # 配置模型文件路径
    log_dir = f'./model/{current_time}'  # 你可以根据需要更改日志文件存储的目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # 如果你还想在控制台输出日志，可以保留这一行
        ]
    )

    # 示例记录训练信息的函数
    def log_training_info(epoch, deloss, count, mean_psnr, mean_ss, pre_loss, best_psnr, mean_sam):
        logging.info(
            'epoch {} : train denoise_loss:{:.5f}, psnr:{:.5f}, ss:{:.5f}, min_loss:{:.5f}, best_psnr:{:.5f}, sam:{:.5f}'.format(
                epoch + 1, deloss / count, mean_psnr, mean_ss, pre_loss, best_psnr, mean_sam)
        )

    for epoch in range(epochs):
        model = model.to(device)
        model = model.train()
        deloss = 0
        count = 0

        for i, data in enumerate(tqdm(trainDataloader)):
            optimizer.zero_grad()
            denoise_gt, noise_image = data
            denoise_gt, noise_image = denoise_gt.float().to(device), noise_image.float().to(device)
            denoise_pre = model(noise_image)

            denoise_loss, grad_loss = cal_loss(denoise_gt, denoise_pre, device)
            loss = denoise_loss + 0.001 * grad_loss
            deloss += denoise_loss.item()
            grad_loss += grad_loss.item()


            count += 1
            loss.backward()
            optimizer.step()

            # print('epoch {} denoise_loss:{:.5f}'.format(epoch + 1, denoise_loss.item()))

        scheduler.step()
        mean_psnr, mean_ss, mean_sam, val_loss = valite(model, testDataloader, device)
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
        print(
            'epoch {} : train denoise_loss:{:.5f}, psnr:{:.5f}, ss:{:.5f}, min_loss:{:.5f}, best_psnr:{:.5f}, sam:{:.5f}'.format(
                epoch + 1, deloss / count, mean_psnr, mean_ss, pre_loss, best_psnr, mean_sam))

        # 示例记录一次训练的信息
        log_training_info(epoch, deloss, count, mean_psnr, mean_ss, pre_loss, best_psnr, mean_sam)
        # Save the current training parameter changes
        data_df['epoch'].append(epoch)
        data_df['denoise_loss'].append(deloss / count)
        data_df['psnr'].append(mean_psnr)
        data_df['best_psnr'].append(best_psnr)
        data_df['ss'].append(mean_ss)
        data_df['val_loss'].append(val_loss)
        data_df['sam'].append(mean_sam)

        torch.save(
            {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
            './model/{}/{}.pkl'.format(current_time,name_now))
        print('psnr increase, now_model save successfully.')

        if (deloss / count) < pre_loss:
        # if mean_psnr > best_psnr:
            # best_psnr = mean_psnr
            pre_loss = deloss / count
            torch.save(
                {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                './model/{}/{}.pkl'.format(current_time, name_best))
        print('psnr increase, best_model save successfully.')

    print('model train complete')
    # os.rename('./model/{}.pkl'.format(name), './model/{}_{}.pkl'.format(name, float('%.3f' % best_psnr) ))

    # save to file
    # data_df = pd.DataFrame(data_df)
    # data_df.to_csv('up_data_df.csv')


def valite(model, testDataloader, device):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    model = model.to(device)
    model = model.eval()
    val_toloss = 0
    count = 0
    c = 0
    ps = 0
    ss = 0
    sam = 0
    for i, data in enumerate(testDataloader):
        denoise_gt, noise_image = data
        denoise_gt, noise_image = denoise_gt.float().to(device), noise_image.float().to(device)

        denoise_pre = model(noise_image)

        deloss, grad_loss = cal_loss(denoise_gt, denoise_pre, device)
        val_loss = deloss + 0.001 * grad_loss
        deloss += deloss.item()
        grad_loss += grad_loss.item()

        dg = denoise_gt.detach().cpu().numpy()
        dp = denoise_pre.detach().cpu().numpy()
        count += 1
        # # 打印 dg[0] 的形状和范围
        # print(f"icvl第 {i + 1} 批次中 dg[0] 的形状: {dg[0].shape}")
        # print(f"第 {i + 1} 批次中 dg[0] 的数据范围: 最小值 {dg[0].min()}，最大值 {dg[0].max()}")
        # # 打印 dg[0] 的形状和范围
        # print(f"第 {i + 1} 批次中 dp[0] 的形状: {dp[0].shape}")
        # print(f"第 {i + 1} 批次中 dp[0] 的数据范围: 最小值 {dp[0].min()}，最大值 {dp[0].max()}")
        for l in range(data[0].size(0)):
            ss += mssim(dg[l], dp[l])
            ps += mpsnr(dg[l], dp[l])
            sam += cal_sam(dg[l], dp[l])
            c += 1
    return ps / c, ss / c, sam / c, val_toloss / count


