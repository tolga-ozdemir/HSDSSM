import numpy as np
from skimage.util import view_as_windows
import cv2
# coding=utf-8
# Version:python 3.7
from setup import *
from utils import *
import h5py
import cv2
import os
import glob
from torchvision.transforms import transforms
import yaml


def add_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)


def crop_patches(image, patch_size, strides):
    patches = view_as_windows(image, patch_size, strides)
    return patches


def denoise_patch(patch):

    denoised_patch = cv2.GaussianBlur(patch, (3, 3), 0)
    return denoised_patch


def reconstruct_image_from_patches(patches, image_shape, strides):
    reconstructed_image = np.zeros(image_shape)
    patch_count = np.zeros(image_shape)
    patch_size = patches.shape[-3:]

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                patch = patches[i, j, k]
                reconstructed_image[
                i * strides[0]:i * strides[0] + patch_size[0],
                j * strides[1]:j * strides[1] + patch_size[1],
                k * strides[2]:k * strides[2] + patch_size[2]] += patch
                patch_count[
                i * strides[0]:i * strides[0] + patch_size[0],
                j * strides[1]:j * strides[1] + patch_size[1],
                k * strides[2]:k * strides[2] + patch_size[2]] += 1

    return reconstructed_image / patch_count


# 中心裁剪函数
def center_crop(array, crop_size):
    original_shape = array.shape
    start = [(o - c) // 2 for o, c in zip(original_shape, crop_size)]
    end = [start[i] + crop_size[i] for i in range(len(crop_size))]

    slices = tuple(slice(start[i], end[i]) for i in range(len(crop_size)))
    return array[slices]


def main():
    arg = option()
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    with open('./training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    model = getmodel(opt)
    model = model.to(device)

    model = load_model(model, arg.pretrain_modle, mode='test')


    # ICVL-gauss
    matpath = "./data/Mat_test/"
    #matpath = "./data/CAVE_test/"
    searchpath = matpath + '*.mat'

    mat_list = glob.glob(searchpath)
    n_psnr = 0
    n_ssim = 0
    n_sam = 0
    output_psnr = 0
    output_ssim = 0
    output_sam = 0
    count = 0
    lock = arg.blind
    if lock == "gauss":
        name = "blind"
    else:
        name = arg.noise
    for mat_file in mat_list:
        save_path = "./result_icvl_2025111/"
        #save_path = "./result_cave_1layer/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path + f'sigma_{name}/'#sigma_{name}/{lock}
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        filename_with_ext = os.path.basename(mat_file)
        # 移除.mat后缀
        file_name = os.path.splitext(filename_with_ext)[0]
        print("Start processing:", file_name)
        save_path = save_path + f'{file_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        noisepsnr, noisessims, noisesam, psnrs, ssims, sam = process(mat_file,  gettransform(arg.blind, arg.noise),model, device,save_path,)  # gavyam_0823-0933.mat

        n_psnr += noisepsnr
        n_ssim += noisessims
        n_sam += noisesam
        output_psnr += psnrs
        output_ssim += ssims
        output_sam += sam
        count += 1
    n_psnr = n_psnr / count
    n_ssim = n_ssim / count
    n_sam = n_sam / count
    output_psnr = output_psnr / count
    output_ssim = output_ssim / count
    output_sam = output_sam / count

    print('noise : psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(n_psnr, n_ssim, n_sam))
    print('denoise : psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(output_psnr, output_ssim, output_sam))


def process(mat_file, transformer, model, device, save_path):

    model = model.to(device)
    model = model.eval()
    ########读取高光谱图像###########

    img = np.array(h5py.File(mat_file)['rad'])  # (31, 1392, 1300)
    img = center_crop(img, (31, 896, 896))
    #img = center_crop(img, (31, 512, 512))
    img = minmax_normalize(img)

    ##########添加噪声#############

    noise_image = transformer(img).float()  # (B,H,W)

    noise_copy = noise_image.numpy().copy()
    noisepsnr = mpsnr(img, noise_copy)  # img (31, 1392, 1300) chw
    noisessims = mssim(img, noise_copy)
    noisesam = cal_sam(img, noise_copy)

    print('psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(noisepsnr, noisessims, noisesam))

    ########裁剪patch########
    # 裁剪成 (150, 150, 31) 大小的patch
    patch_size = (128, 128, 31)
    strides = (128, 128, 31)
    noise_image = noise_image.permute(1, 2, 0)  # (1392, 1300,31) chw -> hwc
    noise_image = noise_image.numpy()
    image_shape = noise_image.shape
    patches = crop_patches(noise_image, patch_size, strides)
    patches = torch.from_numpy(patches)

    #########去噪############
    # 对patch进行去噪处理
    denoised_patches = np.zeros_like(patches)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            input = patches[i, j].float().to(device).permute(0, 3, 1, 2)
            input = model(input)

            denoised_patches[i, j] = input.permute(0, 2, 3, 1).detach().cpu()

    # 将去噪后的patch拼回原来的高光谱图像
    denoise = reconstruct_image_from_patches(denoised_patches, image_shape, strides)

    denoise = np.transpose(denoise, (2, 0, 1))
    psnrs = mpsnr(img, denoise)  # (340, 340, 31)
    ssims = mssim(img, denoise)
    sam = cal_sam(img, denoise)
    print('psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(psnrs, ssims, sam))

    img = np.transpose(img, [1, 2, 0])  # chw -> hwc
    noise_copy = np.transpose(noise_copy, [1, 2, 0])  # chw -> hwc
    denoise = np.transpose(denoise, [1, 2, 0])  # chw -> hwc

    # save mat file
    io.savemat(f'{save_path}/gt.mat', {'data': img})  # hwc
    if noise_copy is not None:
        io.savemat(f'{save_path}/noise.mat', {'data': noise_copy})  # hwc
    io.savemat(f'{save_path}/denoise.mat', {'data': denoise})  # hwc

    save(img * 255, f'{save_path}/gt/')
    save(noise_copy * 255, f'{save_path}/noise/')
    save(denoise * 255, f'{save_path}/denoise/')

    show_bands = [3, 13, 23]  # icvl
    # image RGB
    img = np.stack([img[:, :, show_bands[0]], img[:, :, show_bands[1]], img[:, :, show_bands[2]]], 2)



    cv2.imwrite(f'{save_path}/show_gt.jpg', img * 255)

    if noise_copy is not None:
        # noise RGB
        noise_copy = np.stack(
            [noise_copy[:, :, show_bands[0]], noise_copy[:, :, show_bands[1]], noise_copy[:, :, show_bands[2]]], 2)


        cv2.imwrite(f'{save_path}/show_noise.jpg', noise_copy * 255)

        # denoise RGB

        denoise = np.stack(
            [denoise[:, :, show_bands[0]], denoise[:, :, show_bands[1]], denoise[:, :, show_bands[2]]], 2)

        cv2.imwrite(f'{save_path}/show_denoise.jpg', denoise * 255)
    return noisepsnr, noisessims, noisesam, psnrs, ssims, sam



def save(image_array, save_dir):
    """
    将NumPy数组中的每个通道保存为单独的BMP文件。

    参数:
    - image_array: NumPy数组，形状为(height, width, num_channels)。
    - output_dir: 字符串，输出目录的路径，用于保存BMP文件。

    注意: 确保image_array的数据类型为np.uint8，否则在保存之前需要转换。
    """

    #### save bmp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果输出目录不存在，则创建它

    height, width, num_channels = image_array.shape

    # 遍历所有通道
    for channel in range(num_channels):
        # 提取当前通道的数据
        channel_image = image_array[:, :, channel]

        # # 确保数据类型为np.uint8（如果需要的话）
        # if image_array.dtype != np.uint8:
        #     channel_image = channel_image.astype(np.uint8)
        # 注意：这里我们对整个image_array进行了检查，但在实际中可能只需要检查channel_image
        # 如果其他通道已经是np.uint8，则这种转换是不必要的

        # 将NumPy数组转换为PIL图像对象，并保存为BMP文件
        # pil_image = Image.fromarray(channel_image, mode='L')
        filename = os.path.join(save_dir, f"channel_{channel + 1}.bmp")
        # pil_image.save(filename)
        cv2.imwrite(filename, channel_image)



if __name__ == '__main__':
    main()


