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
from torchvision.transforms import transforms
import scipy.io
import glob
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
    matpath = "/home/zqcz4080/下载/Pycharmprojects/dataset/Indian_pines/output_sub_images/"
    searchpath = matpath + '*.mat'

    mat_list = glob.glob(searchpath)
    lock = arg.blind
    if lock == "gauss":
        name = "blind"
    else:
        name = arg.noise
    for mat_file in mat_list:
        save_path = "./result_Indian_Pines/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        filename_with_ext = os.path.basename(mat_file)
        # 移除.mat后缀
        file_name = os.path.splitext(filename_with_ext)[0]
        print("Start processing:", file_name)
        save_path = save_path + f'{file_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        process(mat_file,model, device,save_path)  # gavyam_0823-0933.mat
        print(file_name,"over")
    print("over")


def process(mat_file, model, device, save_path):

    model = model.to(device)
    model = model.eval()
    ########读取高光谱图像###########
    mat_data = scipy.io.loadmat(mat_file)
    img = mat_data['Indian_pines']

    img = center_crop(img, (128, 128, 31))
    img = minmax_normalize(img)

    #########去噪############
    # 将 NumPy 数组转换为 PyTorch 张量
    img_ = torch.from_numpy(img)

    # 使用 permute 调整维度顺序，将形状从 (140, 140, 31) 变为 (31, 140, 140)
    img_ = img_.permute(2, 0, 1)

    # 添加一个批量维度，变为形状 (1, 31, 140, 140)
    img_ = img_.unsqueeze(0).to(device).float()

    denoise = model(img_)

    # save mat file
    denoise = denoise.detach().cpu().numpy()
    # 先移除维度为1的第一个维度 (1, 31, 140, 140) -> (31, 140, 140)
    denoise = np.squeeze(denoise, axis=0)

    # 调整维度顺序 (31, 140, 140) -> (140, 140, 31)
    denoise = np.transpose(denoise, (1, 2, 0))
    io.savemat(f'{save_path}/denoise.mat', {'data': denoise})  # hwc
    io.savemat(f'{save_path}/gt.mat', {'data': img})  # hwc

    save(denoise * 255, f'{save_path}/denoise/')
    save(img * 255, f'{save_path}/gt/')

    # denoise RGB
    show_bands = [3, 13, 23]  # icvl

    img = np.stack(
        [img[:, :, show_bands[0]], img[:, :, show_bands[1]], img[:, :, show_bands[2]]], 2)

    cv2.imwrite(f'{save_path}/show_gt.jpg', img * 255)

    denoise = np.stack(
        [denoise[:, :, show_bands[0]], denoise[:, :, show_bands[1]], denoise[:, :, show_bands[2]]], 2)

    cv2.imwrite(f'{save_path}/show_denoise.jpg', denoise * 255)




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


