import os
import sys
sys.path.append('../..')

import cv2
from tqdm import tqdm
from data.scripts.util import _get_paths_from_images, imresize_np
import torch
import torch.nn.functional as F
import torchvision
from fast_pytorch_kmeans import KMeans
import numpy as np
from multiprocessing import Pool
from os import path as osp
from utils.utils_io import scandir

def numpy2tensor(input, rgb_range=1.):
    img = np.array(input).astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)
    return tensor.unsqueeze(0)

def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img

def spatial_pooling(image1, image2, kernel_size=(3, 3), stride=(1, 1)):
    assert image1.dim() == 4 and image2.dim() == 4, "Images must be 4-dimensional"
    B, C, H, W = image1.shape
    oh = (H + 2 - kernel_size[0]) // stride[0] + 1
    ow = (W + 2 - kernel_size[1]) // stride[1] + 1
    patches1 = F.unfold(image1, kernel_size=kernel_size, stride=stride, padding=1)
    patches2 = F.unfold(image2, kernel_size=kernel_size, stride=stride, padding=1)
    patches1 = patches1.view(B, C, kernel_size[0], kernel_size[1], oh, ow)
    patches2 = patches2.view(B, C, kernel_size[0], kernel_size[1], oh, ow)
    std1 = torch.std(patches1, dim=(2, 3))
    std2 = torch.std(patches2, dim=(2, 3))
    ans = torch.log((1 + std1 * std1) * (1 + std2 * std2))
    return ans

def main():

    opt = {}
    opt['n_thread'] = 20

    # HR images
    opt['input_folder'] = '/data1/shangwei/dataset/DIV2K/DIV2K_train_HR_aug' ### set path
    opt['save_folder1'] = '/data1/shangwei/dataset/DIV2K/DIV2K_train_HR_aug_mask'
    # opt['save_folder2'] = '/data1/shangwei/dataset/DIV2K/DIV2K_train_HR_aug_masksp'
    extract_mask(opt)



def extract_mask(opt):
    """
    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder1 = opt['save_folder1']
    # save_folder2 = opt['save_folder2']

    if not osp.exists(save_folder1):
        os.makedirs(save_folder1)
        print(f'mkdir {save_folder1} ...')
    else:
        print(f'Folder {save_folder1} already exists. Exit.')
        sys.exit(1)

    # if not osp.exists(save_folder2):
    #     os.makedirs(save_folder2)
    #     print(f'mkdir {save_folder2} ...')
    # else:
    #     print(f'Folder {save_folder2} already exists. Exit.')
    #     sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    """
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, 0)

    a = numpy2tensor(img[:, :, np.newaxis])
    gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
    # # 1111111111111
    # hf = spatial_pooling(a, gaussian(a), (3, 3), (1, 1))
    # b, c, h, w = hf.shape
    #
    # kmeans = KMeans(n_clusters=2, max_iter=10, mode='euclidean', verbose=0)
    # diff = hf.view(-1, 1)
    # diff = (diff - diff.min()) / (diff.max() - diff.min())
    # centroids = torch.FloatTensor([[0.5]])
    # flag = kmeans.fit_predict(diff, centroids=centroids)
    # _, min_index = torch.min(diff.flatten(), dim=0)
    # if flag[min_index] == 1:
    #     flag = 1 - flag
    # mask = flag.view(b, c, h, w)
    # mask_save = tensor2numpy(mask)
    # cv2.imwrite(osp.join(opt['save_folder2'], f'{img_name}_masksp{extension}'), mask_save)

    # 2222222222
    hf = a - gaussian(a)
    b, c, h, w = hf.shape

    kmeans = KMeans(n_clusters=2, max_iter=10, mode='euclidean', verbose=0)
    diff = hf.view(-1, 1)
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    centroids = torch.FloatTensor([[0.5]])
    flag = kmeans.fit_predict(diff, centroids=centroids)
    _, min_index = torch.min(diff.flatten(), dim=0)

    mask = flag.view(b, c, h, w)
    mask_save = tensor2numpy(mask)
    cv2.imwrite(osp.join(opt['save_folder1'], f'{img_name}_mask{extension}'), mask_save)

    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
