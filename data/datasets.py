import cv2
import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import utils.utils as utils
import imageio
import math
import random
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

# def make_coord(shape):
#     """ Make coordinates at grid centers.
#     """
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         # v0, v1 = -1, 1
#
#         r = 1 / n
#         seq = -1 + r + (2 * r) * torch.arange(n).float()
#         coord_seqs.append(seq)
#     # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
#     ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
#     return ret

def gaussian_blur(image, kernel_size):
    """
    使用高斯滤波器对图像进行模糊处理。

    参数:
    - image: 输入图像，大小为 (B, C, H, W)
    - kernel_size: 高斯核的大小，格式为 (kh, kw)

    返回:
    - blurred_image: 模糊后的图像
    """
    # 确保输入图像的维度是四维的
    assert image.dim() == 4, "Image must be 4-dimensional"

    # 计算高斯核的标准差
    sigma = (kernel_size[0] // 2 - 1) * 0.3 + 0.8

    # 创建高斯滤波器
    kernel = torch.tensor(
        [[np.exp(-((i - kernel_size[0] // 2) ** 2 + (j - kernel_size[1] // 2) ** 2) / (2 * sigma ** 2))
          for j in range(kernel_size[1])] for i in range(kernel_size[0])])
    kernel = kernel / kernel.sum()
    kernel = kernel.to(torch.float32)

    # 将滤波器转换为卷积核
    kernel = kernel.view(1, 1, kernel_size[0], kernel_size[1])
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)

    # 使用卷积运算实现高斯滤波
    blurred_image = F.conv2d(image, kernel, padding=kernel_size[0] // 2)
    return blurred_image

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

class VIDEODATA(data.Dataset):
    #  train path:  '/data1/shangwei/dataset/DIV2K/DIV2K_train_HR_Aug_sub'
    #  test path:  '/data1/shangwei/dataset/DIV2K/test/test2k'
    def __init__(self, path, args, name='DIV2K', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.path = path

        if self.train:
            self.dir_gt = os.path.join(path, 'HR', 'X4')
            self.dir_input = os.path.join(path, 'LR_bicubic', 'X4')
            print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)
            self.inputs = os.listdir(self.dir_input)
            self.gts = os.listdir(self.dir_gt)
        else:
            self.dir_gt = os.path.join(path, 'HR', 'X4')
            self.dir_input = os.path.join(path, 'LR', 'X4')
            print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)
            self.inputs = os.listdir(self.dir_input)
            self.gts = os.listdir(self.dir_gt)

        assert len(self.inputs) == len(self.gts)
        self.num_frame = len(self.gts)
        print("Number of frames to load:", self.num_frame)

    def __getitem__(self, idx):
        input, gt, file_name, inp_gray = self._load_file(idx)   #

        input, gt, inp_gray = self.get_patch(input, gt, inp_gray)
        input_tensor = utils.singlenp2Tensor(input, rgb_range=1, n_colors=3)
        gt_tensor = utils.singlenp2Tensor(gt, rgb_range=1, n_colors=3)
        inp_gray_tensor = utils.singlenp2Tensor(inp_gray, rgb_range=1, n_colors=3)

        b_ = inp_gray_tensor.unsqueeze(0)
        blur = gaussian_blur(b_ * 255, (5, 5))
        mask_ = spatial_pooling(b_ * 255, blur, (3, 3), (1, 1))
        struc_p = (mask_ - mask_.min()) / (mask_.max() - mask_.min())  #mask_ / 255.

        mask = struc_p.clone()
        prob = self.args.get('mask_prob', 0.5)
        # print(prob)
        if prob == 0:
            mask = torch.ones_like(struc_p)
        else:
            min = mask.quantile(q=prob).item()  # mask.median().item()  #
            mask[mask < min] = 0
            mask[mask != 0] = 1

        return input_tensor, gt_tensor, struc_p.squeeze(0), mask.squeeze(0), file_name


    def __len__(self):
        # if self.train:
        #     return self.num_frame
        # else:
        return self.num_frame

    def _load_file(self, idx):
        file_name = self.inputs[idx]
        input = np.array(imageio.imread(os.path.join(self.dir_input, file_name)))
        gt = np.array(imageio.imread(os.path.join(self.dir_gt, file_name)))
        inp_gray = np.array(cv2.imread(os.path.join(self.dir_input, file_name), 0))[:, :, np.newaxis]

        return input, gt, file_name, inp_gray


    def get_patch(self, input, gt, inp_gray):   #
        if self.train:
            input, gt, inp_gray = utils.augment([input, gt, inp_gray])

        return input, gt, inp_gray


