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
import torch.nn.functional as F
import torchvision
from fast_pytorch_kmeans import KMeans

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
            self.dir_mask = os.path.join(path.replace('DIV2K_train_HR_Aug_sub256','DIV2K_train_HR_Aug_mask_sub256'), 'LR_bicubic', 'X4')
            print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)
            print("DataSet Mask path:", self.dir_mask)
            self.inputs = os.listdir(self.dir_input)
            self.gts = os.listdir(self.dir_gt)
            self.masks = os.listdir(self.dir_mask)
            assert len(self.gts) == len(self.masks)
        else:
            self.dir_gt = os.path.join(path, 'HR', 'X4')
            self.dir_input = os.path.join(path, 'LR', 'X4')
            print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)
            self.inputs = os.listdir(self.dir_input)
            self.gts = os.listdir(self.dir_gt)

        if self.args['mask_kernel_range'] is None:
            self.args['mask_kernel_range'] = [5, 1, -1]
            self.flag = 0
        else:
            self.flag = 1

        assert len(self.inputs) == len(self.gts)
        self.num_frame = len(self.gts)
        print("Number of frames to load:", self.num_frame)
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        self.kmeans = KMeans(n_clusters=2, max_iter=10, mode='euclidean', verbose=0)

    def __getitem__(self, idx):
        if self.train:
            input, gt, mask, file_name = self._load_file(idx)   #
            input, gt, mask = self.get_patch(input, gt, mask)
            input_tensor = utils.singlenp2Tensor(input, rgb_range=1, n_colors=3)
            gt_tensor = utils.singlenp2Tensor(gt, rgb_range=1, n_colors=3)
            mask_tensor = utils.singlenp2Tensor(mask, rgb_range=1, n_colors=1)
            mask_tensor_list = [torch.ones_like(mask_tensor)]
            if self.flag:
                mask_tensor_list.append(torch.ones_like(mask_tensor))
            mask_tensor[mask_tensor != 0] = 1

            for ks in range(self.args['mask_kernel_range'][0], self.args['mask_kernel_range'][1], self.args['mask_kernel_range'][2]): #range(5, 1, -1):  # 5,4,3,2
                k = np.ones((ks, ks), np.uint8)
                mask = mask_tensor.mul(255)
                mask = np.transpose(mask.numpy(), (1, 2, 0)).astype(np.uint8)
                mask = cv2.dilate(mask, k)[:, :, np.newaxis]
                mask_tensor = utils.singlenp2Tensor(mask, rgb_range=1, n_colors=1)
                mask_tensor_list.append(mask_tensor)
            # assert len(mask_tensor_list) == 5
        else:
            input, gt, gray_input, file_name = self._load_file(idx)
            input_tensor = utils.singlenp2Tensor(input, rgb_range=1, n_colors=3)
            gt_tensor = utils.singlenp2Tensor(gt, rgb_range=1, n_colors=3)
            gray_input = utils.singlenp2Tensor(gray_input, rgb_range=1, n_colors=1)
            hf = gray_input.unsqueeze(0) - self.gaussian(gray_input.unsqueeze(0))
            b, c, h, w = hf.shape
            diff = hf.view(-1, 1)
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            centroids = torch.FloatTensor([[0.5]])
            flag = self.kmeans.fit_predict(diff, centroids=centroids)
            mask_tensor = flag.view(c, h, w)

            k = np.ones((5, 5), np.uint8)
            mask = mask_tensor.mul(255)
            mask = np.transpose(mask.numpy(), (1, 2, 0)).astype(np.uint8)
            mask = cv2.dilate(mask, k)[:, :, np.newaxis]
            mask_tensor = utils.singlenp2Tensor(mask, rgb_range=1, n_colors=1)
            mask_tensor_list = [mask_tensor]

        return input_tensor, gt_tensor, mask_tensor_list, file_name


    def __len__(self):
        # if self.train:
        #     return self.num_frame
        # else:
        return self.num_frame

    def _load_file(self, idx):
        file_name = self.inputs[idx]
        if self.train:
            input = np.array(imageio.imread(os.path.join(self.dir_input, file_name)))
            gt = np.array(imageio.imread(os.path.join(self.dir_gt, file_name)))
            file_names = file_name.split('_')
            file_name1 = ''
            for name in file_names:
                if '.png' not in name:
                    file_name1 += (name + '_')
                else:
                    file_name1 += ('mask_' + name)
            mask = np.array(imageio.imread(os.path.join(self.dir_mask, file_name1)))[:, :, 0:1]
            return input, gt, mask, file_name
        else:
            input = np.array(imageio.imread(os.path.join(self.dir_input, file_name)))
            gt = np.array(imageio.imread(os.path.join(self.dir_gt, file_name)))
            gray_input = np.array(cv2.imread(os.path.join(self.dir_input, file_name), 0))[:, :, np.newaxis]
            return input, gt, gray_input, file_name


    def get_patch(self, input, gt, mask):   #
        # input, gt, mask = utils.get_list_patch([input, gt, mask], patch_size=self.args['patch_size'])
        input, gt, mask = utils.augment([input, gt, mask])

        return input, gt, mask

