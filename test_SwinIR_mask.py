import os
import torch
import glob
import numpy as np
import imageio
# from PIL import Image
import cv2
import math
import time
import argparse
# from models.SwinIR_arch import *
from models.SwinIR_mask_arch import *
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.nn as nn
import torchvision
from fast_pytorch_kmeans import KMeans
# from thop import profile
# from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore")

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	# print(net)
	print('Total number of parameters: %.2f K' % (num_params / 1e3))

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
    blurred_image = F.conv2d(image, kernel.to(image.device), padding=kernel_size[0] // 2)
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

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1
        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret

class Inference:
    def __init__(self, args):
        self.args = args
        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.device = 'cuda'
        self.GPUs = args.n_GPUs
        self.dilate = args.dilate
        self.model_name = args.model_name
        self.thres = args.thres
        self.crop = args.crop

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = self.data_path

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('device: {}'.format(self.device))
        self.window_size = 8
        self.net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=self.window_size,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv', thres=self.thres)
        self.net.load_state_dict(torch.load(self.model_path))  # , strict=False
        self.net = self.net.to(self.device)
        if args.n_GPUs > 1:
            self.net = nn.DataParallel(self.net, range(args.n_GPUs))
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        # self.net.eval()
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        self.kmeans = KMeans(n_clusters=2, max_iter=10, mode='euclidean', verbose=0)
        print_network(self.net)

    def infer(self):
        with torch.no_grad():
            total_psnr = []
            total_ssim = []
            total_y_psnr = []
            total_y_ssim = []
            Gflops = []
            # Ori_Gflops = []
            total_time = 0
            LR_frames = sorted(glob.glob(os.path.join(self.input_path, "LR", "X4", "*")))
            HR_frames = sorted(glob.glob(os.path.join(self.input_path, "HR", "X4", "*")))
            crop_sz, step = 32, 28
            for i in range(len(LR_frames)):
                start_time = time.time()

                input = imageio.imread(LR_frames[i])
                lh, lw, _ = input.shape
                if self.crop:
                    h = ((lh - crop_sz + step) // step - 1) * step + crop_sz
                    w = ((lw - crop_sz + step) // step - 1) * step + crop_sz
                    h, w = 4 * h, 4 * w
                else:
                    h = lh * 4
                    w = lw * 4

                gray_input = np.array(cv2.imread(LR_frames[i], 0))[:, :, np.newaxis]
                gt = imageio.imread(HR_frames[i])[:h, :w, :]

                in_tensor = self.numpy2tensor(input, self.device)
                gray_input = self.numpy2tensor(gray_input, self.device)

                hf = gray_input - self.gaussian(gray_input)
                _, _, h_, w_ = hf.shape
                diff = hf.view(-1, 1)
                diff = (diff - diff.min()) / (diff.max() - diff.min())
                centroids = torch.FloatTensor([[0.5]]).cuda()
                flag = self.kmeans.fit_predict(diff, centroids=centroids)
                mask_tensor = flag.view(1, h_, w_)

                k = np.ones((self.dilate, self.dilate), np.uint8)
                mask = mask_tensor.mul(255)
                mask = np.transpose(mask.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                mask = cv2.dilate(mask, k)[:, :, np.newaxis]
                mask = self.numpy2tensor(mask, self.device)

                mod_pad_h = (8 - h//4 % 8) % 8
                mod_pad_w = (8 - w//4 % 8) % 8
                paddings = (0, mod_pad_w, 0, mod_pad_h)
                save_mask = torch.nn.ReflectionPad2d(paddings)(mask[:,:,:h//4,:w//4])
                ratio_mask = window_partition(save_mask.permute(0, 2, 3, 1), self.window_size).mean([1, 2], keepdim=True)  # nW*B, 1, 1, 1
                policy_cur = (ratio_mask[..., 0:1] >= self.thres).float()
                H, W = h // 4 + mod_pad_h, w // 4 + mod_pad_w

                if 'test8k' in self.data_path:
                    out, infer_time, total_mask_flops = self.forward_chop(in_tensor[:,:,:h//4,:w//4], mask[:,:,:h//4,:w//4])
                else:
                    torch.cuda.synchronize()
                    preprocess_time = time.time()
                    out = self.net(in_tensor, mask)   #
                    torch.cuda.synchronize()
                    forward_time = time.time()
                    infer_time = (forward_time - preprocess_time)

                    ratio = policy_cur.sum() / ((H // self.window_size) * (W // self.window_size))
                    print(ratio)
                    total_mask_flops = SwinIR(img_size=(H, W)).flops(ratio)

                # print(f"FLOPs: {total_mask_flops / 1e9} G, Ori_FLOPs: {total_flops / 1e9} G, Percent: {percent} %")
                print(f"FLOPs: {total_mask_flops / 1e9} G")
                Gflops.append(total_mask_flops)
                # Ori_Gflops.append(total_flops)
                # if i >= self.net.count:
                output_img = self.tensor2numpy(out)[:h, :w, :]
                filename = os.path.basename(LR_frames[i])
                print(filename)
                # print(output_img.shape, gt.shape)

                psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                total_psnr.append(psnr)
                total_ssim.append(ssim)

                sr_img_y = self.rgb2ycbcr(output_img / 255., only_y=True)
                gt_img_y = self.rgb2ycbcr(gt / 255., only_y=True)
                psnr_y, ssim_y = self.get_PSNR_SSIM(sr_img_y * 255, gt_img_y * 255)
                total_y_psnr.append(psnr_y)
                total_y_ssim.append(ssim_y)

                if self.save_image:
                    imageio.imwrite(os.path.join(self.result_path, filename), output_img)

                    policy_cur = policy_cur.repeat(1, self.window_size, self.window_size, 1)
                    mask = window_reverse(policy_cur, self.window_size, H, W).permute(0,3,1,2)
                    save_mask = F.interpolate(mask, scale_factor=4, mode='nearest')
                    save_mask = save_mask.squeeze().detach().cpu()[:h, :w]
                    # red = np.array([255, 0, 0])
                    # green = np.array([0, 255, 0])
                    opacity = 0.65
                    red, greeen, blue = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))

                    greeen[save_mask == 0] = 255
                    red[save_mask == 1] = 255
                    vis_img = np.stack([red, greeen, blue], axis=-1)
                    vis_img = vis_img * (1 - opacity) + output_img * opacity
                    imageio.imwrite(os.path.join(self.result_path, filename.split('.')[0]+'_mask.'+filename.split('.')[1]), vis_img.astype('uint8'))
                postprocess_time = time.time()

                self.logger.write_log(
                    '> {} PSNR={:.5}, SSIM={:.4}, PSNR_Y={:.5}, SSIM_Y={:.4}, forward_time:{:.3}s, GFLOPs: {:.4f}'
                        .format(os.path.basename(filename), psnr, ssim, psnr_y, ssim_y, infer_time, total_mask_flops / 1e9))
                if i != 0:
                    total_time += infer_time
                    # else:
                    #     total_time = (forward_time - preprocess_time)
            total_time = total_time / (len(LR_frames) - 1)
            self.logger.write_log('> model_inference_time:{:.5}ms'.format(total_time*1e3))
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum(total_psnr) / len(LR_frames), sum(total_ssim) / len(LR_frames)))
            self.logger.write_log("# Total AVG-PSNR_Y={:.5}, AVG-SSIM_Y={:.4}".format(sum(total_y_psnr) / len(LR_frames),
                                                                                  sum(total_y_ssim) / len(LR_frames)))
            self.logger.write_log('> model GFlops:{:.2f}G'.format(sum(Gflops) / len(Gflops) / 1e9))
            # self.logger.write_log('> model Ori_GFlops:{:.2f}G'.format(sum(Ori_Gflops) / len(Ori_Gflops) / 1e9))
            # self.logger.write_log('> Percent:{:.2f}%'.format(sum(Gflops) / sum(Ori_Gflops)))

    def rgb2ycbcr(self, img, only_y=True):
        """
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        """
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                  [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)

    def forward_chop(self, in_tensor, mask, chop_size=800):
        n, c, h, w = in_tensor.shape
        total_time = 0
        num_h = h // chop_size + 1
        num_w = w // chop_size + 1
        new_h = num_h * chop_size
        new_w = num_w * chop_size
        total_flops = 0
        total_mask_flops = 0
        pad_h = new_h - h
        pad_w = new_w - w

        # pad_top = int(pad_h / 2.)
        # pad_bottom = pad_h - pad_top
        # pad_left = int(pad_w / 2.)
        # pad_right = pad_w - pad_left

        paddings = (0, pad_w, 0, pad_h)
        new_input0 = torch.nn.ReflectionPad2d(paddings)(in_tensor)
        mask = torch.nn.ReflectionPad2d(paddings)(mask)

        out = torch.zeros([1, c, new_h*4, new_w*4], dtype=torch.float32, device=in_tensor.device)
        for i in range(num_h):
            for j in range(num_w):
                if (i + 1) * chop_size >= h:
                    t_h = h
                else:
                    t_h = (i + 1) * chop_size
                if (j + 1) * chop_size >= w:
                    t_w = w
                else:
                    t_w = (j + 1) * chop_size
                with torch.no_grad():
                    torch.cuda.synchronize()
                    preprocess_time = time.time()
                    out[:, :, i * chop_size * 4:t_h * 4, j * chop_size * 4:t_w * 4] = self.net(
                        new_input0[:, :, i * chop_size:t_h, j * chop_size:t_w], mask[:, :, i * chop_size:t_h, j * chop_size:t_w])
                    torch.cuda.synchronize()
                    forward_time = time.time()
                total_time += (forward_time - preprocess_time)

                mod_pad_h = (8 - (t_h-i * chop_size) % 8) % 8
                mod_pad_w = (8 - (t_w-j * chop_size) % 8) % 8

                ratio_mask = window_partition(mask[:, :, i * chop_size:t_h + mod_pad_h, j * chop_size:t_w + mod_pad_w].permute(0, 2, 3, 1), self.window_size).mean([1, 2],
                                                                                               keepdim=True)  # nW*B, 1, 1, 1
                policy_cur = (ratio_mask[..., 0:1] >= self.thres).float()
                ratio = policy_cur.sum() / (
                            ((t_h-i * chop_size + mod_pad_h) // self.window_size) * ((t_w-j * chop_size + mod_pad_w) // self.window_size))

                flops = SwinIR(img_size=(t_h-i * chop_size + mod_pad_h, t_w-j * chop_size + mod_pad_w)).flops(ratio)
                total_flops += flops
        return out[:, :, :h*4, :w*4], total_time, total_flops

    def numpy2tensor(self, input, device='cuda', rgb_range=1.):
        img = np.array(input).astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
        return tensor.unsqueeze(0).to(device)

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=0):
        cropped_output = output  #[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt  #[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float(80)
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArbVSR')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=False, help='restore border images of video if true')
    parser.add_argument('--crop', action='store_true', default=True, help='for fair comparison')
    parser.add_argument('--dilate', type=int, default=3,
                        help='dilation kernel size')
    parser.add_argument('--thres', type=float, default=0.5,
                        help='window size thres')
    parser.add_argument('--data_path', type=str, default='/data1/shangwei/dataset/DIV2K/test/test2k',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str,
                        default='/data1/shangwei/Pixelwise_SR_experiments/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
                        help='the path of pretrain model')
    parser.add_argument('--model_name', type=str,
                        default='SwinIR',
                        help='the name of model')
    parser.add_argument('--result_path', type=str,
                        default='/data1/shangwei/dataset/DIV2K/test/results_test2k/SwinIR_mask_d3',
                        help='the path of deblur result')
    args = parser.parse_args()
    args.n_GPUs = 1

    Infer = Inference(args)
    Infer.infer()

