import functools
import torch.nn as nn
import torch.nn.functional as F
import models.arch_util as arch_util
import torch
from einops import rearrange

def get_thres_coords(tensor_, thres):
    coords_list_all = []
    ori_coords_list_all = []
    for i in range(tensor_.shape[0]):
        tensor = tensor_[i:i+1]
        B, H, W = tensor.shape
        # 展平每个批次的H和W维度
        flat_tensor = tensor.view(tensor.shape[0], -1)

        mask1 = flat_tensor != thres
        mask2 = flat_tensor == thres
        coords_list = []
        ori_coords_list = []

        mask = mask1.contiguous().view(B, H, W).clone()
        # 获取坐标
        coords = torch.nonzero(mask, as_tuple=True)
        ver_cors = torch.chunk(coords[1].float(), B)  # H坐标
        # print(coords[1].shape, torch.stack(ver_cors, dim=0).shape)
        ver_cor_ = [(-1 + 1 / H + 2 / H * ver_cor) for ver_cor in ver_cors]
        hor_cors = torch.chunk(coords[2].float(), B)  # W坐标
        hor_cor_ = [(-1 + 1 / W + 2 / W * hor_cor) for hor_cor in hor_cors]
        # print(coords[2].shape, torch.stack(hor_cors, dim=0).shape)
        assert len(ver_cors) == len(hor_cors)
        ori_coords = torch.stack([torch.stack(ver_cors, dim=0), torch.stack(hor_cors, dim=0)], dim=-1)
        coords = torch.stack([torch.stack(ver_cor_, dim=0), torch.stack(hor_cor_, dim=0)], dim=-1)
        coords_list.append(coords.clone())
        ori_coords_list.append(ori_coords.to(torch.int64).clone())
        # print(ori_coords.to(torch.int64).dtype)

        mask = mask2.contiguous().view(B, H, W).clone()
        # 获取坐标
        coords = torch.nonzero(mask, as_tuple=True)
        ver_cors = torch.chunk(coords[1].float(), B)  # H坐标
        # print(coords[1].shape, torch.stack(ver_cors, dim=0).shape)
        ver_cor_ = [(-1 + 1 / H + 2 / H * ver_cor) for ver_cor in ver_cors]
        hor_cors = torch.chunk(coords[2].float(), B)  # W坐标
        hor_cor_ = [(-1 + 1 / W + 2 / W * hor_cor) for hor_cor in hor_cors]
        # print(coords[2].shape, torch.stack(hor_cors, dim=0).shape)
        assert len(ver_cors) == len(hor_cors)
        ori_coords = torch.stack([torch.stack(ver_cors, dim=0), torch.stack(hor_cors, dim=0)], dim=-1)
        coords = torch.stack([torch.stack(ver_cor_, dim=0), torch.stack(hor_cor_, dim=0)], dim=-1)
        coords_list.append(coords.clone())
        ori_coords_list.append(ori_coords.to(torch.int64).clone())
        # if coords_list[0].shape[1] + coords_list[1].shape[1] == 0:
        #     print(flat_tensor)
        # print(H*W, coords_list[0].shape[1] + coords_list[1].shape[1], coords_list[0].shape[1], coords_list[1].shape[1])
        assert (coords_list[0].shape[1] + coords_list[1].shape[1]) == H * W
        assert (ori_coords_list[0].shape[1] + ori_coords_list[1].shape[1]) == H * W
        coords_list_all.append(coords_list)
        ori_coords_list_all.append(ori_coords_list)
    return coords_list_all, ori_coords_list_all


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=58, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN_dyconv_v1_da, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x, mask):
        if not self.training:
            coord_, indices = get_thres_coords(mask.squeeze(1).clone(), 0)
        else:
            coord_, indices = None, None

        fea = self.lrelu(self.conv_first(x))
        for block in self.recon_trunk:
            fea = block(fea, mask, coord_, indices)
        out = fea
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


