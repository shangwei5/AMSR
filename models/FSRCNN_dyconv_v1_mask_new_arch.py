import functools
import torch.nn as nn
import torch.nn.functional as F
import models.arch_util as arch_util
import torch
from einops import rearrange
import torchvision

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

class FSRCNN_net(torch.nn.Module):
    def __init__(self, input_channels=3, upscale=4, d=48, s=12, m=4):
        super(FSRCNN_net, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s*9, out_channels=s, kernel_size=1, stride=1, padding=0))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)

        self.unfold = torch.nn.Sequential(
                    torch.nn.Unfold(3, 1, 0, 1),
                    torch.nn.Unfold(3, 2, 0, 1),
                    torch.nn.Unfold(3, 3, 0, 1),
                    torch.nn.Unfold(3, 4, 0, 1)
                                          )
        self.rpad = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                    torch.nn.ReflectionPad2d((2, 2, 2, 2)),
                    torch.nn.ReflectionPad2d((3, 3, 3, 3)),
                    torch.nn.ReflectionPad2d((4, 4, 4, 4))
                                          )
        self.thres = 0.5
        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x, mask):
        #
        fea = self.head_conv(x)
        fea = self.body_conv[0](fea)

        b, c, h, w = fea.shape
        fea_ori = fea.clone()
        fea_p = self.rpad[0](fea)
        fea_f = self.unfold[0](fea_p)

        if self.training:
            fea = rearrange(fea_f, 'b c (h w) -> b c h w', h=h, w=w)
            for i in range(1, 5):
                fea = self.body_conv[i](fea) * mask + fea_ori * (1 - mask)
                if i != 4:
                    fea_p = self.rpad[i](fea)
                    fea_f = self.unfold[i](fea_p)
                    fea = rearrange(fea_f, 'b c (h w) -> b c h w', h=h, w=w)
        else:
            if mask.sum() != 0:
                fea_ori = rearrange(fea_ori, 'b c h w -> (b h w) c', h=h, w=w)
                mask = mask.flatten()  #b*1*h*w
                for i in range(1, 5):
                    fea = rearrange(fea_f, 'b c q -> (b q) c')
                    x_1 = fea[mask > 0]
                    x_2 = self.body_conv[i](x_1.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                    fea_ori[mask > 0] = x_2
                    fea = rearrange(fea_ori, '(b h w) c -> b c h w', h=h, w=w)
                    if i != 4:
                        fea_p = self.rpad[i](fea)
                        fea_f = self.unfold[i](fea_p)
            else:
                fea = fea_ori

        fea = self.body_conv[-2:](fea)
        out = self.tail_conv(fea)
        return out
