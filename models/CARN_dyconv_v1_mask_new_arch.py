import torch
import torch.nn as nn
import models.arch_util as arch_util


def get_thres_coords(tensor_, thres):
    coords_list_all = []
    ori_coords_list_all = []
    for i in range(tensor_.shape[0]):
        tensor = tensor_[i:i + 1]
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
        assert (coords_list[0].shape[1] + coords_list[1].shape[1]) == H * W
        assert (ori_coords_list[0].shape[1] + ori_coords_list[1].shape[1]) == H * W
        coords_list_all.append(coords_list)
        ori_coords_list_all.append(ori_coords_list)
    return coords_list_all, ori_coords_list_all


class Block(nn.Module):
    def __init__(self, nf,
                 group=1):
        super(Block, self).__init__()

        self.b1 = arch_util.EResidualBlock_dyconv_v1(nf, nf, group=group)
        self.c1 = arch_util.BasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = arch_util.BasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = arch_util.BasicBlock(nf * 4, nf, 1, 1, 0)

    def forward(self, x, mask, coord_, indices):
        c0 = o0 = x

        b1 = self.b1(o0, mask, coord_, indices)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1, mask, coord_, indices)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2, mask, coord_, indices)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class CARN_M(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=52, scale=4, multi_scale=False, group=4):
        super(CARN_M, self).__init__()
        self.scale = scale
        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = arch_util.BasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = arch_util.BasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = arch_util.BasicBlock(nf * 4, nf, 1, 1, 0)

        self.upsample = arch_util.UpsampleBlock(nf, scale=scale,
                                                multi_scale=multi_scale,
                                                group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x, mask):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        if not self.training:
            coord_, indices = get_thres_coords(mask.squeeze(1).clone(), 0)
        else:
            coord_, indices = None, None

        b1 = self.b1(o0, mask, coord_, indices)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1, mask, coord_, indices)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2, mask, coord_, indices)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
