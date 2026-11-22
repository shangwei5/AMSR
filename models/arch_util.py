import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from einops import rearrange
from torch.autograd import Variable

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_dyconv_v1_new(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_dyconv_v1_new, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.unfold = nn.Unfold(3, 1, 1, 1)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask, coord_, indices):
        b, c, h, w = x.shape
        identity = x

        if self.training:
            res = self.conv1(x) * mask + identity * (1 - mask)
            res = F.relu(res, inplace=True)
            res = self.conv2(res) * mask + identity  # * (1 - mask)
            return res
        else:
            if coord_[0][0].shape[1] != 0:
                x_inp = self.unfold(x).view(b, -1, h, w)
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :]  #b,c,q
                x_2 = self.conv1.bias.view(1, c, 1) + (self.conv1.weight.view(c, -1) @ x_1).view(b, c, -1)
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                res = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_2.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)

                res = F.relu(res, inplace=True)

                res = self.unfold(res).view(b, -1, h, w)
                x_3 = F.grid_sample(
                    res, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :]  #b,c,q
                x_4 = self.conv2.bias.view(1, c, 1) + (self.conv2.weight.view(c, -1) @ x_3).view(b, c, -1)

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                res = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_4.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                res = res + identity * mask
            else:
                res = identity
            return res

class ResidualBlock_noBN_dyconv_v1(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_dyconv_v1, self).__init__()
        self.conv1 = nn.Conv2d(nf*9, nf, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*9, nf, 1, 1, 0, bias=True)
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask, coord_, indices):
        identity = x
        b, c, h, w = x.shape
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        x_inp = rearrange(x_f, 'b c (h w) -> b c h w', h=h, w=w)
        if self.training:
            out = F.relu(self.conv1(x_inp), inplace=True) * mask + x * (1 - mask)
            out_p = self.rpad(out)
            out_f = self.unfold(out_p)
            out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)
            out = self.conv2(out) * mask
            out = out + identity
        else:
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  # b,c,1,q
                x_2 = F.relu(self.conv1(x_1), inplace=True)
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                out = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_2.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out_p = self.rpad(out)
                out_f = self.unfold(out_p)
                out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)

                x_3 = F.grid_sample(
                    out, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  # b,c,1,q
                x_4 = self.conv2(x_3)  # .squeeze(-2).permute(0, 2, 1)  # b,q,c

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                out = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_4.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)

                out = out + identity * mask
            else:
                out = identity

        return out

class ResidualBlock_noBN_dyconv_v1_da(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_dyconv_v1_da, self).__init__()
        self.conv1 = nn.Conv2d(nf*9, nf, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*9, nf, 1, 1, 0, bias=True)
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))
        self.unfold2 = torch.nn.Unfold(3, 2, 0, 1)
        self.rpad2 = torch.nn.ReflectionPad2d((2, 2, 2, 2))
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask, coord_, indices):
        identity = x
        b, c, h, w = x.shape
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        x_inp = rearrange(x_f, 'b c (h w) -> b c h w', h=h, w=w)
        if self.training:
            out = F.relu(self.conv1(x_inp), inplace=True) * mask + x * (1 - mask)
            out_p = self.rpad2(out)
            out_f = self.unfold2(out_p)
            out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)
            out = self.conv2(out) * mask
            out = out + identity
        else:
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  # b,c,1,q
                x_2 = F.relu(self.conv1(x_1), inplace=True)
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                out = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_2.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out_p = self.rpad2(out)
                out_f = self.unfold2(out_p)
                out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)

                x_3 = F.grid_sample(
                    out, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  # b,c,1,q
                x_4 = self.conv2(x_3)  # .squeeze(-2).permute(0, 2, 1)  # b,q,c

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                out = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_4.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)

                out = out + identity * mask
            else:
                out = identity

        return out

class ResidualBlock_noBN_dyconv_v1_new1(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_dyconv_v1_new1, self).__init__()
        self.conv1 = nn.Conv1d(nf*9, nf, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv1d(nf*9, nf, 1, 1, 0, bias=True)
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask):
        identity = x
        b, c, h, w = x.shape
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        if self.training:
            out = F.relu(self.conv1(x_f), inplace=True).contiguous().view(b, -1, h, w) * mask + x * (1 - mask)
            out_p = self.rpad(out)
            out_f = self.unfold(out_p)
            out = self.conv2(out_f).contiguous().view(b, -1, h, w) * mask
            out = out + identity
        else:
            if mask.sum() != 0:
                x = rearrange(identity, 'b c h w -> (b h w) c')
                mask_ = mask.flatten()  # b*1*h*w
                fea = rearrange(x_f, 'b c q -> (b q) c')
                x_1 = fea[mask_ > 0]
                x_2 = F.relu(self.conv1(rearrange(x_1, '(b q) c -> b c q', b=b)), inplace=True)
                x[mask_ > 0] = rearrange(x_2, 'b c q -> (b q) c')
                out = rearrange(x, '(b h w) c -> b c h w', h=h, w=w)
                out_p = self.rpad(out)
                out_f = self.unfold(out_p)

                x_3 = rearrange(out_f, 'b c q -> (b q) c')
                x_4 = self.conv2(rearrange(x_3[mask_ > 0], '(b q) c -> b c q', b=b))
                x[mask_ > 0] = rearrange(x_4, 'b c q -> (b q) c')
                out = rearrange(x, '(b h w) c -> b c h w', h=h, w=w)

                out = out + identity * mask
            else:
                out = identity

        return out

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output



# rcan arch
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):   # bn=True

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class BasicBlock1d(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):   # bn=True

        m = [nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm1d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock1d, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class EResidualBlock_v1_new(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_v1_new, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.group = group
        self.unfold = torch.nn.Unfold(3, 1, 1, 1)

    def forward(self, x, mask, coord_, indices):
        b, c, h, w = x.shape
        x_ori = x.clone()
        if self.training:
            res = self.body[0](x) * mask + x_ori * (1 - mask)
            res = self.body[1](res)
            res = self.body[2](res) * mask + x_ori * (1 - mask)
            res = self.body[3](res)
        else:
            if coord_[0][0].shape[1] != 0:
                x_inps = torch.chunk(x, chunks=self.group, dim=1)
                x_inps = [self.unfold(x_inp).view(b, -1, h, w) for x_inp in x_inps]
                x_1s = [F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] for x_inp in x_inps]    # b,c,q
                x_2 = self.body[0].bias.view(1, c, 1) + torch.cat([weight.view(c//self.group, -1) @ x_1 for weight, x_1 in zip(torch.chunk(self.body[0].weight, chunks=self.group, dim=0), x_1s)], dim=1).view(b, c, -1)
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                res = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_2.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)

                res = self.body[1](res)

                res_s = torch.chunk(res, chunks=self.group, dim=1)
                res_s = [self.unfold(res).view(b, -1, h, w) for res in res_s]
                x_3s = [F.grid_sample(
                    res, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] for res in res_s] #b,c,q
                x_4 = self.body[2].bias.view(1, c, 1) + torch.cat([weight.view(c//self.group, -1) @ x_3 for weight, x_3 in zip(torch.chunk(self.body[2].weight, chunks=self.group, dim=0), x_3s)], dim=1).view(b, c, -1)

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                res = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_4.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                res = self.body[3](res)
            else:
                res = self.body[3](x_ori)
        out = self.body[4](res)
        out = F.relu(out + x)
        return out

class EResidualBlock_v1(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_v1, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        # out = F.relu(out + x)
        return out

class EResidualBlock_dyconv(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_dyconv, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x, mask, coord_, indices):
        b,c,h,w = x.shape
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        x_inp = rearrange(x_f, 'b c (h w) -> b c h w', h=h, w=w)

        if self.training:
            out = self.body(x_inp) * mask
            out = F.relu(out + x)
        else:
            # print(mask.sum(), coord_[0][0].shape[1])
            # zeros = torch.zeros_like(x).to(x.device)
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  #b,c,1,q
                x_1 = self.body(x_1)  #.squeeze(-2).permute(0,2,1)  # b,q,c
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                new_img = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_1.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out = F.relu(new_img + x * mask)
            else:
                out = F.relu(x)

        return out

class EResidualBlock_dyconv_v1(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_dyconv_v1, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))


    def forward(self, x, mask, coord_, indices):
        b,c,h,w = x.shape
        x_ori = x.clone()
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        x_inp = rearrange(x_f, 'b c (h w) -> b c h w', h=h, w=w)

        if self.training:
            out = self.body[0:2](x_inp) * mask + x * (1 - mask)
            out_p = self.rpad(out)
            out_f = self.unfold(out_p)
            out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)
            out = self.body[2:](out) * mask
            out = F.relu(out + x_ori)
        else:
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  #b,c,1,q
                x_2 = self.body[0:2](x_1)  #.squeeze(-2).permute(0, 2, 1)  # b,q,c
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                new_img = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_2.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out = new_img
                out_p = self.rpad(out)
                out_f = self.unfold(out_p)
                out = rearrange(out_f, 'b c (h w) -> b c h w', h=h, w=w)

                x_3 = F.grid_sample(
                    out, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  # b,c,1,q
                x_4 = self.body[2:](x_3)  #.squeeze(-2).permute(0, 2, 1)  # b,q,c

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                new_img = x.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_4.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out = new_img

                out = F.relu(out + x_ori * mask)
            else:
                out = F.relu(x_ori)

        return out

class EResidualBlock_dyconv_v1_new(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_dyconv_v1_new, self).__init__()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, 1, 0),
        )
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))


    def forward(self, x, mask):
        b,c,h,w = x.shape
        x_ori = x.clone()
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)

        if self.training:
            out = self.body[0:2](x_f).contiguous().view(b, -1, h, w) * mask + x * (1 - mask)
            out_p = self.rpad(out)
            out_f = self.unfold(out_p)
            out = self.body[2:](out_f).contiguous().view(b, -1, h, w) * mask
            out = F.relu(out + x_ori)
        else:
            if mask.sum() != 0:
                mask_ = mask.flatten()
                x_1 = rearrange(x_f, 'b c q -> (b q) c')
                x_2 = self.body[0:2](x_1[mask_ > 0].unsqueeze(-1)).squeeze(-1)
                x = rearrange(x_ori, 'b c h w -> (b h w) c')
                x[mask_ > 0] = x_2
                out_p = self.rpad(rearrange(x, '(b h w) c -> b c h w', h=h, w=w))
                out_f = self.unfold(out_p)

                x_3 = rearrange(out_f, 'b c q -> (b q) c')
                x_4 = self.body[2:](x_3[mask_ > 0].unsqueeze(-1)).squeeze(-1)
                x[mask_ > 0] = x_4
                out = rearrange(x, '(b h w) c -> b c h w', h=h, w=w)
                out = F.relu(out + x_ori * mask)
            else:
                out = F.relu(x_ori)

        return out

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        # B, N, C = x.shape
        # feat_token = x
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(x) + x
        else:
            x = self.proj(x)
        # x = x.flatten(2).transpose(1, 2)
        return x

class EResidualBlock_dyconv_v2(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock_dyconv_v2, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels*9, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.unfold = torch.nn.Unfold(3, 1, 0, 1)
        self.rpad = torch.nn.ReflectionPad2d((1, 1, 1, 1))
        self.PEG = PosCNN(in_channels, in_channels)

    def forward(self, x, mask, coord_, indices):
        x_ori = x
        x = self.PEG(x)
        b,c,h,w = x.shape
        x_p = self.rpad(x)
        x_f = self.unfold(x_p)
        x_inp = rearrange(x_f, 'b c (h w) -> b c h w', h=h, w=w)

        if self.training:
            out = self.body(x_inp) * mask
            out = F.relu(out + x_ori)
        else:
            # print(mask.sum(), coord_[0][0].shape[1])
            # zeros = torch.zeros_like(x).to(x.device)
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_inp, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)  #b,c,1,q
                x_1 = self.body(x_1)  #.squeeze(-2).permute(0,2,1)  # b,q,c
                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                new_img = x_ori.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), x_1.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out = F.relu(new_img + x_ori * mask)
            else:
                out = F.relu(x_ori)

        return out

class EResidualBlock1d(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock1d, self).__init__()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 9, 1, 4, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 9, 1, 4, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class EResidualBlock1d_v1(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock1d_v1, self).__init__()

        # self.body = nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels, 9, 1, 4, groups=group),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels, out_channels, 9, 1, 4, groups=group),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, 1, 1, 0),
        # )
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=9, padding=4, stride=1,
                               groups=out_channels*2,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.sg = SimpleGate()
        self.norm = LayerNorm2d(in_channels)

    def forward(self, x):
        x_ = self.norm(x.unsqueeze(-1)).squeeze(-1)
        out = self.conv2(self.conv1(x_))
        out = self.conv3(self.sg(out))
        out = out + x
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
