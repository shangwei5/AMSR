import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from models import rep_utils as utils


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "gelu": 
        layer = nn.GELU()
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


class ResBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(ResBlock, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, int(ratio*n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv2d(int(ratio*n_feats), int(ratio*n_feats), 3, 1, 0)
        self.reduce_conv = nn.Conv2d(int(ratio*n_feats), n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = utils.pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class ResBlock_1d(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(ResBlock_1d, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, int(ratio * n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv1d(int(ratio * n_feats), int(ratio * n_feats), 9, 1, 0)
        self.reduce_conv = nn.Conv2d(int(ratio * n_feats), n_feats, 1, 1, 0)

    def forward(self, x, coord_, indices):
        b, c, h, w = x.shape
        x_ = x.clone()
        if self.training:
            new_inp = []
            for i in range(b):
                x_1 = F.grid_sample(
                    x_[i:i + 1], coord_[i][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                x_2 = F.grid_sample(
                    x_[i:i + 1], coord_[i][1].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                new_x = torch.cat([x_1, x_2], dim=1)  # 1,h*w,c
                new_inp.append(new_x)

            x1 = to_4d(torch.cat(new_inp, dim=0), h, w)  # b,c,h,w

            out = self.expand_conv(x1)
            out_identity = out

            # explicitly padding with bias for reparameterizing in the test phase
            b0 = self.expand_conv.bias
            out = utils.pad_tensor_1d(to_3d(out).permute(0,2,1), b0)

            out = to_4d(self.fea_conv(out).permute(0,2,1), h, w) + out_identity
            out = self.reduce_conv(out)
            new_out = []
            for i in range(b):
                x1 = to_3d(out[i:i + 1])[:, :coord_[i][0].shape[1]]
                rows = indices[i][0].clone()[:, :, 0]
                cols = indices[i][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                new_img = x_[i:i + 1].contiguous().view(1 * c, -1).scatter(1, new_cor.contiguous().view(
                    1 * c, -1), x1.contiguous().view(1 * c, -1)).contiguous().view(1, c, h, w)
                new_out.append(new_img)
            out = torch.cat(new_out, dim=0)
        else:
            if coord_[0][0].shape[1] != 0:
                x_1 = F.grid_sample(
                    x_, coord_[0][0].clone().flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                x_1 = x_1.permute(0, 2, 1).unsqueeze(-1)  # b,c,q,1

                out = self.expand_conv(x_1)  # b,c,q,1
                out_identity = out

                # explicitly padding with bias for reparameterizing in the test phase
                b0 = self.expand_conv.bias
                out = utils.pad_tensor_1d(out.squeeze(-1), b0)

                out = self.fea_conv(out).unsqueeze(-1) + out_identity
                out = self.reduce_conv(out).squeeze(-1)

                rows = indices[0][0].clone()[:, :, 0]
                cols = indices[0][0].clone()[:, :, 1]
                new_cor = (rows * w + cols).unsqueeze(1).repeat(1, c, 1)
                # print(new_cor.shape, x.shape, x_.shape)
                new_img = x_.contiguous().view(b * c, -1).scatter(1, new_cor.contiguous().view(
                    b * c, -1), out.contiguous().view(b * c, -1)).contiguous().view(b, c, h, w)
                out = new_img
            else:
                out = x_

        out += x

        return out


class RepResBlock(nn.Module):
    def __init__(self, n_feats):
        super(RepResBlock, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out
    

class ECALayer(nn.Module):
    """
    Constructs an efficient channel attention layer.
    """
    def __init__(self, channels, gamma=1, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    
class SimplifiedNAFBlock(nn.Module):
    def __init__(self, in_c, exp, act, eca_gamma, layernorm, residual) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = ResBlock(in_c, exp)
        
        # activation
        self.act = act        
        # channel attention
        if eca_gamma > 0:
            self.ca = ECALayer(in_c, gamma=eca_gamma)
        else:
            self.ca = None
        # layernorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None
        
    def forward(self, x):
        res = x.clone()
        
        if self.norm is not None:
            x = self.norm(x)
            
        x = self.conv1(x)
        
        if self.act is not None:
            x = self.act(x)
        
        if self.ca is not None:
            x = self.ca(x) * x
            
        if self.residual:
            x += res
        else:
            x = x
        return x


class SimplifiedNAFBlock_1d(nn.Module):
    def __init__(self, in_c, exp, act, eca_gamma, layernorm, residual) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = ResBlock_1d(in_c, exp)

        # activation
        self.act = act
        # channel attention
        if eca_gamma > 0:
            self.ca = ECALayer(in_c, gamma=eca_gamma)
        else:
            self.ca = None
        # layernorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None

    def forward(self, zz):
        x, coord_, indices = zz

        res = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        x = self.conv1(x, coord_, indices)

        if self.act is not None:
            x = self.act(x)

        if self.ca is not None:
            x = self.ca(x) * x

        if self.residual:
            x += res
        else:
            x = x
        return [x, coord_, indices]
    

class SimplifiedRepNAFBlock(nn.Module):
    def __init__(self, in_c, exp, act, eca_gamma, layernorm, residual) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = RepResBlock(in_c)
        
        # activation
        self.act = act
        
        # channel attention
        if eca_gamma > 0:
            self.ca = ECALayer(in_c, gamma=eca_gamma)
        else:
            self.ca = None
        
        # LayerNorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None
        
    def forward(self, x):
        res = x.clone()
        
        if self.norm is not None:
            x = self.norm(x)
            
        x = self.conv1(x)
        
        if self.act is not None:
            x = self.act(x)
        
        if self.ca is not None:
            x = self.ca(x) * x
            
        if self.residual:
            x += res
        else:
            x = x
        return x
    
    
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
