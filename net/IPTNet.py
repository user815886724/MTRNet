import torch.nn as nn
import torch
from net.Transformer import VisionTransformer
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        padding=(kernel_size // 2), bias=bias
    )

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale-1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4*n_feats, kernel_size=3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9*n_feats, kernel_size=3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError



class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res





class IPT(nn.Module):
    def __init__(self, patch_size=48, patch_dim=3, n_feats=64, num_heads=12, num_layers=12, num_queries=1, rgb_range=255, dropout_rate=0.,
                 no_mlp=False, pos_every=False, no_pos=False, n_colors=3, no_norm=False, scale=(4,), conv=default_conv):
        super(IPT, self).__init__()
        self.scale_index = 0
        self.n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(n_colors, n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act)
            ) for _ in scale
        ])
        self.body = VisionTransformer(img_dim=patch_size, patch_dim=patch_dim, num_channels=n_feats,
                                      embedding_dim=n_feats*(patch_dim**2), num_heads=num_heads, num_layers=num_layers,
                                      hidden_dim=n_feats*(patch_dim**2)*4, num_queries=num_queries, dropout_rate=dropout_rate,
                                      mlp=no_mlp,pos_every=pos_every, no_pos=no_pos, no_norm=no_norm)
        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ) for s in scale
        ])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head[self.scale_index](x)
        res = self.body(x, self.scale_index)
        res += x

        x = self.tail[self.scale_index](res)

        x = self.add_mean(x)
        return x

    def set_scale(self, scale_index):
        self.scale_index = scale_index









