import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# 哈尔小波变化：将输入图像分解为垂直、水平和对角方向的一个低频和三个高频表示
class HaarWaveletTransformation(nn.Module):
    def __init__(self, in_channels):
        super(HaarWaveletTransformation, self).__init__()
        self.in_channels = in_channels
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False
        self.last_jac = None


    def forward(self, x, reverse=False):
        elements = x.shape[1] * x.shape[2] * x.shape[3]
        if not reverse:
            self.last_jac = (elements / 4 )* np.log(1 / 16.)
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.last_jac = (elements / 4) * np.log(16.)
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)

    def jacobian(self):
        return self.last_jac

x = torch.rand(3, 3, 128, 128)
har = HaarWaveletTransformation(3)
print(har(x))