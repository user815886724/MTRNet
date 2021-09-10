from utils.data_util import TrainDataLoader
from model.MTRNet import InputProjection
from model.MTRNet import OutputProjection,SepConv2d, LinearProjection, InputProjection, ConvProjection,\
    SELayer, image_matrix_mul, WindowAttention,TransformerVision, TransformerBlocks, MTR_Model_1
import torch

from torch.utils.data import DataLoader


# 测试 SepConv2d功能
def test_SepConv():
    x = torch.rand(3, 3, 5, 3)
    sep = SepConv2d(3, 12, 1)
    sep(x)

def test_InputProjection():
    x = torch.rand(2, 3, 64, 64)
    poj = InputProjection(3, 16)
    y = poj(x)
    print(y)


def test_ConvProjection():
    dim = 16
    head = 8
    x = torch.rand(2, 64, 64, dim)
    poj = ConvProjection(dim, head, dim // head)
    q, k, v = poj(x)
    print(q.shape, k.shape, v.shape)
    print(image_matrix_mul(q,k).shape)



def test_SELayer():
    dim = 16
    head = 8
    x = torch.rand(2, 64, 64, dim)
    se = SELayer(dim)
    y = se(x)
    print(y)


# 测试 LinearProjection 功能
def test_LinearProjection():
    dim = 16
    head = 8
    lp = LinearProjection(dim, head, dim // head)
    x = torch.rand(2, 64, 64, dim)
    lp(x)

def test_Windows():
    dim = 16
    head = 8
    x = torch.rand(2, 64, 64, dim)
    win = WindowAttention(dim, win_size=(64,64), num_heads=head)
    y = win(x)
    print(y.shape)

def test_Transformer():
    dim = 16
    head = 8
    x = torch.rand(3, 88, 72, dim)
    transformer = TransformerVision(dim, (88, 72), head, win_size=8, shift_size=4, mlp_ratio=4.)
    y = transformer(x)


def test_TransformerBlocks():
    dim = 16
    head = 8
    x = torch.rand(3, 88, 72, dim)
    transformer = TransformerBlocks(dim, (88, 72), depth=2,num_heads=head, win_size=8, mlp_ratio=4.)
    y = transformer(x)
    output_projection = OutputProjection(in_channel= dim, out_channel=3, kernel_size=3,
                                              stride=1)
    y = output_projection(y)
    print(y.shape)

def test_MTR(x):
    MTR = MTR_Model_1()
    MTR.cuda()
    MTR(x)


if __name__ == '__main__':
    # test_TransformerBlocks()

    train_path = 'data/train'
    loader = DataLoader(dataset=TrainDataLoader(train_path, {'patch_size':256}), batch_size=4)
    for input_img, target_img, file_name in loader:
        B, C, H, W = input_img.shape
        InputProjection()(input_img)
        input_img.cuda()
        test_MTR(input_img)
        print('success')