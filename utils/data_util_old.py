import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
import torch

class TrainDataLoader(Dataset):
    def __init__(self, train_path, img_options=None):
        super(TrainDataLoader, self).__init__()
        input_files = sorted(os.listdir(os.path.join(train_path, 'input')))
        target_files = sorted(os.listdir(os.path.join(train_path, 'target')))

        self.input_filenames = [os.path.join(train_path, 'input', x) for x in input_files if is_image_file(x)]
        self.target_filenames = [os.path.join(train_path, 'target', x) for x in target_files if is_image_file(x)]

        self.img_options = img_options
        self.file_size = len(self.target_filenames)

        self.patch_size = self.img_options['patch_size']

    def __len__(self):
        return self.file_size


    def __getitem__(self, index):
        index_ = index % self.file_size
        patch_size = self.patch_size

        input_path = self.input_filenames[index_]
        target_path = self.target_filenames[index_]

        input_img = Image.open(input_path)
        target_img = Image.open(target_path)

        w, h = target_img.size
        padding_w = patch_size - w if w < patch_size else 0
        padding_h = patch_size - h if h < patch_size else 0

        # Reflect Pad in case image is smaller than patch_size
        if padding_w != 0 or padding_h != 0:
            input_img = TF.pad(input_img, [0, 0, padding_w, padding_h], padding_mode='reflect')
            target_img = TF.pad(target_img, [0, 0, padding_w, padding_h], padding_mode='reflect')

        # 随机对训练数据进行处理
        aug = random.randint(0, 2)
        # 图像亮度与对比度的调整（对输入图像进行gamma校正，也称作幂律变换）
        if aug == 1:
            input_img = TF.adjust_gamma(input_img, 1)
            target_img = TF.adjust_gamma(target_img, 1)

        aug = random.randint(0, 2)
        # 调整照片饱和度
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            input_img = TF.adjust_saturation(input_img, sat_factor)
            target_img = TF.adjust_saturation(target_img, sat_factor)

        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)

        hh, ww = target_img.shape[1], target_img.shape[2]

        rr = random.randint(0, hh - patch_size)
        cc = random.randint(0, ww - patch_size)
        aug = random.randint(0, 8)

        # Crop patch
        # 如果所要求的 patch_size 小于图片大小则只截取那一部分大小
        input_img = input_img[:, rr:rr + patch_size, cc:cc + patch_size]
        target_img = target_img[:, rr:rr + patch_size, cc:cc + patch_size]

        # 随机增加图像的复杂度，例如将图像翻转、旋转、翻转且旋转
        if aug == 1:
            input_img = input_img.flip(1)
            target_img = target_img.flip(1)
        elif aug == 2:
            input_img = input_img.flip(2)
            target_img = target_img.flip(2)
        elif aug == 3:
            input_img = torch.rot90(input_img, dims=(1, 2))
            target_img = torch.rot90(target_img, dims=(1, 2))
        elif aug == 4:
            input_img = torch.rot90(input_img, dims=(1, 2), k=2)
            target_img = torch.rot90(target_img, dims=(1, 2), k=2)
        elif aug == 5:
            input_img = torch.rot90(input_img, dims=(1, 2), k=3)
            target_img = torch.rot90(target_img, dims=(1, 2), k=3)
        elif aug == 6:
            input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
            target_img = torch.rot90(target_img.flip(1), dims=(1, 2))
        elif aug == 7:
            input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
            target_img = torch.rot90(target_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(target_path)[-1])[0]
        return input_img, target_img, filename


def get_window_size(img_size, patch_num):
    assert img_size > patch_num, "windows size must more than patch_num"
    assert img_size % patch_num == 0, 'img_size must be divided by patch_num'
    win_size = img_size // patch_num
    return win_size

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])