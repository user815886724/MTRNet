# -*- coding: utf-8 -*
import os
from torch.utils.data import Dataset
from utils.image_util import load_img
import random
import numpy as np
import torch
from utils.file_util import is_png_file
from utils.dataset_util import Augment_RGB_torch

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, train_path, input_dir, gt_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        input_files = sorted(os.listdir(os.path.join(train_path, input_dir)))
        target_files = sorted(os.listdir(os.path.join(train_path, gt_dir)))

        self.input_filenames = [os.path.join(train_path, input_dir, x) for x in input_files if is_image_file(x)]
        self.target_filenames = [os.path.join(train_path, gt_dir, x) for x in target_files if is_image_file(x)]

        self.img_options = img_options
        self.file_size = len(self.target_filenames)


    def __len__(self):
        return self.file_size


    def __getitem__(self, index):
        index_ = index % self.file_size

        input_file = torch.from_numpy(np.float32(load_img(self.input_filenames[index_])))
        target_file = torch.from_numpy(np.float32(load_img(self.target_filenames[index_])))

        input_file = input_file.permute(2, 0, 1)
        target_file = target_file.permute(2, 0, 1)

        input_filename = os.path.split(self.input_filenames[index_])[-1]
        target_filename = os.path.split(self.target_filenames[index_])[-1]

        # Crop Input and Target
        patch_size = self.img_options['patch_size']
        C, H, W = input_file.shape

        if H - patch_size == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - patch_size)
            c = np.random.randint(0, W - patch_size)

        input_file = input_file[:, r:r+patch_size, c:c+patch_size]
        target_file = target_file[:, r:r+patch_size, c:c+patch_size]
        apply_trans = transforms_aug[random.getrandbits(3)]
        input_file = getattr(augment, apply_trans)(input_file)
        target_file = getattr(augment, apply_trans)(target_file)

        return  target_file, input_file, target_filename, input_filename



def get_window_size(img_size, patch_num):
    assert img_size > patch_num, "windows size must more than patch_num"
    assert img_size % patch_num == 0, 'img_size must be divided by patch_num'
    win_size = img_size // patch_num
    return win_size

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, file_dir, input_dir='input', gt_dir='groundtruth'):
        super(DataLoaderVal, self).__init__()
        input_files = sorted(os.listdir(os.path.join(file_dir, input_dir)))
        target_files = sorted(os.listdir(os.path.join(file_dir, gt_dir)))
        self.input_filenames = [os.path.join(file_dir, input_dir, x) for x in input_files if is_png_file(x)]
        self.target_filenames = [os.path.join(file_dir, gt_dir, x) for x in target_files if is_png_file(x)]
        self.file_size = len(input_files)

    def __len__(self):
        return self.file_size

    def __getitem__(self, index):
        index_ = index % self.file_size
        input_file = torch.from_numpy(np.float32(load_img(self.input_filenames[index_])))
        target_file = torch.from_numpy(np.float32(load_img(self.target_filenames[index_])))
        input_filename = os.path.split(self.input_filenames[index_])[-1]
        target_filename = os.path.split(self.target_filenames[index_])[-1]
        input_file = input_file.permute(2, 0, 1)
        target_file = target_file.permute(2, 0, 1)
        return target_file, input_file, target_filename, input_filename


##################################################################################################
class DataLoaderTest(Dataset):
    def __init__(self, file_dir, input_dir='input'):
        super(DataLoaderTest, self).__init__()
        input_files = sorted(os.listdir(os.path.join(file_dir, input_dir)))
        self.input_filenames = [os.path.join(file_dir, input_dir, x) for x in input_files if is_png_file(x)]
        self.file_size = len(self.input_filenames)

    def __len__(self):
        return self.file_size


    def __getitem__(self, index):
        index_ = index % self.file_size
        input_file = torch.from_numpy(np.float32(load_img(self.input_filenames[index_])))
        input_filename = os.path.split(self.input_filenames[index_])[-1]
        input_file = input_file.permute(2, 0, 1)
        return input_file, input_filename




