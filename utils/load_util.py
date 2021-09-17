import os
from utils.data_util import DataLoaderTrain, DataLoaderVal, DataLoaderTest

def get_training_data(file_dir, img_option, input_dir='input', gt_dir='groundtruth'):
    assert os.path.exists(file_dir)
    return DataLoaderTrain(file_dir, input_dir=input_dir, gt_dir=gt_dir, img_options=img_option)

def get_validation_data(file_dir, input_dir='input', gt_dir='groundtruth'):
    assert os.path.exists(file_dir)
    return DataLoaderVal(file_dir, input_dir=input_dir, gt_dir=gt_dir)

def get_test_data(file_dir, input_dir='input'):
    assert os.path.exists(file_dir)
    return DataLoaderTest(file_dir, input_dir=input_dir)


