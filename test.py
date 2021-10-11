#### Evaluation ####
import argparse
import options
from utils import model_util, load_util, image_util
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


option = options.Options(argparse.ArgumentParser(description='image denoising')).init().parse_args()
val_dataset = load_util.get_validation_data(option.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=option.batch_size, shuffle=True, num_workers=option.eval_workers, pin_memory=True, drop_last=False)

epoch = 1

from net.MTRNet import Signal_MPRNet
model = model_util.get_architecture(option)
path_chk_rest = "model/models/model_latest.pth"
model_util.load_checkpoint(model, path_chk_rest)
model.eval()
psnr_val_rgb = []
len_valset = val_dataset.__len__()
for ii, data_val in enumerate(tqdm(val_loader), 0):
    target = data_val[0]
    input_ = data_val[1]

    with torch.no_grad():
        sam, restored = model(input_)
        psnr_val_rgb.append(image_util.batch_PSNR(restored, target, False).item())

psnr_val_rgb = sum(psnr_val_rgb)/ len_valset
print("[epoch %d PSNR: %.4f]" % (epoch, psnr_val_rgb))
