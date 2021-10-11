import argparse
import options
from utils import load_util, model_util, image_util
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from skimage import img_as_ubyte
import os
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


option = options.Options(argparse.ArgumentParser(description='image denoising')).init().parse_args()
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, option.save_dir,'log')

log_name = os.path.join(log_dir, 'test_' + datetime.now().isoformat().replace(':', '.') +'.txt')
val_dataset = load_util.get_validation_data(option.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=option.batch_size, shuffle=True, num_workers=option.eval_workers, pin_memory=True, drop_last=False)

model = model_util.get_architecture(option)
if option.use_gpu:
    model = torch.nn.DataParallel(model)
    model.cuda()
path_chk_rest = "model/models/model_best_5_38_812.pth"
model_util.load_checkpoint(model, path_chk_rest)
model.eval()
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    best_psnr = 0
    best_ssim = 0
    for i, data_val in enumerate(tqdm(val_loader), 0):
        target_img = data_val[0]
        input_img = data_val[1]
        filenames = data_val[2]


        if option.use_gpu:
            target_img = target_img.cuda()
            input_img = input_img.cuda()
        sam, restored_img = model(input_img)
        restored = torch.clamp(restored_img, 0, 1)

        # psnr_val_rgb.append(image_util.batch_PSNR(restored, target_img).item())

        restored = restored.permute(0, 2, 3, 1)
        if option.use_gpu:
            restored = restored.cuda()
        if option.save_images:
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                tar_img = img_as_ubyte(target_img[batch].permute(1,2,0))
                p_loss = psnr_loss(restored_img, tar_img)
                s_loss = ssim_loss(restored_img, tar_img, multichannel=True)
                if best_psnr < p_loss:
                    best_psnr = p_loss
                if best_ssim < s_loss:
                    best_ssim = s_loss
                psnr_val_rgb.append(p_loss)
                ssim_val_rgb.append(s_loss)
                image_util.save_img((os.path.join(option.result_dir, filenames[batch])), restored_img)

psnr_val_rgb = sum(psnr_val_rgb)/len(val_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(val_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb, ssim_val_rgb))
with open(log_name, 'a') as f:
    f.write("PSNR: {:.4f}, SSIM: {:.4f}, Best_PSNR: {:.4f}, Best_SSIM: {:.4f}".format(
        psnr_val_rgb, ssim_val_rgb, best_psnr, best_ssim))


