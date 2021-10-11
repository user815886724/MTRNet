# -*- coding: utf-8 -*
import os
import sys
import argparse
import options
import torch
from datetime import datetime
import random
import numpy as np
from utils import model_util
import torch.optim as optim
import losses
import time
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from utils import load_util, image_util, dataset_util
from timm.utils import NativeScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

########### dir path ###########
dir_name = os.path.dirname(os.path.abspath(__file__))

########### setting option ###########
option = options.Options(argparse.ArgumentParser(description='image denoising')).init().parse_args()

########### create dir ###########
log_dir = os.path.join(dir_name, option.save_dir,'log')
result_dir = os.path.join(dir_name, option.save_dir, 'results')
model_dir  = os.path.join(dir_name, option.save_dir, 'models')
# mkdir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_name = os.path.join(log_dir, datetime.now().isoformat().replace(':', '.') +'.txt')


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


######### Set GPUs ###########
if not option.use_gpu:
    pass
elif not torch.cuda.is_available():
    sys.exit('There are not GPU can used!!!')
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in device_ids])
    # 在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.benchmark = True



if __name__ == '__main__':
    train_dir = os.path.join(dir_name, option.train_dir)
    val_dir = option.val_dir


    ######### Model ###########
    model = model_util.get_architecture(option)
    with open(log_name, 'a') as f:
        f.write(str(option) + '\n\n')
        f.write(str(model) + '\n\n')

    ######### Optimizer ###########
    start_epoch = 1
    if option.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=option.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=option.weight_decay)
    elif option.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=option.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=option.weight_decay)
    else:
        raise Exception("Error optimizer...")

    ######### DataParallel ###########
    if option.use_gpu:
        model = torch.nn.DataParallel(model)
        model.cuda()

    ######### Resume ###########
    if option.resume:
        path_chk_rest = option.pretrain_weights
        model_util.load_checkpoint(model, path_chk_rest)
        start_epoch = model_util.load_start_epoch(path_chk_rest) + 1
        lr = model_util.load_optim(optimizer, path_chk_rest)

        for p in optimizer.param_groups:
            p['lr'] = lr
        warmup = False
        new_lr = lr
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, option.n_epoch - warmup_epochs+40,eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,after_scheduler=scheduler_cosine)
    scheduler.step()


    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss(opt=option)
    if option.use_gpu:
        criterion_char = criterion_char.cuda()
        # criterion_edge = criterion_edge.cuda()

    ######### DataLoaders ###########
    print('===> Loading datasets')
    # loader = DataLoader(dataset=DataLoaderTrain('data/train', input_dir='input', gt_dir='target', img_options={'patch_size': 256}), batch_size=4)
    # for input_img, target_img, file_name in loader:
    #     print(input_img)
    img_options_train = {'patch_size': option.train_ps}

    train_dataset = load_util.get_training_data(option.train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=option.batch_size, shuffle=True, num_workers=option.train_workers, pin_memory=True, drop_last=True)

    val_dataset = load_util.get_validation_data(option.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=option.batch_size, shuffle=True, num_workers=option.eval_workers, pin_memory=True, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

    eval_now = len(train_loader) // 4

    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, option.n_epoch+1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        model.train()
        psnr_val = []
        for i, data in enumerate(tqdm(train_loader)):
            # zero_grad
            for param in model.parameters():
                param.grad = None
            target_img = data[0]
            input_img = data[1]
            if option.use_gpu:
                target_img = target_img.cuda()
                input_img = input_img.cuda()

            if epoch > 5:
                target_img, input_img = dataset_util.MixUp_AUG().aug(target_img, input_img, option)
            if option.use_gpu:
                with torch.cuda.amp.autocast():
                    sam, restored = model(input_img)
            else:
                sam, restored = model(input_img)
            # Limit its range from 0 to 1, If use this for training, it will cause the gradient to disappear.
            sam = torch.clamp(sam, 0, 1)
            restored = torch.clamp(restored, 0, 1)

            # if epoch > 50:
            #     loss_char_sam = criterion_char(restored, target_img)
            # else:
            #     loss_char_sam = criterion_char(sam, target_img)
            loss_char_sam = criterion_char(sam, target_img)
            # loss_edge_sam = criterion_edge(sam, target_img)
            loss_char = criterion_char(restored, target_img)
            # loss_edge = criterion_edge(restored, target_img)
            # loss = loss_char + (0.05 * loss_edge) + loss_char_sam + (0.05*loss_edge_sam)
            loss = loss_char_sam + loss_char

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            #### Evaluation ####
            if not option.use_gpu:
                psnr_val.append(image_util.batch_PSNR(restored, target_img, False).item())

            if option.use_gpu and ( (i+1) % eval_now == 0 and i > 0):
                with torch.no_grad():
                    model.eval()
                    psnr_val_rgb = []
                    for ii, data_val in enumerate(tqdm(val_loader), 0):
                        target_img = data_val[0]
                        input_img = data_val[1]
                        target_img = target_img.cuda()
                        input_img = input_img.cuda()
                        filenames = data_val[2]
                        with torch.cuda.amp.autocast():
                            sam_restored, restored = model(input_img)
                        restored = torch.clamp(restored, 0, 1)
                        psnr_val_rgb.append(image_util.batch_PSNR(restored, target_img, False).item())
                    psnr_val_rgb = sum(psnr_val_rgb) / len_valset
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i
                        torch.save({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()
                                    }, os.path.join(model_dir, "model_best.pth"))
                    print(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                    with open(log_name, 'a') as f:
                        f.write(
                            "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                            % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                    model.train()
                    torch.cuda.empty_cache()

            scheduler.step()

        if not option.use_gpu:
            psnr_vals = sum(psnr_val)/ len_trainset
            print("------------------------------------------------------------------")
            print("Epoch: {} the psnr is {}".format(epoch, psnr_vals))
            with open(log_name, 'a') as f:
                f.write("------------------------------------------------------------------\n")
                f.write("Epoch: {} the psnr is {}".format(epoch, psnr_vals))
            if psnr_vals > best_psnr:
                best_psnr = psnr_vals
                print("It's Best psnr and it will be save")
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      epoch_loss,
                                                                                      scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(log_name, 'a') as f:
            f.write("------------------------------------------------------------------\n")
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}\n".format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      epoch_loss,
                                                                                      scheduler.get_lr()[0]))
            f.write("------------------------------------------------------------------\n")

        torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_latest.pth"))




