# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import os
from collections import OrderedDict


# 固定部分参数进行网络训练
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

# 接触固定参数进行网络训练
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

# 检查所有参数是否固定
def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

# 缓存模型中断位置
def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

# 加载模型中断位置
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    # noinspection PyBroadException
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_multi_gpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

# 加载上次模型缓存截止的代数
def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

# 加载优化器
def load_optim(optimizer, weights):
    lr = None
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

# 获得模型的网络结构
def get_architecture(opt):
    from net.MTRNet import MPTR_SuperviseNet, Signal_MPRNet, MPTR_SuperviseNet_New
    arch = opt.arch
    if arch == 'MPTR_SuperviseNet':
        model = MPTR_SuperviseNet(image_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size, token_mlp=opt.token_mlp)
    elif arch == 'Signal_MPRNet':
        model = Signal_MPRNet(embed_dim=opt.embed_dim)
    elif arch == 'MPTR_SuperviseNet_New':
        model = MPTR_SuperviseNet_New(image_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                                  token_mlp=opt.token_mlp)
    else:
        raise Exception("Arch error!")
    return model