import torch
import numpy as np

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

if __name__ == '__main__':
    window_size=2
    shift_size=1
    x = torch.randn(1,8,8,3)
    #x.shape
    H = 8
    W = 8

    # calculate attention mask for SW-MSA
    Hp = int(np.ceil(H / window_size)) * window_size
    Wp = int(np.ceil(W / window_size)) * window_size
    img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
    # print("img_mask:",img_mask)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    # print("h_slices:",h_slices)
    # print("w_slices:",w_slices)

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            print("img_mask", img_mask)
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    # print("mask_windows:",mask_windows)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    # print("mask_windows:",mask_windows)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # print("mask_windows:",attn_mask)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    # print("mask_windows:",attn_mask)