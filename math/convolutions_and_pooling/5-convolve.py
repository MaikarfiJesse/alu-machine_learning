#!/usr/bin/env python3
"""A function that performs a convolution on images using multiple kernels"""
import numpy as np



def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Compute multiple convolutions of images using custom padding and stride"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        top, lft = btm, rt = padding
    elif padding == 'same':
        top, lft, btm, rt = (kh-1)//2, (kw-1//2), kh//2, kw//2
    elif padding == 'valid':
        top =lft = btm = rt = 0
    out = np.zeros((
        m,
        (h + top + btm - kh)//sh + 1,
        (w + lft + rt - kw)//sw + 1,
        nc,
    ))

    images = np.pad(
        images,
        (
            (0, 0),  # no padding img axis
            (top, btm),
            (lft, rt),
            (0, 0),  # no padding ch axis
        ),
        mode='constant'
    )
    
    for row in range(0, images.shape[-3] - kh + 1, sh):
        for col in range(0, images.shape[-2] - kw + 1, sw):
            for k in range(nc):
                img = images[:, row:row+kh, col:col+kw, :]
                conv = (img * kernels[:, :, :, k]).sum(axis=(-1, -2, -3))
                out[:, row//sh, col//sw, k] = conv
    return out
