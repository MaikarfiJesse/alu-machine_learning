#!/usr/bin/env python3
"""A function that performs a convolution on images witj channels"""
import numpy as np



def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolution of images using custom padding and stride"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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
            img = images[:, row:row+kh, col:col+kw, :]
            conv = (img * kernel).sum(axis=(-1, -2, -3))
            out[:, row//sh, col//sw] = conv
    return out
