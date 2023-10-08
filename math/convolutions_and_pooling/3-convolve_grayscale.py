#!/usr/bin/env python3
"""A Function that performs a convolution on images with channels"""
import numpy as np



def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Convolution of images using custom padding and stride"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        (phtop, pwlft), (phbtm, pwrt) = padding, padding
    elif padding == 'same':
        phtop, pwlft, phbtm, pwrt = (kh-1)//2, (kw-1//2), kh//2, kw//2
    elif padding == 'valid':
        phtop, pwlft, phbtm, pwrt = 0, 0, 0, 0
    out = np.zeros((
        m,
        (h + phtop + phbtm - kh)//sh + 1,
        (w + pwlft + pwrt - kw)//sw + 1,
    ))

    images = np.pad(
        images,
        (
            (0, 0),  # no padding img axis
            (phtop, phbtm),
            (pwlft, pwrt),
        ),
        mode='constant'
    )
    
    for row in range(0, images.shape[-2] - kh + 1, sh):
        for col in range(0, images.shape[-1] - kw + 1, sw):
            img = images[:, row:row+kh, col:col+kw]
            conv = (img * kernel).sum(axis=(-1, -2))
            out[:, row//sh, col//sw] = conv
    return out
