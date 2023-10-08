#!/usr/bin/env python3
"""A function that performs pooling on images"""
import numpy as np



def pool(images, kernel_shape, stride, mode='max'):
    """Convolution of images using custom padding and stride"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out = np.zeros((
        m,
        (h - kh)//sh + 1,
        (w - kw)//sw + 1,
        c,
    ))

    if mode == 'max':
        fn = np.max
    elif mode == 'avg':
        fn = np.mean
    
    for row in range(0, images.shape[-3] - kh + 1, sh):
        for col in range(0, images.shape[-2] - kw + 1, sw):
            img = images[:, row:row+kh, col:col+kw]
            poolled = fn(img, axis=(-2, -3))
            out[:, row//sh, col//sw] = poolled
    return out
