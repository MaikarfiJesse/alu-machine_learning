#!/usr/bin/env python3
"""A function that performs a same convolution on grayscale images"""
import numpy as np



def convolve_grayscale_same(images, kernel):
    """Compute convolution of images on grayscale"""
    kh, kw = kernel.shape[-2:]
    out = np.zeros_like(images)

    # for odd length kernels, pad equally on both sides
    # for even length kernels pad one extra on the right
    images = np.pad(
        images,
        (
            (0, 0),  # no padding img axis
            ((kh-1)//2, kh//2),
            ((kw-1)//2, kw//2),
        ),
        mode='constant'
    )
    
    for row in range(out.shape[-2] - kernel.shape[0] + 1):
        for col in range(out.shape[-1] - kernel.shape[1] + 1):
            img = images[:, row:row+kernel.shape[0], col:col+kernel.shape[1]]
            conv = (img * kernel).sum(axis=(-1, -2))
            out[:, row, col] = conv
    return out
