#!/usr/bin/env python3
"""A function that concatenates two matrices alog a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate mat1 and mat2 along the specified axis"""
    return np.concatenate([mat1, mat2], axis=axis)
