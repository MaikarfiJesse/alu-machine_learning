#!/usr/bin/env python3
"""A function that concatenates two matrices along a specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [arr[:] for arr in mat1] + [arr[:] for arr in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [arr1[:] + arr2[:] for arr1, arr2 in zip(mat1, mat2)]
    return None
