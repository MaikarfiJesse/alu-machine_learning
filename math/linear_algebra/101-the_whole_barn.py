#!/usr/bin/env python3
"""A Function that adds two matrices"""


def add_matrices(mat1, mat2):
    """Add mat1 and mat2 element-wise"""
    if len(mat1) != len(mat2):
        return None
    if isinstance(mat1[0], (float, int)) != isinstance(mat2[0], (float, int)):
        return None
    if isinstance(mat1[0], (float, int)):
        return [x + y for x, y in zip(mat1, mat2)]
    ret = [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    if any(r is None for r in ret):
        return None
    return ret
