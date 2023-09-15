#!/usr/bin/env python3
"""A function that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Slice matrix along specified axes"""
    return matrix[
        tuple([slice(*axes.get(ax, (None, None)))
               for ax in range(max(axes)+1)])]
