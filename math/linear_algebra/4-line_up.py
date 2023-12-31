#!/usr/bin/env python3
"""A function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Add two equal length arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [sum(pair) for pair in zip(arr1, arr2)]
