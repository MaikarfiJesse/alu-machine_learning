#!/usr/bin/env python3
""" Function that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """Concatenate two arrays into a new array"""
    arr3 = arr1[:]
    arr3.extend(arr2[:])
    return arr3
