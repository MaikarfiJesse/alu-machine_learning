#!/usr/bin/env python3
"""Create a class MultiNormal that represents a Multivariate Normal Distribution
"""


import numpy as np


class MultiNormal:
    """A class that represents Multivariate Normal Distribution

    class constructor:
        def __init__(self, data)

    public instance variables:
        mean [numpy.ndarray of shape (d, 1)]:
            contains the mean of data
        cov [numpy.ndarray of shape (d, d)]:
            contains the covariance matrix of data

    public instance method:
        def pdf(self, x):
            calculates the PDF at a data point
    """
    def __init__(self, data):
        """
        Class constructor

        parameters:
            data [numpy.ndarray of shape (d, n)]:
                contains the data set
                d: number of dimensions in each data point
                n: number of data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov
