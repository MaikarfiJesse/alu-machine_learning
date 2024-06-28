#!/usr/bin/env python3
"""Write a function that performs K-means on a dataset"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    perfoms K-means on a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    k: positive integer - the no. of clusters
    iterations: +ve(int) - max no. of iterations perfomed

    return:
        - C: numpy.ndarray (k, d) containing the centroid
        for each cluster
        - clss: numpy.ndarray (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    clust = np.random.uniform(np.min(X, axis=0),
                          np.max(X, axis=0), (k, X.shape[1]))
    clss = np.zeros(X.shape[0])
    clust_prev = clust.copy()
    for i in range(iterations):
        dist = np.linalg.norm(X[:, None] - clust, axis=-1)
        clss = np.argmin(dist, axis=-1)
        for j in range(k):
            if len(X[clss == j]) == 0:
                clust[j] = np.random.uniform(np.min(X, axis=0),
                                             np.max(X, axis=0))
            else:
                clust[j] = np.mean(X[clss == j], axis=0)
        if (clust == clust_prev).all():
            return clust, clss
        clust_prev = clust.copy()
    return clust, clss
