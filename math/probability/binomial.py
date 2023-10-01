#!/usr/bin/env python3
"""Create a class that represent a binomial distribution"""


def n_choose_k(n, k):
    """Calculate the number of k combinations drawn from n elements"""
    n_fac_minus_k_fac = 1
    for i in range(k+1, n+1):
        n_fac_minus_k_fac *= i
    n_minus_k_fac = 1
    for i in range(1, n - k + 1):
        n_minus_k_fac *= i
    return n_fac_minus_k_fac // n_minus_k_fac


class Binomial:
    """Class to store parameters and methods of the binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """create binomial distribution from data or set parameters
        using n and p."""
        if data is None:
            try:
                self.n = int(n)
                assert n > 0
            except (TypeError, AssertionError):
                raise ValueError('n must be a positive value')
            try:
                self.p = float(p)
                assert 0 < self.p < 1
            except (TypeError, AssertionError):
                raise ValueError('p must be greater than 0 and less than 1')
        elif type(data) != list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            mean = sum(data)/len(data)
            var = sum((d - mean)**2 for d in data)/len(data)
            p = 1.0 - var/mean
            n = sum(d/p for d in data)/len(data)
            if n % 1 < 0.5:
                n = int(n)
            else:
                n = int(n + 1)
            p = sum(d/n for d in data)/len(data)
            self.n = n
            self.p = float(p)

     def pmf(self, k):
        """Calculate the value of the pmf for a given number of "successes"""
        try:
            k = int(k)
            assert 0 <= k <= self.n
        except (TypeError, AssertionError):
            return 0
        return n_choose_k(self.n, k) * self.p**k * (1-self.p)**(self.n-k)
