#!/usr/bin/env python3
"""Create a class that represents a poisson distribution"""
e = 2.7182818285


class Poisson:
    """Class to represent the Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize class with data and rate parameter 'lambtha'"""
        self.lambtha = float(lambtha)
        if data is None and self.lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Compute the PMF at k"""
        k = int(k)
        if k < 0:
            return 0
        k_fac = 1
        for i in range(1, k+1):
            k_fac *= i
        return (self.lambtha**k * e**-self.lambtha)/k_fac

    def cdf(self, k):
        """Compute CDF at k"""
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k+1))
