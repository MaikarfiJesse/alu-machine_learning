#!/usr/bin/env python3
"""Create a class that represents a normal distribution"""
π = 3.1415926536
e = 2.7182818285


def erf(x, n=4):
    """input the Maclaurin approximation of erf"""
    return 2/π**0.5*sum([x, -1/3*x**3, 1/10*x**5, -1/42*x**7, 1/216*x**9])


class Normal:
    """Class to represent the normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Normal distribution using data or with mean and stddev
        if data is missing"""
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None and self.stddev <= 0:
            raise ValueError('stddev must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data)/len(data)
            self.stddev = (sum((self.mean - x)**2 for x in data)/len(data))**.5
