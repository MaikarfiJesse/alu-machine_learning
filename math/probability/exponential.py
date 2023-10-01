#!/usr/bin/env python3
"""Create a class that represents an exponential distribution"""
e = 2.7182818285


class Exponential:
    """Class to represent the exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize an exponential from data or lambtha if data is missing"""
        self.lambtha = float(lambtha)
        if data is None and self.lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = (sum(data)/len(data))**-1

    def pdf(self, x):
        """Calculate the value of the PDF for a given time period"""
        if x < 0:
            return 0
        return self.lambtha*(e**(-x*self.lambtha))

     def cdf(self, x):
        """Calculate the value of the CDF for a given time period"""
        if x < 0:
            return 0
        return 1 - e**(-x*self.lambtha)
