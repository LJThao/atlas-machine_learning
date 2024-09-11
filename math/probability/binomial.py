#!/usr/bin/env python3
"""Class Binomial that represents a binomial distribution"""


class Binomial():
    """Class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initializing data, n, p"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculating n and p from data
            p = sum(data) / len(data)
            n = round(sum(data) / p)
            p = sum(data) / (n * len(data))

        # making sure n is an int and p is a float
        self.n = int(n)
        self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of
        sucesses"""

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of
        sucessess"""
