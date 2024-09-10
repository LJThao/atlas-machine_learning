#!/usr/bin/env python3
"""Class Exponential that represents an exponential
distribution"""


class Exponential():
    """Class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize data and lambtha"""
        self.lambtha = float(lambtha)
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            lambtha = 1 / (sum(data) / len(data))
        elif lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
