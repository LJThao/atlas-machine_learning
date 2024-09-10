#!/usr/bin/env python3
"""Class Normal that represents a normal distribution"""


class Normal():
    """Class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initiaizing data, mean, stddev"""
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = (sum(data)) / (len(data))
            self.stddev = (sum((x - self.mean) ** 2 for x in data)) ** (1/2)

    def z_score(self, x):
        """Calculates the z_score of a given x-value"""


    def x_value(self, z):
        """Calculates x-value of a given z-score"""
