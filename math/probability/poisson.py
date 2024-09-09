#!/usr/bin/env python3
"""Class Poission that represents a distribution"""


class Poisson():
    """Class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """Initializing data and lambtha"""
        # if the data is not given
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            # check the data and if not a list then raise
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # check the data for at least two then raise if it isn't
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha
            self.lambtha = float(sum(data) / len(data))
