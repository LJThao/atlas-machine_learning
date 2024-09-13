#!/usr/bin/env python3
"""Class Exponential that represents an exponential
distribution"""


class Exponential():
    """Class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize data and lambtha"""
        # checks data to see if its a list, if not raise
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # check for two data or more, if not raise
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate lambtha
            lambtha = 1 / (sum(data) / len(data))
        elif lambtha <= 0:
            # checks if lambtha is less than or equal to 0, if it is raise
            raise ValueError("lambtha must be a positive value")
        # turning lambtha to a float
        self.lambtha = float(lambtha)

    def pdf(self, x):
        """Calculates the value of the PDF for a given time

        PDF = Probability Density Function

        """
        # setting e and assigning lamb
        e = 2.7182818285
        lamb = self.lambtha
        # check if x is less than 0, if so return
        if x < 0:
            return 0
        # calculate the value using the formula
        return (lamb * (e ** (-lamb * x)))

    def cdf(self, x):
        """Calculates the value of the CDF for a given time

        CDF = Cumulative Distribuion Function

        """
        # setting approximation for e and assigning lamb
        e = 2.7182818285
        lamb = self.lambtha
        # check if x is less than 0, if so return
        if x < 0:
            return 0
        # calculate the value using the formula
        return (1 - (e ** (-lamb * x)))
