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
            self.lambtha = float(lambtha)
        else:
            # check the data and if not a list then raise
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # check the data for at least two then raise if it isn't
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the value of PMF for a given number of
        sucessess using the posson distribution

        PMF = Probability Mass Function
        
        """
        # setting approximation of constant
        e = 2.7182818285
        # check if k is less, then return
        if k < 0:
            return 0
        # check if k is an integer, convert if it isn't
        if not isinstance(k, int):
            k = int(k)
        # calulate the factorial of k!
        fact = 1
        if k > 1:
            for i in range(2, k + 1):
                fact *= i
        # calculate the # of successes using the poisson formula
        return (e ** (-self.lambtha)) * (self.lambtha ** k) / (fact)

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of
        successes

        CDF = Cumulative Distribution Function

        """
        if k < 0:
            return 0
        k = int(k)
        value = 0
        for i in range(k + 1):
            value += self.pmf(i)
        return value
