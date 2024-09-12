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
            self.stddev = (((sum((x - self.mean) ** 2 for x in data))
                            / len(data))) ** (1/2)

    def z_score(self, x):
        """Calculates the z_score of a given x-value:

        z-score formula: z = x - μ / σ
                         z = x-value - mean / standard deviation

        """
        return ((x - self.mean) / (self.stddev))

    def x_value(self, z):
        """Calculates x-value of a given z-score:

        x-value formula: x = standard deviation + mean

        """
        return ((z * self.stddev) + (self.mean))

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value

        PDF = Probability Density Function

        """
        # setting up approximations for pi and e
        pi = 3.1415926536
        e = 2.7182818285
        std = self.stddev
        mean = self.mean

        # calculating the value of PDF
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        coef = 1 / (std * (2 * pi) ** (1/2))
        return (coef) * (e ** exponent)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value

        CDF = Cumulative Distribution Function
        
        """
        pi = 3.1415926536
        mean = self.mean
        std = self.stddev

        # calculating scaled x for erf function
        x_scaled = (x - mean) / (std * 2 ** 0.5)

        # calculating erf function 
        erf = (2 / (pi ** 0.5)) * (x_scaled - (x_scaled**3) / 3 +
                                   (x_scaled**5) / 10 -
                                   (x_scaled**7) / 42 +
                                   (x_scaled**9) / 216)

        # calculating the cdf with the erf function
        return (0.5 * (1 + erf))
