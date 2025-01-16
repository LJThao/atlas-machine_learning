#!/usr/bin/env python3
"""Likelihood Module"""
import numpy as np


def likelihood(x, n, P):
    """Function that calculates the likelihood of obtaining this
    data given various hypothetical probabilities of developing
    severe side effects:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects

    """
    # validating all inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # computing the likelihood
    fact = np.math.factorial
    bino_coef = fact(n) / (fact(x) * fact(n - x))
    likeli_val = bino_coef * (P ** x) * ((1 - P) ** (n - x))

    return (likeli_val)