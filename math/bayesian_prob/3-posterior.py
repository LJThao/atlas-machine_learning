#!/usr/bin/env python3
"""Posterior Module - Based on 2-marginal.py"""
import numpy as np


def posterior(x, n, P, Pr):
    """Function that calculates the posterior probability for the
    various hypothetical probabilities of developing severe side
    effects given the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P

    """
    return (intersection(x, n, P, Pr) / marginal(x, n, P, Pr))


def marginal(x, n, P, Pr):
    """Function that calculates the marginal probability of obtaining
    the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of patients developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs about P

    """
    return np.sum(intersection(x, n, P, Pr))


def intersection(x, n, P, Pr):
    """Function that calculates the intersection of obtaining this
    data with the various hypothetical probabilities:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P

    """
    # validating all inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    # calculate the likelihood
    lh = likelihood(x, n, P)

    # calculate the intersection
    inter = lh * Pr

    return (inter)


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
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
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
