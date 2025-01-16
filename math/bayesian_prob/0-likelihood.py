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