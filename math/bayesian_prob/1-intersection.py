#!/usr/bin/env python3
"""Intersection Module - Based on 0-likelihood.py"""
import numpy as np


def intersection(x, n, P, Pr):
    """Function that calculates the intersection of obtaining this
    data with the various hypothetical probabilities:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P

    """
    
