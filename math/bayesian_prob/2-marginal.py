#!/usr/bin/env python3
"""Marginal Probability Module - Based on 1-intersection.py"""
import numpy as np


def marginal(x, n, P, Pr):
    """Function that calculates the marginal probability of obtaining
    the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of patients developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs about P

    """
    