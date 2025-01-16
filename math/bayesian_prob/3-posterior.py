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