#!/usr/bin/env python3
"""Gaussian Process Module"""
import numpy as np


class GaussianProcess():
    """Class that that represents a noiseless 1D Gaussian process:

    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """

    X_init is a numpy.ndarray of shape (t, 1) representing the inputs already
    ampled with the black-box function
    Y_init is a numpy.ndarray of shape (t, 1) representing the outputs of the
    black-box function for each input in X_init
    t is the number of initial samples
    l is the length parameter for the kernel
    sigma_f is the standard deviation given to the output of the black-box
    function

        """

    def kernel(self, X1, X2):
        """Function that calculates the covariance kernel matrix between
        two matrices:

        """
