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
        self.X, self.Y, self.l, self.sigma_f = X_init, Y_init, l, sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Function that calculates the covariance kernel matrix between
        two matrices"""
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + (
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        )
        return self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)
