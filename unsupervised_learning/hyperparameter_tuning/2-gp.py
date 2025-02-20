#!/usr/bin/env python3
"""Gaussian Process Module - Based on 1-gp.py"""
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

    def predict(self, X_s):
        """Function that predicts the mean and standard deviation of points
        in a Gaussian process:

        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
         -> s is the number of sample points

        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing the mean for each
        point in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing the variance
        for each point in X_s, respectively

        """
        # calculate covariance
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        # calculate inverses of the covariance mat
        K_inv = np.linalg.inv(self.K)

        # calculate the mean of predictive dist
        mu = K_s.T @ K_inv @ self.Y

        # calculate the variance of predictive dist
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)

        # return the mu and sigma
        return mu.flatten(), sigma

    def update(self, X_new, Y_new):
        """Function that updates a Gaussian Process

        X_new is a numpy.ndarray of shape (1,) that represents the new
        sample point
        Y_new is a numpy.ndarray of shape (1,) that represents the new
        sample function value

        """
        # appending the new sample to the dataset
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))

        # update covariance mat with the new dataset
        self.K = self.kernel(self.X, self.X)
