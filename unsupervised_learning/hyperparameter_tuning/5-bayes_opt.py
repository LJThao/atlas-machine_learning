#!/usr/bin/env python3
"""Bayesian Optimization Module - Based on 4-bayes_opt.py"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Class that performs Bayesian optimization on a noiseless 1D
    Gaussian process:"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Init Bayesian Optimization:

        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the
        inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the
        outputs of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of
        the space in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed
        during acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)

        """
        # storing the black box function to be optimized
        self.f = f

        # init a GP with given sampled data and kernel parameters
        self.gp = GP(X_init, Y_init, l, sigma_f)

        b_min, b_max = bounds

        # acquisition sample points
        self.X_s = np.linspace(b_min, b_max, ac_samples).reshape(-1, 1)

        # exploration-exploration factor
        self.xsi = xsi

        # optimization mode
        self.minimize = minimize

    def acquisition(self):
        """Function that calculates the next best sample location:

        Returns: X_next, EI
        X_next is a numpy.ndarray of shape (1,) representing the next
        best sample point
        EI is a numpy.ndarray of shape (ac_samples,) containing the
        expected improvement of each potential sample

        """
        # mean and std for all sampling prediction points and reshaping
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1)

        # find the best observed value
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        # expected improvement calculation
        Z = np.zeros_like(imp)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]

        EI = np.zeros_like(imp)
        EI[mask] = (
            imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
        )

        # select the best sample point
        X_next = self.X_s[np.argmax(EI)]

        # return next best sample point, containing EI of each sample
        return X_next, EI

    def optimize(self, iterations=100):
        """Function that optimizes the black-box function:

        iterations is the maximum number of iterations to perform
        Returns: X_opt, Y_opt
        X_opt is a numpy.ndarray of shape (1,) representing the
        optimal point
        Y_opt is a numpy.ndarray of shape (1,) representing the
        optimal function value

        """
        # iterating over iterations without using the index
        for _ in range(iterations):
            # getting the next best sample point
            X_next, _ = self.acquisition()

            # stop if the point has already been sampled
            if np.any(np.isclose(self.gp.X, X_next)):
                break

            # evaluating the function at X_next and updating GP
            Y_next = self.f(X_next).reshape(-1, 1)
            self.gp.update(X_next.reshape(-1, 1), Y_next)

        # finding the best observed input point
        X_opt = (
            self.gp.X[np.argmin(self.gp.Y)]
            if self.minimize else self.gp.X[np.argmax(self.gp.Y)]
        )
        # finding the best observed function value
        Y_opt = (
            np.min(self.gp.Y)
            if self.minimize else np.max(self.gp.Y)
        )
        Y_opt = np.array([Y_opt])

        # return the opt point and the opt function value
        return X_opt, Y_opt
