#!/usr/bin/env python3
"""EM Module"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """Function that perform the expectation maximization for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    pi is priors for each cluster
    m is centroid means for each cluster
    S is covariance matrices for each cluster
    g is probabilities for each data point in each cluster
    log_l is log likelihood of the model
    Returns: pi, m, S, g, l, or None, None, None, None, None on
    failure

    """
    # validate all input
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    # init all parameters
    pi, m, S = initialize(X, k)
    g, log_l = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    # iterate range, update parameters, compute probabilities
    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, new_log_l = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        # print the log if verbose every 10
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(
                f"Log Likelihood after {i} iterations: "
                f"{new_log_l:.5f}"
            )

        # if log stops improving, loop breaks
        if abs(new_log_l - log_l) <= tol:
            break

        # update likelihood
        log_l = new_log_l

    return pi, m, S, g, log_l
