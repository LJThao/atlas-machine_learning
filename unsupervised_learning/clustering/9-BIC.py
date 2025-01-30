#!/usr/bin/env python3
"""BIC Module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """Function that finds the best number of clusters for
    a GMM using the Bayesian Information Criterion:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
    clusters to check for (inclusive)
    pi is priors for each cluster
    m is centroid means for each cluster
    S is covariance matrices for each cluster
    l is log likelihood for each cluster size tested
    b is the BIC value for each cluster size tested
    p is the number of parameters
    n is the number of data points
    best_k is the best value for k based on its BIC
    best_result is the tuple containing pi, m, S
    Returns: best_k, best_result, l, b, or None, None, None, None
    on failure

    """
    # validate all parameters
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin < 1 or
            (kmax is not None and (not isinstance(kmax, int) or
                                   kmax < kmin)) or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    # init lists
    best_k_values = []
    best_results = []
    l = []
    b = []

    # run expectation max for each k in range
    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(X, k,
                                                      iterations,
                                                      tol, verbose)

        # store values for each k
        best_k_values.append(k)
        best_results.append((pi, m, S))
        l.append(log_l)

        # compute parameters p
        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)

        # compute BIC
        bic = p * np.log(n) - (2 * log_l)
        b.append(bic)

    # convert lists to arrays
    b = np.array(b)
    l = np.array(l)

    # finding best k
    best_idx = np.argmin(b)
    best_k = best_k_values[best_idx]
    best_result = best_results[best_idx]

    return best_k, best_result, l, b
