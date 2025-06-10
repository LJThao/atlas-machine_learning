#!/usr/bin/env python3
"""Simple Policy Function Module"""
import numpy as np


def policy(matrix, weight):
    """Function that computes the policy with a weight of a matrix"""
    # raw scores
    scores = matrix @ weight
    # for numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)
    # exponentiation
    exp_scores = np.exp(scores)
    # normalizing
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs
