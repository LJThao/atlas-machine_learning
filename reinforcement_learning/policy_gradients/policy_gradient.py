#!/usr/bin/env python3
"""Simple Policy Function & Monte-Carlo Policy Gradient Function Module"""
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


def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy gradient based on a
    state and a weight matrix.

    state: matrix representing the current observation of the environment
    weight: matrix of random weight
    Return: the action and the gradient (in this order)

    """
    if state.ndim == 1:
        state = state[np.newaxis, :]
    # getting the action probs and sampling action
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs.ravel())
    # one-hot encoding of the action
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1
    # computing the gradient
    grad = state.T @ (one_hot - probs)

    return action, grad
