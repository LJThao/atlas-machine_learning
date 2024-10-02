#!/usr/bin/env python3
"""Evaluate Function"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network:
    
    X = numpy.ndarray containing the input data to evaluate
    Y = numpy.ndarray containing the one-hot labels for X
    save_path = the location to load the model from
    
    """
    