#!/usr/bin/env python3
"""Loss Function"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss
    of a prediction

    y = placeholder for the labels of the input data
    y_pred = tensor containing the network's predictions

    """
    # calculating the soft max cross-entropy loss
    tensor = tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
    # returning a tensor containing the network's predictions
    return tensor
