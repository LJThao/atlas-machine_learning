#!/usr/bin/env python3
"""Accuracy Function"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:

    y = placeholder for the labels of the input data
    y_pred = a tensor containing the network's predictions

    """
    # converting one-hot encoded true and predicted labels
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    # comparing the prediction to the labels
    prediction = tf.equal(y, y_pred)
    # casting boolean values to floats then calculates the mean
    tensor = tf.reduce_mean(tf.cast(prediction, tf.float32))
    # returns a tensor of the accuracy of a prediction
    return tensor
