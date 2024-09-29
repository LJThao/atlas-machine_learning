#!/usr/bin/env python3
"""Placeholder Function"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """returns two placeholders x and y
    for the neural network

    nx = number of feature columns in the data
    classes = number of classes in the classifier

    """
    # x = placeholder for the input data to the neural network
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, nx), name="x")
    # y = placeholder for the one-hot labels for the input data
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
