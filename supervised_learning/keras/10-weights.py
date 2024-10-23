#!/usr/bin/env python3
"""Save and Load Weights Module"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Function saves a model's weights:

    network = the model whose weights should be saved
    filename = the path of the file that the weights should be saved to
    save_format = the format in which the weights should be saved
    Returns: None

    """
    # save weights
    network.save_weights(filename,
                         save_format=save_format)

    return (None)


def load_weights(network, filename):
    """Function loads a model's weights:

    network = the model to which the weights should be loaded
    filename = the path of the file that the weights should be loaded from
    Returns: None

    """
    # load weights
    network.load_weights(filename)

    return (None)
