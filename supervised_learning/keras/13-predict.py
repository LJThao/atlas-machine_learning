#!/usr/bin/env python3
"""The Predict Module"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network:

    network = the network model to make the prediction with
    data = the input data to make the prediction with
    verbose = a boolean that determines if output should be printed
    during the prediction process
    Returns: the prediction for the data

    """
    # makes predictions on the input data using the model
    prediction = network.predict(data,
                                 verbose=verbose)

    # returns the prediction
    return (prediction)
