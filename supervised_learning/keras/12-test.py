#!/usr/bin/env python3
"""The Test Model with Keras Module"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network:

    network = the network model to test
    data = the input data to test the model with
    labels are the correct one-hot labels of data
    verbose = a boolean that determines if output should be printed during the testing process
    Returns: the loss and accuracy of the model with the testing data, respectively

    """
    # test the model of the input data and one-hot labels
    test_loss_accuracy = network.evaluate(data,
                                          labels,
                                          verbose=verbose)
    
    # return loss and accuracy of the model with the testing data
    return (test_loss_accuracy)
