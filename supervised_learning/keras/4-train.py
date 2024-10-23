#!/usr/bin/env python3
"""Train Model using Keras Module"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """Function that trains a model using mini-batch gradient descent:

    network = the model to train
    data = a numpy.ndarray of shape (m, nx) containing the input data
    labels = a one-hot numpy.ndarray of shape (m, classes) containing the
    labels of data
    batch_size = the size of the batch used for mini-batch gradient descent
    epochs = the number of passes through data for mini-batch gradient descent
    verbose = a boolean that determines if output should be printed during
    training
    shuffle = a boolean that determines whether to shuffle the batches every
    epoch.
    Normally, it is a good idea to shuffle, but for reproducibility, we have
    chosen to set the default to False.
    Returns: the History object generated after training the model

    """
    
