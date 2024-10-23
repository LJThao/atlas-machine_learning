#!/usr/bin/env python3
"""The Build Model with Keras Module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library:

    nx = the number of input features to the network
    layers = a list containing the number of nodes in each layer of the
    network
    activations = a list containing the activation functions used for each
    layer of the network
    lambtha = the L2 regularization parameter
    keep_prob = the probability that a node will be kept for dropout
    """
    # init the sequential model
    model = K.models.Sequential()
    # add the input dense layer
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha),
                             input_shape=(nx,)))
    # iterate through the layers
    for layer in range(1, len(layers)):
        # adding dropout to the hidden layers
        model.add(K.layers.Dropout(rate=1-keep_prob))
        # then add the dense layer for current layer
        model.add(K.layers.Dense(
            units=layers[layer],
            activation=activations[layer],
            kernel_regularizer=K.regularizers.l2(lambtha))
            )

    # returning the model
    return (model)
