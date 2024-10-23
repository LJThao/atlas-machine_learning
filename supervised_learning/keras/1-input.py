#!/usr/bin/env python3
"""The Input Build Model with Keras Module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library:

    nx = the number of input features to the network
    layers = a list containing the number of nodes in each layer of the
    network
    activations = a list containing the activation functions used for
    each layer of the network
    lambtha = the L2 regularization parameter
    keep_prob = the probability that a node will be kept for dropout
    ** You are not allowed to use the Sequential class **

    """
    # setting X and inputs to the layer
    X = inputs = K.layers.Input(shape=(nx,))
    # set the L2 regularization
    regularizer = K.regularizers.l2(lambtha)

    # iterate through layers, add dense layer, dropout layers
    for layer in range(len(layers)):
        X = K.layers.Dense(units=layers[layer],
                           activation=activations[layer],
                           kernel_regularizer=regularizer)(X)
        if layer < len(layers) - 1:
            X = K.layers.Dropout(1 - keep_prob)(X)

    # set model
    model = K.Model(inputs=inputs, outputs=X)

    # returns the model
    return (model)
