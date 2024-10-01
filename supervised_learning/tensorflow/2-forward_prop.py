#!/usr/bin/env python3
"""Forward Propagation Function"""


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the
    neural network:
    
    x = placeholder for the input data
    layer_sizes = list containing the number of nodes in
    each layer of the network
    activations = list containing the activation functions for
    each layer of the network
    
    """
    # importing my create_layer function
    create_layer = __import__('1-create_layer').create_layer
    # setting prediction to x
    prediction = x
    # forward propagation for each layer using zip to pair iterables
    for nodes, activation in zip(layer_sizes, activations):
        prediction = create_layer(prediction, nodes, activation)
    # return the prediction of the network in tensor form
    return prediction
