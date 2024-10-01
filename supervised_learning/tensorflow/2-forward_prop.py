#!/usr/bin/env python3
"""Forward Propagation Function"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """returns the prediction of the network in tensor
    form:
    
    x = placeholder for the input data
    layer_sizes = list containing the number of nodes in
    each layer of the network
    activations = list containing the activation functions for
    each layer of the network
    
    """
    