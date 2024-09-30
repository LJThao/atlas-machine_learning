#!/usr/bin/env python3
"""Layers Function"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """returns the tensor output of the layer:
    
    prev = the previous layer of the tensor output
    n = number of nodes in the layer to create
    activation = activation function that the layer should
    use
    
    """
    