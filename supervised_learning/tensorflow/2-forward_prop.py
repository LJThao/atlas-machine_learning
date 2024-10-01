#!/usr/bin/env python3
"""Forward Propagation Function"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """returns the prediction of the network in tensor
    form"""