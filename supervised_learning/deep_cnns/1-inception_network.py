#!/usr/bin/env python3
"""Inception Network Module"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network as described in Going
    Deeper with Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use a
    rectified linear activation (ReLU)
    You may use inception_block = __import__('0-inception_block').
    inception_block
    Returns: the keras model

    """
