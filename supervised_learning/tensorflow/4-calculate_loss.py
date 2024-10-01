#!/usr/bin/env python3
"""Loss Function"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss
    of a prediction
    
    y = placeholder for the labels of the input data
    y_pred = tensor containing the network's predictions
    
    """
    