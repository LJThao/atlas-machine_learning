#!/usr/bin/env python3
"""Accuracy Function"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:
    
    y = placeholder for the labels of the input data
    y_pred = a tensor containing the network's predictions
    
    """
