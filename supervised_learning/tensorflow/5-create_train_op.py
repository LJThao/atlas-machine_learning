#!/usr/bin/env python3
"""Train Op Function"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network:
    
    loss = the loss of the network's prediction
    alpha = the learning rate
    
    """
    