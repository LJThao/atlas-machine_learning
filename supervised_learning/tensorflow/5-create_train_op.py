#!/usr/bin/env python3
"""Train Op Function"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network:
    
    loss = the loss of the network's prediction
    alpha = the learning rate
    
    """
    # create optimizer using the gradient descent with alpha
    opt_op = tf.compat.v1.train.GradientDescentOptimizer(alpha)
    # return operation that trains the network after minimizing loss
    return opt_op.minimize(loss)
