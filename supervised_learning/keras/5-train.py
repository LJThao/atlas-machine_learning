#!/usr/bin/env python3
"""Updated Train Model using Keras Module ---
Based on 4-train.py"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):

    """Function to also analyze validation data:

    validation_data = the data to validate the model with, if not None

    """
    