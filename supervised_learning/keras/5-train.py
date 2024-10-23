#!/usr/bin/env python3
"""Updated Train Model using Keras Module ---
Based on 4-train.py"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):

    """Function to also analyze validation data:

    validation_data = the data to validate the model with, if not None

    """
    # trains the model using keras fit function -- now added validation data
    history_obj = network.fit(data,
                              labels,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data)

    # returns the history object generated after training
    return (history_obj)
