#!/usr/bin/env python3
"""Updated Train Model using Keras Module -
Based on 5-train.py - Early Stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Function to also train the model using early stopping:

    early_stopping = a boolean that indicates whether early stopping
    should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience = the patience used for early stopping

    """
    # add early stopping callback if valid data exists
    callbacks = (
        [K.callbacks.EarlyStopping(patience=patience)]
        if early_stopping and validation_data
        else []
    )

    # trains the model using keras fit function -- added callbacks
    history_obj = network.fit(data,
                              labels,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callbacks)

    # returns the history object generated after training
    return (history_obj)
