#!/usr/bin/env python3
"""Updated Train Model using Keras Module -
Based on 6-train.py - Learning Rate Decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Function to also train the model with learning rate decay:

    learning_rate_decay = a boolean that indicates whether learning rate
    decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha = the initial learning rate
    decay_rate = the decay rate

    """
    # add early stopping callback if valid data exists
    callbacks = (
        [K.callbacks.EarlyStopping(patience=patience)]
        if early_stopping and validation_data
        else []
    )

    # add learning rate decay call back
    if learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(
            lambda epoch: alpha / (1 + decay_rate * epoch),
            verbose=1
        ))

    # trains the model using keras fit function
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
