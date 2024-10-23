#!/usr/bin/env python3
"""Updated Train Model using Keras Module -
Based on 7-train.py - Save Best"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Function to also save the best iteration of the model:

    save_best = a boolean indicating whether to save the model after each
    epoch if it is the best
    a model = considered the best if its validation loss is the lowest that
    the model has obtained
    filepath = the file path where the model should be saved

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

    # add save best to save the best model
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                     save_best_only=True))

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
