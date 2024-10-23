#!/usr/bin/env python3
"""Save and Load Model Module"""
import tensorflow.keras as K


def save_model(network, filename):
    """Function saves an entire model:
    
    network = the model to save
    filename = the path of the file that the model should be saved to
    Returns: None

    """
    # saves the model
    network.save(filename)

    return (None)

def load_model(filename):
    """Function loads an entire model:

    filename = the path of the file that the model should be loaded from
    Returns: the loaded model

    """
    # loads the model
    loaded_model = K.models.load_model(filename)

    return (loaded_model)
