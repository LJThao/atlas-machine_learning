#!/usr/bin/env python3
"""Save and Load Configuration Module"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function saves a model's configuration in JSON format:

    network = the model whose configuration should be saved
    filename = the path of the file that the configuration should be saved to
    Returns: None

    """
    with open(filename, "w") as f:
        f.write(network.to_json())

    return (None)


def load_config(filename):
    """Function loads a model with a specific configuration:

    filename is the path of the file containing the model's configuration
    in JSON format
    Returns: the loaded model

    """
    with open(filename, "r") as f:
        model = f.read()

    loaded_model = K.models.model_from_json(model)

    return (loaded_model)
